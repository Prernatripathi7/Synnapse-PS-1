import os
import io
import time
import threading
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from huggingface_hub import hf_hub_download

from src.build_model import build_model
from src.feature_extraction import ImageEncoder
from src.similarity_scoring_and_retrieval import FaissRetriever

app = FastAPI(title="Synnapse Retrieval API")

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = os.getenv("HF_REPO_ID", "mugglewizard/Synnapse-ps1-checkpoints")

CKPT_IN_REPO = os.getenv("HF_CKPT_PATH_IN_REPO", "models/reid_model3_best.pth")
EMB_IN_REPO  = os.getenv("HF_EMB_PATH_IN_REPO",  "features/gallery_embeddings.npy")
IDS_IN_REPO  = os.getenv("HF_IDS_PATH_IN_REPO",  "features/gallery_item_ids.npy")
REFS_IN_REPO = os.getenv("HF_REFS_PATH_IN_REPO", "features/gallery_refs.npy")

MODELS_DIR   = os.getenv("MODELS_DIR", "models")
FEATURES_DIR = os.getenv("FEATURES_DIR", "features")

CKPT_PATH = os.path.join(MODELS_DIR, "reid_model3_best.pth")
EMB_PATH  = os.path.join(FEATURES_DIR, "gallery_embeddings.npy")
IDS_PATH  = os.path.join(FEATURES_DIR, "gallery_item_ids.npy")
REFS_PATH = os.path.join(FEATURES_DIR, "gallery_refs.npy")

# -------------------------
# Global state (lazy-loaded)
# -------------------------
encoder = None
retriever = None
gallery_refs = None

_init_lock = threading.Lock()
_init_started_at = None
_init_error = None


def ensure_hf_file(path_in_repo: str, local_path: str):
    """
    Download file from HF if local_path doesn't exist.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return local_path

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=path_in_repo,
        repo_type="model",
        local_dir="hf_cache_downloads",
        local_dir_use_symlinks=False,
    )

    # Copy to a stable location for your code
    if downloaded != local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(downloaded, "rb") as src_f, open(local_path, "wb") as dst_f:
            dst_f.write(src_f.read())

    return local_path


def preprocess_pil(pil_img, size=224):
    pil_img = pil_img.convert("RGB").resize((size, size))
    x = np.asarray(pil_img).astype(np.float32) / 255.0  # HWC
    x = np.transpose(x, (2, 0, 1))                      # CHW
    x = torch.from_numpy(x)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)  # [1,3,H,W]


def init_everything_if_needed():
    """
    Lazy initialization: downloads files + builds model + loads FAISS.
    Runs once (thread-safe). If it fails, stores error.
    """
    global encoder, retriever, gallery_refs, _init_started_at, _init_error

    if encoder is not None and retriever is not None and gallery_refs is not None:
        return

    with _init_lock:
        # double-check after acquiring lock
        if encoder is not None and retriever is not None and gallery_refs is not None:
            return

        if _init_started_at is None:
            _init_started_at = time.time()

        try:
            # 1) Download assets if missing
            ensure_hf_file(CKPT_IN_REPO, CKPT_PATH)
            ensure_hf_file(EMB_IN_REPO,  EMB_PATH)
            ensure_hf_file(IDS_IN_REPO,  IDS_PATH)
            ensure_hf_file(REFS_IN_REPO, REFS_PATH)

            # 2) Build model architecture
            model = build_model(
                variant="dinov2_vits14",
                emb_dim=512,
                num_classes=1,   # classifier head not needed for retrieval
                device=DEVICE
            )

            # 3) Load checkpoint into encoder wrapper
            encoder_local = ImageEncoder(
                model=model,
                ckpt_path=CKPT_PATH,
                device=DEVICE,
                normalize=True
            )

            # 4) Load FAISS index + refs
            retriever_local = FaissRetriever(
                emb_path=EMB_PATH,
                ids_path=IDS_PATH,
                normalize=True
            )
            refs_local = np.load(REFS_PATH, allow_pickle=True)

            # Commit to globals only when everything succeeded
            encoder = encoder_local
            retriever = retriever_local
            gallery_refs = refs_local

        except Exception as e:
            _init_error = repr(e)
            raise


@app.get("/health")
def health():
    ready = (encoder is not None and retriever is not None and gallery_refs is not None)
    return {
        "status": "ok",
        "ready": ready,
        "device": str(DEVICE),
        "hf_repo": HF_REPO_ID,
        "init_started_seconds_ago": (time.time() - _init_started_at) if _init_started_at else None,
        "init_error": _init_error,
        "gallery_size": int(retriever.emb.shape[0]) if retriever is not None else 0,
    }


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=50),
    extra: int = Query(200, ge=0, le=2000),
):
    """
    Upload image -> returns Top-K nearest items with similarity scores and refs.
    Lazy-loads model/index on first request to avoid Render startup timeout.
    """
    # lazy init
    try:
        init_everything_if_needed()
    except Exception:
        return {"error": "Initialization failed", "detail": _init_error}

    data = await file.read()
    pil = Image.open(io.BytesIO(data))

    x = preprocess_pil(pil).to(DEVICE)
    q_emb = encoder.encode_batch(x)[0].numpy().astype("float32")

    # search returns (top_ids, scores, idxs)
    top_ids_all, scores_all, idxs_all = retriever.search(q_emb, k=k, extra=extra)

    results = []
    for item_id, score, idx in zip(top_ids_all, scores_all, idxs_all):
        idx = int(idx)
        results.append({
            "item_id": str(item_id),
            "score": float(score),
            "ref": str(gallery_refs[idx]),
        })
        if len(results) >= k:
            break

    return {"k": k, "results": results}
