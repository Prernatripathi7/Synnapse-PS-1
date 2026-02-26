import os
import io
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from huggingface_hub import hf_hub_download

# ✅ IMPORTANT: use package imports (works in Colab + deployment)
from src.build_model import build_model
from src.feature_extraction import ImageEncoder
from src.similarity_scoring_and_retrieval import FaissRetriever

app = FastAPI(title="Synnapse Retrieval API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- HF Repo that stores BOTH checkpoint + features ----------
HF_REPO_ID = os.getenv("HF_REPO_ID", "mugglewizard/Synnapse-ps1-checkpoints")

# Paths inside HF repo (keep these exactly matching your HF repo structure)
CKPT_IN_REPO = os.getenv("HF_CKPT_PATH_IN_REPO", "models/reid_model3_best.pth")
EMB_IN_REPO  = os.getenv("HF_EMB_PATH_IN_REPO",  "features/gallery_embeddings.npy")
IDS_IN_REPO  = os.getenv("HF_IDS_PATH_IN_REPO",  "features/gallery_item_ids.npy")
REFS_IN_REPO = os.getenv("HF_REFS_PATH_IN_REPO", "features/gallery_refs.npy")

# Local folders
MODELS_DIR   = os.getenv("MODELS_DIR", "models")
FEATURES_DIR = os.getenv("FEATURES_DIR", "features")

# Local output paths
CKPT_PATH = os.path.join(MODELS_DIR, "reid_model3_best.pth")
EMB_PATH  = os.path.join(FEATURES_DIR, "gallery_embeddings.npy")
IDS_PATH  = os.path.join(FEATURES_DIR, "gallery_item_ids.npy")
REFS_PATH = os.path.join(FEATURES_DIR, "gallery_refs.npy")

# Globals loaded once at startup
encoder = None
retriever = None
gallery_refs = None


def ensure_hf_file(path_in_repo: str, local_path: str):
    """
    Download file from HF if local_path doesn't exist.
    hf_hub_download returns the downloaded file path.
    We then copy/move it to local_path for consistent paths.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return local_path

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=path_in_repo,          # includes subfolders
        repo_type="model",
        local_dir="hf_cache_downloads", # keeps cache in a folder
        local_dir_use_symlinks=False,
    )

    # Copy to local_path (so our code always uses the same local paths)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if downloaded != local_path:
        # copy bytes (safer than rename across filesystems)
        with open(downloaded, "rb") as src_f, open(local_path, "wb") as dst_f:
            dst_f.write(src_f.read())

    return local_path


def preprocess_pil(pil_img, size=224):
    """
    Minimal preprocessing: resize -> tensor -> ImageNet normalize
    """
    pil_img = pil_img.convert("RGB").resize((size, size))
    x = np.asarray(pil_img).astype(np.float32) / 255.0  # HWC
    x = np.transpose(x, (2, 0, 1))                      # CHW
    x = torch.from_numpy(x)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)  # [1,3,H,W]


@app.on_event("startup")
def startup():
    """
    Runs once when server starts:
      1) download checkpoint + features from HF
      2) build model
      3) load weights into ImageEncoder
      4) build FAISS index from gallery embeddings
    """
    global encoder, retriever, gallery_refs

    # 1) Download assets if missing
    ensure_hf_file(CKPT_IN_REPO, CKPT_PATH)
    ensure_hf_file(EMB_IN_REPO,  EMB_PATH)
    ensure_hf_file(IDS_IN_REPO,  IDS_PATH)
    ensure_hf_file(REFS_IN_REPO, REFS_PATH)

    # 2) Build model architecture
    model = build_model(
        variant="dinov2_vits14",
        emb_dim=512,
        num_classes=1,   # classifier not used for retrieval, keep 1
        device=DEVICE
    )

    # 3) Load trained weights (embedding layer + backbone etc.)
    encoder = ImageEncoder(model=model, ckpt_path=CKPT_PATH, device=DEVICE, normalize=True)

    # 4) Build FAISS index
    retriever = FaissRetriever(emb_path=EMB_PATH, ids_path=IDS_PATH, normalize=True)

    # 5) Load refs/paths
    gallery_refs = np.load(REFS_PATH, allow_pickle=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "has_encoder": encoder is not None,
        "has_retriever": retriever is not None,
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
    """
    if encoder is None or retriever is None or gallery_refs is None:
        return {"error": "Server not ready yet (startup still running?)"}

    data = await file.read()
    pil = Image.open(io.BytesIO(data))

    x = preprocess_pil(pil).to(DEVICE)

    # Extract embedding
    emb = encoder.encode_batch(x)[0].numpy().astype("float32")

    # Search FAISS (ask for extra neighbors for safer filtering if needed)
    top_ids_all, scores_all, idxs_all = retriever.search(emb, k=k, extra=extra)

    results = []
    for item_id, score, idx in zip(top_ids_all, scores_all, idxs_all):
        idx = int(idx)
        results.append({
            "item_id": str(item_id),
            "score": float(score),
            "ref": str(gallery_refs[idx])  # path/reference of retrieved image
        })
        if len(results) >= k:
            break

    return {"k": k, "results": results}
