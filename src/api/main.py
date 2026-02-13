import os
import io
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel

# your project imports
from feature_extraction import ImageEncoder
from similarity_scoring_and_retrieval.retriever import FaissRetriever

# ---- YOU must provide this in your notebook or create a build_model.py ----
# from your_training_code import build_model


app = FastAPI(title="Synnapse Module B Retrieval API")

# Global objects loaded once
encoder = None
retriever = None
gallery_refs = None

# Paths (set these correctly)
CKPT_PATH = os.getenv("CKPT_PATH", "checkpoints/best_model.pt")
EMB_PATH  = os.getenv("EMB_PATH", "features/gallery_embeddings.npy")
IDS_PATH  = os.getenv("IDS_PATH", "features/gallery_item_ids.npy")
REFS_PATH = os.getenv("REFS_PATH", "features/gallery_refs.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pil_to_tensor_rgb(pil_img, size=224):
    """
    Minimal transform: resize + to float tensor in [0,1], CHW.
    Replace with your exact training transforms if needed.
    """
    pil_img = pil_img.convert("RGB").resize((size, size))
    arr = np.asarray(pil_img).astype("float32") / 255.0  # HWC
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw)


@app.on_event("startup")
def startup():
    global encoder, retriever, gallery_refs

    # ---- IMPORTANT: you must define build_model() somewhere importable ----
    from build_model import build_model  # create this file or adjust import

    model = build_model()
    encoder = ImageEncoder(model=model, ckpt_path=CKPT_PATH, device=DEVICE, normalize=True)

    retriever = FaissRetriever(emb_path=EMB_PATH, ids_path=IDS_PATH, normalize=True)

    if os.path.exists(REFS_PATH):
        gallery_refs = np.load(REFS_PATH, allow_pickle=True)
    else:
        gallery_refs = None

    print("✅ API ready")
    print("Device:", DEVICE)
    print("CKPT:", CKPT_PATH)
    print("EMB:", EMB_PATH)


@app.post("/search")
@torch.no_grad()
def search(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=50)
):
    """
    Upload an image -> returns Top-K matches with scores.
    """
    if encoder is None or retriever is None:
        return {"error": "Server not initialized"}

    img_bytes = file.file.read()
    pil = Image.open(io.BytesIO(img_bytes))

    x = pil_to_tensor_rgb(pil)                    # [C,H,W]
    emb = encoder.encode_batch(x.unsqueeze(0).to(DEVICE)).numpy()[0]  # [D]

    top_ids, scores, idxs = retriever.search(emb, k=k)

    results = []
    for rank, (iid, score, row_idx) in enumerate(zip(top_ids, scores, idxs), start=1):
        ref = None
        if gallery_refs is not None and int(row_idx) < len(gallery_refs):
            ref = gallery_refs[int(row_idx)]
        results.append({
            "rank": rank,
            "item_id": str(iid),
            "score": float(score),
            "row_index": int(row_idx),
            "ref": None if ref is None else str(ref)
        })

    return {"k": k, "results": results}
