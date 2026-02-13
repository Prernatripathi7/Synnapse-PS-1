import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query

from feature_extraction import ImageEncoder
from similarity_scoring_and_retrieval.retriever import FaissRetriever
from build_model import build_model


app = FastAPI(title="Synnapse Retrieval API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# File paths (Render will use these)
CKPT_PATH = os.getenv("CKPT_PATH", "checkpoints/best_model.pt")
EMB_PATH  = os.getenv("EMB_PATH",  "features/gallery_embeddings.npy")
IDS_PATH  = os.getenv("IDS_PATH",  "features/gallery_item_ids.npy")
REFS_PATH = os.getenv("REFS_PATH", "features/gallery_refs.npy")

encoder = None
retriever = None
gallery_refs = None


def pil_to_tensor_rgb(pil_img, size=224):
    pil_img = pil_img.convert("RGB").resize((size, size))
    arr = np.asarray(pil_img).astype("float32") / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw)


@app.on_event("startup")
def startup():
    global encoder, retriever, gallery_refs

    print("🚀 Loading model and FAISS index...")

    model = build_model()
    encoder = ImageEncoder(model=model, ckpt_path=CKPT_PATH, device=DEVICE, normalize=True)
    retriever = FaissRetriever(emb_path=EMB_PATH, ids_path=IDS_PATH, normalize=True)

    if os.path.exists(REFS_PATH):
        gallery_refs = np.load(REFS_PATH, allow_pickle=True)
    else:
        gallery_refs = None

    print("✅ API ready on device:", DEVICE)


@app.post("/search")
@torch.no_grad()
def search(file: UploadFile = File(...), k: int = Query(5, ge=1, le=50)):

    img_bytes = file.file.read()
    pil = Image.open(io.BytesIO(img_bytes))

    x = pil_to_tensor_rgb(pil)
    emb = encoder.encode_batch(x.unsqueeze(0).to(DEVICE)).numpy()[0]

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
