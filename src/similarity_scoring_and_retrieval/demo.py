import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.feature_extraction import ImageEncoder
from src.similarity_scoring_and_retrieval.retriever import FaissRetriever


def preprocess_pil(pil_img, size=224):
    pil_img = pil_img.convert("RGB").resize((size, size))
    x = np.asarray(pil_img).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std

    return x.unsqueeze(0)


@torch.no_grad()
def run_retrieval_demo_any_image(
    model_ctor,
    ckpt_path,
    query_image_path,
    k=5,
    device="cuda",
    emb_path="features/gallery_embeddings.npy",
    ids_path="features/gallery_item_ids.npy",
    refs_path="features/gallery_refs.npy",
    output_path="sample_output/retrieval_demo.png",
    extra=300,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = model_ctor()
    encoder = ImageEncoder(model=model, ckpt_path=ckpt_path, device=device, normalize=True)

    retriever = FaissRetriever(emb_path=emb_path, ids_path=ids_path, normalize=True)
    gallery_refs = np.load(refs_path, allow_pickle=True)

    pil_q = Image.open(query_image_path).convert("RGB")
    x = preprocess_pil(pil_q).to(device)
    q_emb = encoder.encode_batch(x)[0].cpu().numpy().astype("float32")

    top_ids, scores, idxs = retriever.search(q_emb, k=k, extra=extra)

    top_ids = top_ids[:k]
    scores = scores[:k]
    idxs = idxs[:k]

    plt.figure(figsize=(3 * (len(idxs) + 1), 4))

    plt.subplot(1, len(idxs) + 1, 1)
    plt.imshow(pil_q)
    plt.title("Query")
    plt.axis("off")

    for i, idx in enumerate(idxs):
        img = Image.open(str(gallery_refs[int(idx)])).convert("RGB")
        plt.subplot(1, len(idxs) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"{top_ids[i]}\n{scores[i]:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    return top_ids, scores, idxs
