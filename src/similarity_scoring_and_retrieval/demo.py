import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import ImageEncoder
from similarity_scoring_and_retrieval.retriever import FaissRetriever


def _extract_sample(sample):
    # supports tuple/list: (img, item_id, ...)
    # supports dict: {"image":..., "item_id":...}
    if isinstance(sample, dict):
        return sample["image"], sample["item_id"]
    return sample[0], sample[1]


def _to_hwc(img):
    img = img.detach().cpu().float()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def run_retrieval_demo(
    model_ctor,
    ckpt_path,
    dataset,
    query_index=0,
    k=5,
    device="cuda",
    emb_path="features/gallery_embeddings.npy",
    ids_path="features/gallery_item_ids.npy",
    output_path="sample_output/sample_retrieval.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load encoder (best model)
    model = model_ctor()
    encoder = ImageEncoder(model, ckpt_path, device)

    # Load retriever
    retriever = FaissRetriever(emb_path, ids_path)

    # Query image
    sample = dataset[query_index]
    q_img, q_id = _extract_sample(sample)

    q_emb = encoder.encode_batch(q_img.unsqueeze(0).to(device))
    q_emb = q_emb.numpy().astype("float32")[0]

    top_ids, scores, idxs = retriever.search(q_emb, k=k)

    print("Query ID:", q_id)
    print("Top IDs:", top_ids)
    print("Scores:", scores)
    print("Gallery idxs:", idxs)

    # Collect retrieved images
    retrieved_imgs = []
    retrieved_ids = []
    for idx in idxs:
        img, iid = _extract_sample(dataset[int(idx)])
        retrieved_imgs.append(img)
        retrieved_ids.append(iid)

    # Plot: Query + Top-K
    plt.figure(figsize=(3*(k+1), 4))

    # Query
    plt.subplot(1, k+1, 1)
    plt.imshow(_to_hwc(q_img))
    plt.title(f"Query\nID={q_id}")
    plt.axis("off")

    # Results with highlight
    for i in range(k):
        plt.subplot(1, k+1, i+2)
        plt.imshow(_to_hwc(retrieved_imgs[i]))

        is_correct = (top_ids[i] == q_id)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_edgecolor("green" if is_correct else "red")

        plt.title(f"ID={top_ids[i]}\n{scores[i]:.3f}",
                  color=("green" if is_correct else "red"))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print("✅ Saved to:", output_path)
    return top_ids, scores, idxs
