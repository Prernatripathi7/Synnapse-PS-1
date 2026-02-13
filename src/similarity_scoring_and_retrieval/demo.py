import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from feature_extraction import ImageEncoder
from similarity_scoring_and_retrieval.retriever import FaissRetriever


def _extract_sample(sample):
    # dataset item supports dict or tuple/list
    if isinstance(sample, dict):
        return sample["image"], sample["item_id"]
    return sample[0], sample[1]


def _to_hwc_tensor(img_chw, mean=None, std=None):
    x = img_chw.detach().cpu().float()

    # optional unnormalize for display
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        x = x * std_t + mean_t

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def _load_image_from_ref(dataset, ref, mean=None, std=None):
    """
    ref can be:
      - int : dataset index
      - str : path to image file
      - None: cannot display reliably
    Returns displayable HWC numpy image.
    """
    if ref is None:
        return None

    # dataset index
    if isinstance(ref, (int, np.integer)):
        img, _ = _extract_sample(dataset[int(ref)])
        return _to_hwc_tensor(img, mean=mean, std=std)

    # path
    if isinstance(ref, str):
        im = Image.open(ref).convert("RGB")
        return np.asarray(im)

    # unknown type
    return None


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
    refs_path="features/gallery_refs.npy",
    output_path="sample_output/sample_retrieval.png",
    mean=None,
    std=None,
):
    """
    Produces sample retrieval output:
      - Query image
      - Top-K retrieved images
      - Correct instances highlighted (green)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load encoder (best model)
    model = model_ctor()
    encoder = ImageEncoder(model=model, ckpt_path=ckpt_path, device=device, normalize=True)

    # Load retriever
    retriever = FaissRetriever(emb_path, ids_path, normalize=True)

    # Load refs (mapping from embedding row -> dataset index OR file path)
    gallery_refs = np.load(refs_path, allow_pickle=True)

    # Query
    q_sample = dataset[query_index]
    q_img, q_id = _extract_sample(q_sample)

    q_emb = encoder.encode_batch(q_img.unsqueeze(0).to(device))
    q_emb = q_emb.numpy().astype("float32")[0]

    top_ids, scores, idxs = retriever.search(q_emb, k=k)

    print("Query index:", query_index, "Query ID:", q_id)
    print("Top IDs:", top_ids)
    print("Scores:", scores)
    print("FAISS row idxs:", idxs)

    # Display query
    query_disp = _to_hwc_tensor(q_img, mean=mean, std=std)

    # Display retrieved using refs
    retrieved_disp = []
    for row_idx in idxs:
        ref = gallery_refs[int(row_idx)] if int(row_idx) < len(gallery_refs) else None
        img_disp = _load_image_from_ref(dataset, ref, mean=mean, std=std)
        retrieved_disp.append(img_disp)

    # Plot
    plt.figure(figsize=(3 * (k + 1), 4))

    plt.subplot(1, k + 1, 1)
    plt.imshow(query_disp)
    plt.title(f"Query\nID={q_id}")
    plt.axis("off")

    for i in range(k):
        plt.subplot(1, k + 1, i + 2)

        if retrieved_disp[i] is None:
            plt.text(0.5, 0.5, "No ref\n(saved)", ha="center", va="center")
        else:
            plt.imshow(retrieved_disp[i])

        is_correct = (top_ids[i] == q_id)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_edgecolor("green" if is_correct else "red")

        plt.title(f"ID={top_ids[i]}\n{scores[i]:.3f}", color=("green" if is_correct else "red"))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print("✅ Saved:", output_path)
    return top_ids, scores, idxs
