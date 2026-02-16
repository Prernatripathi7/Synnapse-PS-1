import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from feature_extraction import ImageEncoder
from similarity_scoring_and_retrieval.retriever import FaissRetriever


def _extract_sample(sample):
    """
    Supports:
      - dict: {"image":..., "item_id":..., optional "index"/"idx"/"path"/"img_path"/"image_path"}
      - tuple/list: (image, item_id, optional ref)
    Returns: (image_tensor_CHW, item_id, ref_or_None)
    """
    if isinstance(sample, dict):
        img = sample["image"]
        item_id = sample["item_id"]
        ref = sample.get("index", sample.get("idx", None))
        if ref is None:
            ref = sample.get("path", sample.get("img_path", sample.get("image_path", None)))
        return img, item_id, ref

    # tuple/list
    img = sample[0]
    item_id = sample[1]
    ref = sample[2] if len(sample) >= 3 else None
    return img, item_id, ref


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
        img, _, _ = _extract_sample(dataset[int(ref)])
        return _to_hwc_tensor(img, mean=mean, std=std)

    # path
    if isinstance(ref, str):
        im = Image.open(ref).convert("RGB")
        return np.asarray(im)

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
    extra_neighbors=200,   # how many extra candidates to pull before filtering
):
    """
    Produces sample retrieval output:
      - Query image
      - Top-K retrieved images
      - Correct instances highlighted (green)
      - ✅ Excludes the query image itself (same ref/path) from retrieval results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load encoder (best model)
    model = model_ctor()
    encoder = ImageEncoder(model=model, ckpt_path=ckpt_path, device=device, normalize=True)

    # Load retriever
    retriever = FaissRetriever(emb_path, ids_path, normalize=True)

    # Load refs (mapping from embedding row -> dataset index OR file path)
    gallery_refs = np.load(refs_path, allow_pickle=True)

    # Query sample
    q_sample = dataset[query_index]
    q_img, q_id, q_ref = _extract_sample(q_sample)

    # Decide what we will use to exclude self:
    # Prefer explicit ref from dataset item; else fallback to query_index
    query_ref = q_ref if q_ref is not None else query_index

    # Encode query
    q_emb = encoder.encode_batch(q_img.unsqueeze(0).to(device))
    q_emb = q_emb.numpy().astype("float32")[0]

    # --- Retrieve MORE than k, then filter out self ---
    # IMPORTANT: this requires your FaissRetriever.search supports extra neighbors.
    # If your retriever.search does NOT have 'extra', we will call it with (k + extra_neighbors)
    try:
        top_ids_all, scores_all, idxs_all = retriever.search(q_emb, k=k, extra=extra_neighbors)
    except TypeError:
        top_ids_all, scores_all, idxs_all = retriever.search(q_emb, k=k + extra_neighbors)

    filtered = []
    for tid, sc, row in zip(top_ids_all, scores_all, idxs_all):
        row = int(row)
        ref = gallery_refs[row] if row < len(gallery_refs) else None

        # ✅ exclude exact same image ref/path/index
        if ref == query_ref:
            continue

        filtered.append((tid, float(sc), row))
        if len(filtered) == k:
            break

    if len(filtered) < k:
        print(f"⚠️ Only found {len(filtered)} results after excluding self. "
              f"Increase extra_neighbors (currently {extra_neighbors}).")

    top_ids = np.array([x[0] for x in filtered], dtype=object)
    scores = np.array([x[1] for x in filtered], dtype=float)
    idxs = np.array([x[2] for x in filtered], dtype=int)

    print("Query index:", query_index, "Query ID:", q_id, "Query ref used for exclusion:", query_ref)
    print("Top IDs (self-excluded):", top_ids)
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
    plt.figure(figsize=(3 * (len(idxs) + 1), 4))

    plt.subplot(1, len(idxs) + 1, 1)
    plt.imshow(query_disp)
    plt.title(f"Query\nID={q_id}")
    plt.axis("off")

    for i in range(len(idxs)):
        plt.subplot(1, len(idxs) + 1, i + 2)

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
