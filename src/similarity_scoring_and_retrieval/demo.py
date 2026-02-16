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
      - dict: {"image":..., "item_id":..., optional "path"/"img_path"/"image_path"}
      - tuple/list: (image, item_id, optional path)
    Returns: (image_tensor_CHW, item_id, path_or_None)
    """
    if isinstance(sample, dict):
        img = sample["image"]
        item_id = sample["item_id"]
        path = sample.get("path", sample.get("img_path", sample.get("image_path", None)))
        return img, item_id, path

    img = sample[0]
    item_id = sample[1]
    path = sample[2] if len(sample) >= 3 else None
    return img, item_id, path


def _to_hwc_tensor(img_chw, mean=None, std=None):
    x = img_chw.detach().cpu().float()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        x = x * std_t + mean_t
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def _load_image_from_path(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def _get_query_path(dataset, query_index, q_path_from_sample=None):
    """
    We MUST match the exact path string format saved in gallery_refs.npy.
    Priority:
      1) path returned by dataset[query_index] (best)
      2) dataset.df.iloc[query_index]["image"] if available
    """
    if isinstance(q_path_from_sample, str) and len(q_path_from_sample) > 0:
        return q_path_from_sample

    if hasattr(dataset, "df"):
        df = dataset.df
        if "image" in df.columns:
            return str(df.iloc[int(query_index)]["image"])

    raise ValueError(
        "Could not determine query path. Make sure dataset[query_index] returns (img, id, path) "
        "or dataset has .df with column 'image'."
    )


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
    extra_neighbors=300,   # pull extra, then filter
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load encoder (best model)
    model = model_ctor()
    encoder = ImageEncoder(model=model, ckpt_path=ckpt_path, device=device, normalize=True)

    # Load retriever
    retriever = FaissRetriever(emb_path, ids_path, normalize=True)

    # Load refs (embedding row -> image path)
    gallery_refs = np.load(refs_path, allow_pickle=True)
    assert isinstance(gallery_refs[0], str), "Expected gallery_refs to be paths (strings)."

    # Query
    q_sample = dataset[int(query_index)]
    q_img, q_id, q_path_from_sample = _extract_sample(q_sample)

    query_path = _get_query_path(dataset, query_index, q_path_from_sample)

    # Encode query
    q_emb = encoder.encode_batch(q_img.unsqueeze(0).to(device))
    q_emb = q_emb.numpy().astype("float32")[0]

    # Search more than k
    # (your current retriever.search does NOT filter self, so we will)
    top_ids_all, scores_all, idxs_all = retriever.search(q_emb, k=k + extra_neighbors)

    # Filter out exact same path
    filtered = []
    for tid, sc, row in zip(top_ids_all, scores_all, idxs_all):
        row = int(row)
        ref_path = gallery_refs[row]
        if ref_path == query_path:
            continue  # ✅ exclude same exact image file
        filtered.append((tid, float(sc), row))
        if len(filtered) == k:
            break

    if len(filtered) < k:
        print(f"⚠️ Only got {len(filtered)} results after excluding self. "
              f"Try increasing extra_neighbors (now {extra_neighbors}).")

    top_ids = np.array([x[0] for x in filtered], dtype=object)
    scores = np.array([x[1] for x in filtered], dtype=float)
    idxs = np.array([x[2] for x in filtered], dtype=int)

    print("Query index:", query_index, "Query ID:", q_id)
    print("Query path:", query_path)
    print("Top IDs (self-excluded):", top_ids)
    print("Scores:", scores)
    print("FAISS row idxs:", idxs)
    print("Retrieved paths:", [gallery_refs[i] for i in idxs])

    # Display query
    query_disp = _to_hwc_tensor(q_img, mean=mean, std=std)

    # Display retrieved (paths)
    retrieved_disp = [_load_image_from_path(gallery_refs[int(r)]) for r in idxs]

    # Plot
    plt.figure(figsize=(3 * (len(idxs) + 1), 4))

    plt.subplot(1, len(idxs) + 1, 1)
    plt.imshow(query_disp)
    plt.title(f"Query\nID={q_id}")
    plt.axis("off")

    for i in range(len(idxs)):
        plt.subplot(1, len(idxs) + 1, i + 2)
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
