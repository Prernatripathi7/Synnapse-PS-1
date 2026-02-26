import os
import numpy as np
import torch
from tqdm import tqdm
from feature_extraction import ImageEncoder
def _extract_from_batch(batch):
    if isinstance(batch, dict):
        imgs = batch.get("image", batch.get("images"))
        item_ids = batch.get("item_id", batch.get("item_ids"))
        refs = batch.get("index", batch.get("idx", None))
        if refs is None:
            refs = batch.get("path", batch.get("img_path", None))
        return imgs, item_ids, refs
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must be at least (images, item_id).")
        imgs = batch[0]
        item_ids = batch[1]
        refs = batch[2] if len(batch) >= 3 else None
        return imgs, item_ids, refs
    raise TypeError(f"Unsupported batch type: {type(batch)}")
def _to_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

@torch.no_grad()
def build_gallery_database(
    model_ctor,
    ckpt_path,
    gallery_loader,
    device,
    out_dir="features",
    normalize=True
):
    os.makedirs(out_dir, exist_ok=True)
    model = model_ctor()
    encoder = ImageEncoder(model=model, ckpt_path=ckpt_path, device=device, normalize=normalize)
    all_emb, all_ids, all_refs = [], [], []

    for batch in tqdm(gallery_loader, desc="Building gallery embeddings"):
        imgs, item_ids, refs = _extract_from_batch(batch)

        emb = encoder.encode_batch(imgs)
        all_emb.append(emb.numpy().astype("float32"))

        item_ids_list = _to_list(item_ids)
        if item_ids_list is None:
            raise ValueError("item_ids missing in batch.")
        all_ids.extend(item_ids_list)

        refs_list = _to_list(refs)
        B = imgs.shape[0]
        if refs_list is None:
            all_refs.extend([None] * B)
        else:
            if len(refs_list) == 1 and B > 1:
                refs_list = refs_list * B
            if len(refs_list) != B:
                raise ValueError(f"refs length {len(refs_list)} != batch size {B}")
            all_refs.extend(refs_list)

    gallery_emb = np.concatenate(all_emb, axis=0).astype("float32")
    gallery_ids = np.array(all_ids, dtype=object)
    gallery_refs = np.array(all_refs, dtype=object)

    np.save(os.path.join(out_dir, "gallery_embeddings.npy"), gallery_emb)
    np.save(os.path.join(out_dir, "gallery_item_ids.npy"), gallery_ids)
    np.save(os.path.join(out_dir, "gallery_refs.npy"), gallery_refs)

    print("Saved gallery DB in:", out_dir)
    print(" - gallery_embeddings.npy:", gallery_emb.shape)
    print(" - gallery_item_ids.npy:", gallery_ids.shape)
    print(" - gallery_refs.npy:", gallery_refs.shape)

    return gallery_emb, gallery_ids, gallery_refs
