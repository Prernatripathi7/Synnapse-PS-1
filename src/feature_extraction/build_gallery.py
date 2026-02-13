import os
import numpy as np
import torch
from tqdm import tqdm
from feature_extraction import ImageEncoder

def build_gallery_database(model_ctor, ckpt_path, gallery_loader, device, out_dir="features"):
    """
    Creates the embedding database:
      - gallery_embeddings.npy (N,D)
      - gallery_item_ids.npy   (N,)
    """
    os.makedirs(out_dir, exist_ok=True)

    model = model_ctor()
    encoder = ImageEncoder(model=model, ckpt_path=ckpt_path, device=device, normalize=True)

    all_emb = []
    all_ids = []

    for batch in tqdm(gallery_loader, desc="Extracting gallery embeddings"):
        # Works for tuple/list batch: (imgs, item_ids, ...)
        imgs = batch[0].to(device)
        item_ids = batch[1]

        emb = encoder.encode_batch(imgs)                 # [B,D] on CPU
        all_emb.append(emb.numpy().astype("float32"))

        if torch.is_tensor(item_ids):
            item_ids = item_ids.detach().cpu().tolist()
        elif isinstance(item_ids, np.ndarray):
            item_ids = item_ids.tolist()

        all_ids.extend(item_ids)

    gallery_emb = np.concatenate(all_emb, axis=0).astype("float32")
    gallery_ids = np.array(all_ids)

    np.save(os.path.join(out_dir, "gallery_embeddings.npy"), gallery_emb)
    np.save(os.path.join(out_dir, "gallery_item_ids.npy"), gallery_ids)

    print("✅ Saved embedding database:")
    print(" -", os.path.join(out_dir, "gallery_embeddings.npy"), gallery_emb.shape)
    print(" -", os.path.join(out_dir, "gallery_item_ids.npy"), gallery_ids.shape)

    return gallery_emb, gallery_ids
