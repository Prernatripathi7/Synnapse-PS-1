import os
from huggingface_hub import HfApi
REPO_ID = os.getenv("HF_REPO_ID", "mugglewizard/Synnapse-ps1-checkpoints")

def main():
    api = HfApi()
    ckpt_path = "models/reid_model3_best.pth"
    if os.path.exists(ckpt_path):
        print("Uploading checkpoint:", ckpt_path)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo="models/reid_model3_best.pth",
            repo_id=REPO_ID,
            repo_type="model",
        )
    else:
        print("Checkpoint not found at:", ckpt_path)
    feats = [
        "features/gallery_embeddings.npy",
        "features/gallery_item_ids.npy",
        "features/gallery_refs.npy",
    ]
    for f in feats:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing feature file: {f}")

    for f in feats:
        print("Uploading:", f)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f,    
            repo_id=REPO_ID,
            repo_type="model",
        )

    print("Upload complete to:", REPO_ID)

if __name__ == "__main__":
    main()
