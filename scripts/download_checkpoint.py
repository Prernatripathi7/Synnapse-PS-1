import os
from huggingface_hub import hf_hub_download

REPO_ID = "mugglewizard/Synnapse-ps1-checkpoints"
FILENAME = "reid_model3_best.pth"

def main():
    os.makedirs("models", exist_ok=True)

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir="models",
        local_dir_use_symlinks=False,
    )

    print("Checkpoint downloaded to:", path)

if __name__ == "__main__":
    main()
