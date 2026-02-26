import torch
import torch.nn.functional as F
def _clean_state_dict(sd):
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned

def _filter_mismatched_keys(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    dropped = []

    for k, v in state_dict.items():
        if k not in model_sd:
            dropped.append((k, "missing_in_model"))
            continue
        if hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                dropped.append((k, f"shape {tuple(v.shape)} != {tuple(model_sd[k].shape)}"))
                continue
        filtered[k] = v

    return filtered, dropped

class ImageEncoder:
    def __init__(self, model, ckpt_path, device, normalize=True):
        self.model = model.to(device)
        self.device = device
        self.normalize = normalize

        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            raise ValueError("Unknown checkpoint format. Expected dict or dict with 'model_state_dict'.")

        state_dict = _clean_state_dict(state_dict)
        state_dict, dropped = _filter_mismatched_keys(self.model, state_dict)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if dropped:
            print(f"[encoder] Dropped {len(dropped)} keys due to mismatch (OK for retrieval). Example:", dropped[:3])
        if missing:
            print(f"[encoder] Missing keys (OK if classifier differs): {len(missing)}")
        if unexpected:
            print(f"[encoder] Unexpected keys: {len(unexpected)}")

        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, imgs):
        imgs = imgs.to(self.device)
        output = self.model(imgs)
        emb = output[0] if isinstance(output, (tuple, list)) else output
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().cpu()
