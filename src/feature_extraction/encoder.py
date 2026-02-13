import torch
import torch.nn.functional as F

class ImageEncoder:
    def __init__(self, model, ckpt_path, device, normalize=True):
        self.model = model.to(device)
        self.device = device
        self.normalize = normalize

        checkpoint = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, imgs):
        """
        imgs: tensor [B, C, H, W]
        returns: embeddings [B, D] on CPU
        """
        imgs = imgs.to(self.device)

        output = self.model(imgs)
        emb = output[0] if isinstance(output, (tuple, list)) else output

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb.detach().cpu()
