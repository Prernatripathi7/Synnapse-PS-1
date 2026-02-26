import torch
import torch.nn as nn
import torch.nn.functional as F
def load_dinov2_backbone(variant="dinov2_vits14", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = torch.hub.load("facebookresearch/dinov2", variant)
    backbone.eval()
    backbone.to(device)
    return backbone

class DINOv2Embedder(nn.Module):
    def __init__(self, dinov2_backbone, num_classes: int = 1, emb_dim: int = 512):
        super().__init__()
        self.backbone = dinov2_backbone
        feat_dim = self.backbone.embed_dim
        self.embedding = nn.Linear(feat_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        out = self.backbone.forward_features(x)
        feats = out["x_norm_clstoken"]
        emb = F.normalize(self.embedding(feats), p=2, dim=1)
        logits = self.classifier(emb)
        return emb, logits

def build_model(
    variant="dinov2_vits14",
    emb_dim=512,
    num_classes=1,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2_backbone(variant=variant, device=device)
    model = DINOv2Embedder(backbone, num_classes=num_classes, emb_dim=emb_dim)
    return model.to(device)
