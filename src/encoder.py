"""Frozen feature extractor (Preprocessing stage in the diagram).

Pluggable backbone: CLIP (open_clip) or DINOv2 (timm/torch.hub) if installed,
otherwise a torchvision ResNet50 fallback so the pipeline always runs.

All encoders return an L2-normalizable embedding of shape [B, d]; the embedding
dimension is exposed via `.dim`. Weights are frozen (eval mode, no grad).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet stats used by ResNet/DINO; CLIP ships its own transform but these are close.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FrozenEncoder(nn.Module):
    """Wraps a frozen image backbone and normalizes inputs internally.

    Expects input images in [0, 1], shape [B, 3, H, W]. Resizes to `resize`
    and applies the backbone-appropriate normalization before the forward pass.
    """

    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.encoder
        self.resize = cfg.encoder_resize
        self.model, self.dim, mean, std = _build_backbone(cfg)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.resize:
            x = F.interpolate(x, size=self.resize, mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        if self.kind == "clip":
            feats = self.model.encode_image(x)
        elif self.kind == "dino":
            feats = self.model(x)
        else:  # resnet
            feats = self.model(x).flatten(1)
        return feats.float()


def _build_backbone(cfg):
    """Returns (module, dim, mean, std). Falls back to ResNet50 if deps missing."""
    kind = cfg.encoder

    if kind == "clip":
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                cfg.clip_model, pretrained=cfg.clip_pretrained)
            dim = model.visual.output_dim
            return model, dim, CLIP_MEAN, CLIP_STD
        except Exception as e:  # pragma: no cover - depends on optional install
            print(f"[encoder] CLIP unavailable ({e}); falling back to ResNet50.")
            kind = "resnet"

    if kind == "dino":
        try:
            model = torch.hub.load("facebookresearch/dinov2", cfg.dino_model)
            dim = model.embed_dim
            return model, dim, IMAGENET_MEAN, IMAGENET_STD
        except Exception as e:  # pragma: no cover
            print(f"[encoder] DINOv2 unavailable ({e}); falling back to ResNet50.")
            kind = "resnet"

    # torchvision ResNet50, penultimate features (2048-d), always available.
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=weights)
    net.fc = nn.Identity()
    return net, 2048, IMAGENET_MEAN, IMAGENET_STD


def build_encoder(cfg) -> FrozenEncoder:
    return FrozenEncoder(cfg)
