"""Small training utilities: EMA, seeding, image-grid saving."""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model parameters (used for sampling)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def save_grid(images, path, nrow=8):
    """`images`: [N,3,H,W] in [-1,1]. Saves a PNG grid (needs torchvision)."""
    from torchvision.utils import save_image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image((images + 1) / 2, path, nrow=nrow)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
