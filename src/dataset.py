"""CIFAR data loading.

Two views of the same dataset:
  * `raw_loader`   : images in [0, 1], used by the frozen encoder to build the
                     embedding bank (encoder applies its own normalization).
  * `train_loader` : images in [-1, 1] for the flow model, plus the sample's
                     bank index and label so the training loop can retrieve its
                     precomputed embedding and KNN neighbors.

Bank row i corresponds to CIFAR train index i, so no re-encoding is needed
during training (the encoder is frozen, exactly as in the diagram).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def _cifar(cfg, train=True):
    cls = datasets.CIFAR100 if cfg.dataset == "cifar100" else datasets.CIFAR10
    return cls(root=cfg.data_root, train=train, download=True)


class RawImages(Dataset):
    """Returns (image[0,1], index, label) — for embedding precomputation."""

    def __init__(self, cfg, train=True):
        self.base = _cifar(cfg, train)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self.to_tensor(img), i, label


class FlowImages(Dataset):
    """Returns (image[-1,1], index, label) — for training the flow model."""

    def __init__(self, cfg, train=True):
        self.base = _cifar(cfg, train)
        self.tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self.tf(img), i, label


def raw_loader(cfg, train=True, batch_size=256):
    return DataLoader(RawImages(cfg, train), batch_size=batch_size, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=True)


def train_loader(cfg):
    return DataLoader(FlowImages(cfg, train=True), batch_size=cfg.batch_size,
                      shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                      drop_last=True)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> uint8 [0,255] for saving/eval."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
