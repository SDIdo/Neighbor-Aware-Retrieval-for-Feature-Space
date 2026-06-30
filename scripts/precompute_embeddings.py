"""Stage 0 (Preprocessing): build and cache the frozen-encoder embedding bank.

Runs every CIFAR train image through the frozen encoder once and stores an
[N, d] embedding matrix (L2-normalized) + labels to an .npz. Everything
downstream reads this file, so the encoder never runs during training.

Usage:
    python scripts/precompute_embeddings.py --dataset cifar10 --encoder clip
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config, add_cli, config_from_args
from src.dataset import raw_loader
from src.encoder import build_encoder


def main():
    parser = add_cli(argparse.ArgumentParser())
    cfg = config_from_args(parser.parse_args())
    device = cfg.device if torch.cuda.is_available() else "cpu"

    encoder = build_encoder(cfg).to(device)
    print(f"[precompute] encoder={cfg.encoder} dim={encoder.dim} device={device}")

    loader = raw_loader(cfg, train=True, batch_size=256)
    embs, labels = [], []
    for img, _, label in tqdm(loader, desc="encoding"):
        feats = encoder(img.to(device))
        if cfg.emb_normalize:
            feats = F.normalize(feats, dim=1)
        embs.append(feats.cpu())
        labels.append(label)

    emb = torch.cat(embs).numpy().astype("float32")
    lab = torch.cat(labels).numpy().astype("int64")
    np.savez(cfg.emb_path, emb=emb, labels=lab, dim=encoder.dim)
    print(f"[precompute] saved {emb.shape} -> {cfg.emb_path}")


if __name__ == "__main__":
    main()
