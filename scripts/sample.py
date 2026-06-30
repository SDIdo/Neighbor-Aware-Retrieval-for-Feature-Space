"""Inference stage: generate images conditioned on target images.

Mirrors the diagram's Inference row: target image -> frozen encoder -> c ->
KNN -> graph -> frozen GNN -> r_c -> frozen flow model -> generated image.
Everything is frozen; we load EMA weights.

Usage:
    python scripts/sample.py --ckpt runs/gnn/last.pt --num-samples 64
"""
import argparse
import os

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F

from config import Config
from src.dataset import RawImages
from src.encoder import build_encoder
from src.flow import FlowModel
from src.knn import load_bank
from src.retriever import Retriever
from src.utils import save_grid


def load_cfg(d):
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = load_cfg(ckpt["cfg"])
    bank, labels = load_bank(cfg.emb_path)
    retriever = Retriever(cfg, bank, labels)

    model = FlowModel(cfg, bank.shape[1]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    n = args.num_samples
    if cfg.cond_mode == "none":
        batch, originals = None, None
    else:
        # Encode unseen TEST images to obtain target embeddings c.
        encoder = build_encoder(cfg).to(device).eval()
        test = RawImages(cfg, train=False)
        imgs = torch.stack([test[i][0] for i in range(n)]).to(device)
        with torch.no_grad():
            c = encoder(imgs)
            if cfg.emb_normalize:
                c = F.normalize(c, dim=1)
        labs = torch.tensor([test[i][2] for i in range(n)])
        batch = retriever.batch_from_embeddings(c, device, labels=labs)
        originals = imgs * 2 - 1  # [-1,1] for side-by-side grid

    gen = model.sample(batch=batch, n=n, steps=args.steps,
                       guidance=args.guidance, device=device).cpu()

    out = args.out or os.path.join(cfg.out_dir, "inference.png")
    save_grid(gen, out, nrow=8)
    print(f"[sample] generated grid -> {out}")
    if originals is not None:
        save_grid(originals.cpu(), out.replace(".png", "_targets.png"), nrow=8)
        print("[sample] target grid -> " + out.replace('.png', '_targets.png'))


if __name__ == "__main__":
    main()
