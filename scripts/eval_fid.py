"""Evaluation: FID of generated images vs. the CIFAR test set.

Generates a folder of samples (conditioned on test-image targets) and scores
them with clean-fid against the built-in CIFAR statistics. This is the headline
metric for the GNN-vs-raw-vs-label ablation.

Requires: pip install clean-fid

Usage:
    python scripts/eval_fid.py --ckpt runs/gnn/last.pt --num 10000
"""
import argparse
import os

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import RawImages, to_uint8
from src.encoder import build_encoder
from src.flow import FlowModel
from src.knn import load_bank
from src.retriever import Retriever
from sample import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = load_cfg(ckpt["cfg"])
    bank, labels = load_bank(cfg.emb_path)
    retriever = Retriever(cfg, bank, labels)

    model = FlowModel(cfg, bank.shape[1]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    encoder = build_encoder(cfg).to(device).eval() if cfg.cond_mode != "none" else None
    test = RawImages(cfg, train=False)
    n = min(args.num, len(test))

    out_dir = args.out_dir or os.path.join(cfg.out_dir, "fid_samples")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image

    written = 0
    for start in tqdm(range(0, n, args.batch), desc="sampling"):
        idxs = list(range(start, min(start + args.batch, n)))
        if cfg.cond_mode == "none":
            batch = None
        else:
            imgs = torch.stack([test[i][0] for i in idxs]).to(device)
            with torch.no_grad():
                c = encoder(imgs)
                if cfg.emb_normalize:
                    c = F.normalize(c, dim=1)
            batch = retriever.batch_from_embeddings(c, device)
        gen = model.sample(batch=batch, n=len(idxs), guidance=args.guidance, device=device)
        gen = to_uint8(gen).permute(0, 2, 3, 1).cpu().numpy()
        for img in gen:
            Image.fromarray(img).save(os.path.join(out_dir, f"{written:06d}.png"))
            written += 1

    print(f"[eval] wrote {written} samples -> {out_dir}")
    try:
        from cleanfid import fid
        name = "cifar100" if cfg.dataset == "cifar100" else "cifar10"
        score = fid.compute_fid(out_dir, dataset_name=name, dataset_res=32,
                                dataset_split="test", mode="clean")
        print(f"[eval] FID ({name}, n={written}): {score:.2f}")
    except ImportError:
        print("[eval] clean-fid not installed; samples written. "
              "pip install clean-fid to score.")


if __name__ == "__main__":
    main()
