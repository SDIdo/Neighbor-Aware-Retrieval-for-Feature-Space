"""Training stage: jointly train the GNN context network and the flow model.

Reads the cached embedding bank, retrieves KNN neighbors per batch, builds the
local graph, produces r_c, and trains the rectified-flow U-Net end-to-end. The
encoder stays frozen (it was used offline in precompute_embeddings.py).

Usage:
    python scripts/train.py --cond-mode gnn --epochs 200 --out-dir runs/gnn
"""
import argparse
import os

import _bootstrap  # noqa: F401
import torch
from tqdm import tqdm

from config import add_cli, config_from_args
from src.dataset import train_loader
from src.flow import FlowModel
from src.knn import load_bank
from src.retriever import Retriever
from src.utils import EMA, count_params, save_grid, seed_everything


def main():
    cfg = config_from_args(add_cli(argparse.ArgumentParser()).parse_args())
    device = cfg.device if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if not os.path.exists(cfg.emb_path):
        raise FileNotFoundError(
            f"Embedding bank {cfg.emb_path} not found. Run precompute_embeddings.py first.")
    bank, labels = load_bank(cfg.emb_path)
    enc_dim = bank.shape[1]
    retriever = Retriever(cfg, bank, labels)
    print(f"[train] bank={tuple(bank.shape)} cond_mode={cfg.cond_mode} device={device}")

    model = FlowModel(cfg, enc_dim).to(device)
    print(f"[train] trainable params: {count_params(model)/1e6:.1f}M")
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    loader = train_loader(cfg)
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for x0, idx, _ in pbar:
            x0 = x0.to(device)
            batch = retriever.batch_from_indices(idx, device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device == "cuda"):
                loss = model.loss(x0, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            step += 1
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % cfg.sample_every == 0:
            _sample_preview(cfg, ema.shadow, retriever, device,
                            os.path.join(cfg.out_dir, f"samples_ep{epoch+1}.png"))
        if (epoch + 1) % cfg.ckpt_every == 0:
            _save(cfg, model, ema, opt, epoch, os.path.join(cfg.out_dir, "last.pt"))

    _save(cfg, model, ema, opt, cfg.epochs - 1, os.path.join(cfg.out_dir, "last.pt"))
    print("[train] done.")


@torch.no_grad()
def _sample_preview(cfg, model, retriever, device, path):
    model.eval()
    n = min(cfg.num_samples, 64)
    if cfg.cond_mode == "none":
        batch = None
    else:
        # condition on a fixed set of real training targets for a stable preview
        idx = torch.arange(n)
        batch = retriever.batch_from_indices(idx, device)
    imgs = model.sample(batch=batch, n=n, device=device)
    save_grid(imgs.cpu(), path, nrow=8)
    print(f"[train] wrote {path}")


def _save(cfg, model, ema, opt, epoch, path):
    torch.save({
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
    }, path)
    print(f"[train] checkpoint -> {path}")


if __name__ == "__main__":
    main()
