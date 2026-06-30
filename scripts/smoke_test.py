"""Tiny end-to-end shape/sanity check on CPU with random data (no CIFAR, no GPU).

Builds a small config, fakes an embedding bank, and runs the full
retrieve -> graph -> GNN -> flow loss -> sample path on a few random images.
Catches shape/wiring bugs in seconds.

    python scripts/smoke_test.py
"""
import _bootstrap  # noqa: F401
import torch

from config import Config
from src.flow import FlowModel
from src.retriever import Retriever


def main():
    torch.manual_seed(0)
    cfg = Config()
    cfg.device = "cpu"
    cfg.image_size = 32
    cfg.unet_base_ch = 32          # tiny net for speed
    cfg.unet_ch_mult = (1, 2, 2)
    cfg.gnn_hidden = 64
    cfg.gnn_heads = 4
    cfg.cond_dim = 64
    cfg.k = 6

    N, d = 200, 128                # fake bank
    bank = torch.nn.functional.normalize(torch.randn(N, d), dim=1)
    labels = torch.randint(0, cfg.num_classes, (N,))

    for mode in ["gnn", "raw", "label", "none"]:
        cfg.cond_mode = mode
        retriever = Retriever(cfg, bank, labels)
        model = FlowModel(cfg, d)

        idx = torch.randint(0, N, (8,))
        x0 = torch.randn(8, 3, 32, 32)
        batch = retriever.batch_from_indices(idx, "cpu")

        loss = model.loss(x0, batch)
        assert torch.isfinite(loss), f"{mode}: non-finite loss"

        with torch.no_grad():
            sample_batch = None if mode == "none" else batch
            imgs = model.sample(batch=sample_batch, n=8, steps=4, device="cpu")
        assert imgs.shape == (8, 3, 32, 32), f"{mode}: bad sample shape {imgs.shape}"
        print(f"[smoke] cond_mode={mode:5s}  loss={loss.item():.4f}  "
              f"sample={tuple(imgs.shape)}  OK")

    print("[smoke] all conditioning modes passed.")


if __name__ == "__main__":
    main()
