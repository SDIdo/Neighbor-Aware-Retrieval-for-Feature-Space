"""Rectified-flow generative model with pluggable conditioning.

Wires the conditioner (GNN context / raw embedding / class label / none) to the
U-Net velocity field and implements the flow-matching loss and an ODE sampler.

Convention: images live in [-1, 1]. t in [0, 1] with t=0 == data, t=1 == noise.
    x_t      = (1 - t) * x0 + t * noise
    target v = noise - x0            (constant straight-line velocity)
Sampling integrates dx/dt = v from t=1 (noise) down to t=0 (image).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GNNContext
from .unet import UNet


class Conditioner(nn.Module):
    """Produces the conditioning vector consumed by the U-Net (FiLM)."""

    def __init__(self, cfg, enc_dim):
        super().__init__()
        self.mode = cfg.cond_mode
        self.cond_dim = cfg.cond_dim
        if self.mode == "gnn":
            self.gnn = GNNContext(cfg, enc_dim)
        elif self.mode == "raw":
            self.proj = nn.Sequential(
                nn.Linear(enc_dim, cfg.cond_dim), nn.GELU(),
                nn.Linear(cfg.cond_dim, cfg.cond_dim))
        elif self.mode == "label":
            self.emb = nn.Embedding(cfg.num_classes, cfg.cond_dim)
        elif self.mode != "none":
            raise ValueError(f"unknown cond_mode {self.mode}")
        # learned null token for classifier-free guidance / condition dropout
        self.null = nn.Parameter(torch.zeros(cfg.cond_dim))

    def forward(self, batch):
        if self.mode == "none":
            return None
        if self.mode == "gnn":
            X, A, mask = batch["graph"]
            return self.gnn(X, A, mask)
        if self.mode == "raw":
            return self.proj(batch["c"])
        return self.emb(batch["label"])

    def null_like(self, n, device):
        return self.null.unsqueeze(0).expand(n, -1).to(device)


class FlowModel(nn.Module):
    def __init__(self, cfg, enc_dim):
        super().__init__()
        self.cfg = cfg
        self.cond = Conditioner(cfg, enc_dim)
        self.unet = UNet(cfg, cfg.cond_dim)

    # ---- training ----
    def loss(self, x0, batch):
        B = x0.size(0)
        noise = torch.randn_like(x0)
        t = torch.rand(B, device=x0.device)
        tb = t.view(B, 1, 1, 1)
        x_t = (1 - tb) * x0 + tb * noise
        target = noise - x0

        cond = self.cond(batch)
        cond = self._cond_dropout(cond, B, x0.device)
        v = self.unet(x_t, t, cond)
        return F.mse_loss(v, target)

    def _cond_dropout(self, cond, B, device):
        """Randomly replace condition rows with the null token (enables CFG)."""
        if cond is None or self.cfg.cond_dropout <= 0:
            return cond
        drop = torch.rand(B, device=device) < self.cfg.cond_dropout
        null = self.cond.null_like(B, device)
        return torch.where(drop.unsqueeze(1), null, cond)

    # ---- sampling ----
    @torch.no_grad()
    def sample(self, batch=None, cond=None, n=None, steps=None, guidance=None,
               sampler=None, device="cuda"):
        cfg = self.cfg
        steps = steps or cfg.sample_steps
        guidance = cfg.guidance_scale if guidance is None else guidance
        sampler = sampler or cfg.sampler

        if cond is None and batch is not None:
            cond = self.cond(batch)
        n = n or (cond.size(0) if cond is not None else cfg.num_samples)
        null = self.cond.null_like(n, device) if cond is not None else None

        def velocity(x, t):
            tt = torch.full((n,), t, device=device)
            if cond is None or guidance == 1.0:
                return self.unet(x, tt, cond)
            v_c = self.unet(x, tt, cond)
            v_u = self.unet(x, tt, null)
            return v_u + guidance * (v_c - v_u)

        x = torch.randn(n, 3, cfg.image_size, cfg.image_size, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t0, t1 = ts[i].item(), ts[i + 1].item()
            dt = t1 - t0                       # negative: integrating toward data
            v = velocity(x, t0)
            if sampler == "heun" and i < steps - 1:
                x_pred = x + dt * v
                v2 = velocity(x_pred, t1)
                x = x + dt * 0.5 * (v + v2)
            else:
                x = x + dt * v
        return x.clamp(-1, 1)
