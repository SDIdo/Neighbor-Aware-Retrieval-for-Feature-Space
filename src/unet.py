"""Small DDPM-style U-Net backbone for 32x32 images (the Generative Flow Model).

Predicts a velocity field v(x_t, t, cond). Conditioning (timestep + r_c) is
injected into every residual block via FiLM (AdaGN-style scale/shift), which is
how the GNN context steers generation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding for continuous t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)        # FiLM scale + shift
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1 + scale) + shift)
        h = self.conv2(self.dropout(h))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(B, C, H * W).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, cfg, cond_dim):
        super().__init__()
        ch = cfg.unet_base_ch
        emb_dim = ch * 4
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # Project the (already-built) condition vector into the embedding space.
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.base_ch = ch

        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)

        # ---- encoder ----
        self.down = nn.ModuleList()
        chans = [ch]
        cur = ch
        res = cfg.image_size
        for i, mult in enumerate(cfg.unet_ch_mult):
            out = ch * mult
            for _ in range(cfg.unet_num_res_blocks):
                blocks = nn.ModuleList([ResBlock(cur, out, emb_dim, cfg.unet_dropout)])
                cur = out
                if res in cfg.unet_attn_res:
                    blocks.append(AttnBlock(cur))
                self.down.append(blocks)
                chans.append(cur)
            if i != len(cfg.unet_ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(cur)]))
                chans.append(cur)
                res //= 2

        # ---- middle ----
        self.mid = nn.ModuleList([
            ResBlock(cur, cur, emb_dim, cfg.unet_dropout),
            AttnBlock(cur),
            ResBlock(cur, cur, emb_dim, cfg.unet_dropout),
        ])

        # ---- decoder ----
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(cfg.unet_ch_mult))):
            out = ch * mult
            for _ in range(cfg.unet_num_res_blocks + 1):
                blocks = nn.ModuleList(
                    [ResBlock(cur + chans.pop(), out, emb_dim, cfg.unet_dropout)])
                cur = out
                if res in cfg.unet_attn_res:
                    blocks.append(AttnBlock(cur))
                self.up.append(blocks)
            if i != 0:
                self.up.append(nn.ModuleList([Upsample(cur)]))
                res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(32, cur), nn.SiLU(), nn.Conv2d(cur, 3, 3, padding=1))

    def forward(self, x, t, cond):
        emb = self.time_mlp(timestep_embedding(t, self.base_ch))
        if cond is not None:
            emb = emb + self.cond_mlp(cond)

        h = self.in_conv(x)
        hs = [h]
        for blocks in self.down:
            h = _apply(blocks, h, emb)
            hs.append(h)            # one skip per down entry (resblock or downsample)
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        for blocks in self.up:
            if isinstance(blocks[0], Upsample):
                h = blocks[0](h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = _apply(blocks, h, emb)
        return self.out(h)


def _apply(blocks, h, emb):
    for b in blocks:
        h = b(h, emb) if isinstance(b, ResBlock) else b(h)
    return h
