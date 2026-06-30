"""Central configuration for the Neighbor-Aware Retrieval flow model.

A single dataclass keeps every hyperparameter in one place so scripts stay thin.
Override any field from the CLI, e.g. `python scripts/train.py --k 16 --epochs 100`.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- paths ----
    data_root: str = "./data"            # where torchvision downloads CIFAR
    out_dir: str = "./runs/default"      # checkpoints, samples, logs
    emb_path: str = "./data/cifar10_emb.npz"  # cached embedding bank

    # ---- dataset ----
    dataset: str = "cifar10"             # cifar10 | cifar100
    image_size: int = 32
    num_workers: int = 4

    # ---- frozen encoder (Preprocessing stage) ----
    encoder: str = "clip"                # clip | dino | resnet  (auto-fallback to resnet)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    dino_model: str = "dinov2_vits14"
    encoder_resize: int = 224            # ViT encoders expect 224x224 input
    emb_normalize: bool = True           # L2-normalize embeddings (cosine KNN)

    # ---- KNN retrieval ----
    k: int = 12                          # neighbors per target
    exclude_self: bool = True            # drop the query itself during training
    knn_backend: str = "torch"           # torch | faiss

    # ---- local graph ----
    # "star"   : every neighbor connects to the target only
    # "mutual" : star + kNN edges among neighbors (sees neighbor-neighbor structure)
    graph_topology: str = "mutual"
    graph_inner_k: int = 4               # neighbor-neighbor edges per node (mutual mode)
    edge_sim_bias: bool = True           # feed cosine similarity as attention bias

    # ---- GNN context network (produces r_c) ----
    gnn_layers: int = 3
    gnn_hidden: int = 256
    gnn_heads: int = 4
    gnn_dropout: float = 0.0
    cond_dim: int = 256                  # dimensionality of r_c

    # ---- generative flow model (rectified flow U-Net) ----
    unet_base_ch: int = 128
    unet_ch_mult: tuple = (1, 2, 2, 2)
    unet_num_res_blocks: int = 2
    unet_attn_res: tuple = (16,)         # resolutions that get self-attention
    unet_dropout: float = 0.1

    # ---- training ----
    epochs: int = 200
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    amp: bool = True                     # mixed precision
    seed: int = 0
    log_every: int = 100
    sample_every: int = 5                # epochs
    ckpt_every: int = 10                 # epochs

    # ---- conditioning ablation ----
    # gnn  : condition on r_c from the GNN (the proposed method)
    # raw  : condition on the raw target embedding c (baseline, no graph)
    # label: condition on the class label embedding (baseline)
    # none : unconditional
    cond_mode: str = "gnn"
    cond_dropout: float = 0.1            # prob of dropping condition (classifier-free guidance)

    # ---- sampling ----
    sample_steps: int = 50
    sampler: str = "heun"                # euler | heun
    guidance_scale: float = 1.0          # >1 enables classifier-free guidance
    num_samples: int = 64

    # ---- runtime ----
    device: str = "cuda"

    @property
    def num_classes(self) -> int:
        return 100 if self.dataset == "cifar100" else 10


def add_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Expose every Config field as an optional CLI flag."""
    for f in dataclasses.fields(Config):
        name = f"--{f.name.replace('_', '-')}"
        default = getattr(Config, f.name, None)
        if f.type == "bool" or isinstance(default, bool):
            parser.add_argument(name, type=lambda x: x.lower() in ("1", "true", "yes"),
                                default=None)
        elif isinstance(default, tuple):
            parser.add_argument(name, type=str, default=None,
                                help="comma-separated, e.g. 1,2,2,2")
        else:
            ptype = type(default) if default is not None else str
            parser.add_argument(name, type=ptype, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    for f in dataclasses.fields(Config):
        val = getattr(args, f.name, None)
        if val is None:
            continue
        if isinstance(getattr(cfg, f.name), tuple):
            val = tuple(int(x) for x in str(val).split(","))
        setattr(cfg, f.name, val)
    return cfg
