# Neighbor-Aware Retrieval for Feature-Space Conditioning

A retrieval-augmented **rectified-flow** image generator whose conditioning
vector `r_c` is produced by a **GNN over a KNN graph of frozen-encoder
embeddings**, instead of a raw embedding or class label. Implemented on
**CIFAR-10 / CIFAR-100**.

This is the architecture from `docs/Diagramflow.jpeg`, made concrete.

```
image ──[frozen DINO/CLIP]──► c ──[KNN over embedding bank]──► local graph
                                                                   │
                                                          [GNN context] ──► r_c
                                                                   │
                noise ───────────[rectified-flow U-Net | FiLM(t, r_c)]──► image
```

## Pipeline (matches the diagram)

| Stage | Script | Frozen | Trained |
|---|---|---|---|
| 0. Preprocessing — cache embedding bank | `scripts/precompute_embeddings.py` | encoder | — |
| 1. Training — GNN + flow, joint | `scripts/train.py` | encoder | GNN, U-Net |
| 2. Inference — generate from target image | `scripts/sample.py` | everything | — |
| 3. Evaluation — FID | `scripts/eval_fid.py` | everything | — |

## Components

- `src/encoder.py` — frozen feature extractor. CLIP (`open_clip`) or DINOv2
  (`torch.hub`) if installed; **auto-falls back to a torchvision ResNet50** so it
  always runs. Inputs in `[0,1]`, resized to 224, normalized internally.
- `src/knn.py` — KNN over the cached bank. Pure-torch batched cosine (default) or
  FAISS. Excludes the query itself during training (no self-leakage).
- `src/graph.py` — builds the per-sample local graph densely: node 0 = target,
  nodes 1..K = neighbors. `star` or `mutual` topology; edge weights = cosine sim.
- `src/gnn.py` — dense multi-head **GATv2** with similarity as an attention bias;
  reads out the **target node** and projects to `r_c` (`cond_dim`).
- `src/unet.py` — small 32×32 DDPM-style U-Net; `r_c` + timestep injected via
  **FiLM (AdaGN)** in every ResBlock; self-attention at 16×16.
- `src/flow.py` — rectified flow: `x_t=(1-t)x0+t·noise`, target velocity
  `noise−x0`, MSE loss; Euler/Heun ODE sampler with classifier-free guidance.
- `src/retriever.py` — glues bank + KNN + graph into the `batch` dict.

No `torch-geometric` or `faiss` required — the GNN runs as batched matmuls on the
fixed-size `K+1` graphs.

## Setup

```bash
pip install -r requirements.txt
# optional, better encoders / metrics:
# pip install open_clip_torch timm faiss-cpu clean-fid
```

> Needs PyTorch (a CUDA GPU is strongly recommended for training). PyTorch
> currently ships wheels up to Python 3.12–3.13; use one of those if `pip install
> torch` fails on a newer interpreter.

## Quick check (CPU, seconds, no data)

```bash
python scripts/smoke_test.py     # runs all 4 conditioning modes on random data
```

## Run on CIFAR-10

```bash
# 0. cache embeddings (downloads CIFAR-10 the first time)
python scripts/precompute_embeddings.py --dataset cifar10 --encoder clip \
    --emb-path data/cifar10_emb.npz

# 1. train the proposed model (GNN conditioning)
python scripts/train.py --dataset cifar10 --cond-mode gnn \
    --emb-path data/cifar10_emb.npz --out-dir runs/gnn --epochs 200

# 2. inference: generate from unseen test-image targets
python scripts/sample.py --ckpt runs/gnn/last.pt --num-samples 64

# 3. FID
python scripts/eval_fid.py --ckpt runs/gnn/last.pt --num 10000
```

## The core experiment (ablation)

The contribution is `r_c` from the GNN. Train identical models that differ only
in `--cond-mode` and compare FID / diversity:

```bash
python scripts/train.py --cond-mode gnn   --out-dir runs/gnn      # proposed
python scripts/train.py --cond-mode raw   --out-dir runs/raw      # baseline: raw c
python scripts/train.py --cond-mode label --out-dir runs/label    # baseline: class label
python scripts/train.py --cond-mode none  --out-dir runs/uncond   # unconditional
```

If `gnn` does not beat `raw`, the graph adds nothing — so run this first.
Useful secondary knobs: `--k`, `--graph-topology {star,mutual}`, `--gnn-layers`,
`--cond-dim`, `--guidance-scale`.

## Defaults

CLIP ViT-B/32 encoder · K=12 · mutual graph · GATv2 (3 layers, 256 hidden, 4
heads) · `cond_dim`=256 · 32×32 U-Net (~35M params, base 128) · rectified flow ·
AdamW 2e-4 · EMA 0.9999 · Heun 50-step sampling. All in `config.py`.
