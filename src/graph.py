"""Build the per-sample local similarity graph (the "LOCAL GRAPH" box).

Each graph has N = K+1 nodes: node 0 is the target `c`, nodes 1..K are its
retrieved neighbors. Graphs are small and fixed-size, so we represent a batch of
them densely:

    X    : [B, N, d]   node features (embeddings)
    A    : [B, N, N]   edge weights (cosine similarity) on existing edges
    mask : [B, N, N]   boolean, True where an edge exists (incl. self-loops)

This dense form lets the GNN run as plain batched matmuls -- no torch-geometric.
"""
from __future__ import annotations

import torch


def build_graph(c: torch.Tensor, neighbors: torch.Tensor, cfg):
    """`c`: [B, d] target embeddings. `neighbors`: [B, K, d]. Returns X, A, mask."""
    B, K, d = neighbors.shape
    N = K + 1
    X = torch.cat([c.unsqueeze(1), neighbors], dim=1)        # [B, N, d]

    # Cosine similarity between every pair of nodes (embeddings are normalized).
    A = torch.bmm(X, X.transpose(1, 2)).clamp(-1.0, 1.0)     # [B, N, N]

    mask = torch.zeros(B, N, N, dtype=torch.bool, device=X.device)
    eye = torch.eye(N, dtype=torch.bool, device=X.device)
    mask |= eye                                              # self-loops
    mask[:, 0, :] = True                                     # target -> all
    mask[:, :, 0] = True                                     # all -> target

    if cfg.graph_topology == "mutual":
        # Connect each neighbor to its `inner_k` most-similar other neighbors.
        inner_k = min(cfg.graph_inner_k, K - 1) if K > 1 else 0
        if inner_k > 0:
            nbr_sim = A[:, 1:, 1:].clone()                   # [B, K, K]
            nbr_sim.diagonal(dim1=1, dim2=2).fill_(-2.0)     # exclude self
            top = nbr_sim.topk(inner_k, dim=2).indices       # [B, K, inner_k]
            rows = torch.arange(K, device=X.device).view(1, K, 1).expand(B, K, inner_k)
            bdx = torch.arange(B, device=X.device).view(B, 1, 1).expand(B, K, inner_k)
            # offset by 1 because node 0 is the target
            mask[bdx, rows + 1, top + 1] = True
            mask[bdx, top + 1, rows + 1] = True              # keep symmetric

    A = A.masked_fill(~mask, 0.0)
    return X, A, mask
