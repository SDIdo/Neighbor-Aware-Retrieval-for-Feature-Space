"""GNN context network: turns the local graph into the conditioning vector r_c.

A stack of dense multi-head GATv2 layers operates on the [B, N, N] adjacency
produced by graph.py. The cosine-similarity edge weights are injected as an
additive attention bias ("Similarity-Based Edges"). The readout is the target
node's (node 0) final representation, projected to `cond_dim` -> r_c.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGATv2Layer(nn.Module):
    """GATv2 attention over a dense batch of graphs.

    Attention logit follows Brody et al. (2022):
        e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)
    plus an optional bias from the edge's cosine similarity. Non-edges are masked
    to -inf before the softmax over neighbors j.
    """

    def __init__(self, in_dim, out_dim, heads, edge_bias=True, dropout=0.0):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W_l = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, 1, 1, out_dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.edge_bias = nn.Parameter(torch.zeros(heads)) if edge_bias else None
        nn.init.xavier_uniform_(self.att)

    def forward(self, h, A, mask):
        B, N, _ = h.shape
        H, F_ = self.heads, self.out_dim
        hl = self.W_l(h).view(B, N, H, F_).permute(0, 2, 1, 3)   # [B,H,N,F]
        hr = self.W_r(h).view(B, N, H, F_).permute(0, 2, 1, 3)   # [B,H,N,F]

        # pairwise sum h_i + h_j -> [B,H,N,N,F]
        e = self.leaky(hl.unsqueeze(3) + hr.unsqueeze(2))
        e = (e * self.att).sum(-1)                               # [B,H,N,N]

        if self.edge_bias is not None:
            e = e + self.edge_bias.view(1, H, 1, 1) * A.unsqueeze(1)

        m = mask.unsqueeze(1)                                    # [B,1,N,N]
        e = e.masked_fill(~m, float("-inf"))
        alpha = F.softmax(e, dim=-1)                             # over j
        alpha = torch.nan_to_num(alpha)                         # isolated nodes
        alpha = self.dropout(alpha)

        out = torch.einsum("bhij,bhjf->bhif", alpha, hr)         # [B,H,N,F]
        return out.permute(0, 2, 1, 3).reshape(B, N, H * F_)     # concat heads


class GNNContext(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        H, hid = cfg.gnn_heads, cfg.gnn_hidden
        assert hid % H == 0, "gnn_hidden must be divisible by gnn_heads"
        per_head = hid // H

        self.input = nn.Linear(in_dim, hid)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.gnn_layers):
            self.layers.append(
                DenseGATv2Layer(hid, per_head, H, cfg.edge_sim_bias, cfg.gnn_dropout))
            self.norms.append(nn.LayerNorm(hid))

        self.readout = nn.Sequential(
            nn.LayerNorm(hid), nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, cfg.cond_dim),
        )

    def forward(self, X, A, mask):
        h = self.input(X)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.gelu(layer(h, A, mask)))      # residual GAT block
        r_c = self.readout(h[:, 0])                       # target node readout
        return r_c
