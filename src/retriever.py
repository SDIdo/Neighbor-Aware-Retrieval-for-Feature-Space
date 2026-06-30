"""Ties the embedding bank, KNN index, and graph builder into one helper.

Given target bank indices (training) or arbitrary query embeddings (inference),
it returns the `batch` dict consumed by FlowModel: {c, graph, label}.
"""
from __future__ import annotations

import torch

from .graph import build_graph
from .knn import KNNIndex


class Retriever:
    def __init__(self, cfg, bank: torch.Tensor, labels: torch.Tensor):
        self.cfg = cfg
        self.bank = bank
        self.labels = labels
        self.index = KNNIndex(bank, cfg.knn_backend, cfg.device)

    def _assemble(self, c, self_idx, label, device):
        sim, nidx = self.index.search(c, self.cfg.k, self_idx)
        neighbors = self.index.gather(nidx).to(device)
        c = c.to(device)
        X, A, mask = build_graph(c, neighbors, self.cfg)
        return {"c": c, "graph": (X, A, mask),
                "label": label.to(device) if label is not None else None}

    def batch_from_indices(self, idx: torch.Tensor, device):
        """Training: targets are existing bank rows; exclude self from neighbors."""
        c = self.bank[idx.to(self.bank.device)]
        self_idx = idx if self.cfg.exclude_self else None
        return self._assemble(c, self_idx, self.labels[idx.to(self.bank.device)], device)

    def batch_from_embeddings(self, c: torch.Tensor, device, labels=None):
        """Inference: targets are new/unseen embeddings; keep all neighbors."""
        return self._assemble(c, None, labels, device)
