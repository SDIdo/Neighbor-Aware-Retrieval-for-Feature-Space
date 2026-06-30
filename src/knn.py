"""K-nearest-neighbor retrieval over the cached embedding bank.

Two backends:
  * "torch"  : batched cosine similarity (matmul + topk). No extra dependency,
               fine for CIFAR-scale banks (50k x 512).
  * "faiss"  : exact inner-product index, if faiss is installed.

Embeddings are assumed L2-normalized, so inner product == cosine similarity.
"""
from __future__ import annotations

import numpy as np
import torch


class KNNIndex:
    def __init__(self, bank: torch.Tensor, backend: str = "torch", device: str = "cuda"):
        """`bank`: [N, d] L2-normalized embeddings of the whole dataset."""
        self.backend = backend
        self.N, self.d = bank.shape
        if backend == "faiss":
            import faiss
            self._index = faiss.IndexFlatIP(self.d)
            self._index.add(bank.detach().cpu().numpy().astype("float32"))
            self._bank = bank
        else:
            self.device = device
            self._bank = bank.to(device)

    def search(self, queries: torch.Tensor, k: int, exclude_self_idx: torch.Tensor | None = None):
        """Return (sim, idx), each [B, k].

        `exclude_self_idx`: optional [B] tensor of bank indices to drop from each
        query's results (used during training so a sample never retrieves itself).
        We over-fetch by one and filter to keep exactly k.
        """
        kk = k + 1 if exclude_self_idx is not None else k
        if self.backend == "faiss":
            import faiss  # noqa: F401
            sim_np, idx_np = self._index.search(
                queries.detach().cpu().numpy().astype("float32"), kk)
            sim = torch.from_numpy(sim_np)
            idx = torch.from_numpy(idx_np)
        else:
            q = queries.to(self._bank.device, self._bank.dtype)
            sims = q @ self._bank.t()                       # [B, N]
            sim, idx = torch.topk(sims, kk, dim=1)          # [B, kk]
            sim, idx = sim.cpu(), idx.cpu()

        if exclude_self_idx is not None:
            sim, idx = _drop_self(sim, idx, exclude_self_idx.cpu(), k)
        return sim, idx

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Fetch embeddings for [B, k] indices -> [B, k, d]."""
        flat = self._bank[idx.reshape(-1).to(self._bank.device)]
        return flat.view(*idx.shape, self.d)


def _drop_self(sim: torch.Tensor, idx: torch.Tensor, self_idx: torch.Tensor, k: int):
    """Remove the query's own bank index from each row, keep top-k of the rest."""
    keep_sim = torch.empty(idx.size(0), k, dtype=sim.dtype)
    keep_idx = torch.empty(idx.size(0), k, dtype=idx.dtype)
    for i in range(idx.size(0)):
        row_i, row_s = idx[i], sim[i]
        mask = row_i != self_idx[i]
        ri, rs = row_i[mask][:k], row_s[mask][:k]
        if ri.numel() < k:  # self wasn't in the fetched set; just trim
            ri, rs = row_i[:k], row_s[:k]
        keep_idx[i], keep_sim[i] = ri, rs
    return keep_sim, keep_idx


def load_bank(emb_path: str):
    """Load a cached embedding bank saved by precompute_embeddings.py."""
    data = np.load(emb_path)
    bank = torch.from_numpy(data["emb"]).float()
    labels = torch.from_numpy(data["labels"]).long()
    return bank, labels
