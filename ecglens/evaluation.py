"""Shared signal loading + retrieval helpers (used by training-time validation and the final eval)."""
import numpy as np
import torch

SCALE = 200.0  # gain is 1000 ADC units per mV; per-lead std is ~150 units, so this gives ~unit variance with amplitude preserved


def prep(batch_int16: np.ndarray) -> torch.Tensor:
    """[B,12,1000] int16 -> float tensor; subtract per-lead baseline, keep absolute amplitude (voltage criteria matter)."""
    x = batch_int16.astype(np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    return torch.from_numpy(x / SCALE)


@torch.no_grad()
def embed_ecgs(model, X, rows, bs=256, features=False):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        xb = prep(X[rows[i : i + bs]])
        out.append((model.ecg.features(xb) if features else model.ecg(xb)).numpy())
    return np.concatenate(out)


def strict_ranks(sim: np.ndarray, true_col: np.ndarray) -> np.ndarray:
    """1-based rank of the paired (identical-report) gallery item for each query; sim is [n_query, n_gallery].

    Gallery = unique report strings, so exactly one item is strict-relevant per query: no ties in the target.
    Rank = 1 + number of gallery items scoring strictly higher than the target.
    """
    tgt = sim[np.arange(len(sim)), true_col][:, None]
    return 1 + (sim > tgt).sum(1)


def recall_at(ranks: np.ndarray, ks=(1, 5, 10)) -> dict:
    d = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
    d["median_rank"] = float(np.median(ranks))
    d["MRR"] = float((1.0 / ranks).mean())
    return d


def label_precision_at(sim: np.ndarray, query_set: list, gallery_set: list, k=10) -> float:
    """Mean fraction of the top-k retrieved gallery reports whose superclass set equals the query ECG's set."""
    top = np.argsort(-sim, axis=1)[:, :k]
    g = np.array([hash(s) for s in gallery_set])
    q = np.array([hash(s) for s in query_set])
    return float((g[top] == q[:, None]).mean())
