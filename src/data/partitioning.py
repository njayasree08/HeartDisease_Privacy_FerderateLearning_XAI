"""Reproducible partitioning of the TRAINING set into simulated hospital clients."""
from __future__ import annotations
import numpy as np


def iid_partition(y, n_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Stratified round-robin split: each client sees roughly the global class mix."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[i].extend(chunk.tolist())
    return [np.array(sorted(p)) for p in parts]


def non_iid_partition(y, n_clients: int, alpha: float = 0.5, seed: int = 42, min_size: int = 10) -> list[np.ndarray]:
    """Dirichlet label-skew partition (smaller alpha = more heterogeneous clients)."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    for _ in range(200):
        parts = [[] for _ in range(n_clients)]
        for cls in np.unique(y):
            idx = rng.permutation(np.where(y == cls)[0])
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size and all(len(set(y[p])) >= 1 for p in parts):
            return [np.array(sorted(p)) for p in parts]
    raise ValueError("Could not build a non-IID partition meeting min_size; lower min_size, alpha, or client count.")


def partition(y, n_clients: int, kind: str = "iid", alpha: float = 0.5, seed: int = 42) -> list[np.ndarray]:
    if n_clients < 2:
        raise ValueError("At least two clients are required.")
    if len(y) < n_clients * 2:
        raise ValueError("Too few training records for this many clients.")
    return iid_partition(y, n_clients, seed) if kind == "iid" else non_iid_partition(y, n_clients, alpha, seed)
