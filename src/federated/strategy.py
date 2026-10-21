"""Aggregation strategies. FedAvg applies ONLY to neural-network weight tensors (never to tree models)."""
from __future__ import annotations
import numpy as np


def fedavg(client_weights: list[list[np.ndarray]], client_sizes: list[int]) -> list[np.ndarray]:
    """Federated Averaging: layer-wise mean of client tensors weighted by local sample counts."""
    if not client_weights:
        raise ValueError("No client updates to aggregate.")
    n_layers = len(client_weights[0])
    if any(len(w) != n_layers for w in client_weights):
        raise ValueError("Clients returned a different number of parameter tensors.")
    for i in range(n_layers):
        shapes = {w[i].shape for w in client_weights}
        if len(shapes) != 1:
            raise ValueError(f"Incompatible tensor shapes in layer {i}: {shapes}")
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Total client sample count must be positive.")
    return [sum(w[i] * (n / total) for w, n in zip(client_weights, client_sizes)) for i in range(n_layers)]


def flower_fedavg(client_weights, client_sizes):
    """Same aggregation through Flower's own primitive (used to cross-check our implementation)."""
    from flwr.server.strategy.aggregate import aggregate
    return aggregate(list(zip(client_weights, client_sizes)))
