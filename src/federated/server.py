
"""Aggregation server for NumPy neural-network parameters."""
from __future__ import annotations

import numpy as np
from src.federated.strategy import fedavg

PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


class FederatedServer:
    def __init__(self, initial_weights):
        if isinstance(initial_weights, dict):
            self.global_weights = [
                initial_weights[name].copy()
                for name in PARAMETER_NAMES
            ]
        else:
            self.global_weights = [
                np.asarray(w).copy() for w in initial_weights
            ]

        self.rounds_done = 0

    def aggregate(self, updates):
        if not updates:
            raise ValueError("No client updates received.")

        weights, sizes = zip(*updates)
        self.global_weights = fedavg(list(weights), list(sizes))
        self.rounds_done += 1
        return self.global_weights

    @staticmethod
    def n_params(weights):
        return int(sum(w.size for w in weights))
