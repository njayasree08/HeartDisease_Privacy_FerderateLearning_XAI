
"""Simulated hospital client using a NumPy neural network."""
from __future__ import annotations

import numpy as np

PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class NumPyModel(dict):
    """Dictionary model with get_weights and set_weights compatibility."""
    def get_weights(self):
        return [self[name].copy() for name in PARAMETER_NAMES if name in self]

    def set_weights(self, weights):
        for name, w in zip(PARAMETER_NAMES, weights):
            self[name] = np.asarray(w).copy()


def build_model(n_features: int, lr: float = 0.01, seed: int = 42):
    """Initialize a small neural network without TensorFlow or Keras."""
    rng = np.random.default_rng(seed)
    return NumPyModel({
        "W1": rng.normal(0, np.sqrt(2 / n_features), (n_features, 16)),
        "b1": np.zeros((1, 16)),
        "W2": rng.normal(0, np.sqrt(2 / 16), (16, 1)),
        "b2": np.zeros((1, 1)),
        "lr": float(lr),
    })


def predict(model, X):
    X = np.asarray(X, dtype=np.float64)
    h = np.maximum(X @ model["W1"] + model["b1"], 0)
    return _sigmoid(h @ model["W2"] + model["b2"]).ravel()


def _train(model, X, y, epochs, batch_size):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    if len(y) == 0:
        raise ValueError("Hospital client has no training records.")

    lr = model["lr"]
    rng = np.random.default_rng(42)
    batch_size = max(1, min(int(batch_size), len(y)))

    for _ in range(int(epochs)):
        for start in range(0, len(y), batch_size):
            idx = rng.permutation(len(y))[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            z1 = xb @ model["W1"] + model["b1"]
            h1 = np.maximum(z1, 0)
            p = _sigmoid(h1 @ model["W2"] + model["b2"])

            dz2 = (p - yb) / len(idx)
            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)

            dz1 = (dz2 @ model["W2"].T) * (z1 > 0)
            dW1 = xb.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)

            model["W1"] -= lr * dW1
            model["b1"] -= lr * db1
            model["W2"] -= lr * dW2
            model["b2"] -= lr * db2

    return model


class HospitalClient:
    """A simulated hospital that trains on its local records."""

    def __init__(self, cid, X, y, lr, seed):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self.cid = cid
        self.lr = float(lr)
        self.seed = seed
        self.model = build_model(self._X.shape[1], lr, seed)

    @property
    def n_examples(self):
        return len(self._y)

    def fit(self, global_weights, epochs, batch_size):
        """Load server parameters, train locally, and return updated parameters."""
        if isinstance(global_weights, dict):
            weights = global_weights
        else:
            weights = dict(zip(PARAMETER_NAMES, global_weights))

        self.model = NumPyModel({
            name: np.asarray(weights[name]).copy()
            for name in PARAMETER_NAMES
        })
        self.model["lr"] = self.lr

        _train(self.model, self._X, self._y, epochs, batch_size)

        p = np.clip(predict(self.model, self._X), 1e-7, 1 - 1e-7)
        loss = -np.mean(
            self._y * np.log(p) + (1 - self._y) * np.log(1 - p)
        )
        accuracy = np.mean((p >= 0.5) == self._y)

        returned_weights = [
            self.model[name].copy() for name in PARAMETER_NAMES
        ]

        return returned_weights, self.n_examples, {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
