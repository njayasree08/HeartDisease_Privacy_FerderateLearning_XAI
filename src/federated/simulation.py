
"""Federated learning simulation using a NumPy neural network."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src import config
from src.data.partitioning import partition
from src.data.preprocessing import build_preprocessor
from src.federated.client import (
    HospitalClient,
    build_model,
    predict,
    _train,
)
from src.federated.server import FederatedServer, PARAMETER_NAMES
from src.models.evaluate import compute_metrics


def _as_model(weights, lr):
    return {
        **{
            name: np.asarray(weight).copy()
            for name, weight in zip(PARAMETER_NAMES, weights)
        },
        "lr": float(lr),
    }


def _eval(model, X, y):
    p = np.clip(predict(model, X), 1e-7, 1 - 1e-7)
    metrics = compute_metrics(y, p)
    metrics["loss"] = float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    )
    return metrics


def run_federated(
    splits: dict,
    n_clients=3,
    kind="iid",
    alpha=0.5,
    rounds=10,
    local_epochs=2,
    batch_size=16,
    lr=0.01,
    seed=config.SEED,
    outlier_strategy="keep",
    progress=None,
    save_dir: Path | None = None,
):
    """Run simulated FedAvg and compare it with a centralized baseline."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (
        splits["train"], splits["val"], splits["test"]
    )

    prep = build_preprocessor(outlier_strategy).fit(Xtr)
    A, V, T = (
        prep.transform(x).astype("float64")
        for x in (Xtr, Xva, Xte)
    )

    y_tr = np.asarray(ytr, dtype="float64")
    y_va = np.asarray(yva, dtype="float64")
    y_te = np.asarray(yte, dtype="float64")

    parts = partition(y_tr, n_clients, kind, alpha, seed)
    clients = [
        HospitalClient(i, A[p], y_tr[p], lr, seed + i)
        for i, p in enumerate(parts)
    ]

    initial_model = build_model(A.shape[1], lr, seed)
    server = FederatedServer(initial_model)
    n_params = server.n_params(server.global_weights)

    history = []
    start_time = time.perf_counter()

    for round_number in range(1, rounds + 1):
        updates = []
        local_metrics = []

        for client in clients:
            weights, count, metrics = client.fit(
                server.global_weights, local_epochs, batch_size
            )
            updates.append((weights, count))
            local_metrics.append(metrics)

        server.aggregate(updates)
        global_model = _as_model(server.global_weights, lr)

        val_metrics = _eval(global_model, V, y_va)
        test_metrics = _eval(global_model, T, y_te)

        history.append({
            "round": round_number,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_recall": val_metrics["recall"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_recall": test_metrics["recall"],
            "test_roc_auc": test_metrics["roc_auc"],
            "local_loss_mean": float(np.mean([
                item["loss"] for item in local_metrics
            ])),
            "participating": len(updates),
            "bytes_up": n_params * 8 * len(updates),
            "bytes_down": n_params * 8 * len(updates),
            "status": "completed",
            "client_losses": [
                round(item["loss"], 4) for item in local_metrics
            ],
        })

        if progress:
            progress(f"Round {round_number}/{rounds}", round_number / rounds)

    final_federated = _eval(
        _as_model(server.global_weights, lr), T, y_te
    )

    central_model = build_model(A.shape[1], lr, seed)
    _train(
        central_model, A, y_tr,
        rounds * local_epochs, batch_size
    )
    final_centralized = _eval(central_model, T, y_te)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "config": {
            "n_clients": n_clients,
            "partition": kind,
            "alpha": alpha if kind != "iid" else None,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": "Mini-batch gradient descent (NumPy)",
            "aggregation": "FedAvg weighted by local sample count",
            "model_params": n_params,
            "model": "NumPy neural network with 16-unit ReLU hidden layer",
        },
        "client_sizes": [len(p) for p in parts],
        "client_positive_rate": [
            round(float(y_tr[p].mean()), 3) for p in parts
        ],
        "history": history,
        "final_federated": final_federated,
        "final_centralized": final_centralized,
        "comm": {
            "total_bytes": sum(
                item["bytes_up"] + item["bytes_down"]
                for item in history
            ),
            "raw_records_sent_to_server": 0,
        },
        "seconds": time.perf_counter() - start_time,
        "notes": (
            "Simulated clients on one machine. Preprocessing is fitted on "
            "pooled training data. No secure aggregation or differential "
            "privacy is applied. Model implemented using NumPy, not Keras."
        ),
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "federated.json").write_text(
            json.dumps(result, indent=2, default=float),
            encoding="utf-8",
        )
        np.savez(
            save_dir / "federated_weights.npz",
            *server.global_weights,
        )

    return result
