"""CSV exports built from saved experiment records."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src import config


def metrics_csv(table: pd.DataFrame) -> bytes:
    return table.round(4).to_csv(index=False).encode()


def experiment_history() -> pd.DataFrame:
    """One row per saved experiment that actually exists on disk."""
    rows = []
    for p in sorted((config.ARTIFACT_DIR / "experiments").glob("*/experiment.json")):
        e = json.loads(p.read_text())
        ok = {n: r for n, r in e["models"].items() if "val" in r}
        rows.append(dict(run_id=e["run_id"], timestamp=e["timestamp"], mode=e["mode"], seed=e["seed"], tuned=e["tune"],
                         models_trained=len(ok), models_failed=len(e["models"]) - len(ok), selected=e["selected_model"],
                         val_selection_score=ok[e["selected_model"]]["selection_score"] if e["selected_model"] else None,
                         federated_saved=(p.parent / "federated.json").exists()))
    return pd.DataFrame(rows)
