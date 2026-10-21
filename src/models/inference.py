"""Integrity-checked artifact loading and single-patient prediction."""
from __future__ import annotations
import hashlib, json, logging
from pathlib import Path
import joblib, pandas as pd, sklearn
from src import config
from src.data.validation import validate_patient_input

log = logging.getLogger(__name__)


class ArtifactError(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_experiment_dir(root: Path = config.ARTIFACT_DIR) -> Path | None:
    runs = sorted((root / "experiments").glob("*/experiment.json")) if (root / "experiments").exists() else []
    return runs[-1].parent if runs else None


def load_experiment(run_dir: Path) -> dict:
    p = Path(run_dir) / "experiment.json"
    if not p.exists():
        raise ArtifactError(f"No experiment.json in {run_dir}")
    return json.loads(p.read_text())


def load_model(run_dir: Path, name: str):
    """Load a model ONLY after verifying its SHA-256 against the experiment record.
    Only load artifacts produced by this application; joblib files can execute code when loaded."""
    meta = load_experiment(run_dir)
    info = meta["models"].get(name)
    if not info or "file" not in info:
        raise ArtifactError(f"Model '{name}' was not saved in this experiment.")
    path = Path(run_dir) / info["file"]
    if not path.exists():
        raise ArtifactError(f"Artifact file missing: {path.name}. Retrain the models.")
    if sha256(path) != info["sha256"]:
        raise ArtifactError("Artifact checksum mismatch; refusing to load.")

    v_saved = meta.get("versions", {}).get("sklearn", "")
    if v_saved and v_saved.split(".")[0] != sklearn.__version__.split(".")[0]:
        raise ArtifactError(f"scikit-learn major version mismatch (trained {v_saved}, "
                            f"installed {sklearn.__version__}). Retrain the models.")
    elif v_saved and v_saved != sklearn.__version__:
        log.warning("scikit-learn minor version differs: trained on %s, running %s", v_saved, sklearn.__version__)

    return joblib.load(path)


def predict_patient(model, values: dict, threshold: float = 0.5) -> dict:
    """Evaluate patient values with transparent probabilities, thresholds, and risk levels."""
    problems = validate_patient_input(values)
    if problems:
        raise ValueError("Invalid input: " + "; ".join(problems))
    X = pd.DataFrame([{c: float(values[c]) for c in config.FEATURES}])

    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else (len(classes) - 1)
    prob = float(model.predict_proba(X)[0, pos_idx])
    threshold = float(threshold)
    higher = bool(prob >= threshold)

    if prob >= 0.70:
        risk_category = "High Risk"
        risk_badge = "High Risk (Significant Risk Factors)"
    elif prob >= threshold:
        risk_category = "Elevated Risk"
        risk_badge = "Elevated Risk (Above Threshold)"
    elif prob >= 0.35:
        risk_category = "Moderate / Borderline Risk"
        risk_badge = "Moderate / Borderline (Below Threshold)"
    else:
        risk_category = "Low Risk"
        risk_badge = "Low Risk (Favorable Clinical Profile)"

    confidence = prob if higher else (1.0 - prob)

    return dict(
        probability=prob,
        threshold=threshold,
        predicted_class=1 if higher else 0,
        higher_risk=higher,
        risk_category=risk_category,
        risk_badge=risk_badge,
        confidence=confidence,
        label="Model-estimated HIGHER-risk classification" if higher else "Model-estimated LOWER-risk classification",
        frame=X,
    )
