import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data.loader import load_prepared


@pytest.fixture(scope="session")
def frame():
    """Real Cleveland data when present; otherwise a TEST-ONLY schema-compatible frame (never used by the app)."""
    if config.DEFAULT_DATASET.exists():
        df, _ = load_prepared()
        return df
    rng = np.random.default_rng(0); n = 160
    d = {c: rng.integers(0, len(config.FEATURE_INFO[c][3]), n) for c in config.CATEGORICAL}
    d["thal"] = rng.integers(1, 4, n)
    d.update(age=rng.integers(30, 75, n), trestbps=rng.integers(100, 180, n), chol=rng.integers(150, 350, n),
             thalach=rng.integers(100, 200, n), oldpeak=rng.random(n) * 4)
    df = pd.DataFrame(d)
    df["target"] = ((df.cp == 0).astype(int) + (df.ca > 0) + (rng.random(n) > .6) >= 2).astype(int)
    return df


@pytest.fixture(scope="session")
def experiment(frame, tmp_path_factory):
    from src.models.train import run_training
    root = tmp_path_factory.mktemp("art")
    exp = run_training(frame, ["Logistic Regression", "Random Forest", "XGBoost", "Soft Voting", "Stacking"], mode="fast",
                       tune=False, artifact_root=root)
    return exp, root / "experiments" / exp["run_id"]
