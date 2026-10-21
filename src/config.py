"""Central configuration: paths, seeds, feature metadata, and training modes."""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, ARTIFACT_DIR, REPORT_DIR = ROOT / "data", ROOT / "artifacts", ROOT / "reports"
DEFAULT_DATASET = DATA_DIR / "heart_cleveland.csv"
# Public GitHub mirror of the recoded Cleveland data (the UCI host may be unreachable).
DATASET_URL = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"

SEED = 42
TARGET_CANDIDATES = ["target", "num", "output", "condition", "heart_disease", "disease"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [f for f in FEATURES if f not in NUMERIC]

# Encoding = the widely circulated recoded Cleveland file (alphabetical label encoding; verified against UCI
# category counts). UCI-original uploads are harmonised to this encoding by the loader.
# label, description, kind, allowed values / (min, max) for the recoded 0-based variant
FEATURE_INFO = {
    "age": ("Age", "Age in years", "num", (18, 100)),
    "sex": ("Sex", "Recorded sex", "cat", {0: "Female", 1: "Male"}),
    "cp": ("Chest pain type", "Chest pain category", "cat",
           {0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"}),
    "trestbps": ("Resting blood pressure", "mm Hg, at admission", "num", (60, 260)),
    "chol": ("Cholesterol", "Serum cholesterol, mg/dl", "num", (80, 700)),
    "fbs": ("Fasting blood sugar > 120 mg/dl", "Binary indicator", "cat", {0: "No", 1: "Yes"}),
    "restecg": ("Resting ECG", "Resting electrocardiographic result", "cat",
                {0: "Left ventricular hypertrophy", 1: "Normal", 2: "ST-T wave abnormality"}),
    "thalach": ("Maximum heart rate", "Maximum heart rate achieved, bpm", "num", (60, 230)),
    "exang": ("Exercise-induced angina", "Angina during exercise", "cat", {0: "No", 1: "Yes"}),
    "oldpeak": ("ST depression", "Exercise-induced ST depression vs. rest", "num", (0.0, 10.0)),
    "slope": ("ST slope", "Slope of peak-exercise ST segment", "cat",
              {0: "Downsloping", 1: "Flat", 2: "Upsloping"}),
    "ca": ("Major vessels (fluoroscopy)", "Number of major vessels coloured by fluoroscopy",
           "cat", {0: "0", 1: "1", 2: "2", 3: "3"}),
    "thal": ("Thalassemia (thal)", "Thallium stress test result", "cat",
             {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}),
}
LABELS = {k: v[0] for k, v in FEATURE_INFO.items()}

MODES = {  # fast = quick demo, research = broader search
    "fast": dict(cv_folds=3, search_iter=6, optuna_trials=0, fed_rounds=10, local_epochs=2),
    "research": dict(cv_folds=5, search_iter=25, optuna_trials=20, fed_rounds=30, local_epochs=3),
}
SELECTION_RULE = "validation score = 0.5 * recall + 0.5 * ROC-AUC (recall prioritised: false negatives are costly)"
DISCLAIMER = ("Research and educational use only. This tool is not clinically validated, does not diagnose "
              "or rule out heart disease, and cannot replace professional medical assessment.")
