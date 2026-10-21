"""SHAP on the FULL prediction function (raw features -> P(disease)); works for any fitted pipeline/ensemble."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap
from src import config


def _fn(model):
    idx = list(model.classes_).index(1)
    return lambda d: model.predict_proba(pd.DataFrame(d, columns=config.FEATURES))[:, idx]


def explain(model, X_background: pd.DataFrame, X_explain: pd.DataFrame, max_background: int = 50,
            max_evals: int = 120, seed: int = config.SEED) -> shap.Explanation:
    """Permutation-based SHAP (model-agnostic). Output values are in probability units for class 'disease'."""
    bg = X_background.sample(min(max_background, len(X_background)), random_state=seed)[config.FEATURES]
    masker = shap.maskers.Independent(bg.values, max_samples=max_background)
    ex = shap.Explainer(_fn(model), masker, algorithm="permutation", feature_names=[config.LABELS[c] for c in config.FEATURES],
                        seed=seed)
    return ex(X_explain[config.FEATURES].values, max_evals=max_evals, silent=True)


def global_importance(exp: shap.Explanation) -> pd.DataFrame:
    imp = np.abs(exp.values).mean(axis=0)
    return pd.DataFrame({"feature": exp.feature_names, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summary_fig(exp):
    plt.figure(figsize=(7, 5)); shap.plots.beeswarm(exp, show=False, max_display=13); plt.tight_layout()
    return plt.gcf()


def bar_fig(exp):
    plt.figure(figsize=(7, 4.5)); shap.plots.bar(exp, show=False, max_display=13); plt.tight_layout()
    return plt.gcf()


def waterfall_fig(exp, i: int = 0):
    plt.figure(figsize=(7, 5)); shap.plots.waterfall(exp[i], show=False, max_display=10); plt.tight_layout()
    return plt.gcf()


def describe(exp, i: int = 0, top: int = 3) -> str:
    """Plain-language summary. Associations with the MODEL output only - not causal medical effects."""
    order = np.argsort(-np.abs(exp.values[i]))[:top]
    parts = [f"{exp.feature_names[j]} ({'raised' if exp.values[i][j] > 0 else 'lowered'} the model estimate by "
             f"{abs(exp.values[i][j]) * 100:.1f} percentage points)" for j in order]
    return "Most influential inputs for this model output: " + "; ".join(parts) + ". These describe model behaviour, not causes."
