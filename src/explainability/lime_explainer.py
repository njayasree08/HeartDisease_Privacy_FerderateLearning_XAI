"""LIME for single-record explanations using the same fitted pipeline and raw feature order."""
from __future__ import annotations
import numpy as np, pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from src import config


def build_explainer(X_train: pd.DataFrame, seed: int = config.SEED) -> LimeTabularExplainer:
    cat_idx = [config.FEATURES.index(c) for c in config.CATEGORICAL]
    names = {config.FEATURES.index(c): [config.FEATURE_INFO[c][3].get(k, str(k)) for k in range(max(config.FEATURE_INFO[c][3]) + 1)]
             for c in config.CATEGORICAL}
    return LimeTabularExplainer(X_train[config.FEATURES].values.astype(float), feature_names=[config.LABELS[c] for c in config.FEATURES],
                                class_names=["lower risk", "higher risk"], categorical_features=cat_idx, categorical_names=names,
                                discretize_continuous=True, random_state=seed, mode="classification")


def explain_instance(model, explainer, x_row: pd.DataFrame, n_features: int = 8, num_samples: int = 1500):
    """Return (list of (condition, weight) for the 'higher risk' class, model probability, local-fit R^2)."""
    idx = list(model.classes_).index(1)

    def fn(arr):
        p = model.predict_proba(pd.DataFrame(np.asarray(arr), columns=config.FEATURES))
        return np.column_stack([1 - p[:, idx], p[:, idx]])

    exp = explainer.explain_instance(x_row[config.FEATURES].values[0].astype(float), fn, num_features=n_features,
                                     num_samples=num_samples, labels=(1,))
    prob = float(fn(x_row[config.FEATURES].values.astype(float))[0, 1])
    return exp.as_list(label=1), prob, float(exp.score)
