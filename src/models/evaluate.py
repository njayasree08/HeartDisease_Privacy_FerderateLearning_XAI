"""Metric and curve computation from real predictions."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from src import config


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return dict(accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
                f1=f1_score(y_true, y_pred, zero_division=0), roc_auc=auc,
                avg_precision=average_precision_score(y_true, y_prob),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def find_optimal_threshold(y_true, y_prob) -> float:
    """Find the optimal threshold on ROC curve via Youden's J statistic (max TPR - FPR)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_th = float(thresholds[best_idx])
    return float(np.clip(round(best_th, 3), 0.1, 0.9))


def selection_score(m: dict) -> float:
    """Documented rule (config.SELECTION_RULE): 0.5*recall + 0.5*ROC-AUC."""
    return 0.5 * m["recall"] + 0.5 * m["roc_auc"]


def roc_points(y_true, y_prob) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def pr_points(y_true, y_prob) -> pd.DataFrame:
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return pd.DataFrame({"recall": r, "precision": p})


def report_text(y_true, y_prob, threshold: float = 0.5) -> str:
    return classification_report(y_true, (np.asarray(y_prob) >= threshold).astype(int),
                                 target_names=["no disease", "disease"], zero_division=0)


def confusion_df(m: dict) -> pd.DataFrame:
    return pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                        index=["Actual: no disease", "Actual: disease"],
                        columns=["Pred: lower risk", "Pred: higher risk"])
