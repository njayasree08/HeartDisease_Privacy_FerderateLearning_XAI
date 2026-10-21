"""Leakage-safe preprocessing: all fitting happens inside sklearn Pipelines on training data only."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


class IQRClipper(BaseEstimator, TransformerMixin):
    """Optional winsorising to IQR bounds learned from training data (off by default)."""
    def __init__(self, k: float = 1.5):
        self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        self.lo_, self.hi_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def build_preprocessor(outlier_strategy: str = "keep") -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot categoricals (unknown codes ignored)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if outlier_strategy == "winsorize_iqr":
        num_steps.append(("clip", IQRClipper()))
    num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", Pipeline(num_steps), config.NUMERIC),
                              ("cat", cat, config.CATEGORICAL)], verbose_feature_names_out=False)


def clean_frame(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    out = df[config.FEATURES + ["target"]].copy()
    if drop_duplicates:
        out = out.drop_duplicates()
    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int = config.SEED, val_size: float = 0.2, test_size: float = 0.2):
    """Stratified train/validation/test split. Returns dict of (X, y) frames/series."""
    X, y = df[config.FEATURES], df["target"]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=val_size / (1 - test_size),
                                              stratify=y_tmp, random_state=seed)
    return {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)}


def split_summary(splits: dict) -> pd.DataFrame:
    rows = [{"split": k, "records": len(y), "positives": int(y.sum()),
             "positive_rate": round(float(y.mean()), 3)} for k, (_, y) in splits.items()]
    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Reference bounds only; nothing is removed."""
    rows = []
    for c in config.NUMERIC:
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi, m, sd = q1 - 1.5 * iqr, q3 + 1.5 * iqr, s.mean(), s.std()
        rows.append({"feature": c, "iqr_low": lo, "iqr_high": hi, "iqr_outliers": int(((s < lo) | (s > hi)).sum()),
                     "3sd_low": m - 3 * sd, "3sd_high": m + 3 * sd,
                     "3sd_outliers": int(((s < m - 3 * sd) | (s > m + 3 * sd)).sum())})
    return pd.DataFrame(rows).round(2)
