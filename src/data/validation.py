"""Schema, range and patient-input validation."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from src import config


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    n_rows: int = 0
    n_missing: int = 0
    n_duplicates: int = 0
    missing_by_col: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_schema(df: pd.DataFrame, target_col: str = "target") -> ValidationReport:
    r = ValidationReport(n_rows=len(df))
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        r.errors.append(f"Missing required feature columns: {missing}")
    if target_col not in df.columns:
        r.errors.append(f"Missing target column '{target_col}'.")
    if r.errors:
        return r
    feats = df[config.FEATURES]
    r.missing_by_col = feats.isna().sum().to_dict()
    r.n_missing = int(sum(r.missing_by_col.values()))
    r.n_duplicates = int(df.duplicated().sum())
    for c in config.FEATURES:
        if not pd.api.types.is_numeric_dtype(df[c]):
            r.errors.append(f"Column '{c}' must be numeric (found {df[c].dtype}).")
    if r.errors:
        return r
    for c, (_, _, kind, spec) in config.FEATURE_INFO.items():
        col = df[c].dropna()
        bad = (~col.isin(spec.keys())).sum() if kind == "cat" else ((col < spec[0]) | (col > spec[1])).sum()
        if bad:
            r.warnings.append(f"{bad} value(s) in '{c}' fall outside the expected range/codes.")
    if r.n_missing:
        r.warnings.append(f"{r.n_missing} missing feature value(s); they are imputed inside the pipeline.")
    if r.n_duplicates:
        r.warnings.append(f"{r.n_duplicates} duplicate record(s) detected.")
    if df[target_col].nunique() < 2:
        r.errors.append("Target has fewer than two classes.")
    return r


def validate_patient_input(values: dict) -> list[str]:
    """Return a list of problems with a single patient's inputs (empty list = valid)."""
    problems = []
    for c in config.FEATURES:
        if c not in values or values[c] is None:
            problems.append(f"'{c}' is required.")
            continue
        _, _, kind, spec = config.FEATURE_INFO[c]
        v = values[c]
        try:
            v = float(v)
        except (TypeError, ValueError):
            problems.append(f"'{c}' must be numeric.")
            continue
        if kind == "cat" and v not in spec:
            problems.append(f"'{c}' must be one of {list(spec)}.")
        if kind == "num" and not (spec[0] <= v <= spec[1]):
            problems.append(f"'{c}' must be between {spec[0]} and {spec[1]}.")
    return problems
