"""Dataset loading (CSV & Excel), target normalisation, clinical verification, and variant detection."""
from __future__ import annotations
import io, logging, urllib.request
from pathlib import Path
import pandas as pd
from src import config

log = logging.getLogger(__name__)


class DatasetError(Exception):
    """Raised for unreadable, missing, or malformed datasets."""


def load_dataset_file(source, file_type: str | None = None) -> pd.DataFrame:
    """Load dataset from CSV or Excel (file path or file-like / UploadedFile object)."""
    try:
        # Determine format
        name = getattr(source, "name", str(source)).lower() if hasattr(source, "name") or isinstance(source, (str, Path)) else ""
        is_excel = (file_type in ("excel", "xlsx", "xls")) or name.endswith((".xlsx", ".xls"))

        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise DatasetError(f"Dataset not found at {source}. Download it from the sidebar, upload a CSV/Excel file, "
                                   f"or obtain the original data from {config.UCI_PAGE}.")
            if is_excel:
                df = pd.read_excel(src_path, na_values=["?", "NA", "null", ""])
            else:
                df = pd.read_csv(src_path, encoding="utf-8-sig", na_values=["?", "NA", "null", ""])
        else:
            # File-like or UploadedFile
            if hasattr(source, "seek"):
                source.seek(0)
            if is_excel:
                df = pd.read_excel(source, na_values=["?", "NA", "null", ""])
            else:
                try:
                    df = pd.read_csv(source, encoding="utf-8-sig", na_values=["?", "NA", "null", ""])
                except Exception:
                    if hasattr(source, "seek"):
                        source.seek(0)
                    df = pd.read_excel(source, na_values=["?", "NA", "null", ""])
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Could not read dataset file: {exc}") from exc

    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_csv(source) -> pd.DataFrame:
    """Load a CSV or Excel dataset from a path or an uploaded file-like object (backward compatibility)."""
    return load_dataset_file(source)


def download_dataset(url: str = config.DATASET_URL, dest: Path = config.DEFAULT_DATASET) -> Path:
    """Download the public CSV. Raises DatasetError with guidance on failure."""
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as exc:
        raise DatasetError(f"Download failed ({exc}). Upload the CSV manually; original data: {config.UCI_PAGE}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def find_target_column(df: pd.DataFrame) -> str:
    """Identify the target column from candidates or heuristic fallback."""
    for c in config.TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # Secondary heuristics
    for c in ["diag", "diagnosis", "class", "label", "outcome", "heartdisease"]:
        if c in df.columns:
            return c
    # If the last column looks binary or discrete
    last_col = df.columns[-1]
    if df[last_col].nunique(dropna=True) <= 5:
        return last_col
    raise DatasetError(f"No target column found. Expected one of {config.TARGET_CANDIDATES} or a binary outcome column.")


def verify_target_semantics(df: pd.DataFrame, target_col: str) -> dict:
    """Clinically verify whether target=1 represents heart disease or no disease.

    Established clinical correlations for coronary artery disease:
    - oldpeak (ST depression): higher -> disease (+)
    - exang (exercise angina): present -> disease (+)
    - ca (major fluoroscopy vessels): higher -> disease (+)
    - thalach (max HR achieved): lower in disease, higher in healthy (-)
    - age: higher -> disease (+)
    """
    raw_s = pd.to_numeric(df[target_col], errors="coerce")
    if raw_s.isna().all() or raw_s.nunique() < 2:
        return {
            "verified": False,
            "positive_class": "unknown",
            "recommended_invert": False,
            "correlations": {},
            "evidence": "Target has insufficient variation to verify.",
        }

    # Binarize for correlation if multiclass
    b_target = (raw_s > 0).astype(float) if raw_s.max() > 1 else raw_s.astype(float)

    # Marker expected direction for DISEASE (positive class):
    # +1 means higher feature value indicates disease
    # -1 means higher feature value indicates healthy
    markers = {"oldpeak": +1, "exang": +1, "ca": +1, "thalach": -1, "age": +1}
    correlations = {}
    disease_votes = 0
    healthy_votes = 0

    for col, expected_dir in markers.items():
        if col in df.columns:
            feat_s = pd.to_numeric(df[col], errors="coerce")
            valid = (~feat_s.isna()) & (~b_target.isna())
            if valid.sum() >= 3:
                corr = float(feat_s[valid].corr(b_target[valid]))
                correlations[col] = round(corr, 3)
                # If corr aligns with expected direction, raw 1 = disease
                if corr * expected_dir > 0.05:
                    disease_votes += 1
                elif corr * expected_dir < -0.05:
                    healthy_votes += 1

    total_votes = disease_votes + healthy_votes
    if healthy_votes > disease_votes:
        # In this dataset, raw target=1 corresponds to HEALTHY, and raw target=0 corresponds to DISEASE!
        # Therefore, we MUST invert so that 1 = HEART DISEASE.
        invert = True
        pos_meaning = "Raw target 1 represents NO DISEASE (inversion required so positive class = Heart Disease)"
        summary = (f"Clinical verification detected inverted target: {healthy_votes}/{total_votes} cardiac markers "
                   f"(e.g., oldpeak={correlations.get('oldpeak', 0):+.2f}, exang={correlations.get('exang', 0):+.2f}, "
                   f"thalach={correlations.get('thalach', 0):+.2f}) correlate negatively with raw 1. "
                   "Target is inverted so 1 = Heart Disease.")
    elif disease_votes > healthy_votes:
        invert = False
        pos_meaning = "Raw target 1 represents HEART DISEASE (standard positive class)"
        summary = (f"Clinical verification confirmed standard target: {disease_votes}/{total_votes} cardiac markers "
                   f"(e.g., oldpeak={correlations.get('oldpeak', 0):+.2f}, exang={correlations.get('exang', 0):+.2f}, "
                   f"thalach={correlations.get('thalach', 0):+.2f}) correlate positively with raw 1. "
                   "Target 1 correctly represents Heart Disease.")
    else:
        invert = False
        pos_meaning = "Target 1 taken as Heart Disease (default)"
        summary = "Cardiac marker correlation was neutral or features missing. Defaulting to target 1 = Heart Disease."

    return {
        "verified": True,
        "positive_class": pos_meaning,
        "recommended_invert": invert,
        "correlations": correlations,
        "disease_votes": disease_votes,
        "healthy_votes": healthy_votes,
        "evidence": summary,
    }


def detect_variant(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """Identify dataset encoding and default target semantics using clinical verification."""
    target_col = target_col or find_target_column(df)
    vals = set(pd.Series(df[target_col]).dropna().unique().tolist())

    if vals - {0, 1} or (target_col == "num" and len(vals) > 2):
        return dict(
            variant="uci_original",
            invert=False,
            note="Multiclass 'num' detected; values > 0 mapped to disease.",
            verification={"evidence": "Multiclass UCI encoding: values 1-4 mapped to disease present (target=1)."},
        )

    # Perform clinical verification on binary target
    verif = verify_target_semantics(df, target_col)
    variant_name = "recoded_0based" if ("cp" in df and df["cp"].min() == 0) else "binary_dataset"

    return dict(
        variant=variant_name,
        invert=verif["recommended_invert"],
        note=verif["evidence"],
        verification=verif,
    )


def to_binary_target(df: pd.DataFrame, target_col: str, invert: bool = False) -> pd.DataFrame:
    """Return a copy with a clean binary column 'target' (1 = disease present). Never used as a predictor."""
    out = df.copy()
    raw = pd.to_numeric(out[target_col], errors="coerce")
    if raw.isna().any():
        raise DatasetError("Target contains non-numeric or missing values.")
    y = (raw > 0).astype(int) if raw.max() > 1 else raw.astype(int)
    if target_col in out.columns:
        out = out.drop(columns=[target_col])
    out["target"] = (1 - y) if invert else y
    return out


def harmonise(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Bring data to the project encoding and turn placeholder codes into real missing values.
    recoded_0based: ca==4 and thal==0 are placeholders for missing -> NaN.
    uci_original: map UCI codes (cp 1-4, restecg 0-2, slope 1-3, thal 3/6/7) onto the recoded encoding."""
    out = df.copy()
    if variant == "uci_original":
        maps = {"cp": {1: 3, 2: 1, 3: 2, 4: 0}, "restecg": {0: 1, 1: 2, 2: 0},
                "slope": {1: 2, 2: 1, 3: 0}, "thal": {3: 2, 6: 1, 7: 3}}
        for c, m in maps.items():
            if c in out:
                out[c] = out[c].map(m)
    else:
        if "ca" in out:
            out["ca"] = out["ca"].where(out["ca"] != 4)
        if "thal" in out:
            out["thal"] = out["thal"].where(out["thal"] != 0)
    return out


def load_prepared(source=config.DEFAULT_DATASET, invert: bool | None = None, target_col: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load, detect variant, and return (frame with binary 'target', info dict)."""
    raw = load_dataset_file(source)
    tcol = target_col or find_target_column(raw)
    info = detect_variant(raw, tcol)
    if invert is not None:
        info["invert"] = invert
    return harmonise(to_binary_target(raw, tcol, info["invert"]), info["variant"]), info
