"""Training orchestration: baseline, tuning, ensembles, validation-based selection, artifact saving."""
from __future__ import annotations
import json, logging, platform, time
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn, xgboost
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from src import config
from src.data.preprocessing import clean_frame, split_data
from src.models import ensembles, evaluate, tuning
from src.models.inference import sha256

log = logging.getLogger(__name__)


def run_training(df, model_names=None, mode="fast", seed=config.SEED, tune=True, drop_duplicates=True,
                 outlier_strategy="keep", artifact_root: Path = config.ARTIFACT_DIR, progress=None, data_info=None):
    """Train requested models; select on VALIDATION; report TEST once. Returns the experiment record dict."""
    cfg = config.MODES[mode]
    names = list(model_names or ensembles.ALL_NAMES)
    data = clean_frame(df, drop_duplicates)
    splits = split_data(data, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["val"], splits["test"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(artifact_root) / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    fitted, records, preds = {}, {}, {"val": {"y_true": yva.values}, "test": {"y_true": yte.values}}

    def step(msg, frac):
        if progress:
            progress(msg, frac)

    base_names = [n for n in ensembles.BASE_NAMES if n in names or any(e in names for e in ensembles.ENSEMBLE_NAMES)]
    bases = {}
    total = len(base_names) + len([n for n in names if n in ensembles.ENSEMBLE_NAMES])
    done = 0
    for n in base_names:
        step(f"Training {n}", done / total)
        t0 = time.perf_counter()
        try:
            pipe, info = ensembles.make_pipeline(n, seed, outlier_strategy), {}
            if tune:
                pipe, info = tuning.tune(n, pipe, Xtr, ytr, cfg["cv_folds"], cfg["search_iter"], seed)
                if n == "XGBoost" and cfg["optuna_trials"] > 0:
                    pipe, oinfo = tuning.optuna_xgb(pipe, Xtr, ytr, cfg["optuna_trials"], cfg["cv_folds"], seed)
                    info.update(oinfo)
            else:
                pipe.fit(Xtr, ytr)
            bases[n] = pipe
            if n in names:
                fitted[n] = pipe
                records[n] = {"tuning": info, "train_seconds": time.perf_counter() - t0}
        except Exception as exc:  # keep the app alive; record the failure
            log.exception("Model %s failed", n)
            records[n] = {"error": f"{type(exc).__name__}: {exc}"}
        done += 1
    for n in [m for m in ensembles.ENSEMBLE_NAMES if m in names]:
        step(f"Training {n}", done / total)
        t0 = time.perf_counter()
        try:
            need = {"Logistic Regression", "Random Forest", "XGBoost"}
            if not need <= set(bases):
                raise RuntimeError(f"Requires base models {sorted(need - set(bases))}, which failed or are missing.")
            ens = ensembles.make_voting(bases) if n == "Soft Voting" else ensembles.make_stacking(bases, seed, cfg["cv_folds"])
            ens.fit(Xtr, ytr)
            fitted[n] = ens
            records[n] = {"tuning": {"note": "base estimators reuse tuned pipelines"}, "train_seconds": time.perf_counter() - t0}
        except Exception as exc:
            log.exception("Ensemble %s failed", n)
            records[n] = {"error": f"{type(exc).__name__}: {exc}"}
        done += 1

    for n, model in fitted.items():
        t0 = time.perf_counter()
        pv = model.predict_proba(Xva)[:, list(model.classes_).index(1)]
        pt = model.predict_proba(Xte)[:, list(model.classes_).index(1)]
        latency_ms = (time.perf_counter() - t0) / (len(Xva) + len(Xte)) * 1000
        preds["val"][n], preds["test"][n] = pv, pt
        mv, mt = evaluate.compute_metrics(yva, pv), evaluate.compute_metrics(yte, pt)
        opt_thresh = evaluate.find_optimal_threshold(yva, pv)
        # Cross-validation of the already-selected configuration on TRAIN only (fresh clone per fold).
        cvres = cross_validate(clone(model), Xtr, ytr, cv=cv, scoring=["accuracy", "recall", "roc_auc"], n_jobs=1)
        records[n].update(val=mv, test=mt, optimal_threshold=opt_thresh, selection_score=evaluate.selection_score(mv), latency_ms_per_row=latency_ms,
                          cv={k: [float(np.mean(cvres[f"test_{k}"])), float(np.std(cvres[f"test_{k}"]))]
                              for k in ("accuracy", "recall", "roc_auc")})
        joblib.dump(model, run_dir / f"{n.replace(' ', '_')}.joblib")
        records[n]["file"] = f"{n.replace(' ', '_')}.joblib"
        records[n]["sha256"] = sha256(run_dir / records[n]["file"])

    ok = {n: r for n, r in records.items() if "val" in r}
    selected = max(ok, key=lambda n: ok[n]["selection_score"]) if ok else None
    pd.DataFrame({"y_true": preds["test"]["y_true"], **{n: preds["test"][n] for n in ok}}).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"y_true": preds["val"]["y_true"], **{n: preds["val"][n] for n in ok}}).to_csv(run_dir / "val_predictions.csv", index=False)
    for k, (X, y) in splits.items():
        X.assign(target=y).to_csv(run_dir / f"split_{k}.csv", index=True)  # index = row id in cleaned frame

    exp = dict(run_id=run_id, timestamp=datetime.now().isoformat(timespec="seconds"), mode=mode, seed=seed,
               tune=tune, drop_duplicates=drop_duplicates, outlier_strategy=outlier_strategy,
               selection_rule=config.SELECTION_RULE, selected_model=selected, data_info=data_info or {},
               n_records=len(data), split_sizes={k: len(v[1]) for k, v in splits.items()}, settings=cfg,
               models=records, versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                             xgboost=xgboost.__version__, pandas=pd.__version__, numpy=np.__version__))
    (run_dir / "experiment.json").write_text(json.dumps(exp, indent=2, default=float))
    step("Done", 1.0)
    return exp


def metrics_table(exp: dict, split: str = "test") -> pd.DataFrame:
    rows = [{"model": n, **{k: v for k, v in r[split].items()}, "train_seconds": r["train_seconds"],
             "val_selection_score": r["selection_score"]} for n, r in exp["models"].items() if split in r]
    return pd.DataFrame(rows)
