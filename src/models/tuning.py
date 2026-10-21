"""Hyperparameter search (always on the TRAINING split only)."""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from src import config

GRIDS = {
    "Logistic Regression": ("grid", {"clf__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": ("grid", {"clf__max_depth": [2, 3, 4, 5, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "KNN": ("grid", {"clf__n_neighbors": [3, 5, 7, 9, 13], "clf__weights": ["uniform", "distance"]}),
    "SVM": ("random", {"clf__C": np.logspace(-2, 2, 20), "clf__gamma": ["scale", 0.01, 0.05, 0.1], "clf__kernel": ["rbf", "linear"]}),
    "Random Forest": ("random", {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 5, 8, None],
                                 "clf__min_samples_leaf": [1, 2, 4]}),
    "XGBoost": ("random", {"clf__n_estimators": [50, 100, 200, 300], "clf__max_depth": [2, 3, 4],
                           "clf__learning_rate": [0.01, 0.03, 0.1, 0.2], "clf__subsample": [0.7, 0.85, 1.0],
                           "clf__colsample_bytree": [0.7, 0.85, 1.0]}),
}


def tune(name, pipe, X, y, cv_folds=3, n_iter=6, seed=config.SEED):
    """Return (best_pipeline, info). GridSearchCV for small grids, RandomizedSearchCV otherwise."""
    kind, space = GRIDS[name]
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    if kind == "grid":
        s = GridSearchCV(pipe, space, scoring="roc_auc", cv=cv, n_jobs=1)
    else:
        s = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv, random_state=seed, n_jobs=1)
    s.fit(X, y)
    return s.best_estimator_, dict(search=type(s).__name__, best_params={k: _py(v) for k, v in s.best_params_.items()},
                                   best_cv_roc_auc=float(s.best_score_))


def optuna_xgb(pipe, X, y, trials, cv_folds=3, seed=config.SEED):
    """Optional Optuna refinement of XGBoost; returns (pipeline, info) or the input pipeline if trials<=0."""
    if trials <= 0:
        return pipe, {}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)

    def objective(t):
        p = clone(pipe).set_params(clf__n_estimators=t.suggest_int("n", 50, 300), clf__max_depth=t.suggest_int("d", 2, 5),
                                   clf__learning_rate=t.suggest_float("lr", 0.01, 0.3, log=True),
                                   clf__subsample=t.suggest_float("ss", 0.6, 1.0),
                                   clf__colsample_bytree=t.suggest_float("cs", 0.6, 1.0))
        return cross_val_score(p, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=trials)
    best = clone(pipe).set_params(clf__n_estimators=st.best_params["n"], clf__max_depth=st.best_params["d"],
                                  clf__learning_rate=st.best_params["lr"], clf__subsample=st.best_params["ss"],
                                  clf__colsample_bytree=st.best_params["cs"])
    return best.fit(X, y), dict(optuna_trials=trials, optuna_best_cv_roc_auc=float(st.best_value))


def _py(v):
    return v.item() if hasattr(v, "item") else v
