"""Model factories. Every model is a Pipeline (or ensemble of Pipelines) that takes RAW feature frames."""
from __future__ import annotations
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src import config
from src.data.preprocessing import build_preprocessor

BASE_NAMES = ["Logistic Regression", "Decision Tree", "SVM", "KNN", "Random Forest", "XGBoost"]
ENSEMBLE_NAMES = ["Soft Voting", "Stacking"]
ALL_NAMES = BASE_NAMES + ENSEMBLE_NAMES


def make_estimator(name: str, seed: int = config.SEED):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "SVM": SVC(probability=True, random_state=seed),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="logloss",
                                 random_state=seed, n_jobs=1, verbosity=0),
    }[name]


def make_pipeline(name: str, seed: int = config.SEED, outlier_strategy: str = "keep") -> Pipeline:
    return Pipeline([("prep", build_preprocessor(outlier_strategy)), ("clf", make_estimator(name, seed))])


def make_voting(bases: dict, weights=None) -> VotingClassifier:
    """Soft voting over probability outputs of the three primary bases."""
    est = [(n, clone(bases[n])) for n in ("Logistic Regression", "Random Forest", "XGBoost")]
    return VotingClassifier(est, voting="soft", weights=weights, n_jobs=1)


def make_stacking(bases: dict, seed: int = config.SEED, cv: int = 5) -> StackingClassifier:
    """LR / RF / XGB base learners, Logistic Regression meta-learner on out-of-fold probabilities."""
    est = [(n, clone(bases[n])) for n in ("Logistic Regression", "Random Forest", "XGBoost")]
    return StackingClassifier(est, final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
                              stack_method="predict_proba", cv=cv, n_jobs=1, passthrough=False)
