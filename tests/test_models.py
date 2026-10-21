import numpy as np, pytest
from src.models import evaluate
from src.models.ensembles import make_pipeline


def test_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0]); p = np.array([.9, .8, .3, .6, .2, .1])  # tp2 fn1 fp1 tn2
    m = evaluate.compute_metrics(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6) and m["recall"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(8 / 9)
    assert evaluate.selection_score(m) == pytest.approx(0.5 * 2 / 3 + 0.5 * 8 / 9)


def test_all_models_trained_and_probabilities_valid(experiment):
    exp, run = experiment
    assert {"Logistic Regression", "Random Forest", "XGBoost", "Soft Voting", "Stacking"} <= {n for n, r in exp["models"].items() if "val" in r}
    for n, r in exp["models"].items():
        assert 0 <= r["test"]["roc_auc"] <= 1 and r["test"]["tp"] + r["test"]["fn"] > 0


def test_selection_follows_validation_scores(experiment):
    exp, _ = experiment
    scores = {n: r["selection_score"] for n, r in exp["models"].items() if "val" in r}
    assert exp["selected_model"] == max(scores, key=scores.get)


def test_class_order_and_probability_column(experiment, frame):
    from src import config
    from src.models.inference import load_model
    exp, run = experiment
    m = load_model(run, "Stacking"); assert list(m.classes_) == [0, 1]
    pr = m.predict_proba(frame[config.FEATURES].head(5)); assert np.allclose(pr.sum(axis=1), 1)


def test_pipeline_fits_with_missing(frame):
    from src import config
    X = frame[config.FEATURES].copy(); X.iloc[:4, 3] = np.nan
    assert len(make_pipeline("Logistic Regression").fit(X, frame["target"]).predict(X)) == len(X)


def test_optuna_xgboost_path(frame):
    from src import config
    from src.data.preprocessing import clean_frame, split_data
    from src.models.tuning import optuna_xgb
    (X, y) = split_data(clean_frame(frame))["train"]
    pipe, info = optuna_xgb(make_pipeline("XGBoost"), X, y, trials=2, cv_folds=3)
    assert info["optuna_trials"] == 2 and pipe.predict_proba(X).shape == (len(X), 2)
    assert optuna_xgb(make_pipeline("XGBoost"), X, y, trials=0)[1] == {}
