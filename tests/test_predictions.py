import json, pytest
from src import config
from src.explainability import lime_explainer as le, shap_explainer as se
from src.models.inference import ArtifactError, load_model, predict_patient
from src.data.preprocessing import clean_frame, split_data

P = dict(age=55, sex=1, cp=0, trestbps=130, chol=240, fbs=0, restecg=1, thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=2)


def test_predict_valid_and_invalid(experiment):
    exp, run = experiment; m = load_model(run, exp["selected_model"])
    r = predict_patient(m, P)
    assert 0 <= r["probability"] <= 1 and "classification" in r["label"]
    assert "threshold" in r and "predicted_class" in r and "risk_category" in r
    assert r["threshold"] == 0.5
    r_cust = predict_patient(m, P, threshold=0.35)
    assert r_cust["threshold"] == 0.35
    with pytest.raises(ValueError):
        predict_patient(m, {**P, "age": 500})


def test_tampered_artifact_refused(experiment):
    exp, run = experiment
    with open(run / exp["models"]["Logistic Regression"]["file"], "ab") as f:
        f.write(b"x")
    with pytest.raises(ArtifactError):
        load_model(run, "Logistic Regression")


def test_shap_and_lime_work_on_stacking(experiment, frame):
    exp, run = experiment; m = load_model(run, "Stacking")
    sp = split_data(clean_frame(frame)); Xtr, Xte = sp["train"][0], sp["test"][0]
    e = se.explain(m, Xtr, Xte.iloc[:3], max_background=20, max_evals=60)
    assert e.values.shape == (3, 13) and len(se.global_importance(e)) == 13
    pairs, prob, _ = le.explain_instance(m, le.build_explainer(Xtr), Xte.iloc[[0]], 5, num_samples=300)
    assert len(pairs) == 5 and 0 <= prob <= 1
    assert prob == pytest.approx(predict_patient(m, Xte.iloc[0].to_dict())["probability"], abs=1e-6)
