import json, pytest
from src.models.train import metrics_table
from src.reporting import csv_export
from src.reporting.pdf_report import build_pdf


def test_pdf_built_from_experiment(experiment):
    exp, _ = experiment
    pdf = build_pdf(exp, metrics_table(exp, "test"), patient=dict(label="x", probability=.4, inputs={"age": 50}, narrative="n"))
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000


def test_pdf_requires_experiment():
    with pytest.raises(ValueError):
        build_pdf({}, None)


def test_metrics_csv(experiment):
    exp, _ = experiment
    assert b"roc_auc" in csv_export.metrics_csv(metrics_table(exp, "test"))
