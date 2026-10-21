"""PDF report built from real experiment outputs (reportlab)."""
from __future__ import annotations
import io
from datetime import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from src import config


def _table(df: pd.DataFrame, col_w=None):
    data = [list(df.columns)] + [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r] for r in df.values.tolist()]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16325c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#c8d1db")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")])]))
    return t


def build_pdf(exp: dict, metrics: pd.DataFrame, fed: dict | None = None, shap_importance: pd.DataFrame | None = None,
              shap_png: bytes | None = None, patient: dict | None = None) -> bytes:
    """Return PDF bytes. Raises ValueError if there is no experiment to report."""
    if not exp:
        raise ValueError("No experiment available; train models first.")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    ss = getSampleStyleSheet(); H, B = ss["Heading2"], ss["BodyText"]
    s = [Paragraph("Privacy-Preserving and Explainable Heart Disease Risk Prediction", ss["Title"]),
         Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} | experiment {exp['run_id']}", B),
         Paragraph(f"<b>Notice:</b> {config.DISCLAIMER}", B), Spacer(1, 8)]
    s += [Paragraph("1. Dataset and configuration", H),
          Paragraph(f"Cleveland heart-disease data, {exp['n_records']} records after cleaning; splits {exp['split_sizes']}. "
                    f"Mode: {exp['mode']}; seed {exp['seed']}; hyperparameter tuning: {exp['tune']}; duplicates dropped: "
                    f"{exp['drop_duplicates']}; outlier strategy: {exp['outlier_strategy']}. "
                    f"Target note: {exp['data_info'].get('note', 'n/a')}", B),
          Paragraph(f"Selection rule: {exp['selection_rule']}. Selected model: <b>{exp['selected_model']}</b>.", B)]
    cols = ["model", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "fp", "fn"]
    s += [Paragraph("2. Held-out test metrics (all models)", H), _table(metrics[cols].round(3)), Spacer(1, 6),
          Paragraph("Accuracy alone is insufficient for medical screening: false negatives (missed cases) and false positives "
                    "carry different costs, so recall, specificity and ROC-AUC are reported alongside it. The test set is small, "
                    "so differences of one or two patients are within noise.", B)]
    sel = exp["models"].get(exp["selected_model"], {}).get("test")
    if sel:
        s += [Paragraph("3. Confusion matrix (selected model, test set)", H),
              _table(pd.DataFrame([["Actual: no disease", sel["tn"], sel["fp"]], ["Actual: disease", sel["fn"], sel["tp"]]],
                                  columns=["", "Pred lower risk", "Pred higher risk"]))]
    if fed:
        fc, ff, fz = fed["config"], fed["final_federated"], fed["final_centralized"]
        s += [Paragraph("4. Federated learning (simulated hospitals)", H),
              Paragraph(f"{fc['n_clients']} simulated clients, {fc['partition']} partition, {fc['rounds']} rounds, {fc['local_epochs']} local epochs; "
                        f"aggregation: {fc['aggregation']}. Client sizes: {fed['client_sizes']}. Raw records sent to server: "
                        f"{fed['comm']['raw_records_sent_to_server']}. Total simulated traffic: {fed['comm']['total_bytes'] / 1024:.0f} KiB.", B),
              _table(pd.DataFrame([["Federated NN", ff["accuracy"], ff["recall"], ff["roc_auc"]],
                                   ["Centralised NN", fz["accuracy"], fz["recall"], fz["roc_auc"]]],
                                  columns=["model", "accuracy", "recall", "roc_auc"])),
              Paragraph("Model updates can leak information; this prototype uses no secure aggregation or differential privacy.", B)]
    if shap_importance is not None:
        s += [Paragraph("5. SHAP global importance (selected model)", H), _table(shap_importance.head(10).round(4))]
        if shap_png:
            s.append(Image(io.BytesIO(shap_png), width=13 * cm, height=8.5 * cm))
    if patient:
        s += [Paragraph("6. Patient-specific explanation (explicitly requested)", H),
              Paragraph(f"{patient['label']} - model estimate {patient['probability'] * 100:.1f}% (uncalibrated). "
                        f"{patient.get('narrative', '')}", B),
              _table(pd.DataFrame([(config.LABELS[k], v) for k, v in patient["inputs"].items()], columns=["input", "value"]))]
    s += [Paragraph("Limitations", H),
          Paragraph("Small single-source dataset; not externally validated; probabilities not calibrated; SHAP/LIME describe model "
                    "behaviour and are not causal; simulated clients are not real hospitals; no formal privacy guarantee; "
                    "no regulatory compliance assessment has been performed.", B), Paragraph(config.DISCLAIMER, B)]
    doc.build(s)
    return buf.getvalue()
