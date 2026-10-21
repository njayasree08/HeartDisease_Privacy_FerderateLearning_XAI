
"""Clinical AI model evaluation dashboard."""
import pandas as pd
import streamlit as st

from src import config, ui
from src.models import evaluate
from src.models.train import metrics_table
from src.visualization import charts

ui.setup("Model Evaluation")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">CLINICAL AI · PERFORMANCE ANALYSIS</div>
        <h1>Model Evaluation</h1>
        <p>Compare predictive performance, examine classification errors,
        and review the validation-based model selection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

exp, run = ui.require_experiment()
tp = pd.read_csv(run / "test_predictions.csv")
models = [column for column in tp.columns if column != "y_true"]
tab = metrics_table(exp, "test")

st.info(
    "Evaluation uses the held-out TEST split, which is separate from "
    "model fitting, tuning, and selection. "
    f"Test size: {len(tp)} records. With a small test set, differences "
    "of one or two cases can be due to sampling variation."
)

st.markdown("### 01 · Performance overview")

metric_cols = st.columns(4)
overview_metrics = [
    ("Models evaluated", str(len(models))),
    ("Test records", str(len(tp))),
    ("Selected model", str(exp["selected_model"])),
    ("Selection basis", "Validation"),
]

for col, (label, value) in zip(metric_cols, overview_metrics):
    col.metric(label, value)

st.plotly_chart(
    charts.metric_bars(tab),
    width="stretch",
)

st.markdown("### 02 · Model comparison")
st.caption(
    "Values are calculated on the held-out test split. "
    "Scores are rounded to three decimal places for readability."
)

display_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "fp",
    "fn",
    "train_seconds",
]

st.dataframe(
    tab[display_columns].round(3),
    hide_index=True,
    width="stretch",
)

with st.expander("How should these metrics be interpreted?"):
    st.markdown(
        """
        - **Accuracy:** proportion of all predictions that are correct.
        - **Precision:** proportion of positive predictions that are correct.
        - **Recall:** proportion of actual positive cases identified.
        - **Specificity:** proportion of actual negative cases identified.
        - **F1-score:** balance between precision and recall.
        - **ROC-AUC:** ability to distinguish the two classes across thresholds.
        - **FP (false positives):** negative cases incorrectly predicted positive.
        - **FN (false negatives):** positive cases incorrectly predicted negative.
        - **Training time:** measured training duration recorded by the project.
        
        For screening applications, false negatives and false positives have
        different consequences. Accuracy alone is therefore insufficient.
        These results are for research and education, not clinical validation.
        """
    )

st.divider()
st.markdown("### 03 · Detailed model analysis")

default_model = (
    models.index(exp["selected_model"])
    if exp["selected_model"] in models
    else 0
)

selected = st.selectbox(
    "Choose a model to inspect",
    models,
    index=default_model,
)

eval_opt_thresh = float(exp["models"][selected].get("optimal_threshold", 0.50))

th_c1, th_c2 = st.columns([1.5, 1], gap="large")
with th_c1:
    eval_threshold = st.slider(
        f"Evaluation Decision Threshold for {selected}",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Simulate the effect of changing the probability threshold on precision, recall, and confusion matrix.",
        key=f"eval_thresh_{selected}",
    )
with th_c2:
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-top:6px;font-size:0.83rem;">
            <div style="font-weight:700;color:#0f172a;">Threshold Benchmark</div>
            <div style="color:#64748b;margin-top:4px;">
                Active Threshold: <strong>{eval_threshold:.2f}</strong><br>
                Validation Optimal (Youden's J): <strong>{eval_opt_thresh:.2f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Compute metrics at user-selected threshold
sim_metrics = evaluate.compute_metrics(tp["y_true"], tp[selected], threshold=eval_threshold)

sim_cols = st.columns(5)
ui.card(sim_cols[0], "Accuracy", f"{sim_metrics['accuracy']:.3f}", subtext=f"@ threshold {eval_threshold:.2f}", accent="teal")
ui.card(sim_cols[1], "Sensitivity (Recall)", f"{sim_metrics['recall']:.3f}", subtext=f"True Pos: {sim_metrics['tp']}", accent="navy")
ui.card(sim_cols[2], "Specificity", f"{sim_metrics['specificity']:.3f}", subtext=f"True Neg: {sim_metrics['tn']}", accent="teal")
ui.card(sim_cols[3], "Precision", f"{sim_metrics['precision']:.3f}", subtext=f"False Pos: {sim_metrics['fp']}", accent="navy")
ui.card(sim_cols[4], "F1-Score", f"{sim_metrics['f1']:.3f}", subtext=f"False Neg: {sim_metrics['fn']}", accent="teal")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### Confusion matrix")
    st.plotly_chart(
        charts.confusion_heatmap(
            evaluate.confusion_df(sim_metrics),
            f"{selected}: Confusion Matrix (Threshold = {eval_threshold:.2f})",
        ),
        width="stretch",
    )

with right:
    st.markdown("#### Classification report")
    st.code(
        evaluate.report_text(tp["y_true"], tp[selected], threshold=eval_threshold),
        language="text",
    )

st.divider()
st.markdown("### 04 · ROC and precision–recall curves")
st.caption(
    "Select models to compare their curves. These plots use the test-set "
    "predictions and are intended for descriptive evaluation."
)

chosen = st.multiselect(
    "Models to compare",
    models,
    default=models[: min(4, len(models))],
)

if chosen:
    roc = {
        model_name: (
            evaluate.roc_points(tp["y_true"], tp[model_name]),
            exp["models"][model_name]["test"]["roc_auc"],
        )
        for model_name in chosen
    }

    pr = {
        model_name: (
            evaluate.pr_points(tp["y_true"], tp[model_name]),
            exp["models"][model_name]["test"]["avg_precision"],
        )
        for model_name in chosen
    }

    roc_col, pr_col = st.columns(2, gap="large")

    with roc_col:
        st.markdown("#### ROC curve")
        st.plotly_chart(
            charts.roc_fig(roc),
            width="stretch",
        )

    with pr_col:
        st.markdown("#### Precision–recall curve")
        st.plotly_chart(
            charts.pr_fig(pr),
            width="stretch",
        )
else:
    st.warning("Select at least one model to display the curves.")

st.divider()
st.markdown("### 05 · Cross-validation results")
st.caption(
    "Cross-validation statistics are calculated on the training split. "
    "The values below show mean ± standard deviation."
)

cv = pd.DataFrame(
    [
        {
            "model": model_name,
            **{
                f"{metric} cv": f"{values[0]:.3f} ± {values[1]:.3f}"
                for metric, values in model_result["cv"].items()
            },
        }
        for model_name, model_result in exp["models"].items()
        if "cv" in model_result
    ]
)

if not cv.empty:
    st.dataframe(
        cv,
        hide_index=True,
        width="stretch",
    )
else:
    st.info("Cross-validation results are not available for this experiment.")

st.markdown("### 06 · Model selection")

st.success(
    f"Selected model: {exp['selected_model']}. "
    f"Selection was based on validation results using the rule: "
    f"“{exp['selection_rule']}”."
)

st.caption(
    "The selected model is chosen using validation results, rather than "
    "selecting a model based on its held-out test performance. "
    "These metrics alone do not establish clinical safety or effectiveness."
)
