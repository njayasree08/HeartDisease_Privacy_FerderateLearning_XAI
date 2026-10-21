"""Modern Healthcare Dashboard: Heart Disease Risk Prediction, Federated Learning & Explainable AI."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src import config, ui
from src.models import inference
from src.models.train import metrics_table
from src.visualization import charts

ui.setup("Clinical AI Dashboard", icon="🫀")

# -----------------------------------------------------------------------------
# Data & Experiment Retrieval
# -----------------------------------------------------------------------------
df = ui.get_data()
run = ui.latest_run()
exp = None
if run:
    try:
        exp = inference.load_experiment(run)
    except Exception:
        exp = None

fed = ui.latest_federated()
info = st.session_state.get("info", {})

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <section class="clinical-hero">
        <div class="eyebrow">
            <span>🫀</span> CLINICAL AI RESEARCH · PRIVACY-AWARE FEDERATED LEARNING · EXPLAINABLE AI
        </div>
        <h1>Heart Disease Risk Prediction & Healthcare Dashboard</h1>
        <p>
            An end-to-end clinical machine learning workspace for cardiovascular risk prediction,
            model benchmarking, simulated cross-hospital federated learning, transparent inference,
            and SHAP/LIME feature attribution.
        </p>
    </section>
    <div class="status-row">
        <div class="status-chip teal">● Verified Risk Classification (1 = Disease)</div>
        <div class="status-chip">● CSV & Excel Upload Enabled</div>
        <div class="status-chip">● Federated Learning Simulation (FedAvg)</div>
        <div class="status-chip">● SHAP & LIME Interpretability</div>
        <div class="status-chip">● Clinical PDF Reporting</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Active Dataset Status Banner
# -----------------------------------------------------------------------------
active_name = st.session_state.get("_upload_name", "Cleveland Heart Disease (Standard Benchmark)")
data_count = len(df) if df is not None else 0
pos_rate = df["target"].mean() if df is not None else 0.0
verif_text = info.get("verification", {}).get("evidence", "Class 1 verified as Heart Disease via clinical marker correlation.")

st.markdown(
    f"""
    <div class="notice">
        <strong>📂 Active Dataset:</strong> {active_name} &nbsp;|&nbsp;
        <strong>Records:</strong> {data_count:,} patients &nbsp;|&nbsp;
        <strong>Positive Rate (Disease):</strong> {pos_rate:.1%} &nbsp;|&nbsp;
        <strong>Status:</strong> {verif_text}
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Executive KPI Metrics
# -----------------------------------------------------------------------------
st.markdown("### 📊 Executive Performance Overview")
st.caption("Live metrics from active dataset, trained model experiments, and simulated federated learning.")

models_dict = exp.get("models", {}) if exp else {}
trained_count = sum(1 for r in models_dict.values() if isinstance(r, dict) and "val" in r)
selected_model = exp.get("selected_model", "None trained") if exp else "None trained"
selected_meta = models_dict.get(selected_model, {}) if exp else {}
best_score = selected_meta.get("selection_score")
opt_thresh = selected_meta.get("optimal_threshold", 0.50)

fed_rounds = fed.get("config", {}).get("rounds", 0) if fed else 0
fed_acc = fed.get("final_federated", {}).get("accuracy") if fed else None

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
ui.card(kpi1, "Dataset Patients", f"{data_count:,}", subtext="Rows in active dataset", accent="teal")
ui.card(kpi2, "Clinical Predictors", f"{len(config.FEATURES)}", subtext="Standard cardiac markers", accent="navy")
ui.card(kpi3, "Models Evaluated", f"{trained_count}", subtext=f"Selected: {selected_model}", accent="teal")
ui.card(kpi4, "Federated Rounds", f"{fed_rounds}", subtext=f"Fed Acc: {fed_acc*100:.1f}%" if fed_acc else "Simulated FedAvg", accent="navy")

kpi_sub1, kpi_sub2 = st.columns(2)
ui.card(kpi_sub1, "Validation Selection Score", f"{best_score:.3f}" if best_score is not None else "Not Trained",
        subtext="Score = 0.5×Recall + 0.5×ROC-AUC (Prioritizes Sensitivity)", accent="teal")
ui.card(kpi_sub2, "Selected Model & Threshold", f"{selected_model}",
        subtext=f"Decision Cutoff: 0.50 (ROC Optimal: {opt_thresh:.2f})", accent="navy")

# -----------------------------------------------------------------------------
# Interactive Model Comparison Chart (if experiment exists)
# -----------------------------------------------------------------------------
if exp:
    st.markdown("---")
    st.markdown("### 📈 Model Evaluation Benchmark (Held-out Test Split)")
    st.caption("Multi-metric comparison across evaluated classifiers on held-out test data.")
    try:
        metrics_df = metrics_table(exp, "test")
        st.plotly_chart(charts.metric_bars(metrics_df), width="stretch")
    except Exception as exc:
        st.warning(f"Could not render model comparison chart: {exc}")

# -----------------------------------------------------------------------------
# Research Workflow Pipeline
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔄 Research & Clinical Workflow")
st.caption("Follow the standardized stages of the diagnostic pipeline.")

steps = [
    ("01", "Upload & Inspect", "Upload CSV or Excel data, detect missing values, and verify target class semantics."),
    ("02", "Leakage-Safe Preprocessing", "Impute missing values and encode categoricals strictly inside sklearn Pipelines."),
    ("03", "Model Training & Selection", "Train baselines and ensembles; select optimal model on validation score."),
    ("04", "Evaluation & Thresholds", "Inspect held-out test metrics, confusion matrix, ROC/PR curves, and cutoffs."),
    ("05", "Federated Learning", "Simulate hospital client data isolation and model aggregation with FedAvg."),
    ("06", "Patient Assessment & XAI", "Evaluate patients with transparent probabilities, thresholds, and SHAP/LIME."),
]

s_cols = st.columns(3)
for idx, (num, title, desc) in enumerate(steps):
    with s_cols[idx % 3]:
        st.markdown(
            f"""
            <div class="hc-card accent-teal" style="margin-bottom:14px;">
                <div class="section-tag">STEP {num}</div>
                <div style="font-weight:750;color:#0f172a;font-size:1.02rem;margin:4px 0 6px 0;">{title}</div>
                <div style="font-size:0.84rem;color:#64748b;line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Module Navigation Directory
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🧭 Platform Modules Directory")
st.caption("Select any module to navigate directly to its functionality.")

modules = [
    ("DATA", "1. Dataset Explorer", "Upload CSV/Excel, preview records, inspect missing values, and select target.", "/Dataset_Explorer"),
    ("PIPELINE", "2. Data Preprocessing", "Review train/val/test splits, outlier reports, and data leakage safeguards.", "/Data_Preprocessing"),
    ("TRAINING", "3. Model Training", "Train 6 base algorithms and 2 ensembles with cross-validation tuning.", "/Model_Training"),
    ("EVALUATION", "4. Model Evaluation", "Review test accuracy, recall, specificity, ROC curves, and confusion matrix.", "/Model_Evaluation"),
    ("PRIVACY", "5. Federated Learning", "Inspect simulated hospital networks, local training, and FedAvg aggregation.", "/Federated_Learning"),
    ("INTERPRET", "6. Explainable AI", "Compute global and local SHAP waterfall and LIME contribution plots.", "/Explainable_AI"),
    ("CLINICAL", "7. Patient Risk Assessment", "Input patient vitals, view transparent probability vs threshold, and export PDF.", "/Patient_Risk_Assessment"),
    ("HISTORY", "9. Reports & History", "Review previous experiment artifacts, download CSV metrics, and generate PDF.", "/Reports_History"),
]

m_cols = st.columns(4)
for idx, (cat, title, desc, link) in enumerate(modules):
    with m_cols[idx % 4]:
        st.markdown(
            f"""
            <a href="{link}" target="_self" class="nav-card">
                <div class="nav-badge">{cat}</div>
                <div class="nav-title">{title}</div>
                <div class="nav-desc">{desc}</div>
            </a>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Research Notes & Clinical Disclaimer
# -----------------------------------------------------------------------------
st.markdown("---")
n1, n2 = st.columns(2, gap="large")

with n1:
    st.markdown(
        """
        <div class="hc-card accent-navy">
            <div style="font-weight:750;color:#0f172a;margin-bottom:8px;">💡 Research & Methodology Notes</div>
            <div style="font-size:0.86rem;color:#475569;line-height:1.65;">
                • <strong>Target Verification:</strong> Automatic clinical correlation confirms the positive class aligns with disease markers (e.g. ST depression, angina).<br>
                • <strong>Validation-Based Selection:</strong> Models are selected on validation splits (0.5×Recall + 0.5×ROC-AUC) before test evaluation.<br>
                • <strong>Leakage Prevention:</strong> Preprocessing scalers and imputers are fit strictly on training splits.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with n2:
    st.markdown(
        f"""
        <div class="hc-card accent-teal">
            <div style="font-weight:750;color:#0f172a;margin-bottom:8px;">⚠️ Medical Disclaimer</div>
            <div style="font-size:0.86rem;color:#475569;line-height:1.65;">
                {config.DISCLAIMER}<br><br>
                This software is intended for machine learning research, educational demonstration, and algorithm validation.
                It does not constitute medical advice, diagnosis, or clinical protocol.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
