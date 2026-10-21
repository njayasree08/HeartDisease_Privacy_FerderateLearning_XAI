"""Clinical AI Research Results Dashboard: Consolidated Findings & Reproducibility."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src import ui
from src.models.train import metrics_table
from src.visualization import charts

ui.setup("Research Results")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 08 · EMPIRICAL RESEARCH SYNTHESIS</div>
        <h1>Consolidated Research Results</h1>
        <p>Comprehensive overview of algorithm performance, federated learning convergence,
        feature importance hierarchies, and experimental reproducibility specifications.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

exp, run = ui.require_experiment()
tab = metrics_table(exp, "test").sort_values("roc_auc", ascending=False)

st.markdown("### 01 · Consolidated Test-Set Performance")
st.dataframe(
    tab[["model", "accuracy", "recall", "specificity", "f1", "roc_auc", "fp", "fn"]].round(3),
    hide_index=True,
    width="stretch",
)

sel = exp["selected_model"]
st_ = exp["models"][sel]
top_acc = tab.sort_values("accuracy", ascending=False).iloc[0]

# Key Findings Cards
st.markdown("### 02 · Key Scientific Findings")
c_f1, c_f2 = st.columns(2, gap="large")

with c_f1:
    st.markdown(
        f"""
        <div class="hc-card accent-teal">
            <div style="font-weight:750;color:#0f172a;margin-bottom:8px;">🎯 Validation Selected Model: {sel}</div>
            <div style="font-size:0.88rem;color:#334155;line-height:1.65;">
                • <strong>Validation Selection Score:</strong> {st_['selection_score']:.3f}<br>
                • <strong>Held-out Test Accuracy:</strong> {st_['test']['accuracy']*100:.1f}%<br>
                • <strong>Held-out Test Sensitivity (Recall):</strong> {st_['test']['recall']*100:.1f}%<br>
                • <strong>Held-out Test Specificity:</strong> {st_['test']['specificity']*100:.1f}%<br>
                • <strong>Test Area Under ROC:</strong> {st_['test']['roc_auc']:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_f2:
    st.markdown(
        f"""
        <div class="hc-card accent-navy">
            <div style="font-weight:750;color:#0f172a;margin-bottom:8px;">🏆 Benchmark Highlights</div>
            <div style="font-size:0.88rem;color:#334155;line-height:1.65;">
                • <strong>Highest Test Accuracy:</strong> {top_acc['model']} ({top_acc['accuracy']*100:.1f}%)<br>
                • <strong>Highest Test ROC-AUC:</strong> {tab.iloc[0]['model']} ({tab.iloc[0]['roc_auc']:.3f})<br>
                • <strong>Accuracy Span:</strong> {tab['accuracy'].min()*100:.1f}% – {tab['accuracy'].max()*100:.1f}%<br>
                • <strong>Sample Cohort:</strong> {exp['n_records']} patients across train/val/test splits.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Federated vs Centralized
st.markdown("### 03 · Federated Learning vs Centralized Baseline")
fed = ui.latest_federated()
if fed:
    h = pd.DataFrame(fed["history"])
    st.plotly_chart(charts.fed_history(h), width="stretch")
    f, c = fed["final_federated"], fed["final_centralized"]
    fc1, fc2, fc3 = st.columns(3)
    ui.card(fc1, "Federated Accuracy", f"{f['accuracy']*100:.1f}%", subtext=f"Centralized: {c['accuracy']*100:.1f}%", accent="teal")
    ui.card(fc2, "Federated Recall", f"{f['recall']*100:.1f}%", subtext=f"Centralized: {c['recall']*100:.1f}%", accent="navy")
    ui.card(fc3, "Federated ROC-AUC", f"{f['roc_auc']:.3f}", subtext=f"Centralized: {c['roc_auc']:.3f}", accent="teal")
else:
    st.info("No federated experiment saved yet. Launch a simulation in Module 05 to display results.")

# Global SHAP Feature Importance
imp = st.session_state.get("shap_importance")
if imp:
    st.markdown(f"### 04 · Most Influential Clinical Features ({imp[0]})")
    st.plotly_chart(charts.importance_bar(imp[1]), width="stretch")

# Reproducibility & Environment
st.markdown("### 05 · Reproducibility & Environment Manifest")
st.markdown(
    """
    <div class="notice">
        <strong>Reproducibility Guarantee:</strong> All experiments record complete software dependency versions,
        random seeds, pipeline parameters, and dataset provenance hashes.
    </div>
    """,
    unsafe_allow_html=True,
)
st.json({
    "seed": exp["seed"],
    "mode": exp["mode"],
    "settings": exp["settings"],
    "versions": exp["versions"],
    "data": exp.get("data_info", {}),
})
