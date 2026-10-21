"""Clinical AI Data Preprocessing Dashboard."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src import config, ui
from src.data.preprocessing import build_preprocessor, clean_frame, outlier_report, split_data, split_summary

ui.setup("Data Preprocessing")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 02 · PIPELINE ENGINEERING & DATA SPLITS</div>
        <h1>Data Preprocessing & Splitting</h1>
        <p>Prepare raw clinical records for machine learning with strict leakage-safe boundaries:
        impute missing values, encode categorical variables, configure outlier handling, and review stratified partitions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = ui.require_data()

# -----------------------------------------------------------------------------
# Configuration Controls
# -----------------------------------------------------------------------------
st.markdown("### 01 · Preprocessing Settings")
with st.container(border=True):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        drop = st.checkbox("Drop exact duplicate records", value=True, help="Removes identical patient records to prevent split contamination.")
    with c2:
        outl = st.selectbox(
            "Outlier strategy",
            ["keep", "winsorize_iqr"],
            help="'keep' preserves clinical physiological extremes. 'winsorize_iqr' clamps values to training IQR bounds.",
        )
    with c3:
        seed = st.number_input("Random split seed", min_value=0, max_value=99999, value=int(config.SEED))

clean = clean_frame(df, drop)

# -----------------------------------------------------------------------------
# Partitions & Stratification
# -----------------------------------------------------------------------------
st.markdown("### 02 · Stratified Partitions (Train / Validation / Test)")
st.caption("Stratification preserves the exact heart disease class proportion across all three subsets.")

sp = split_data(clean, int(seed))
summary_df = split_summary(sp)

sc1, sc2, sc3 = st.columns(3)
ui.card(sc1, "Training Split (60%)", f"{len(sp['train'][1])} patients", subtext=f"Positives: {int(sp['train'][1].sum())} ({sp['train'][1].mean():.1%})", accent="teal")
ui.card(sc2, "Validation Split (20%)", f"{len(sp['val'][1])} patients", subtext=f"Positives: {int(sp['val'][1].sum())} ({sp['val'][1].mean():.1%})", accent="navy")
ui.card(sc3, "Held-Out Test Split (20%)", f"{len(sp['test'][1])} patients", subtext=f"Positives: {int(sp['test'][1].sum())} ({sp['test'][1].mean():.1%})", accent="teal")

st.dataframe(summary_df, hide_index=True, width="stretch")

# -----------------------------------------------------------------------------
# Missing Values & Outlier Analysis
# -----------------------------------------------------------------------------
st.markdown("### 03 · Missing Values & Outlier Screening")
t_miss, t_outl = st.tabs(["Missing Values Summary", "Outlier Diagnostic Bounds"])

with t_miss:
    st.dataframe(df[config.FEATURES].isna().sum().rename("Missing Count").to_frame().T, width="stretch")
    st.caption("Missing values are automatically imputed within each fold/pipeline (median for numerics, mode for categoricals).")

with t_outl:
    st.dataframe(outlier_report(clean), hide_index=True, width="stretch")
    st.caption("Reference 1.5× IQR and 3-SD bounds are displayed for exploratory screening; legitimate clinical values are preserved.")

# -----------------------------------------------------------------------------
# Transformation Inspection
# -----------------------------------------------------------------------------
st.markdown("### 04 · Feature Transformation (Fit strictly on Training split)")
prep = build_preprocessor(outl).fit(sp["train"][0])
after = pd.DataFrame(prep.transform(sp["train"][0]), columns=prep.get_feature_names_out())

col_raw, col_trans = st.columns(2, gap="large")
with col_raw:
    st.markdown("#### Before: Raw Features (Training Subset)")
    st.dataframe(sp["train"][0][config.NUMERIC].describe().T.round(2), width="stretch")

with col_trans:
    st.markdown(f"#### After: Encoded & Scaled Inputs ({after.shape[1]} Model Dimensions)")
    st.dataframe(after.describe().T.round(2).head(12), width="stretch")

st.markdown(
    """
    <div class="notice">
        <strong>🛡️ Data Leakage Prevention Protocol:</strong> All scalers, imputers, and encoders are encapsulated
        inside Scikit-Learn <code>Pipeline</code> objects and fit strictly on training records. Validation and test sets
        are transformed without refitting, ensuring reliable generalisation estimates.
    </div>
    """,
    unsafe_allow_html=True,
)
