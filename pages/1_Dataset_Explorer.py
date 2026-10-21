"""Dataset Explorer & Dataset Upload Hub: CSV & Excel support, preview, missing values, feature analysis, and target selection."""
from __future__ import annotations
import io
import pandas as pd
import streamlit as st
from src import config, ui
from src.data import loader
from src.data.validation import validate_schema
from src.visualization import charts

ui.setup("Dataset Explorer")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 01 · DATA MANAGEMENT & EXPLORATION</div>
        <h1>Dataset Explorer & Upload Hub</h1>
        <p>Preview datasets, detect missing values, inspect clinical feature distributions,
        select target variables, and verify class semantics for downstream modeling.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Section 1: Dataset Upload & Selection
# -----------------------------------------------------------------------------
st.markdown("### 01 · Dataset Upload & Configuration")
st.caption("Upload a clinical dataset in CSV or Excel format, or use the standard Cleveland Heart Disease dataset.")

upload_col1, upload_col2 = st.columns([1.6, 1], gap="large")

with upload_col1:
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV, XLSX, XLS)",
        type=["csv", "xlsx", "xls"],
        help="Upload a dataset containing patient clinical records and a diagnostic target column.",
        key="main_dataset_uploader",
    )

with upload_col2:
    st.markdown("**Dataset Action**")
    active_source_name = st.session_state.get("_upload_name", "Cleveland Heart Disease (Default)")
    st.info(f"**Currently Active:** {active_source_name}")
    if st.button("🔄 Reset to Default Cleveland Dataset", use_container_width=True):
        st.session_state.pop("_source", None)
        st.session_state.pop("_upload_name", None)
        st.session_state.pop("_invert", None)
        st.session_state.pop("_target_col", None)
        st.session_state.pop("df", None)
        st.session_state.pop("info", None)
        st.success("Reset to default dataset.")
        st.rerun()

# Determine working raw dataframe
raw_df = None
source_to_read = None

if uploaded_file is not None:
    source_to_read = uploaded_file
    st.session_state["_source"] = uploaded_file
    st.session_state["_upload_name"] = uploaded_file.name
else:
    source_to_read = st.session_state.get("_source", config.DEFAULT_DATASET)

try:
    if not hasattr(loader, "load_dataset_file"):
        import importlib
        importlib.reload(loader)
    load_fn = getattr(loader, "load_dataset_file", getattr(loader, "load_csv", None))
    raw_df = load_fn(source_to_read)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Target Selection & Clinical Semantics Verification
# -----------------------------------------------------------------------------
with st.container(border=True):
    t_col1, t_col2, t_col3 = st.columns([1.4, 1.4, 1.2], gap="medium")

    # Target selection
    detected_target = None
    try:
        detected_target = loader.find_target_column(raw_df)
    except Exception:
        detected_target = raw_df.columns[-1]

    col_options = list(raw_df.columns)
    default_idx = col_options.index(detected_target) if detected_target in col_options else (len(col_options) - 1)

    with t_col1:
        target_choice = st.selectbox(
            "Select Target Column",
            col_options,
            index=default_idx,
            help="Select the column representing the heart disease diagnosis or risk outcome.",
        )

    # Verification of selected target
    verif = loader.verify_target_semantics(raw_df, target_choice)
    rec_invert = verif.get("recommended_invert", False)
    current_invert = st.session_state.get("_invert", rec_invert)

    with t_col2:
        invert_toggle = st.checkbox(
            "Invert Target (1 = No Disease)",
            value=bool(current_invert),
            help="Automatically set by clinical marker verification. Check only if your target 1 means healthy.",
        )

    with t_col3:
        if st.button("Apply Dataset Settings", type="primary", use_container_width=True):
            st.session_state["_target_col"] = target_choice
            st.session_state["_invert"] = invert_toggle
            st.session_state.pop("df", None)
            st.session_state.pop("info", None)
            st.success("Target settings applied!")
            st.rerun()

    # Display clinical verification summary
    st.markdown(
        f"""
        <div class="notice">
            <strong>🩺 Target Semantics Verification:</strong> {verif.get('evidence', 'No clinical verification available.')}<br>
            <strong>Active Target:</strong> <code>{target_choice}</code> · 
            <strong>Mapping:</strong> {'Inverted (1 = Heart Disease, 0 = No Disease)' if invert_toggle else 'Standard (1 = Heart Disease, 0 = No Disease)'}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Get prepared data
df = ui.require_data()
rep = validate_schema(df)

for e in rep.errors:
    st.error(e)
for w in rep.warnings:
    st.warning(w)

# -----------------------------------------------------------------------------
# Section 2: Key Metrics
# -----------------------------------------------------------------------------
st.markdown("### 02 · Dataset Overview")
m1, m2, m3, m4 = st.columns(4)
ui.card(m1, "Total Patients", f"{rep.n_rows:,}", subtext="Rows in loaded dataset", accent="teal")
ui.card(m2, "Clinical Predictors", f"{len(config.FEATURES)}", subtext="Standardized input features", accent="navy")
ui.card(m3, "Missing Values", f"{rep.n_missing:,}", subtext="Imputed in pipeline", accent="teal")
ui.card(m4, "Positive Rate", f"{df['target'].mean():.1%}", subtext="Class 1 (Heart Disease)", accent="navy")

# -----------------------------------------------------------------------------
# Section 3: Deep Exploration Tabs
# -----------------------------------------------------------------------------
st.markdown("### 03 · Data Exploration & Feature Analysis")
t1, t2, t3, t4, t5 = st.tabs([
    "📋 Data Preview",
    "🔍 Missing-Value Detection",
    "📊 Statistical Summary",
    "🎯 Target Class Analysis",
    "📈 Clinical Feature Distributions",
])

# Tab 1: Preview
with t1:
    st.markdown("#### Sample Records")
    st.caption("Showing the first 50 records of the prepared dataset with binary target (1 = Heart Disease).")
    st.dataframe(df.head(50), width="stretch")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Prepared Dataset (CSV)", csv_bytes, "prepared_heart_data.csv", "text/csv")

# Tab 2: Missing Values
with t2:
    st.markdown("#### Missing-Value & Placeholder Detection")
    st.caption("Overview of missing values, column data types, and placeholder replacements.")

    missing_series = df.isna().sum()
    missing_pct = (missing_series / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Count": missing_series.values,
        "Missing %": [f"{p:.1f}%" for p in missing_pct.values],
    })

    col_a, col_b = st.columns([1.5, 1], gap="large")
    with col_a:
        st.dataframe(missing_df, hide_index=True, width="stretch")

    with col_b:
        st.markdown(
            """
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:18px;">
                <h4 style="margin-top:0;color:#0f172a;">Imputation Strategy</h4>
                <p style="font-size:0.86rem;color:#475569;line-height:1.6;">
                    • <strong>Numeric features:</strong> Median imputation learned strictly on training splits.<br>
                    • <strong>Categorical features:</strong> Most frequent (mode) imputation within pipelines.<br>
                    • <strong>Placeholder codes:</strong> <code>ca=4</code> and <code>thal=0</code> are converted to NaN and cleanly imputed without data leakage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Tab 3: Statistics
with t3:
    st.markdown("#### Feature Descriptive Statistics")
    st.caption("Distribution statistics for numeric clinical markers.")
    num_cols_present = [c for c in config.NUMERIC if c in df.columns]
    st.dataframe(df[num_cols_present].describe().T.round(2), width="stretch")

    st.markdown("#### Categorical Feature Value Counts")
    cat_summary = []
    for c in config.CATEGORICAL:
        if c in df.columns:
            counts = df[c].value_counts(dropna=False).to_dict()
            spec = config.FEATURE_INFO.get(c, (c, "", "cat", {}))[3]
            labeled_counts = {f"{k} ({spec.get(k, k)})": v for k, v in counts.items()}
            cat_summary.append({"Feature": config.LABELS.get(c, c), "Distribution": str(labeled_counts)})
    st.dataframe(pd.DataFrame(cat_summary), hide_index=True, width="stretch")

# Tab 4: Target
with t4:
    st.markdown("#### Target Class Balance")
    st.caption("Verification that positive class (1) represents Heart Disease.")
    tc1, tc2 = st.columns([1.2, 1], gap="large")
    with tc1:
        st.plotly_chart(charts.class_distribution(df["target"]), width="stretch")
    with tc2:
        pos_count = int(df["target"].sum())
        neg_count = len(df) - pos_count
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin-top:20px;">
                <h4 style="margin-top:0;color:#0f172a;">Class Breakdown</h4>
                <div style="margin:12px 0;">
                    <div style="color:#0d9488;font-weight:750;font-size:1.1rem;">● Heart Disease (Class 1): {pos_count} ({pos_count/len(df):.1%})</div>
                    <div style="color:#0284c7;font-weight:750;font-size:1.1rem;margin-top:6px;">● No Disease (Class 0): {neg_count} ({neg_count/len(df):.1%})</div>
                </div>
                <p style="font-size:0.85rem;color:#64748b;line-height:1.5;">
                    The balanced distribution prevents model bias. Both classes are stratified across training, validation, and testing partitions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Tab 5: Feature Distributions & Heatmap
with t5:
    st.markdown("#### Feature Distribution vs Heart Disease Status")
    feat_selected = st.selectbox("Select Feature to Visualize", config.FEATURES, format_func=lambda c: config.LABELS.get(c, c))
    st.plotly_chart(charts.feature_hist(df, feat_selected), width="stretch")

    st.markdown("#### Clinical Feature Correlation Matrix")
    st.plotly_chart(charts.corr_heatmap(df), width="stretch")
