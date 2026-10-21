"""Home / overview page for the heart-disease research dashboard."""
import streamlit as st
from src import config, ui

# Keep the existing project helpers and data/model loading flow intact.
ui.setup("Privacy-Preserving & Explainable Heart Disease Risk Prediction")

# Dashboard styling: presentation only; does not change model training or predictions.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px;}
      [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e5eaf0; padding: 16px 18px;
        border-radius: 14px; box-shadow: 0 2px 8px rgba(15, 23, 42, .035);
      }
      [data-testid="stMetricLabel"] {color: #526174; font-size: .84rem;}
      [data-testid="stMetricValue"] {color: #12263a; font-weight: 700;}
      .hero {
        padding: 28px 30px; border-radius: 18px; margin: 4px 0 22px 0;
        background: linear-gradient(115deg, #102a43 0%, #176b72 100%);
        color: white; border: 1px solid rgba(255,255,255,.12);
      }
      .hero-kicker {text-transform: uppercase; letter-spacing: .12em; font-size: .76rem;
        color: #b9e7e5; font-weight: 700; margin-bottom: 8px;}
      .hero h1 {font-size: clamp(1.65rem, 3vw, 2.45rem); line-height: 1.15;
        color: white; margin: 0 0 12px 0;}
      .hero p {color: #e4f1f3; font-size: 1rem; max-width: 850px; margin: 0;}
      .section-note {color: #64748b; font-size: .92rem; margin-top: -8px;}
      .step-card {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 15px; min-height: 110px;}
      .step-number {color: #0f766e; font-size: .76rem; font-weight: 800;
        letter-spacing: .08em; text-transform: uppercase;}
      .step-title {font-weight: 700; color: #172b4d; margin: 5px 0;}
      .step-copy {font-size: .84rem; color: #64748b;}
      .status-pill {display: inline-block; padding: 5px 10px; border-radius: 999px;
        background: #e6f6f3; color: #11645e; font-size: .78rem; font-weight: 700;}
      div[data-testid="stVerticalBlock"] > div:has(> div > .hero) {gap: .5rem;}
      @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {background: #17212b; border-color: #334155;}
        [data-testid="stMetricValue"] {color: #f1f5f9;}
        .step-card {background: #17212b; border-color: #334155;}
        .step-title {color: #e2e8f0;}
        .step-copy, .section-note {color: #a8b5c5;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

df = ui.get_data()
exp = None
run = ui.latest_run()
if run:
    from src.models import inference
    exp = inference.load_experiment(run)
fed = ui.latest_federated()

# Hero section
st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">AI research dashboard · Model transparency · Privacy-aware learning</div>
      <h1>Heart Disease Risk Prediction</h1>
      <p>Explore supervised learning, simulated federated training, and SHAP/LIME explanations
      in one reproducible research workspace.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# Status and key metrics
data_count = len(df) if df is not None else None
models = exp.get("models", {}) if exp else {}
trained_count = sum(1 for result in models.values() if isinstance(result, dict) and "val" in result)
selected_model = exp.get("selected_model") if exp else None
selected_result = models.get(selected_model, {}) if selected_model else {}
best = selected_result.get("selection_score") if selected_result else None
fed_config = fed.get("config", {}) if fed else {}
fed_rounds = fed_config.get("rounds", 0)

if df is None:
    st.info("Dataset not loaded yet. Open **Dataset Explorer** and upload a supported CSV or use the public dataset option.")
elif not run:
    st.info("Dataset loaded. Open **Model Training** and select **Start training** to populate experiment metrics.")

metric_cols = st.columns(4, gap="medium")
metric_cols[0].metric("Dataset records", f"{data_count:,}" if data_count is not None else "—")
metric_cols[1].metric("Input features", len(config.FEATURES))
metric_cols[2].metric("Models evaluated", trained_count)
metric_cols[3].metric("Federated rounds", fed_rounds)

metric_cols2 = st.columns(2, gap="medium")
metric_cols2[0].metric(
    "Validation selection score",
    f"{best:.3f}" if best is not None else "—",
    help="Selection score is computed from validation recall and ROC-AUC; it is not the held-out test score.",
)
metric_cols2[1].metric("Selected model", str(selected_model) if selected_model else "Not trained yet")

st.divider()

# Main dashboard content
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Research workflow")
    st.markdown('<p class="section-note">A clear view of how data moves through the project.</p>', unsafe_allow_html=True)
    steps = [
        ("01", "Prepare data", "Validate columns, handle missing values, and create stratified splits."),
        ("02", "Train models", "Fit preprocessing and models on training data; select using validation data."),
        ("03", "Compare results", "Review held-out test metrics and error patterns."),
        ("04", "Explain predictions", "Use SHAP and LIME to inspect model behaviour."),
    ]
    for number, title, copy in steps:
        st.markdown(
            f'<div class="step-card"><div class="step-number">STEP {number}</div>'
            f'<div class="step-title">{title}</div><div class="step-copy">{copy}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")

with right:
    st.subheader("Project at a glance")
    st.markdown(
        """
        **Machine learning**

        Classical models, tree-based models, and ensemble methods.

        **Federated learning**

        Simulated hospital clients train locally and share model updates for FedAvg aggregation.

        **Explainable AI**

        Global and patient-level explanations with SHAP and LIME.

        **Research outputs**

        Reproducible experiment records and CSV/PDF reports.
        """
    )
    st.warning(
        "Educational research prototype only. The dataset is small, federated clients are simulated, "
        "and predictions are not clinically validated."
    )

st.divider()
st.subheader("Explore the modules")
st.caption(
    "Use the sidebar to open the full modules. This home page is an overview; it does not replace "
    "the existing training, evaluation, explanation, or reporting pages."
)
module_cols = st.columns(3, gap="medium")
modules = [
    ("Data", "Dataset Explorer", "Inspect records, columns, and target handling."),
    ("Training", "Model Training", "Run experiments and select a model."),
    ("Evaluation", "Model Evaluation", "Compare performance on the held-out test set."),
    ("Federated", "Federated Learning", "Explore simulated clients and FedAvg rounds."),
    ("Explainability", "Explainable AI", "Inspect SHAP and LIME explanations."),
    ("Reports", "Reports History", "Review and export generated reports."),
]
for index, (category, title, description) in enumerate(modules):
    with module_cols[index % 3]:
        with st.container(border=True):
            st.caption(category.upper())
            st.markdown(f"**{title}**")
            st.write(description)

if df is not None:
    info = st.session_state.get("info", {})
    note = info.get("note")
    if note:
        st.caption(f"Dataset note: {note}")

st.caption(
    "All displayed experiment metrics come from runs performed in this environment. "
    "No model performance values are pre-filled."
)
