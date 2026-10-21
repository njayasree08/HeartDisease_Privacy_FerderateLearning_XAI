
"""Clinical AI model training dashboard."""
import streamlit as st

from src import config, ui
from src.models import ensembles
from src.models.train import metrics_table, run_training

ui.setup("Model Training")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">CLINICAL AI · EXPERIMENT WORKSPACE</div>
        <h1>Model Training</h1>
        <p>Configure machine-learning experiments, monitor training,
        and compare validation performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = ui.require_data()

st.markdown("### 01 · Experiment configuration")
st.caption("Choose the models and training settings for your experiment.")

with st.container(border=True):
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        names = st.multiselect(
            "Models to train",
            ensembles.ALL_NAMES,
            default=ensembles.ALL_NAMES,
            help="Choose one or more models to include in this experiment.",
        )

        mode = st.radio(
            "Training mode",
            ["fast", "research"],
            horizontal=True,
            help=(
                f"Fast mode: {config.MODES['fast']}\n\n"
                f"Research mode: {config.MODES['research']}"
            ),
        )

    with col2:
        tune = st.checkbox(
            "Enable hyperparameter tuning",
            value=True,
            help="Tuning uses cross-validation on the training split only.",
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=int(config.SEED),
            step=1,
            help="A fixed seed helps make supported operations reproducible.",
        )

    st.markdown("**Model selection rule**")
    st.code(config.SELECTION_RULE, language="text")

    st.info(
        "Stacking and Soft Voting require Logistic Regression, Random Forest, "
        "and XGBoost. Those models are trained automatically when either "
        "ensemble is selected."
    )

    start = st.button(
        "Start training experiment",
        type="primary",
        disabled=not names,
        use_container_width=True,
    )

if start:
    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    def progress_callback(message, fraction):
        progress_bar.progress(min(max(float(fraction), 0.0), 1.0))
        progress_text.write(message)

    with st.spinner("Training models. Please wait..."):
        try:
            st.session_state["last_exp"] = run_training(
                df,
                names,
                mode,
                int(seed),
                tune,
                progress=progress_callback,
                data_info=st.session_state["info"],
            )
            st.success("Training completed. Experiment artifacts were saved.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

st.divider()
st.markdown("### 02 · Latest experiment")

exp, run = ui.require_experiment()

st.caption(f"Experiment ID: {exp['run_id']} · Mode: {exp['mode']}")

failed_models = {
    name: result["error"]
    for name, result in exp["models"].items()
    if "error" in result
}

for name, error in failed_models.items():
    st.error(f"{name} failed: {error}")

results = metrics_table(exp, "val")
results["status"] = results["model"].apply(
    lambda name: (
        "Failed"
        if name in failed_models
        else "Completed"
    )
)

completed_count = len(exp["models"]) - len(failed_models)
summary_cols = st.columns(3)

summary_cols[0].metric("Models in experiment", len(exp["models"]))
summary_cols[1].metric("Completed", completed_count)
summary_cols[2].metric("Failed", len(failed_models))

st.markdown("#### Validation performance")
st.caption(
    "These are validation-set metrics used to compare models and select "
    "the final model. They are not test-set results."
)

display_columns = [
    "model",
    "status",
    "accuracy",
    "recall",
    "specificity",
    "roc_auc",
    "val_selection_score",
    "train_seconds",
]

st.dataframe(
    results[display_columns].round(3),
    hide_index=True,
    width="stretch",
)

st.success(
    f"Selected model: {exp['selected_model']} "
    f"— chosen using the validation rule: {exp['selection_rule']}."
)

st.caption(
    "For held-out test performance, open the Model Evaluation page. "
    "The test set should not be used to select or tune the model."
)

with st.expander("View hyperparameter tuning details"):
    tuning_details = {
        name: result.get("tuning")
        for name, result in exp["models"].items()
        if "tuning" in result
    }

    if tuning_details:
        st.json(tuning_details)
    else:
        st.info("No tuning details are available for this experiment.")
