"""Home / overview page."""
import streamlit as st
from src import config, ui

ui.setup("Privacy-Preserving & Explainable Heart Disease Risk Prediction")
df = ui.get_data()
exp = None
run = ui.latest_run()
if run:
    from src.models import inference
    exp = inference.load_experiment(run)
fed = ui.latest_federated()

st.write("An educational research platform combining supervised and ensemble learning, a simulated federated "
         "neural network across hospital clients, and SHAP/LIME explanations for heart-disease risk modelling.")
c = st.columns(6)
ui.card(c[0], "Dataset records", len(df) if df is not None else "–")
ui.card(c[1], "Input features", len(config.FEATURES))
ui.card(c[2], "Models trained", sum("val" in r for r in exp["models"].values()) if exp else 0)
best = exp["models"][exp["selected_model"]]["selection_score"] if exp and exp["selected_model"] else None
ui.card(c[3], "Best validation score", f"{best:.3f}" if best else "–")
ui.card(c[4], "Federated rounds", fed["config"]["rounds"] if fed else 0)
ui.card(c[5], "Selected model", exp["selected_model"] if exp else "–")

st.subheader("Research objectives")
st.markdown("- Compare classical, tree-based and ensemble models on a fixed, leakage-free split.\n"
            "- Show how federated averaging trains a shared network without pooling raw records.\n"
            "- Explain individual and global predictions with SHAP and LIME.\n"
            "- Report honest, reproducible results, including limitations.")
st.subheader("Workflow")
st.code("Dataset → validation → split (train/val/test) → pipelines → tuning → validation-based selection → test report\n"
        "Training split → simulated hospital clients → local training → FedAvg → global model → held-out test\n"
        "Selected model → SHAP (global/local) + LIME (local) → patient assessment → PDF/CSV reports", language="text")
st.subheader("Modules")
st.markdown("Use the sidebar to open: **Dataset Explorer → Data Preprocessing → Model Training → Model Evaluation → "
            "Federated Learning → Explainable AI → Patient Risk Assessment → Research Results → Reports History**.")
if df is not None:
    st.caption(ui.st.session_state["info"]["note"])
st.caption("Results shown anywhere in this app come from experiments actually run on this machine; nothing is pre-filled.")
