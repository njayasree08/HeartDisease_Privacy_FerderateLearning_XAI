"""Clinical AI Federated Learning Dashboard: Simulated Multi-Hospital FedAvg."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from src import config, ui
from src.data.preprocessing import clean_frame, split_data
from src.federated.simulation import run_federated
from src.visualization import charts

ui.setup("Federated Learning")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 05 · DISTRIBUTED & PRIVACY-AWARE LEARNING</div>
        <h1>Federated Learning Simulation</h1>
        <p>Simulate collaborative multi-center training across isolated hospital clients.
        Model weights are aggregated centrally via Federated Averaging (FedAvg) without sharing raw patient records.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = ui.require_data()

# -----------------------------------------------------------------------------
# Conceptual Network Architecture Diagram
# -----------------------------------------------------------------------------
st.markdown("### 01 · Distributed Hospital Topology")
st.markdown(
    """
    <div style="display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:16px;align-items:center;margin:16px 0;">
      <div style="padding:20px;border:1.5px solid #ccfbf1;border-radius:12px;background:#ffffff;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,0.04);">
        <div style="font-size:32px;">🏥</div>
        <h4 style="color:#0f766e;margin:6px 0 2px 0;">Hospital Center A</h4>
        <p style="font-size:0.82rem;color:#64748b;margin:0;">Local isolated EHR partition</p>
        <span style="font-size:0.75rem;font-weight:700;color:#0d9488;background:#f0fdfa;padding:3px 8px;border-radius:9999px;">Client Node 1</span>
      </div>
      <div style="padding:22px;border:2px solid #0d9488;border-radius:14px;background:#f0fdfa;text-align:center;box-shadow:0 4px 12px rgba(13,148,136,0.12);">
        <div style="font-size:34px;">🔄</div>
        <h3 style="color:#0f766e;margin:6px 0 2px 0;">Central Aggregation Server</h3>
        <p style="font-size:0.86rem;color:#334155;margin:0 0 6px 0;">Sample-weighted FedAvg Parameter Sync</p>
        <span style="font-size:0.78rem;font-weight:800;color:#ffffff;background:#0d9488;padding:4px 12px;border-radius:9999px;">Zero Patient Data Exchange</span>
      </div>
      <div style="padding:20px;border:1.5px solid #ccfbf1;border-radius:12px;background:#ffffff;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,0.04);">
        <div style="font-size:32px;">🏥</div>
        <h4 style="color:#0f766e;margin:6px 0 2px 0;">Hospital Center B</h4>
        <p style="font-size:0.82rem;color:#64748b;margin:0;">Local isolated EHR partition</p>
        <span style="font-size:0.75rem;font-weight:700;color:#0d9488;background:#f0fdfa;padding:3px 8px;border-radius:9999px;">Client Node 2</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Configuration Form
# -----------------------------------------------------------------------------
st.markdown("### 02 · Simulation Parameters")
with st.container(border=True):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        n_clients = st.slider("Simulated Hospital Nodes", min_value=2, max_value=5, value=3)
        kind = st.selectbox(
            "Client Data Partitioning",
            ["non_iid", "iid"],
            help="Non-IID simulates natural clinical heterogeneity where patient demographic and disease distributions vary across hospitals.",
        )
        alpha = st.slider(
            "Dirichlet Non-IID Skew (Alpha)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            disabled=(kind == "iid"),
            help="Lower alpha creates higher inter-hospital class imbalance.",
        )

    with col2:
        fed_rounds = st.slider("Federated Communication Rounds", min_value=2, max_value=25, value=10)
        local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=5, value=2)

    with col3:
        batch_size = st.select_slider("Client Batch Size", options=[8, 16, 32, 64], value=16)
        lr = st.select_slider("Client Neural Net Learning Rate", options=[0.005, 0.01, 0.02, 0.05], value=0.01)

    start_sim = st.button("🚀 Launch Federated Training Simulation", type="primary", use_container_width=True)

if start_sim:
    clean = clean_frame(df)
    splits = split_data(clean, seed=int(config.SEED))
    progress_bar = st.progress(0.0)
    progress_status = st.empty()

    def progress_cb(msg, frac):
        progress_bar.progress(min(max(float(frac), 0.0), 1.0))
        progress_status.info(msg)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = config.ARTIFACT_DIR / "federated" / run_id

    with st.spinner("Simulating multi-center training rounds across clients..."):
        try:
            fed_result = run_federated(
                splits=splits,
                n_clients=int(n_clients),
                kind=kind,
                alpha=float(alpha),
                rounds=int(fed_rounds),
                local_epochs=int(local_epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                save_dir=save_path,
                progress=progress_cb,
            )
            st.session_state["latest_fed_result"] = fed_result
            st.success(f"Federated learning completed successfully in {fed_result['seconds']:.1f}s!")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

fed_res = st.session_state.get("latest_fed_result") or ui.latest_federated()

# -----------------------------------------------------------------------------
# Results & Comparison
# -----------------------------------------------------------------------------
if fed_res:
    st.markdown("---")
    st.markdown("### 03 · Federated Convergence & Benchmark")
    f_res = fed_res["final_federated"]
    c_res = fed_res["final_centralized"]
    comm = fed_res.get("comm", {})

    k1, k2, k3, k4 = st.columns(4)
    ui.card(k1, "Federated Accuracy", f"{f_res['accuracy']*100:.1f}%", subtext=f"Centralized: {c_res['accuracy']*100:.1f}%", accent="teal")
    ui.card(k2, "Federated Recall", f"{f_res['recall']*100:.1f}%", subtext=f"Centralized: {c_res['recall']*100:.1f}%", accent="navy")
    ui.card(k3, "Federated ROC-AUC", f"{f_res['roc_auc']:.3f}", subtext=f"Centralized: {c_res['roc_auc']:.3f}", accent="teal")
    ui.card(k4, "Raw Data Privacy", "100% Isolated", subtext=f"{comm.get('raw_records_sent_to_server', 0)} raw records shared", accent="navy")

    h_df = pd.DataFrame(fed_res["history"])
    st.plotly_chart(charts.fed_history(h_df), width="stretch")

    st.markdown("### 04 · Hospital Client Data Partitions")
    client_records = []
    for cid, (size, pos_r) in enumerate(zip(fed_res["client_sizes"], fed_res["client_positive_rate"])):
        client_records.append({
            "Client Node": f"Hospital Client {cid + 1}",
            "Local Records": size,
            "Disease Rate": f"{pos_r*100:.1f}%",
            "Weight Share in FedAvg": f"{size / sum(fed_res['client_sizes'])*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(client_records), hide_index=True, width="stretch")
else:
    st.info("No federated simulation run yet. Click 'Launch Federated Training Simulation' above to run.")