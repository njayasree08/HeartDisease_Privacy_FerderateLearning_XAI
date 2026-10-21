"""Clinical AI Reports & Experiment History Dashboard: PDF export, CSV metrics, and artifact verification."""
from __future__ import annotations
import io, json
from pathlib import Path
import pandas as pd
import streamlit as st
from src import config, ui
from src.explainability import shap_explainer as se
from src.models.inference import sha256
from src.models.train import metrics_table
from src.reporting import csv_export
from src.reporting.pdf_report import build_pdf

ui.setup("Reports & Experiment History")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 09 · AUDIT LOGS & REPORT GENERATION</div>
        <h1>Reports & Experiment History</h1>
        <p>Review historical model experiments, export held-out evaluation metrics to CSV,
        verify cryptographic artifact checksums, and generate comprehensive multi-page clinical PDF reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

hist = csv_export.experiment_history()
if hist.empty:
    st.info("No saved experiments found. Train models in Module 03 to populate the experiment registry.")
    st.stop()

st.markdown("### 01 · Experiment Registry")
st.dataframe(hist, hide_index=True, width="stretch")

exp, run = ui.require_experiment()
tab = metrics_table(exp, "test")

# -----------------------------------------------------------------------------
# Data & PDF Export Suite
# -----------------------------------------------------------------------------
st.markdown("### 02 · Clinical Export Center")
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.download_button(
        "📥 Download Test Metrics (CSV)",
        data=csv_export.metrics_csv(tab),
        file_name=f"metrics_test_{exp['run_id']}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    st.download_button(
        "📥 Download Full History (CSV)",
        data=hist.to_csv(index=False).encode("utf-8"),
        file_name="experiment_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c3:
    if st.button("📑 Compile Clinical PDF Report", type="primary", use_container_width=True):
        with st.spinner("Compiling ReportLab PDF document..."):
            try:
                imp = st.session_state.get("shap_importance")
                png = None
                if imp and st.session_state.get("shap_key") in st.session_state:
                    buf = io.BytesIO()
                    se.bar_fig(st.session_state[st.session_state["shap_key"]]).savefig(
                        buf, format="png", dpi=110, bbox_inches="tight"
                    )
                    png = buf.getvalue()

                st.session_state["pdf"] = build_pdf(
                    exp,
                    tab,
                    ui.latest_federated(),
                    imp[1] if imp else None,
                    png,
                )
                st.success("Clinical PDF report generated successfully!")
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

if "pdf" in st.session_state:
    st.download_button(
        "⬇️ Download Compiled PDF Report",
        data=st.session_state["pdf"],
        file_name=f"Heart_Disease_Clinical_Report_{exp['run_id']}.pdf",
        mime="application/pdf",
        type="primary",
    )

# -----------------------------------------------------------------------------
# Artifact Integrity & Audit Log
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 03 · Cryptographic Artifact Verification (SHA-256)")
rows = []
for n, r in exp["models"].items():
    if "file" in r:
        p = run / r["file"]
        exists = p.exists()
        cksum_ok = exists and sha256(p) == r["sha256"]
        rows.append({
            "Model": n,
            "Artifact File": r["file"],
            "Status": "Verified" if cksum_ok else "Corrupted/Missing",
            "SHA-256 Hash": r["sha256"][:16] + "...",
        })
    else:
        rows.append({
            "Model": n,
            "Artifact File": "-",
            "Status": f"Failed ({r.get('error', 'unknown')})",
            "SHA-256 Hash": "-",
        })

st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("All saved models are verified against cryptographic SHA-256 checksums before loading to guarantee tamper-proof integrity.")

st.markdown("### 04 · System Execution Log (Tail)")
log = config.ARTIFACT_DIR / "app.log"
st.code("\n".join(log.read_text(errors="ignore").splitlines()[-35:]) if log.exists() else "No log entries.", language="text")
