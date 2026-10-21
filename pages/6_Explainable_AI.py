"""Clinical AI Explainability Dashboard: Global & Local Interpretability with SHAP & LIME."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src import config, ui
from src.explainability import lime_explainer as le, shap_explainer as se
from src.visualization import charts

ui.setup("Explainable AI")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 06 · EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI)</div>
        <h1>Model Explainability & Feature Attribution</h1>
        <p>Inspect global and instance-level decision drivers using SHAP (Shapley Additive Explanations)
        and LIME (Local Interpretable Model-agnostic Explanations) to audit model transparency and clinical plausibility.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

exp, run = ui.require_experiment()
fr = ui.split_frames(run)
names = [n for n, r in exp["models"].items() if "file" in r]

m_col1, m_col2 = st.columns([1.5, 1], gap="large")
with m_col1:
    model_name = st.selectbox(
        "Select Model to Explain",
        names,
        index=names.index(exp["selected_model"]) if exp["selected_model"] in names else 0,
        help="Select any successfully trained model to audit its decision logic.",
    )

try:
    model = ui.cached_model(str(run), model_name)
except Exception as exc:
    st.error(f"Could not load model safely: {exc}")
    st.stop()

Xte = fr["test"][config.FEATURES]

# -----------------------------------------------------------------------------
# Global SHAP Explanation
# -----------------------------------------------------------------------------
st.markdown("### 01 · Global Feature Attribution (SHAP)")
st.caption("Quantifies the average marginal contribution of each clinical feature across held-out test patients.")

n = st.slider("Test records to evaluate for global SHAP", 10, len(Xte), min(40, len(Xte)))
key = f"shap_{run.name}_{model_name}_{n}"

if st.button("🔬 Compute Global SHAP Values", type="primary"):
    with st.spinner(f"Computing permutation SHAP attributions for {model_name}..."):
        try:
            st.session_state[key] = se.explain(model, fr["train"][config.FEATURES], Xte.iloc[:n])
            st.session_state["shap_key"] = key
            st.success("Global SHAP computation finished!")
        except Exception as exc:
            st.error(f"SHAP failed for {model_name}: {exc}")

shap_exp = st.session_state.get(key)
if shap_exp is not None:
    imp = se.global_importance(shap_exp)
    st.session_state["shap_importance"] = (model_name, imp)

    col_bar, col_sum = st.columns(2, gap="large")
    with col_bar:
        st.markdown("#### Mean Absolute SHAP Impact")
        st.pyplot(se.bar_fig(shap_exp))
    with col_sum:
        st.markdown("#### SHAP Beeswarm Summary Plot")
        st.pyplot(se.summary_fig(shap_exp))

    st.dataframe(imp.round(4), hide_index=True, width="stretch")
    st.markdown(
        """
        <div class="notice">
            <strong>Clinical Note:</strong> SHAP values are expressed in probability units for the 'Heart Disease' class.
            Features with high positive SHAP values (such as ST depression, fluoroscopy vessels, and exercise angina)
            are primary drivers of elevated cardiovascular risk estimates.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Local Patient Instance Explanation (SHAP + LIME)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 02 · Instance-Level Explanation (Single Record: SHAP & LIME)")
st.caption("Inspect how features interact for a specific individual patient from the test cohort.")

rec_idx = st.number_input("Test cohort record index", min_value=0, max_value=len(Xte) - 1, value=0)
row = Xte.iloc[[int(rec_idx)]]
actual_label = int(fr["test"]["target"].iloc[int(rec_idx)])

st.markdown(
    f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin:8px 0;font-size:0.86rem;">
        <strong>Patient Record #{rec_idx}:</strong> Ground-Truth Diagnosis = 
        <span style="font-weight:800;color:{'#b91c1c' if actual_label == 1 else '#15803d'};">
            {'Heart Disease (Class 1)' if actual_label == 1 else 'No Disease (Class 0)'}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.dataframe(row.rename(columns=config.LABELS), hide_index=True, width="stretch")

n_feat = st.slider("Number of top LIME features to explain", 3, 13, 8)

if st.button("🔍 Explain This Patient Record", type="primary"):
    c_shap, c_lime = st.columns(2, gap="large")

    with c_shap:
        st.markdown("#### SHAP Patient Waterfall Plot")
        try:
            s1 = se.explain(model, fr["train"][config.FEATURES], row)
            c_shap.pyplot(se.waterfall_fig(s1, 0))
            c_shap.info(se.describe(s1, 0))
        except Exception as exc:
            c_shap.error(f"SHAP explanation failed: {exc}")

    with c_lime:
        st.markdown("#### LIME Local Feature Weights")
        try:
            pairs, prob, r2 = le.explain_instance(model, le.build_explainer(fr["train"]), row, n_feat)
            c_lime.plotly_chart(charts.contribution_bar(pairs, f"LIME Local Impact (P(Disease)={prob*100:.1f}%)"), width="stretch")
            c_lime.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;font-size:0.84rem;color:#334155;">
                    <strong>Model Probability:</strong> {prob*100:.1f}% &nbsp;|&nbsp; 
                    <strong>Local Fit R²:</strong> {r2:.2f}<br>
                    Red bars elevate predicted disease probability; teal bars lower predicted disease probability.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            c_lime.error(f"LIME explanation failed: {exc}")
