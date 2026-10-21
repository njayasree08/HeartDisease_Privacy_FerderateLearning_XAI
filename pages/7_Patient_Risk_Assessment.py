"""Clinical Patient Risk Assessment Dashboard with transparent probabilities and thresholds."""
from __future__ import annotations
from datetime import datetime
import streamlit as st
from src import config, ui
from src.data.validation import validate_patient_input
from src.explainability import lime_explainer as le, shap_explainer as se
from src.models.inference import predict_patient
from src.models.train import metrics_table
from src.reporting.pdf_report import build_pdf
from src.visualization import charts

ui.setup("Patient Risk Assessment")

st.markdown(
    """
    <div class="clinical-hero">
        <div class="eyebrow">MODULE 07 · CLINICAL INFERENCE & EXPLAINABILITY</div>
        <h1>Patient Risk Assessment</h1>
        <p>Evaluate individual patient profiles against trained diagnostic models with full transparency:
        view predicted class, raw model probability, classification threshold, and SHAP/LIME explainability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

exp, run = ui.require_experiment()
df = ui.require_data()
model_name = exp.get("selected_model", "Selected Model")
model_meta = exp.get("models", {}).get(model_name, {})
opt_threshold = float(model_meta.get("optimal_threshold", 0.50))

st.markdown(
    f"""
    <div class="notice">
        <strong>Active Model:</strong> {model_name} &nbsp;|&nbsp; 
        <strong>Validation Optimal Threshold:</strong> {opt_threshold:.2f} &nbsp;|&nbsp; 
        <strong>Positive Class Semantics:</strong> Class 1 = Heart Disease Present
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Section 1: Patient Profile & Presets
# -----------------------------------------------------------------------------
st.markdown("### 01 · Patient Profile Input")
st.caption("Load a clinical preset profile or customize each clinical predictor.")

presets = {
    "Custom Profile (Manual Input)": None,
    "High-Risk Cardiac Patient (Symptomatic 64M, ST Depression, Blocked Vessels)": {
        "age": 64.0, "sex": 1.0, "cp": 0.0, "trestbps": 160.0, "chol": 290.0,
        "fbs": 1.0, "restecg": 2.0, "thalach": 115.0, "exang": 1.0, "oldpeak": 3.2,
        "slope": 1.0, "ca": 2.0, "thal": 3.0
    },
    "Low-Risk / Healthy Individual (Asymptomatic 36F, Normal Vitals, Clear Vessels)": {
        "age": 36.0, "sex": 0.0, "cp": 1.0, "trestbps": 112.0, "chol": 175.0,
        "fbs": 0.0, "restecg": 1.0, "thalach": 178.0, "exang": 0.0, "oldpeak": 0.0,
        "slope": 2.0, "ca": 0.0, "thal": 2.0
    },
    "Borderline / Moderate Risk Patient (54M, Mild Angina, Normal Stress Test)": {
        "age": 54.0, "sex": 1.0, "cp": 2.0, "trestbps": 138.0, "chol": 245.0,
        "fbs": 0.0, "restecg": 0.0, "thalach": 142.0, "exang": 0.0, "oldpeak": 1.2,
        "slope": 1.0, "ca": 0.0, "thal": 2.0
    },
}

selected_preset = st.selectbox(
    "Clinical Profile Presets",
    list(presets.keys()),
    index=0,
    help="Select a benchmark patient profile to quickly inspect model behavior across risk categories.",
)

preset_vals = presets[selected_preset]

# Decision Threshold Slider
thresh_col1, thresh_col2 = st.columns([1.5, 1], gap="large")
with thresh_col1:
    decision_threshold = st.slider(
        "Decision Threshold (Classification Cutoff)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probabilities at or above this threshold classify the patient as HIGHER RISK. Standard clinical default is 0.50.",
    )
with thresh_col2:
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-top:6px;font-size:0.83rem;">
            <div style="font-weight:700;color:#0f172a;">Threshold Guidance</div>
            <div style="color:#64748b;margin-top:4px;">
                Standard Default: <strong>0.50</strong><br>
                Validation ROC-Optimal (Youden's J): <strong>{opt_threshold:.2f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

vals = {}

with st.form("patient_form"):
    cols = st.columns(3)

    for k, feature in enumerate(config.FEATURES):
        label, desc, kind, spec = config.FEATURE_INFO[feature]
        col = cols[k % 3]

        if kind == "cat":
            opts = list(spec.keys())
            def_idx = 0
            if preset_vals and feature in preset_vals:
                pval = int(preset_vals[feature])
                if pval in opts:
                    def_idx = opts.index(pval)

            vals[feature] = col.selectbox(
                label,
                opts,
                index=def_idx,
                format_func=lambda v, s=spec: f"{v} – {s[v]}",
                help=f"{desc}. Values: {spec}",
                key=f"f_{feature}_{selected_preset}",
            )
        else:
            def_val = float(df[feature].median())
            if preset_vals and feature in preset_vals:
                def_val = float(preset_vals[feature])

            step = 0.1 if feature == "oldpeak" else 1.0
            vals[feature] = col.number_input(
                label,
                min_value=float(spec[0]),
                max_value=float(spec[1]),
                value=def_val,
                step=step,
                help=desc,
                key=f"f_{feature}_{selected_preset}",
            )

    submitted = st.form_submit_button(
        "Run Patient Risk Assessment",
        type="primary",
        use_container_width=True,
    )

if submitted:
    problems = validate_patient_input(vals)
    if problems:
        for p in problems:
            st.error(p)
    else:
        try:
            model = ui.cached_model(str(run), model_name)
            result = predict_patient(model, vals, threshold=decision_threshold)
            st.session_state["patient_result"] = {
                "res": result,
                "vals": dict(vals),
                "threshold": decision_threshold,
                "model_name": model_name,
            }
        except Exception as exc:
            st.error(f"Assessment could not be completed: {exc}")

patient = st.session_state.get("patient_result")

# -----------------------------------------------------------------------------
# Section 2: Assessment Results
# -----------------------------------------------------------------------------
if patient:
    result = patient["res"]
    patient_vals = patient["vals"]
    prob_val = float(result["probability"])
    thresh_val = float(result.get("threshold", 0.50))
    is_high = result["higher_risk"]
    pred_class = result.get("predicted_class", 1 if is_high else 0)
    risk_cat = result.get("risk_category", "High Risk" if is_high else "Low Risk")

    st.markdown("---")
    st.markdown("### 02 · Assessment Output & Transparent Decision Metrics")
    st.caption(f"Evaluated on {datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p')} using active model: {model_name}")

    # Primary Decision Card
    if is_high:
        st.markdown(
            f"""
            <div style="background:#fef2f2;border:2px solid #f87171;border-radius:12px;padding:20px 24px;margin-bottom:18px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                    <div>
                        <div style="font-size:0.8rem;font-weight:800;color:#991b1b;letter-spacing:0.08em;text-transform:uppercase;">MODEL CLASSIFICATION OUTCOME</div>
                        <h2 style="color:#b91c1c !important;margin:4px 0 0 0 !important;font-size:1.8rem !important;">🚨 HIGHER RISK · Class 1 (Heart Disease Positive)</h2>
                    </div>
                    <div style="background:#fee2e2;border:1px solid #fca5a5;padding:8px 16px;border-radius:9999px;font-weight:800;color:#991b1b;font-size:0.95rem;">
                        Probability: {prob_val*100:.1f}% &ge; Threshold: {thresh_val*100:.1f}%
                    </div>
                </div>
                <p style="color:#7f1d1d;font-size:0.92rem;margin:10px 0 0 0;line-height:1.5;">
                    The estimated probability of heart disease exceeds the active decision threshold by <strong>{(prob_val - thresh_val)*100:+.1f}%</strong>.
                    Clinical markers strongly associate this profile with coronary artery disease.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background:#f0fdf4;border:2px solid #4ade80;border-radius:12px;padding:20px 24px;margin-bottom:18px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                    <div>
                        <div style="font-size:0.8rem;font-weight:800;color:#166534;letter-spacing:0.08em;text-transform:uppercase;">MODEL CLASSIFICATION OUTCOME</div>
                        <h2 style="color:#15803d !important;margin:4px 0 0 0 !important;font-size:1.8rem !important;">✅ LOWER RISK · Class 0 (Low Disease Likelihood)</h2>
                    </div>
                    <div style="background:#dcfce7;border:1px solid #86efac;padding:8px 16px;border-radius:9999px;font-weight:800;color:#166534;font-size:0.95rem;">
                        Probability: {prob_val*100:.1f}% &lt; Threshold: {thresh_val*100:.1f}%
                    </div>
                </div>
                <p style="color:#14532d;font-size:0.92rem;margin:10px 0 0 0;line-height:1.5;">
                    The estimated probability of heart disease is below the active decision threshold by <strong>{(thresh_val - prob_val)*100:.1f}%</strong>.
                    Note: Uncalibrated model output for educational/research use; does not substitute clinical diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Transparent Metric Pillars
    rc1, rc2, rc3, rc4 = st.columns(4)
    ui.card(rc1, "Predicted Class", f"Class {pred_class}", subtext="1 = Disease | 0 = No Disease", accent="teal" if is_high else "navy")
    ui.card(rc2, "Model Probability", f"{prob_val*100:.1f}%", subtext=f"Raw P(Disease) = {prob_val:.4f}", accent="teal")
    ui.card(rc3, "Active Threshold", f"{thresh_val*100:.1f}%", subtext="Cutoff for positive class", accent="navy")
    ui.card(rc4, "Risk Tier", risk_cat, subtext="Clinical interpretation tier", accent="teal" if is_high else "navy")

    # Progress visualizer
    st.markdown("#### Probability vs Threshold Meter")
    st.progress(max(0.0, min(1.0, prob_val)))
    st.caption(f"Zero Probability [0.0%] ◄──────── Active Threshold: {thresh_val*100:.1f}% ────────► Maximum Probability [100.0%]")

    # -------------------------------------------------------------------------
    # Section 3: Explainable AI (SHAP & LIME)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 03 · Model Explanations (SHAP & LIME)")
    st.caption("Inspect why the model arrived at this specific probability for this patient.")

    model = ui.cached_model(str(run), model_name)
    frames = ui.split_frames(run)
    narrative = ""

    shap_col, lime_col = st.columns(2, gap="large")

    with shap_col:
        st.markdown("#### SHAP Local Waterfall Explanation")
        try:
            shap_values = se.explain(
                model,
                frames["train"][config.FEATURES],
                result["frame"],
            )
            st.pyplot(se.waterfall_fig(shap_values, 0))
            narrative = se.describe(shap_values, 0)
            st.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;font-size:0.86rem;line-height:1.6;color:#334155;">
                    <strong>SHAP Narrative:</strong> {narrative}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"SHAP explanation could not be computed: {exc}")

    with lime_col:
        st.markdown("#### LIME Local Feature Contributions")
        try:
            pairs, probability, r2 = le.explain_instance(
                model,
                le.build_explainer(frames["train"]),
                result["frame"],
                8,
            )
            st.plotly_chart(
                charts.contribution_bar(pairs, f"LIME Local Contributions (P={probability*100:.1f}%)"),
                width="stretch",
            )
            st.caption(f"LIME surrogate local-fit R² = {r2:.2f}. Explains local feature directions.")
        except Exception as exc:
            st.error(f"LIME explanation could not be computed: {exc}")

    # -------------------------------------------------------------------------
    # Section 4: Clinical PDF Export
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 04 · Export Patient Assessment Report")

    export_col1, export_col2 = st.columns([1.5, 1])
    with export_col1:
        include_patient = st.checkbox(
            "Include complete patient features, model metrics, and explanation in PDF",
            value=True,
        )

    if include_patient:
        try:
            pdf_bytes = build_pdf(
                exp,
                metrics_table(exp, "test"),
                ui.latest_federated(),
                patient={
                    "label": result["label"],
                    "probability": result["probability"],
                    "inputs": patient_vals,
                    "narrative": narrative,
                },
            )
            st.download_button(
                "📥 Download Patient Assessment PDF Report",
                data=pdf_bytes,
                file_name=f"Patient_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
            )
        except Exception as exc:
            st.error(f"PDF report generation error: {exc}")
else:
    st.markdown(
        """
        <div class="notice">
            <strong>Ready for Assessment:</strong> Select a patient preset profile above or enter custom clinical values,
            then click <strong>Run Patient Risk Assessment</strong> to compute the transparent prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )
