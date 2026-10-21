"""Shared Streamlit helpers: page setup, dataset state, artifact access, and clinical theme."""
from __future__ import annotations
import json, logging
from pathlib import Path
import pandas as pd
import streamlit as st
from src import config
from src.data import loader
from src.models import inference


def _logging():
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        config.ARTIFACT_DIR.mkdir(exist_ok=True)
        h = logging.FileHandler(config.ARTIFACT_DIR / "app.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
        root.setLevel(logging.INFO)


def setup(title: str, icon: str = "🫀"):
    _logging()
    st.set_page_config(
        page_title=f"{title} | Clinical Heart Risk FL-XAI",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css = config.ROOT / "assets" / "custom.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    _sidebar()


def _sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span style="font-size:1.8rem;">🫀</span>
                <div>
                    <div style="font-weight:800;color:#0f172a;font-size:1.05rem;line-height:1.2;">CardioAI Studio</div>
                    <div style="font-size:0.75rem;color:#0d9488;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">FL & Explainable AI</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### 📂 Active Dataset")

        up = st.file_uploader(
            "Upload Dataset (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload a Cleveland-style dataset in CSV (.csv) or Excel (.xlsx, .xls) format.",
            key="sidebar_uploader",
        )

        if up is not None and st.session_state.get("_upload_name") != up.name:
            try:
                st.session_state.update(_source=up, _upload_name=up.name, _invert=None, _target_col=None)
                st.session_state.pop("df", None)
                st.session_state.pop("info", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Error loading uploaded file: {exc}")

        # Fallback download button if default data missing
        if not config.DEFAULT_DATASET.exists() and st.button("📥 Download Cleveland Dataset"):
            try:
                loader.download_dataset()
                st.session_state.pop("df", None)
                st.rerun()
            except loader.DatasetError as exc:
                st.error(str(exc))

        df = get_data()
        info = st.session_state.get("info")

        if df is not None and info:
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;margin:8px 0;font-size:0.83rem;">
                    <div style="color:#64748b;font-weight:700;font-size:0.72rem;text-transform:uppercase;">Records & Features</div>
                    <div style="color:#0f172a;font-weight:800;font-size:1.1rem;margin:2px 0;">{len(df):,} rows · {df.shape[1]} cols</div>
                    <div style="color:#0d9488;font-weight:700;margin-top:4px;">● Positive Class: 1 = Heart Disease</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("⚙️ Dataset Target & Mapping", expanded=False):
                verif = info.get("verification", {})
                if verif and verif.get("evidence"):
                    st.caption(f"**Verification:** {verif['evidence']}")

                curr_inv = info.get("invert", False)
                inv = st.checkbox(
                    "Invert target (1 = No disease)",
                    value=curr_inv,
                    help="Automatic clinical correlation sets this correctly. Override only if necessary.",
                )
                if inv != curr_inv:
                    st.session_state["_invert"] = inv
                    st.session_state.pop("df", None)
                    st.rerun()

                if st.button("🔄 Reset to Default Dataset", use_container_width=True):
                    st.session_state.pop("_source", None)
                    st.session_state.pop("_upload_name", None)
                    st.session_state.pop("_invert", None)
                    st.session_state.pop("_target_col", None)
                    st.session_state.pop("df", None)
                    st.session_state.pop("info", None)
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 🧭 Quick Navigation")
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;gap:5px;font-size:0.86rem;font-weight:600;">
                <a href="/" target="_self" style="color:#0f766e;text-decoration:none;padding:4px 8px;border-radius:6px;background:#f0fdfa;">🏠 Dashboard Home</a>
                <a href="/Dataset_Explorer" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">📊 1. Dataset Explorer</a>
                <a href="/Data_Preprocessing" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">⚙️ 2. Data Preprocessing</a>
                <a href="/Model_Training" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">🧠 3. Model Training</a>
                <a href="/Model_Evaluation" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">📈 4. Model Evaluation</a>
                <a href="/Federated_Learning" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">🌐 5. Federated Learning</a>
                <a href="/Explainable_AI" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">🔍 6. Explainable AI</a>
                <a href="/Patient_Risk_Assessment" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">🧑‍⚕️ 7. Patient Risk Assessment</a>
                <a href="/Research_Results" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">📑 8. Research Results</a>
                <a href="/Reports_History" target="_self" style="color:#334155;text-decoration:none;padding:4px 8px;">🗂️ 9. Reports & History</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption(f"🛡️ **Status:** Research / Educational Mode\n\n{config.DISCLAIMER}")


def get_data() -> pd.DataFrame | None:
    """Return the prepared dataframe (binary 'target': 1 = disease) or None."""
    if "df" not in st.session_state:
        src = st.session_state.get("_source", config.DEFAULT_DATASET)
        inv = st.session_state.get("_invert")
        tcol = st.session_state.get("_target_col")
        try:
            if not hasattr(loader, "load_dataset_file"):
                import importlib
                importlib.reload(loader)
            df, info = loader.load_prepared(src, invert=inv, target_col=tcol)
            st.session_state.update(df=df, info=info)
        except loader.DatasetError as exc:
            st.session_state["df"] = None
            st.session_state["info"] = None
            st.session_state["_data_error"] = str(exc)
        except Exception as exc:
            st.session_state["df"] = None
            st.session_state["info"] = None
            st.session_state["_data_error"] = f"Failed to load dataset: {exc}"
    return st.session_state.get("df")


def require_data() -> pd.DataFrame:
    df = get_data()
    if df is None:
        st.error(st.session_state.get("_data_error", "No dataset loaded."))
        st.info(f"Place the dataset at `data/heart_cleveland.csv`, upload CSV/Excel via the sidebar or Dataset Explorer, or download from {config.UCI_PAGE}.")
        st.stop()
    return df


def latest_run() -> Path | None:
    return inference.latest_experiment_dir()


def require_experiment():
    run = latest_run()
    if run is None:
        st.warning("No trained experiment found. Open **Model Training** and run an experiment first.")
        st.stop()
    return inference.load_experiment(run), run


@st.cache_resource(show_spinner=False)
def cached_model(run_dir: str, name: str):
    return inference.load_model(Path(run_dir), name)


def latest_federated():
    root = config.ARTIFACT_DIR / "federated"
    runs = sorted(root.glob("*/federated.json")) if root.exists() else []
    return json.loads(runs[-1].read_text()) if runs else None


def card(col, label: str, value, subtext: str | None = None, accent: str = "teal"):
    accent_class = "accent-teal" if accent == "teal" else "accent-navy"
    sub_html = f"<div class='card-subtext'>{subtext}</div>" if subtext else ""
    col.markdown(
        f"""
        <div class='hc-card {accent_class}'>
            <div class='card-label'>{label}</div>
            <div class='card-value'>{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def split_frames(run: Path):
    return {k: pd.read_csv(run / f"split_{k}.csv", index_col=0) for k in ("train", "val", "test")}
