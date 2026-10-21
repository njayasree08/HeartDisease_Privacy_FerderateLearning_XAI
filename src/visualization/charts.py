"""Consistent Plotly charts with modern healthcare teal & navy palette."""
from __future__ import annotations
import pandas as pd, plotly.express as px, plotly.graph_objects as go
from src import config

NAVY = "#0f172a"
TEAL = "#0d9488"
TEAL_LIGHT = "#14b8a6"
BLUE = "#0284c7"
SLATE = "#334155"
GRAY = "#94a3b8"
RED = "#ef4444"
SEQ = [TEAL, NAVY, BLUE, TEAL_LIGHT, SLATE, "#38bdf8", "#0f766e", "#475569"]


def _style(fig, title=None, h=380):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(color=NAVY, size=15, weight=700)) if title else None,
        height=h,
        margin=dict(l=15, r=15, t=55 if title else 25, b=20),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12, color=SLATE),
        colorway=SEQ,
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
    return fig


def metric_bars(df: pd.DataFrame, metrics=("accuracy", "precision", "recall", "f1", "roc_auc"), title="Model Comparison (Test Set)"):
    long = df.melt(id_vars="model", value_vars=list(metrics), var_name="metric", value_name="value")
    fig = px.bar(long, x="model", y="value", color="metric", barmode="group", range_y=[0, 1.05],
                 color_discrete_sequence=[TEAL, NAVY, BLUE, TEAL_LIGHT, "#f59e0b"])
    fig.update_traces(marker_line_width=0, opacity=0.92)
    return _style(fig, title, 420)


def confusion_heatmap(cm_df: pd.DataFrame, title="Confusion Matrix"):
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, "#f0fdfa"], [0.5, "#99f6e4"], [1, TEAL]], aspect="auto")
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont=dict(size=14, color=NAVY, weight=700))
    return _style(fig, title, 320)


def roc_fig(curves: dict):
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GRAY, width=1.5), name="Chance"))
    for n, (pts, auc) in curves.items():
        fig.add_trace(go.Scatter(x=pts.fpr, y=pts.tpr, mode="lines", line=dict(width=2.2), name=f"{n} (AUC {auc:.3f})"))
    fig.update_xaxes(title="False Positive Rate", range=[-0.02, 1.02])
    fig.update_yaxes(title="True Positive Rate (Recall)", range=[-0.02, 1.02])
    return _style(fig, "ROC Curves (Test Set)", 420)


def pr_fig(curves: dict):
    fig = go.Figure()
    for n, (pts, ap) in curves.items():
        fig.add_trace(go.Scatter(x=pts.recall, y=pts.precision, mode="lines", line=dict(width=2.2), name=f"{n} (AP {ap:.3f})"))
    fig.update_xaxes(title="Recall", range=[-0.02, 1.02])
    fig.update_yaxes(title="Precision", range=[-0.02, 1.02])
    return _style(fig, "Precision-Recall Curves", 420)


def class_distribution(y: pd.Series):
    d = y.map({0: "No Heart Disease", 1: "Heart Disease"}).value_counts().reset_index()
    d.columns = ["class", "count"]
    fig = px.bar(d, x="class", y="count", text="count", color="class",
                 color_discrete_map={"No Heart Disease": "#0284c7", "Heart Disease": TEAL})
    fig.update_traces(textposition="outside", marker_line_width=0, width=0.45)
    return _style(fig, "Target Class Distribution", 320)


def corr_heatmap(df: pd.DataFrame):
    c = df[config.FEATURES + ["target"]].corr().round(2)
    fig = px.imshow(c, text_auto=True, color_continuous_scale="Tealrose", zmin=-1, zmax=1, aspect="auto")
    return _style(fig, "Clinical Feature Correlation Heatmap", 560)


def feature_hist(df: pd.DataFrame, col: str):
    d = df.assign(group=df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}))
    fig = px.histogram(d, x=col, color="group", barmode="overlay", opacity=0.75,
                       color_discrete_map={"No Heart Disease": "#0284c7", "Heart Disease": TEAL},
                       labels={col: config.LABELS.get(col, col)})
    return _style(fig, f"{config.LABELS.get(col, col)} by Target Class", 340)


def importance_bar(df: pd.DataFrame, value="mean_abs_shap", title="Global Feature Importance (mean |SHAP|)"):
    d = df.sort_values(value)
    fig = px.bar(d, x=value, y="feature", orientation="h", color_discrete_sequence=[TEAL])
    fig.update_traces(marker_line_width=0)
    return _style(fig, title, 440)


def contribution_bar(pairs, title="Local Feature Contributions (LIME)"):
    d = pd.DataFrame(pairs, columns=["feature", "weight"]).iloc[::-1]
    d["direction"] = d.weight.map(lambda w: "Increases Risk Estimate" if w > 0 else "Decreases Risk Estimate")
    fig = px.bar(d, x="weight", y="feature", orientation="h", color="direction",
                 color_discrete_map={"Increases Risk Estimate": RED, "Decreases Risk Estimate": TEAL})
    fig.update_traces(marker_line_width=0)
    return _style(fig, title, 400)


def fed_history(h: pd.DataFrame):
    fig = go.Figure()
    for col, nm in [("val_accuracy", "Accuracy"), ("val_recall", "Recall"), ("val_roc_auc", "ROC-AUC")]:
        fig.add_trace(go.Scatter(x=h["round"], y=h[col], mode="lines+markers", line=dict(width=2.2), name=f"Global {nm} (Validation)"))
    fig.add_trace(go.Scatter(x=h["round"], y=h["val_loss"], mode="lines", name="Global Loss (Validation)",
                             line=dict(dash="dot", color=RED, width=2), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="Metric Score", range=[0, 1.02]),
                      yaxis2=dict(title="Cross-Entropy Loss", overlaying="y", side="right", showgrid=False))
    fig.update_xaxes(title="Federated Round")
    return _style(fig, "Federated Learning Convergence Across Hospital Clients", 420)
