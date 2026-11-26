import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from src.config import DATA_DIR, RESULTS_DIR, WINDOW, HORIZONS, TARGET, SEED
from src.data import load_dataset
from src.preprocess import prepare, to_ugm3
from src.models import MODELS

st.set_page_config(page_title="PM2.5 AI Forecaster", page_icon="🌫️", layout="wide")

BEST_CELL = "background-color:#1B5E20;color:white;font-weight:600"


# ---------- CPCB AQI helper (PM2.5 breakpoints) ----------
def get_aqi_badge(pm25: float):
    if pm25 <= 30:
        return "Good", "#00B050"
    elif pm25 <= 60:
        return "Satisfactory", "#92D050"
    elif pm25 <= 90:
        return "Moderate", "#FFD600"
    elif pm25 <= 120:
        return "Poor", "#FF9900"
    elif pm25 <= 250:
        return "Very Poor", "#FF3B30"
    else:
        return "Severe", "#C00000"


# ---------- Takeaway text computed from metrics.csv (never hard-coded) ----------
def build_takeaway(df: pd.DataFrame) -> str:
    out = []
    for h in ["1h", "3h", "6h"]:
        sub = df[df.Horizon == h]
        if sub.empty or "Persistence" not in set(sub.Model):
            continue
        pers = sub[sub.Model == "Persistence"].iloc[0]
        deep = sub[sub.Model != "Persistence"]
        best = deep.loc[deep["MAE"].idxmin()]
        chg = (pers.MAE - best.MAE) / pers.MAE * 100
        out.append(
            f"• **At +{h}:** best model by MAE is **{best.Model}** "
            f"({best.MAE:.2f} vs Persistence {pers.MAE:.2f}, "
            f"{abs(chg):.0f}% {'lower' if chg >= 0 else 'higher'}). "
            f"R² {best.R2:.3f} vs {pers.R2:.3f}; spike MAE {best.Spike_MAE:.1f} vs {pers.Spike_MAE:.1f}."
        )
    return "\n\n".join(out)


def highlight_best(df: pd.DataFrame) -> pd.DataFrame:
    """Highlight the best value in each metric column, separately for each horizon."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for _, g in df.groupby("Horizon"):
        for c in ["MAE", "RMSE", "Spike_MAE"]:
            styles.loc[g[c].idxmin(), c] = BEST_CELL
        styles.loc[g["R2"].idxmax(), "R2"] = BEST_CELL
    return styles


# ---------- Cached data & model loader ----------
@st.cache_resource
def load_pipeline():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    csv_path = DATA_DIR / "city_hour.csv"
    data_arg = str(csv_path) if csv_path.exists() else None
    df, source = load_dataset(data_arg, "Delhi")
    d = prepare(df)

    nf = d["X_train"].shape[2]
    y_true = to_ugm3(d["y_test"], d["t_mean"], d["t_std"])

    pm_idx = d["features"].index(TARGET)
    last_pm = to_ugm3(d["X_test"][:, -1, pm_idx], d["t_mean"], d["t_std"])
    preds = {"Persistence": np.repeat(last_pm[:, None], len(HORIZONS), axis=1)}

    for name, cls in MODELS.items():
        pt_path = RESULTS_DIR / f"{name.replace(' ', '_')}.pt"
        if pt_path.exists():
            model = cls(nf, len(HORIZONS))
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(d["X_test"])).numpy()
            preds[name] = to_ugm3(out, d["t_mean"], d["t_std"])

    X_pm_ugm3 = to_ugm3(d["X_test"][:, :, pm_idx], d["t_mean"], d["t_std"])

    spike_thr = float(np.percentile(y_true[:, 0], 90))
    if (RESULTS_DIR / "run_info.json").exists():
        info = json.load(open(RESULTS_DIR / "run_info.json"))
        source = info.get("source", source)
        spike_thr = info.get("spike_threshold_ugm3_1h", spike_thr)

    metrics_df = pd.read_csv(RESULTS_DIR / "metrics.csv") if (RESULTS_DIR / "metrics.csv").exists() else pd.DataFrame()
    return d, y_true, preds, X_pm_ugm3, source, spike_thr, metrics_df


d, y_true, preds, X_pm_ugm3, source, spike_thr, metrics_df = load_pipeline()
n_test = len(y_true)

# ---------- Sidebar ----------
st.sidebar.title("🌫️ Forecast Controls")
st.sidebar.caption(f"**Data Source:** `{source}`")
st.sidebar.caption(f"**Test Windows:** `{n_test}` hrs | **90th %ile Spike:** `{spike_thr:.1f} µg/m³`")

avail_models = [m for m in ["GRU", "LSTM", "1D CNN", "Persistence"] if m in preds]
selected_model = st.sidebar.selectbox("Select Primary Model", avail_models, index=0)
compare_all = st.sidebar.checkbox("Overlay All Models on Chart", value=False)

spike_indices = np.where(y_true[:, 2] >= spike_thr)[0]
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = int(spike_indices[len(spike_indices) // 2]) if len(spike_indices) > 0 else n_test - 48

col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("⚡ Jump to Spike", use_container_width=True):
    if len(spike_indices) > 0:
        st.session_state.sample_idx = int(np.random.choice(spike_indices))
if col_b2.button("🎲 Random Hour", use_container_width=True):
    st.session_state.sample_idx = int(np.random.randint(0, n_test))

idx = st.sidebar.slider("Test Set Window Index (t)", 0, n_test - 1, key="sample_idx")

st.sidebar.divider()
st.sidebar.markdown("**Pipeline Architecture**")
st.sidebar.markdown(
    f"- **Input Window:** `{WINDOW}h` × `{len(d['features'])}` features\n"
    f"- **Horizons:** `+1h`, `+3h`, `+6h`\n"
    f"- **Spike Handling:** `log1p` + Huber Loss\n"
    f"- **Split:** 70% Train / 15% Val / 15% Test"
)

# ---------- Header ----------
st.title("Delhi PM2.5 Multi-Horizon Deep Forecaster")
tab1, tab2 = st.tabs(["📈 Live Forecast Simulator", "🏆 Model Benchmarks & Diagnostics"])

CARD = "padding:14px;border-radius:10px;background:#1E1E2E;color:#FFFFFF;border-left:6px solid {c}"

with tab1:
    hist_24 = X_pm_ugm3[idx]
    curr_pm = hist_24[-1]
    curr_cat, curr_color = get_aqi_badge(curr_pm)

    cols = st.columns(4)
    cols[0].markdown(
        f"<div style='{CARD.format(c=curr_color)}'>"
        f"<span style='color:#AAAAAA;font-size:13px'>Current Observed (t=0)</span><br>"
        f"<span style='font-size:26px;font-weight:700;color:#FFFFFF'>{curr_pm:.1f} µg/m³</span><br>"
        f"<span style='color:{curr_color};font-weight:600'>● {curr_cat}</span></div>",
        unsafe_allow_html=True,
    )

    for k, h in enumerate(HORIZONS):
        pred_val = preds[selected_model][idx, k]
        true_val = y_true[idx, k]
        pers_val = preds["Persistence"][idx, k]
        err = pred_val - true_val
        pers_err = abs(pers_val - true_val)
        gain = pers_err - abs(err)
        cat, col = get_aqi_badge(pred_val)

        cols[k + 1].markdown(
            f"<div style='{CARD.format(c=col)}'>"
            f"<span style='color:#AAAAAA;font-size:13px'>+{h}h Forecast ({selected_model})</span><br>"
            f"<span style='font-size:26px;font-weight:700;color:#FFFFFF'>{pred_val:.1f} µg/m³</span> "
            f"<span style='font-size:13px;color:#CCCCCC'>(Actual: {true_val:.1f})</span><br>"
            f"<span style='color:{col};font-weight:600'>● {cat}</span> &nbsp;|&nbsp; "
            f"<span style='font-size:12px;color:{'#4CAF50' if gain >= 0 else '#FF5252'}'>"
            f"{'Beats' if gain >= 0 else 'Trails'} Persistence by {abs(gain):.1f}</span></div>",
            unsafe_allow_html=True,
        )

    st.write("")

    fig = go.Figure()
    t_hist = list(range(-WINDOW + 1, 1))
    t_fut = [0] + HORIZONS

    fig.add_trace(go.Scatter(
        x=t_hist, y=hist_24, mode="lines+markers", name="Last 24h Observed",
        line=dict(color="#00E5FF", width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(
        x=t_fut, y=[curr_pm] + list(y_true[idx]), mode="lines+markers", name="Actual Ground Truth",
        line=dict(color="#FFFFFF", width=3), marker=dict(size=9, symbol="diamond")))
    fig.add_trace(go.Scatter(
        x=t_fut, y=[curr_pm] + list(preds["Persistence"][idx]), mode="lines+markers", name="Persistence Baseline",
        line=dict(color="#888888", width=2, dash="dot"), marker=dict(size=6)))

    model_colors = {"GRU": "#00E676", "LSTM": "#2979FF", "1D CNN": "#FF9100"}
    models_to_plot = [m for m in ["GRU", "LSTM", "1D CNN"] if m in preds] if compare_all else [selected_model]
    for m_name in models_to_plot:
        if m_name == "Persistence":
            continue
        fig.add_trace(go.Scatter(
            x=t_fut, y=[curr_pm] + list(preds[m_name][idx]), mode="lines+markers", name=f"{m_name} Forecast",
            line=dict(color=model_colors.get(m_name, "#00E676"), width=3, dash="dash"),
            marker=dict(size=9, symbol="circle")))

    fig.add_vline(x=0, line_dash="dash", line_color="#888", annotation_text="Forecast Origin (t=0)")
    fig.add_hline(y=spike_thr, line_dash="dot", line_color="#FF5252",
                  annotation_text=f"90th %ile Spike Threshold ({spike_thr:.0f} µg/m³)")
    fig.update_layout(
        title=f"24-Hour Lookback Window & +1h / +3h / +6h Trajectory (Test Window #{idx})",
        xaxis_title="Hours Relative to Forecast Origin (t=0)",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_dark", height=440, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, theme=None)   # theme=None keeps the dark template

    st.subheader("Rolling Test Set Horizon Comparison (Last 7 Days)")
    h_choice = st.radio("Select Forecast Horizon", ["1h", "3h", "6h"], index=2, horizontal=True)
    h_idx = HORIZONS.index(int(h_choice[0]))
    n_tail = min(24 * 7, n_test)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=y_true[-n_tail:, h_idx], name="Actual", line=dict(color="#FFFFFF", width=2)))
    for m_name, col in model_colors.items():
        if m_name in preds:
            fig2.add_trace(go.Scatter(
                y=preds[m_name][-n_tail:, h_idx], name=m_name,
                line=dict(color=col, width=2 if m_name == selected_model else 1.2),
                opacity=1.0 if m_name == selected_model or compare_all else 0.35))
    fig2.update_layout(
        xaxis_title="Hours (Last 7 Days of Test Set)", yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark", height=340, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True, theme=None)

with tab2:
    st.subheader("Test Set Evaluation Metrics (µg/m³)")
    col_m1, col_m2 = st.columns([3, 2])
    with col_m1:
        if not metrics_df.empty:
            h_filter = st.selectbox("Filter by Horizon", ["All", "1h", "3h", "6h"])
            df_show = metrics_df if h_filter == "All" else metrics_df[metrics_df["Horizon"] == h_filter]
            styled = (df_show.style
                      .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}", "Spike_MAE": "{:.1f}"})
                      .apply(highlight_best, axis=None))
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("Green = best value per metric within each horizon.")
        else:
            st.warning("results/metrics.csv not found. Run `python main.py` first.")
    with col_m2:
        if not metrics_df.empty:
            st.info("**Key Takeaways (computed from metrics.csv)**\n\n" + build_takeaway(metrics_df))

    st.divider()
    st.subheader("Saved Matplotlib Diagnostic Plots")
    g1, g2 = st.columns(2)
    if (RESULTS_DIR / "horizon_comparison.png").exists():
        g1.image(str(RESULTS_DIR / "horizon_comparison.png"), caption="Error vs Forecast Horizon", use_container_width=True)
    if (RESULTS_DIR / "training_curves.png").exists():
        g2.image(str(RESULTS_DIR / "training_curves.png"), caption="Huber Loss Training & Validation Curves", use_container_width=True)
    g3, g4 = st.columns(2)
    if (RESULTS_DIR / "scatter_6h.png").exists():
        g3.image(str(RESULTS_DIR / "scatter_6h.png"), caption="6h Predicted vs Actual Scatter", use_container_width=True)
    if (RESULTS_DIR / "predictions.png").exists():
        g4.image(str(RESULTS_DIR / "predictions.png"), caption="7-Day Forecast Trajectories", use_container_width=True)