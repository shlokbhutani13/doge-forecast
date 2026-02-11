import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.features import make_features

import subprocess
import sys

def run_training():
    subprocess.run([sys.executable, "src/train.py"], check=True)


# ----------------------------
# Page config + subtle styling
# ----------------------------
st.set_page_config(
    page_title="DOGE Forecast",
    page_icon="🐶",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .stMetric { background: rgba(255,255,255,0.04); padding: 14px; border-radius: 16px; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.9rem; opacity: 0.9; }
      .small-note { opacity: 0.7; font-size: 0.9rem; }
      .card { background: rgba(255,255,255,0.04); padding: 18px; border-radius: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Loaders
# ----------------------------
@st.cache_data
def load_raw():
    df = pd.read_csv("data/doge.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_models():
    return joblib.load("models/doge_models.joblib")

def file_exists(path: str) -> bool:
    return os.path.exists(path)

# ----------------------------
# Header
# ----------------------------
st.title("🐶 DOGE-USD Forecast")
st.caption("Probabilistic next-day forecast using quantile regression (p10 / p50 / p90) + backtesting on the last 20% of history.")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    show_rows = st.slider("Rows to preview", 5, 50, 10)
    show_raw = st.toggle("Show raw data table", value=True)
    st.divider()
    st.markdown("**Files status**")
    st.write("✅ data/doge.csv" if file_exists("data/doge.csv") else "❌ data/doge.csv missing")
    st.write("✅ models/doge_models.joblib" if file_exists("models/doge_models.joblib") else "❌ models/doge_models.joblib missing")
    st.write("✅ data/backtest_preds.csv" if file_exists("data/backtest_preds.csv") else "⚠️ data/backtest_preds.csv missing (run training)")
    st.divider()
    st.markdown('<div class="small-note">Tip: If charts don’t update after training, restart Streamlit.</div>', unsafe_allow_html=True)

# ----------------------------
# Guard rails
# ----------------------------
if not file_exists("data/doge.csv"):
    st.error("Missing data/doge.csv. Run: `python src/fetch_data.py`")
    st.stop()

if not file_exists("models/doge_models.joblib"):
    st.error("Missing models/doge_models.joblib. Run: `python src/train.py`")
    st.stop()

raw = load_raw()
art = load_models()

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_backtest, tab_forecast = st.tabs(["Overview", "Backtest", "Forecast"])

# ----------------------------
# Overview tab
# ----------------------------
with tab_overview:
    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader("Latest snapshot")
        last = raw.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Date", str(pd.to_datetime(last["Date"]).date()))
        c2.metric("Close", f"${float(last['Close']):.6f}")
        c3.metric("High", f"${float(last['High']):.6f}")
        c4.metric("Volume", f"{int(last['Volume']):,}")

        st.markdown('<div class="small-note">This dashboard forecasts the next-day close. Crypto is volatile; this is not financial advice.</div>', unsafe_allow_html=True)

    with colB:
        st.subheader("Price trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw["Date"], y=raw["Close"], mode="lines", name="Close"))
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)

    if show_raw:
        st.subheader("Raw data (preview)")
        st.dataframe(raw.tail(show_rows), use_container_width=True, height=280)

# ----------------------------
# Backtest tab
# ----------------------------
with tab_backtest:
    st.subheader("Backtest (last 20% of history)")

    bt_path = "data/backtest_preds.csv"

    if not file_exists(bt_path):
        st.warning("Backtest data missing.")
        if st.button("Run training to generate backtest"):
            with st.spinner("Training model… this may take a minute"):
                run_training()
            st.success("Training complete. Restarting app…")
            st.rerun()


    else:
        bt = pd.read_csv(bt_path)
        bt["Date"] = pd.to_datetime(bt["Date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["Date"], y=bt["target_next_close"], mode="lines", name="Actual Next Close"))
        fig.add_trace(go.Scatter(x=bt["Date"], y=bt["pred_p50"], mode="lines", name="Predicted (p50)"))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=bt["Date"], y=bt["pred_p90"],
            mode="lines", line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=bt["Date"], y=bt["pred_p10"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            name="p10–p90 band",
            opacity=0.2
        ))

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="card"><b>How to read this:</b> '
            'The line shows the model’s median prediction (p50). '
            'The shaded area is uncertainty (p10–p90). '
            'If actual prices often fall outside the band, the model is underestimating volatility.</div>',
            unsafe_allow_html=True
        )

# ----------------------------
# Forecast tab
# ----------------------------
with tab_forecast:
    st.subheader("Next-Day Forecast (from latest available day)")

    feat = make_features(raw, horizon=1)
    latest = feat.iloc[-1:]
    X = latest[art["feature_cols"]].values

    p10 = float(art["model_p10"].predict(X)[0])
    p50 = float(art["model_p50"].predict(X)[0])
    p90 = float(art["model_p90"].predict(X)[0])

    last_date = latest["Date"].iloc[0]
    last_close = float(latest["Close"].iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Date", str(pd.to_datetime(last_date).date()))
    c2.metric("Last Close", f"${last_close:.6f}")
    c3.metric("Forecast (p50)", f"${p50:.6f}")
    c4.metric("Confidence Band (p10–p90)", f"${p10:.6f} – ${p90:.6f}")

    # Simple “range bar”
    st.caption("Range visualization (p10 → p90)")
    bar = go.Figure()
    bar.add_trace(go.Scatter(
        x=[p10, p90],
        y=["Tomorrow"],
        mode="lines+markers",
        line=dict(width=10),
        marker=dict(size=14),
        name="p10–p90"
    ))
    bar.add_trace(go.Scatter(
        x=[p50],
        y=["Tomorrow"],
        mode="markers",
        marker=dict(size=16),
        name="p50"
    ))
    bar.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Price (USD)")
    st.plotly_chart(bar, use_container_width=True)

    st.markdown('<div class="small-note">Not financial advice. For learning/demo purposes.</div>', unsafe_allow_html=True)
