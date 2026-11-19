import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.artifacts import make_forecast
from src.strategy import equity_curves, run_backtest

st.set_page_config(page_title="DOGE Forecast Lab", page_icon="DF", layout="wide")
st.markdown(
    """
    <style>
    .block-container{max-width:1200px;padding-top:2.5rem}
    [data-testid="stMetric"]{
      border:1px solid #dbe4df;border-radius:14px;padding:14px;background:#fff
    }
    .note{padding:14px 16px;border-radius:12px;background:#f2f6f3;color:#52605a;line-height:1.55}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    raw = pd.read_csv("data/doge.csv", parse_dates=["Date"])
    backtest = pd.read_csv("data/backtest_preds.csv", parse_dates=["Date"])
    metrics = json.loads(open("data/metrics.json", encoding="utf-8").read())
    return raw, backtest, metrics


@st.cache_resource
def load_model():
    return joblib.load("models/doge_models.joblib")


try:
    raw, backtest, metrics = load_data()
    artifact = load_model()
    forecast = make_forecast(raw, artifact)
except Exception as error:
    st.error(f"Forecast artifacts could not be loaded: {error}")
    st.stop()

st.caption("REPRODUCIBLE CRYPTO FORECASTING STUDY")
st.title("DOGE Forecast Lab")
st.write(
    "A one-day probabilistic forecast evaluated against the persistence baseline. "
    "The model estimates return quantiles, then converts them into price levels."
)

freshness, result = st.columns([1, 1])
with freshness:
    st.subheader("Forecast")
    a, b = st.columns(2)
    a.metric("Data through", str(forecast.data_date.date()))
    b.metric("Forecast date", str(forecast.forecast_date.date()))
    c, d = st.columns(2)
    c.metric("Last close", f"${forecast.last_close:.6f}")
    d.metric(
        "Median forecast",
        f"${forecast.p50:.6f}",
        f"{(forecast.p50 / forecast.last_close - 1) * 100:.2f}%",
    )
    st.write(f"80% model interval: **${forecast.p10:.6f} to ${forecast.p90:.6f}**")

with result:
    st.subheader("Holdout evaluation")
    winner = "Persistence baseline" if metrics["winner"] == "baseline" else "Quantile model"
    st.metric("Lower holdout MAE", winner)
    a, b = st.columns(2)
    a.metric("Model MAE", f"${metrics['model_mae']:.6f}")
    b.metric("Baseline MAE", f"${metrics['baseline_mae']:.6f}")
    c, d = st.columns(2)
    c.metric("Interval coverage", f"{metrics['interval_coverage']:.1%}")
    d.metric("Directional accuracy", f"{metrics['directional_accuracy']:.1%}")

if metrics["winner"] == "baseline":
    st.markdown(
        '<div class="note">The persistence baseline has lower holdout error. '
        "This result is reported as measured; the dashboard does not present "
        "the model as a trading edge.</div>",
        unsafe_allow_html=True,
    )

tab_trading, tab_backtest, tab_history, tab_method = st.tabs(
    ["Would it make money?", "Forecast vs actual", "Price history", "Method"]
)
with tab_trading:
    st.write(
        "Rule: hold DOGE tomorrow only when the median forecast predicts a gain bigger than "
        "the trading fee. Otherwise stay in cash. Same test days as above."
    )
    fee_bps = st.select_slider(
        "Trading fee per trade (basis points, 10 = 0.1%)",
        options=[0, 5, 10, 25, 50],
        value=10,
        key="fee_bps",
    )
    trading = run_backtest(backtest, float(fee_bps))
    strat, hold = trading["forecast_strategy"], trading["buy_and_hold"]
    rand = trading["random_same_exposure"]
    a, b, c, d = st.columns(4)
    a.metric("Strategy return", f"{strat['total_return']:.1%}")
    b.metric("Buy and hold", f"{hold['total_return']:.1%}")
    c.metric("Strategy Sharpe", f"{strat['sharpe']:.2f}", f"buy and hold {hold['sharpe']:.2f}")
    d.metric(
        "Worst drawdown", f"{strat['max_drawdown']:.1%}", f"buy and hold {hold['max_drawdown']:.1%}"
    )
    curves = equity_curves(backtest, float(fee_bps))
    fig = go.Figure()
    for column in ("Forecast strategy", "Buy and hold"):
        fig.add_trace(go.Scatter(x=curves["Date"], y=curves[column], name=column))
    fig.update_layout(
        height=420, margin={"l": 10, "r": 10, "t": 20, "b": 10}, yaxis_title="Value of $1"
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        f'<div class="note">The strategy held DOGE on {strat["time_in_market"]:.0%} of days. '
        "Sitting in cash during a falling market looks smart even without skill, so it is "
        f"compared with {rand['simulations']:,} random strategies that hold DOGE just as often. "
        f"It beat {rand['share_of_random_runs_beaten']:.0%} of them (random median "
        f"{rand['median_total_return']:.1%}). That hints at some signal, but it is not strong "
        "proof of an edge.</div>",
        unsafe_allow_html=True,
    )

with tab_backtest:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest["Date"], y=backtest["actual_next_close"], name="Actual"))
    fig.add_trace(
        go.Scatter(x=backtest["Date"], y=backtest["baseline"], name="Persistence baseline")
    )
    fig.add_trace(go.Scatter(x=backtest["Date"], y=backtest["pred_p50"], name="Model median"))
    fig.add_trace(
        go.Scatter(x=backtest["Date"], y=backtest["pred_p90"], line={"width": 0}, showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=backtest["Date"],
            y=backtest["pred_p10"],
            fill="tonexty",
            line={"width": 0},
            name="p10-p90",
        )
    )
    fig.update_layout(height=480, margin={"l": 10, "r": 10, "t": 20, "b": 10}, yaxis_title="USD")
    st.plotly_chart(fig, width="stretch")
with tab_history:
    fig = go.Figure(go.Scatter(x=raw["Date"], y=raw["Close"], name="DOGE-USD close"))
    fig.update_layout(height=460, margin={"l": 10, "r": 10, "t": 20, "b": 10}, yaxis_title="USD")
    st.plotly_chart(fig, width="stretch")
with tab_method:
    st.markdown(
        """
        The project uses past-only return, volatility, momentum, moving-average, RSI, MACD,
        volume, and lag features. Three gradient-boosting quantile regressors estimate the
        next-day return distribution. The final 20% of observations form a time-ordered holdout.

        The persistence baseline predicts that the next close equals the current close.
        Quantiles are ordered before display, and the original crossing count remains an
        evaluation diagnostic.

        **Educational use only. This is not financial advice or a trading recommendation.**
        """
    )
