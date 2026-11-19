"""Would the forecasts have made money? A simple long-or-cash backtest with trading fees.

Rule: at each day's close, look at the model's middle forecast (p50) for tomorrow.
If it predicts a gain bigger than the cost of trading, hold DOGE for the next day.
Otherwise sit in cash. We compare that with simply buying and holding DOGE.

Only the 363 test days are used, and each decision uses only that day's forecast,
so the backtest never peeks at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365  # crypto trades every day


def next_day_returns(backtest: pd.DataFrame) -> np.ndarray:
    return (backtest["actual_next_close"] / backtest["current_close"] - 1).to_numpy()


def forecast_positions(backtest: pd.DataFrame, fee_bps: float) -> np.ndarray:
    """1 = hold DOGE tomorrow, 0 = cash. Only trade when the forecast beats the fee."""
    expected = (backtest["pred_p50"] / backtest["current_close"] - 1).to_numpy()
    return (expected > fee_bps / 10_000).astype(float)


def strategy_returns(
    positions: np.ndarray, market_returns: np.ndarray, fee_bps: float
) -> np.ndarray:
    """Daily returns after paying `fee_bps` every time the position changes."""
    positions = np.asarray(positions, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    if len(positions) != len(market_returns):
        raise ValueError("positions and returns must have the same length.")
    trades = np.abs(np.diff(np.concatenate([[0.0], positions])))
    return positions * market_returns - trades * fee_bps / 10_000


def performance(daily_returns: np.ndarray, positions: np.ndarray | None = None) -> dict:
    daily_returns = np.asarray(daily_returns, dtype=float)
    equity = np.cumprod(1 + daily_returns)
    years = len(daily_returns) / DAYS_PER_YEAR
    volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(DAYS_PER_YEAR))
    sharpe = (
        float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(DAYS_PER_YEAR))
        if np.std(daily_returns, ddof=1) > 0
        else 0.0
    )
    drawdown = equity / np.maximum.accumulate(equity) - 1
    result = {
        "total_return": float(equity[-1] - 1),
        "annual_return": float(equity[-1] ** (1 / years) - 1),
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        held = positions > 0
        result["time_in_market"] = float(held.mean())
        result["trades"] = int(np.abs(np.diff(np.concatenate([[0.0], positions]))).sum())
        result["win_rate_when_holding"] = (
            float((daily_returns[held] > 0).mean()) if held.any() else 0.0
        )
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in result.items()
    }


def random_exposure_check(
    market_returns: np.ndarray,
    time_in_market: float,
    fee_bps: float,
    *,
    simulations: int = 2_000,
    seed: int = 7,
) -> dict:
    """Is the forecast better than holding DOGE on random days for the same share of time?

    Sitting in cash during a crash looks smart even without any skill. This compares the
    strategy against many random strategies that hold DOGE just as often.
    """
    rng = np.random.default_rng(seed)
    totals = np.empty(simulations)
    for i in range(simulations):
        positions = (rng.random(len(market_returns)) < time_in_market).astype(float)
        totals[i] = np.prod(1 + strategy_returns(positions, market_returns, fee_bps)) - 1
    return {
        "simulations": simulations,
        "median_total_return": round(float(np.median(totals)), 4),
        "p5_total_return": round(float(np.percentile(totals, 5)), 4),
        "p95_total_return": round(float(np.percentile(totals, 95)), 4),
        "_totals": totals,
    }


def run_backtest(backtest: pd.DataFrame, fee_bps: float = 10.0) -> dict:
    """Forecast strategy vs buy-and-hold on the same days, both paying fees."""
    market = next_day_returns(backtest)
    positions = forecast_positions(backtest, fee_bps)
    hold = np.ones(len(market))
    model_daily = strategy_returns(positions, market, fee_bps)
    hold_daily = strategy_returns(hold, market, fee_bps)
    forecast = performance(model_daily, positions)
    random_check = random_exposure_check(market, forecast["time_in_market"], fee_bps)
    totals = random_check.pop("_totals")
    random_check["share_of_random_runs_beaten"] = round(
        float((totals < forecast["total_return"]).mean()), 4
    )
    return {
        "fee_bps": fee_bps,
        "days": len(market),
        "start": str(pd.to_datetime(backtest["Date"]).min().date()),
        "end": str(pd.to_datetime(backtest["Date"]).max().date()),
        "forecast_strategy": forecast,
        "buy_and_hold": performance(hold_daily, hold),
        "random_same_exposure": random_check,
    }


def equity_curves(backtest: pd.DataFrame, fee_bps: float = 10.0) -> pd.DataFrame:
    market = next_day_returns(backtest)
    positions = forecast_positions(backtest, fee_bps)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(backtest["Date"]).to_numpy(),
            "Forecast strategy": np.cumprod(1 + strategy_returns(positions, market, fee_bps)),
            "Buy and hold": np.cumprod(
                1 + strategy_returns(np.ones(len(market)), market, fee_bps)
            ),
        }
    )
