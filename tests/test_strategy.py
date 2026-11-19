import numpy as np
import pandas as pd
import pytest

from src.strategy import forecast_positions, performance, run_backtest, strategy_returns


def make_backtest(closes, forecasts):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=len(closes) - 1),
            "current_close": closes[:-1],
            "actual_next_close": closes[1:],
            "pred_p50": np.asarray(forecasts, dtype=float),
        }
    )


def test_fees_are_charged_only_when_the_position_changes():
    positions = np.array([1, 1, 0, 1])
    market = np.array([0.01, 0.02, 0.05, -0.01])

    returns = strategy_returns(positions, market, fee_bps=10)

    # Buy on day 1, sell on day 3, buy again on day 4: three trades at 0.1%.
    np.testing.assert_allclose(returns, [0.009, 0.02, -0.001, -0.011])


def test_strategy_only_holds_when_the_forecast_beats_the_fee():
    backtest = make_backtest([1.0, 1.0, 1.0], forecasts=[1.0005, 1.01])

    positions = forecast_positions(backtest, fee_bps=10)

    assert positions.tolist() == [0.0, 1.0]


def test_drawdown_and_total_return_follow_the_equity_curve():
    result = performance(np.array([0.10, -0.50, 0.20]))

    assert result["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1, abs=1e-4)
    assert result["max_drawdown"] == pytest.approx(-0.5, abs=1e-4)


def test_backtest_compares_strategy_with_buy_and_hold_and_random_exposure():
    rng = np.random.default_rng(0)
    closes = np.cumprod(1 + rng.normal(0, 0.02, 120))
    backtest = make_backtest(closes, forecasts=closes[:-1] * 1.01)

    result = run_backtest(backtest, fee_bps=10)

    # A forecast that always predicts +1% holds every day, so it matches buy and hold.
    assert result["forecast_strategy"]["time_in_market"] == 1.0
    assert result["forecast_strategy"]["total_return"] == pytest.approx(
        result["buy_and_hold"]["total_return"]
    )
    assert 0 <= result["random_same_exposure"]["share_of_random_runs_beaten"] <= 1
