import numpy as np
import pandas as pd

from src.data import validate_market_data
from src.evaluation import calculate_metrics, order_quantiles, returns_to_prices
from src.features import build_training_frame, latest_inference_row


def sample_data(rows=80):
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(0.10, 0.20, rows) + np.sin(np.arange(rows) / 5) * 0.005
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.arange(rows) + 1000,
        }
    )


def test_market_data_is_sorted_and_rejects_duplicates():
    data = sample_data().iloc[::-1]
    validated = validate_market_data(data)
    assert validated["Date"].is_monotonic_increasing
    duplicate = pd.concat([validated, validated.tail(1)])
    try:
        validate_market_data(duplicate)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("duplicate dates must fail")


def test_training_target_is_next_day_return_and_inference_keeps_latest_row():
    data = sample_data()
    training = build_training_frame(data)
    expected = data["Close"].iloc[-1] / data["Close"].iloc[-2] - 1
    assert np.isclose(training["target_return_1"].iloc[-1], expected)
    inference = latest_inference_row(data)
    assert inference["Date"].iloc[0] == data["Date"].iloc[-1]


def test_quantiles_are_ordered_and_metrics_name_baseline_winner():
    low, mid, high, crossings = order_quantiles(
        np.array([0.3, 0.1]), np.array([0.2, 0.2]), np.array([0.1, 0.3])
    )
    assert crossings == 1
    assert np.all(low <= mid)
    assert np.all(mid <= high)
    assert np.allclose(returns_to_prices(np.array([100, 100]), mid), [120, 120])
    metrics = calculate_metrics(
        actual=np.array([101, 99]),
        model=np.array([103, 97]),
        baseline=np.array([100, 100]),
        low=np.array([98, 96]),
        high=np.array([104, 102]),
        previous=np.array([100, 100]),
        crossings=crossings,
    )
    assert metrics["winner"] == "baseline"
    assert metrics["interval_coverage"] == 1.0
