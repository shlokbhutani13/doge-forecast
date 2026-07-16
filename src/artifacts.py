from dataclasses import dataclass
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from src.evaluation import order_quantiles, returns_to_prices
from src.features import latest_inference_row


@dataclass
class Forecast:
    data_date: pd.Timestamp
    forecast_date: pd.Timestamp
    last_close: float
    p10: float
    p50: float
    p90: float


def make_forecast(raw, artifact):
    if artifact.get("artifact_version") != 2:
        raise ValueError("Model artifact is incompatible.")
    row = latest_inference_row(raw)
    columns = artifact["feature_cols"]
    x = row[columns]
    returns = [artifact[f"model_{name}"].predict(x)[0] for name in ("p10", "p50", "p90")]
    low, mid, high, _ = order_quantiles(*(np.array([value]) for value in returns))
    close = float(row["Close"].iloc[0])
    prices = returns_to_prices(close, np.array([low[0], mid[0], high[0]]))
    date = pd.to_datetime(row["Date"].iloc[0])
    return Forecast(date, date + timedelta(days=1), close, *map(float, prices))


def load_artifact(path="models/doge_models.joblib"):
    return joblib.load(path)
