import numpy as np
import pandas as pd

from src.data import validate_market_data

FEATURE_COLUMNS = [
    "ret_1",
    "ret_7",
    "ret_14",
    "vol_7",
    "vol_14",
    "sma_ratio_7",
    "sma_ratio_14",
    "ema_ratio_7",
    "ema_ratio_14",
    "rsi_14",
    "macd",
    "macd_signal",
    "volume_ratio_7",
    "volume_ret_1",
    "close_lag_return_1",
    "close_lag_return_2",
    "close_lag_return_3",
    "close_lag_return_7",
]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    relative = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + relative)
    return result.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = validate_market_data(frame)
    close = data["Close"]
    returns = close.pct_change()
    data["ret_1"] = returns
    data["ret_7"] = close.pct_change(7)
    data["ret_14"] = close.pct_change(14)
    data["vol_7"] = returns.rolling(7).std()
    data["vol_14"] = returns.rolling(14).std()
    data["sma_ratio_7"] = close / close.rolling(7).mean() - 1
    data["sma_ratio_14"] = close / close.rolling(14).mean() - 1
    data["ema_ratio_7"] = close / close.ewm(span=7, adjust=False).mean() - 1
    data["ema_ratio_14"] = close / close.ewm(span=14, adjust=False).mean() - 1
    data["rsi_14"] = rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["volume_ratio_7"] = data["Volume"] / data["Volume"].rolling(7).mean() - 1
    data["volume_ret_1"] = data["Volume"].pct_change()
    for lag in (1, 2, 3, 7):
        data[f"close_lag_return_{lag}"] = returns.shift(lag)
    return data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def build_training_frame(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if horizon != 1:
        raise ValueError("Only a one-day horizon is supported.")
    raw = validate_market_data(frame)
    target = raw[["Date", "Close"]].copy()
    target["target_return_1"] = target["Close"].shift(-1) / target["Close"] - 1
    features = build_feature_frame(raw)
    return (
        features.merge(target[["Date", "target_return_1"]], on="Date")
        .dropna()
        .reset_index(drop=True)
    )


def latest_inference_row(frame: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(frame)
    if features.empty:
        raise ValueError("At least 30 rows are required.")
    return features.tail(1).copy()
