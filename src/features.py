import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    horizon=1 -> predict next day's Close.
    Creates safe features using only past/current values.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]

    # Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_7"] = close.pct_change(7)
    df["ret_14"] = close.pct_change(14)

    # Volatility
    df["vol_7"] = df["ret_1"].rolling(7).std()
    df["vol_14"] = df["ret_1"].rolling(14).std()

    # Moving averages
    df["sma_7"] = close.rolling(7).mean()
    df["sma_14"] = close.rolling(14).mean()
    df["ema_7"] = close.ewm(span=7, adjust=False).mean()
    df["ema_14"] = close.ewm(span=14, adjust=False).mean()

    # RSI
    df["rsi_14"] = rsi(close, 14)

    # MACD + signal
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Volume features
    df["vol_sma_7"] = df["Volume"].rolling(7).mean()
    df["vol_ret_1"] = df["Volume"].pct_change(1)

    # Lags
    for k in [1, 2, 3, 5, 7, 14]:
        df[f"close_lag_{k}"] = close.shift(k)
        df[f"ret_lag_{k}"] = df["ret_1"].shift(k)

    # Target: next close
    df["target_next_close"] = close.shift(-horizon)

    df = df.dropna().reset_index(drop=True)
    return df
