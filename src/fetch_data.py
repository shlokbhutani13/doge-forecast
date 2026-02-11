import os
import yfinance as yf
import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns like ('Close','DOGE-USD').
    This flattens them to 'Close', 'Open', etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Keep the first level (field name) and drop ticker level
        df.columns = [c[0] for c in df.columns.to_list()]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def fetch(symbol: str = "DOGE-USD", period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance. Check internet or symbol/interval.")

    df = _flatten_columns(df)

    # Index is usually DatetimeIndex; bring it out as a column
    df = df.reset_index()

    # yfinance might name index column 'Date' or 'Datetime'
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    # Prefer 'Adj Close' -> not needed for our project
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Unexpected columns after cleanup. Got: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = fetch(symbol="DOGE-USD", period="5y", interval="1d")
    out_path = os.path.join("data", "doge.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
