from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Market data is empty.")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame[REQUIRED_COLUMNS].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise").dt.tz_localize(None)
    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data["Date"].duplicated().any():
        raise ValueError("Market data contains duplicate dates.")
    if (data["Close"] <= 0).any() or (data["Volume"] < 0).any():
        raise ValueError("Market data contains invalid prices or volume.")
    return data.sort_values("Date").reset_index(drop=True)


def write_market_data(frame: pd.DataFrame, path: str | Path) -> None:
    data = validate_market_data(frame)
    destination = Path(path)
    temporary = destination.with_suffix(".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(destination)
