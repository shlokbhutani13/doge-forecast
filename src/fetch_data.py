import pandas as pd
import yfinance as yf

from src.data import validate_market_data, write_market_data


def fetch_market_data(symbol="DOGE-USD", period="5y", interval="1d", downloader=yf.download):
    frame = downloader(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame = (
        frame.reset_index()
        .rename(columns={"Datetime": "Date"})
        .drop(columns=["Adj Close"], errors="ignore")
    )
    return validate_market_data(frame)


def main():
    data = fetch_market_data()
    write_market_data(data, "data/doge.csv")
    print(f"Saved {len(data)} rows through {data['Date'].max().date()}")


if __name__ == "__main__":
    main()
