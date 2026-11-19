# DOGE Forecast Lab

[![CI](https://github.com/shlokbhutani13/doge-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokbhutani13/doge-forecast/actions/workflows/ci.yml)

A small experiment in forecasting tomorrow's Dogecoin (DOGE-USD) price. Instead of one number, it predicts a range: a low, middle and high estimate. Then it checks whether that's any better than the simplest guess, "tomorrow's price = today's price."

**Try it:** [live dashboard](https://doge-forecast-hzbcysrfyyxfrqzagu6vu7.streamlit.app/) (it can take a few seconds to wake up)

## Result

Tested on the last 363 days, kept in time order so the model never sees the future:

| | Model | "Same as today" baseline |
| --- | ---: | ---: |
| Mean absolute error | 0.00457 | 0.00462 |
| RMSE | 0.00723 | 0.00727 |

- The 10th–90th percentile range contained the real price 84.6% of the time.
- The model called the direction (up or down) right 56% of the time.

So the model beats the baseline, but only by a little. I show both numbers on the dashboard instead of pretending this is a trading edge.

## How it works

1. 1,827 days of DOGE-USD price history.
2. 18 features built only from past data: returns, volatility, moving averages, RSI, MACD and volume.
3. Gradient Boosting quantile regression predicts the 10th, 50th and 90th percentile of the next day's return, which is then turned into prices.
4. A Streamlit dashboard shows the forecast range, recent trend and model vs. baseline.

The live app reads saved data and a saved model. It doesn't download prices or retrain while you use it.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Get fresh data and retrain:

```bash
python -m src.fetch_data
python -m src.train
```

## Tests

```bash
ruff check .
python -m pytest -q
```

The tests check data validation, that the target really is the next day, that the forecast range stays in order (low ≤ middle ≤ high), and the baseline comparison.

## Limits

Crypto prices are noisy, and past results don't predict future ones. This is a learning project, not financial advice.

## License

MIT
