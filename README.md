# DOGE Forecast Lab

[![CI](https://github.com/shlokbhutani13/doge-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokbhutani13/doge-forecast/actions/workflows/ci.yml)

A small experiment in forecasting tomorrow's Dogecoin (DOGE-USD) price. Instead of one number, it predicts a range: a low, middle and high estimate. Then it checks two things: is that better than the simplest guess ("tomorrow's price = today's price"), and would trading on it have made money after fees?

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

## Would it have made money?

A better forecast doesn't automatically mean profit, so I tested a simple trading rule on the same 363 days: hold DOGE tomorrow only if the forecast predicts a gain bigger than the trading fee, otherwise sit in cash. Every trade pays a 0.1% fee.

| | Forecast strategy | Buy and hold |
| --- | ---: | ---: |
| Total return | -2.1% | -69.0% |
| Sharpe ratio | 0.20 | -1.14 |
| Worst drawdown | -24.0% | -75.2% |
| Days holding DOGE | 25% | 100% |

That looks great, but DOGE crashed during the test period, so simply holding less would have helped. To check for that, I compared it with 2,000 random strategies that hold DOGE on random days for the same 25% of the time. The forecast beat 88% of them (random median: -34.9%). That's a hint of real signal, but not strong proof, and a 0.20 Sharpe ratio isn't something you'd trade on. The dashboard lets you change the fee and watch the edge shrink.

## How it works

1. 1,827 days of DOGE-USD price history.
2. 18 features built only from past data: returns, volatility, moving averages, RSI, MACD and volume.
3. Gradient Boosting quantile regression predicts the 10th, 50th and 90th percentile of the next day's return, which is then turned into prices.
4. A Streamlit dashboard shows the forecast range, model vs. baseline, and the trading backtest.

The live app reads saved data and a saved model. It doesn't download prices or retrain while you use it.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Get fresh data, retrain, and rerun the trading backtest:

```bash
python -m src.fetch_data
python -m src.train      # also runs python -m src.backtest
```

## Tests

```bash
ruff check .
python -m pytest -q
```

The tests check data validation, that the target really is the next day, that the forecast range stays in order (low ≤ middle ≤ high), the baseline comparison, and the backtest math (fees, drawdown, positions).

## Limits

Crypto prices are noisy, and past results don't predict future ones. This is a learning project, not financial advice.

## License

MIT
