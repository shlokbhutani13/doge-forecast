# DOGE Forecast Lab

DOGE Forecast Lab is a reproducible next-day forecasting study for DOGE-USD. It predicts return quantiles, converts them into price levels, and compares the median model with a persistence baseline on a time-ordered holdout set.

## See it in action

- [Open the live Streamlit demo](https://doge-forecast-hzbcysrfyyxfrqzagu6vu7.streamlit.app/)
- [Project overview in my portfolio](https://shlokbhutani13.github.io/)

## Current result

On the refreshed time-ordered holdout, the committed model has MAE `0.00457` versus `0.00462` for the persistence baseline. The improvement is small, so the dashboard reports both values instead of presenting the forecast as a trading edge. The p10-p90 interval covers 84.6% of holdout observations, with zero quantile crossings after the return-model redesign.

The pipeline also reports RMSE, directional accuracy, interval coverage, and quantile crossings.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## Refresh and retrain

```bash
python -m src.fetch_data
python -m src.train
```

The deployed app reads committed data and model artifacts. It does not download market data or train a model at runtime.

## Tests

```bash
ruff check .
python -m pytest -q
```

Tests cover data validation, next-day target alignment, retention of the newest inference row, quantile ordering, price conversion, and baseline comparison.

## Structure

```text
src/data.py          Data validation
src/features.py      Past-only features and target alignment
src/evaluation.py    Metrics and quantile ordering
src/train.py         Time-ordered training and artifact output
src/artifacts.py     Forecast generation
app.py               Streamlit dashboard
```

## Limitations

Crypto prices are noisy and nonstationary. A historical holdout result does not imply future performance. This project is educational and does not provide financial advice.

## License

MIT
