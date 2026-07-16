import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.evaluation import calculate_metrics, order_quantiles, returns_to_prices
from src.features import FEATURE_COLUMNS, build_training_frame


@dataclass
class TrainingResult:
    artifact: dict
    backtest: pd.DataFrame
    metrics: dict


def train_models(raw: pd.DataFrame, test_ratio=0.2, random_state=42):
    data = build_training_frame(raw)
    split = int(len(data) * (1 - test_ratio))
    train, test = data.iloc[:split], data.iloc[split:]
    if len(train) < 50 or len(test) < 10:
        raise ValueError("Insufficient data for time-ordered training.")
    models = {}
    predictions = {}
    for name, alpha in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            random_state=random_state,
        )
        model.fit(train[FEATURE_COLUMNS], train["target_return_1"])
        models[name] = model
        predictions[name] = model.predict(test[FEATURE_COLUMNS])
    low_ret, mid_ret, high_ret, crossings = order_quantiles(
        predictions["p10"], predictions["p50"], predictions["p90"]
    )
    current = test["Close"].to_numpy()
    actual = current * (1 + test["target_return_1"].to_numpy())
    low, model_price, high = (
        returns_to_prices(current, low_ret),
        returns_to_prices(current, mid_ret),
        returns_to_prices(current, high_ret),
    )
    metrics = calculate_metrics(
        actual=actual,
        model=model_price,
        baseline=current,
        low=low,
        high=high,
        previous=current,
        crossings=crossings,
    )
    backtest = pd.DataFrame(
        {
            "Date": test["Date"],
            "current_close": current,
            "actual_next_close": actual,
            "baseline": current,
            "pred_p10": low,
            "pred_p50": model_price,
            "pred_p90": high,
        }
    )
    artifact = {
        "artifact_version": 2,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_end_date": str(pd.to_datetime(raw["Date"]).max().date()),
        "feature_cols": FEATURE_COLUMNS,
        "model_p10": models["p10"],
        "model_p50": models["p50"],
        "model_p90": models["p90"],
        "metrics": metrics,
    }
    return TrainingResult(artifact, backtest, metrics)


def main():
    raw = pd.read_csv("data/doge.csv")
    result = train_models(raw)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(result.artifact, "models/doge_models.joblib")
    result.backtest.to_csv("data/backtest_preds.csv", index=False)
    Path("data/metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
