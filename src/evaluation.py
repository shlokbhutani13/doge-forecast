import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def order_quantiles(p10, p50, p90):
    values = np.column_stack([p10, p50, p90])
    crossings = int(np.sum((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])))
    ordered = np.sort(values, axis=1)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2], crossings


def returns_to_prices(current_close, returns):
    return np.asarray(current_close) * (1 + np.asarray(returns))


def calculate_metrics(*, actual, model, baseline, low, high, previous, crossings):
    actual = np.asarray(actual)
    model = np.asarray(model)
    baseline = np.asarray(baseline)
    previous = np.asarray(previous)
    model_mae = float(mean_absolute_error(actual, model))
    baseline_mae = float(mean_absolute_error(actual, baseline))
    return {
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "model_rmse": float(np.sqrt(mean_squared_error(actual, model))),
        "baseline_rmse": float(np.sqrt(mean_squared_error(actual, baseline))),
        "directional_accuracy": float(
            np.mean(np.sign(model - previous) == np.sign(actual - previous))
        ),
        "interval_coverage": float(np.mean((actual >= low) & (actual <= high))),
        "quantile_crossings": int(crossings),
        "winner": "model" if model_mae < baseline_mae else "baseline",
    }
