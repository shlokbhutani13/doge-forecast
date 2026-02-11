import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import make_features


def train_test_split_time(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return mae, rmse, mape


def main():
    raw_path = os.path.join("data", "doge.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError("data/doge.csv not found. Run: python src/fetch_data.py")

    raw = pd.read_csv(raw_path)
    data = make_features(raw, horizon=1)

    drop_cols = {"Date", "target_next_close"}
    feature_cols = [c for c in data.columns if c not in drop_cols]

    train_df, test_df = train_test_split_time(data, test_ratio=0.2)

    X_train = train_df[feature_cols].values
    y_train = train_df["target_next_close"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_next_close"].values

    baseline_pred = test_df["Close"].values
    b_mae, b_rmse, b_mape = metrics(y_test, baseline_pred)

    def fit_quantile(alpha: float):
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    model_p10 = fit_quantile(0.10)
    model_p50 = fit_quantile(0.50)
    model_p90 = fit_quantile(0.90)

    p10 = model_p10.predict(X_test)
    p50 = model_p50.predict(X_test)
    p90 = model_p90.predict(X_test)

    m_mae, m_rmse, m_mape = metrics(y_test, p50)

    print("\n=== RESULTS (Test Set) ===")
    print(f"Baseline MAE: {b_mae:.6f}")
    print(f"Model MAE:    {m_mae:.6f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "model_p10": model_p10,
            "model_p50": model_p50,
            "model_p90": model_p90,
        },
        "models/doge_models.joblib",
    )

    print("Models saved to models/doge_models.joblib")

        # Save backtest predictions for the Streamlit backtest chart
    os.makedirs("data", exist_ok=True)
    out = test_df[["Date", "Close", "target_next_close"]].copy()
    out["pred_p10"] = p10
    out["pred_p50"] = p50
    out["pred_p90"] = p90
    out.to_csv(os.path.join("data", "backtest_preds.csv"), index=False)
    print("Backtest saved to data/backtest_preds.csv")



if __name__ == "__main__":
    main()
