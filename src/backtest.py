"""Run the trading backtest on the saved test-set forecasts.

    python -m src.backtest
"""

import json
from pathlib import Path

import pandas as pd

from src.strategy import run_backtest

FEE_LEVELS_BPS = (0.0, 10.0, 25.0)


def main(
    predictions_path: str = "data/backtest_preds.csv",
    output_path: str = "data/strategy.json",
) -> dict:
    backtest = pd.read_csv(predictions_path)
    results = {f"{fee:g}bps": run_backtest(backtest, fee) for fee in FEE_LEVELS_BPS}
    Path(output_path).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["10bps"], indent=2))
    return results


if __name__ == "__main__":
    main()
