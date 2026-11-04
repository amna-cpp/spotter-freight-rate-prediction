"""Compare a small shortlist on two pre-October forecast windows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import metrics, split_before
from freight_model import ResidualRateModel, TreeRateModel

CANDIDATES = {
    "original": lambda: TreeRateModel(),
    "shallow": lambda: TreeRateModel(
        iterations=2400, parameters={"depth": 4, "learning_rate": 0.06}
    ),
    "medium": lambda: TreeRateModel(
        iterations=2400, parameters={"depth": 5, "learning_rate": 0.06}
    ),
    "residual_4": lambda: ResidualRateModel(depth=4),
    "residual_6": lambda: ResidualRateModel(depth=6),
}
WINDOWS = [("2025-06-01", "2025-08-01"), ("2025-08-01", "2025-10-01")]


def main():
    data = pd.read_csv("data/train_test.csv")
    data, _ = split_before(data, "2025-10-01")
    rows = []
    for cutoff, end in WINDOWS:
        train, check = split_before(data[data.date < end], cutoff)
        for name, factory in CANDIDATES.items():
            model = factory().fit(train)
            predictions = np.asarray(model.predict(check))
            row = {
                "model": name,
                "train_end": train.date.max(),
                "check_start": check.date.min(),
                "check_end": check.date.max(),
                "n": len(check),
                **metrics(check.posted_rate, predictions),
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
    output = Path("results")
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "tuning_comparison.csv", index=False)
    scores = table.groupby("model").mae.mean().sort_values()
    selected = {
        "criterion": "lowest mean MAE across June-July and August-September",
        "model": scores.index[0],
        "mean_window_mae": scores.to_dict(),
        "selection_latest_date": data.date.max(),
        "october_used_for_selection": False,
    }
    (output / "model_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
