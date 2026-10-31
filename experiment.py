"""Development comparison only: January-July -> August-September."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import metrics, split_before
from freight_model import AdditiveRateModel, TreeRateModel

if __name__ == "__main__":
    data = pd.read_csv("data/train_test.csv")
    selection_data, _ = split_before(data, "2025-10-01")
    train, development = split_before(selection_data, "2025-08-01")
    results = []
    candidates = {
        "distance_equipment_median": None,
        "quote_times_distance": "quote",
        "additive_1": AdditiveRateModel(1),
        "additive_2": AdditiveRateModel(2),
        "tree_core": TreeRateModel("core"),
        "tree_market": TreeRateModel("market"),
        "tree_quote": TreeRateModel("quote"),
    }
    for name, model in candidates.items():
        print("Fitting", name, flush=True)
        if model is None:
            medians = (
                (train.posted_rate / train.distance).groupby(train.equipment).median()
            )
            predictions = development.distance * development.equipment.map(medians)
        elif isinstance(model, str):
            predictions = development.distance * development.quote_signal
        else:
            model.fit(train)
            predictions = model.predict(development)
        predictions = np.asarray(predictions)
        for month in ["all", "2025-08", "2025-09"]:
            mask = (
                np.ones(len(development), dtype=bool)
                if month == "all"
                else development.date.str.startswith(month).to_numpy()
            )
            row = {
                "model": name,
                "period": month,
                "n": int(mask.sum()),
                **metrics(development.posted_rate.to_numpy()[mask], predictions[mask]),
            }
            print(json.dumps(row), flush=True)
            results.append(row)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("results/development_comparison.csv", index=False)
