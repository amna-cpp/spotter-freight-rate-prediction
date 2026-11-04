"""Export the fitted model and public summaries for the Vercel demo."""

import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import Pool

ROOT = Path(__file__).resolve().parent


def export_baseline(baseline, training):
    categories, numeric = baseline._raw(training)
    coefficients = baseline.model.coef_
    result = {"categories": {}}
    offset = 0
    for name, levels in zip(categories, baseline.encoder.categories_, strict=True):
        result["categories"][name] = dict(
            zip(levels, coefficients[offset : offset + len(levels)], strict=True)
        )
        offset += len(levels)
    weights = coefficients[offset:] / baseline.scaler.scale_
    result["numeric"] = dict(zip(numeric, weights, strict=True))
    result["intercept"] = float(
        baseline.model.intercept_ - baseline.scaler.mean_ @ weights
    )
    return result


def main():
    model = joblib.load(ROOT / "models/freight.joblib")
    training = pd.read_csv(ROOT / "data/train_test.csv")
    features = model.features.transform(training)
    output = ROOT / "demo/model"
    output.mkdir(parents=True, exist_ok=True)
    categorical = list(features.select_dtypes("object").columns)
    pool = Pool(features, cat_features=categorical)
    model.model.save_model(str(output / "exported.py"), format="python", pool=pool)
    metadata = {
        "numeric_features": list(features.select_dtypes("number").columns),
        "categorical_features": categorical,
        "cities": model.features.cities,
        "equipment_medians": model.features.equipment_medians,
        "weight_median": model.features.weight_median,
    }
    metadata["float_feature_borders"] = [
        model.model.get_borders()[model.model.feature_names_.index(name)]
        for name in metadata["numeric_features"]
    ]
    if hasattr(model, "baseline"):
        metadata["baseline"] = export_baseline(model.baseline, training)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = []
    base = {
        "pickup": "Lexington",
        "delivery": "Fort Wayne",
        "distance": 360,
        "equipment": "Dry Van",
        "weight": 32000,
        "date": "2025-12-04",
    }
    for name in ["distance", "weight"]:
        borders = model.model.get_borders()[model.model.feature_names_.index(name)]
        for border in borders[len(borders) // 2 : len(borders) // 2 + 2]:
            for delta in [-0.000001, 0.000001]:
                load = {**base, name: border + delta}
                expected = round(float(model.predict(pd.DataFrame([load]))[0]), 2)
                cases.append({"load": load, "predicted_rate": expected})
    (output / "parity_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    summaries = {
        "cities": sorted(model.features.cities),
        "metrics": json.loads((ROOT / "results/metrics.json").read_text()),
        "december": pd.read_csv(ROOT / "data/december_chart_inputs.csv")[
            ["date", "predicted_rate"]
        ].to_dict("records"),
    }
    public = ROOT / "demo/public"
    public.mkdir(exist_ok=True)
    (public / "data.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("Exported the fitted model and aggregate summaries; no raw rows copied.")


if __name__ == "__main__":
    main()
