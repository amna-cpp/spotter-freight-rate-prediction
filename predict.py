"""Load a saved model and fill both submission CSVs without changing input features."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from freight_model import validate_features


def validate_ids(frame: pd.DataFrame, label: str) -> None:
    if (
        "load_id" not in frame
        or frame.load_id.isna().any()
        or not frame.load_id.is_unique
        or frame.load_id.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{label}: load_id must be present and unique")


def predict_files(model, validation, template, december):
    for label, frame in [("validation", validation), ("December", december)]:
        validate_features(frame, label)
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Template columns must be load_id,predicted_rate")
    for label, frame in [("validation", validation), ("template", template)]:
        validate_ids(frame, label)
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Validation and template load_id sets differ")
    rates = pd.Series(np.asarray(model.predict(validation)), index=validation.load_id)
    output = template.copy()
    output["predicted_rate"] = output.load_id.map(rates)
    completed_december = december.copy()
    completed_december["predicted_rate"] = np.asarray(model.predict(december))
    for frame in [output, completed_december]:
        if (
            not np.isfinite(frame.predicted_rate).all()
            or not frame.predicted_rate.gt(0).all()
        ):
            raise ValueError("Model produced invalid/nonpositive predictions")
    return output, completed_december


def write_predictions(model, data_dir: Path, output: Path):
    predictions, december = predict_files(
        model,
        pd.read_csv(data_dir / "validation.csv"),
        pd.read_csv(data_dir / "validation_predictions_template.csv"),
        pd.read_csv(data_dir / "december_chart_inputs.csv"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, float_format="%.2f")
    december.to_csv(
        data_dir / "december_chart_inputs.csv", index=False, float_format="%.2f"
    )
    return predictions, december


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/freight.joblib")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("validation_predictions.csv")
    )
    args = parser.parse_args()
    model = joblib.load(args.model)
    output, december = write_predictions(model, args.data_dir, args.output)
    print(f"Wrote {len(output):,} validation and {len(december)} December predictions.")


if __name__ == "__main__":
    main()
