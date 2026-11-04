"""Evaluate the fixed model, refit all labels, and write submission predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluation import (
    audit,
    day_bootstrap_interval,
    error_slices,
    metrics,
    paired_mae_interval,
    split_before,
)
from freight_model import SEED, ResidualRateModel, TreeRateModel, validate_features
from predict import validate_ids, write_predictions


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = pd.read_csv(data_dir / "train_test.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    for label, frame in [("training", labeled), ("validation", validation)]:
        validate_features(frame, label)
        validate_ids(frame, label)
        frame["distance"] = pd.to_numeric(frame.distance)
    if not set(labeled.load_id).isdisjoint(validation.load_id):
        raise ValueError("Training and validation IDs overlap")
    if "posted_rate" not in labeled:
        raise ValueError("Training data must contain posted_rate")
    target = pd.to_numeric(labeled.posted_rate, errors="coerce")
    if not np.isfinite(target).all() or not target.gt(0).all():
        raise ValueError("Training target must be finite and positive")
    labeled["posted_rate"] = target
    if pd.to_datetime(labeled.date).max() >= pd.to_datetime(validation.date).min():
        raise ValueError("Validation dates must follow all training dates")
    return labeled, validation


def evaluate_holdout(train, holdout, output_dir: Path) -> dict:
    model = ResidualRateModel().fit(train)
    predictions = model.predict(holdout)
    original = TreeRateModel().fit(train).predict(holdout)
    medians = (train.posted_rate / train.distance).groupby(train.equipment).median()
    baseline = holdout.distance * holdout.equipment.map(medians)
    residuals = holdout[
        ["load_id", "date", "equipment", "distance", "posted_rate"]
    ].copy()
    residuals["predicted_rate"] = predictions
    residuals["baseline_rate"] = np.asarray(baseline)
    residuals["original_model_rate"] = original
    residuals["absolute_error"] = np.abs(residuals.posted_rate - predictions)
    residuals.to_csv(output_dir / "october_holdout_predictions.csv", index=False)
    error_slices(residuals).to_csv(output_dir / "october_slices.csv", index=False)

    squared = np.sort((residuals.posted_rate - predictions).to_numpy() ** 2)
    largest = max(1, int(np.ceil(0.01 * len(squared))))
    tail_share = (
        float(squared[-largest:].sum() / squared.sum()) if squared.sum() else 0.0
    )
    return {
        "selection": {
            "criterion": "lowest mean MAE across June-July and August-September",
            "model": "residual_4",
            "iterations": model.iterations,
            "depth": model.parameters["depth"],
            "seed": SEED,
        },
        "split": {
            "fit_start": train.date.min(),
            "fit_end": train.date.max(),
            "fit_n": len(train),
            "holdout_start": holdout.date.min(),
            "holdout_end": holdout.date.max(),
            "holdout_n": len(holdout),
        },
        "october": {
            **metrics(holdout.posted_rate, predictions),
            "median_absolute_error": float(residuals.absolute_error.median()),
            "p95_absolute_error": float(residuals.absolute_error.quantile(0.95)),
            "largest_1pct_errors_share_squared_error": tail_share,
        },
        "october_baseline": metrics(holdout.posted_rate, baseline),
        "revision": {
            "original_october": metrics(holdout.posted_rate, original),
            "mae_reduction_pct": float(
                100
                * (
                    1
                    - np.abs(holdout.posted_rate - predictions).mean()
                    / np.abs(holdout.posted_rate - original).mean()
                )
            ),
            "paired_mae_improvement_day_bootstrap_95pct": paired_mae_interval(
                holdout, original, predictions
            ),
            "october_previously_inspected": True,
        },
        "october_mae_day_bootstrap_95pct": day_bootstrap_interval(holdout, predictions),
    }


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    output_dir, model_dir = Path("results"), Path("models")
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    data, validation = load_inputs(args.data_dir)
    train, holdout = split_before(data, "2025-10-01")

    print("Evaluating the fixed model on October...", flush=True)
    summary = evaluate_holdout(train, holdout, output_dir)
    print(f"October MAE: ${summary['october']['mae']:.2f}", flush=True)

    print("Checking the two-month forecast horizon...", flush=True)
    earlier, _ = split_before(train, "2025-09-01")
    stress = ResidualRateModel().fit(earlier)
    summary["october_two_month_horizon"] = metrics(
        holdout.posted_rate, stress.predict(holdout)
    )

    print(f"Refitting on all {len(data):,} labeled loads...", flush=True)
    model = ResidualRateModel().fit(data)
    joblib.dump(model, model_dir / "freight.joblib")
    importance = pd.Series(
        model.model.feature_importances_, index=model.model.feature_names_
    )
    importance.sort_values(ascending=False).rename_axis("feature").reset_index(
        name="importance"
    ).to_csv(
        output_dir / "feature_importance.csv",
        index=False,
    )
    predictions, december = write_predictions(
        model, args.data_dir, Path("validation_predictions.csv")
    )
    summary["final_fit_rows"] = len(data)
    summary["output_rows"] = {"validation": len(predictions), "december": len(december)}
    summary["december_range"] = [
        float(december.predicted_rate.min()),
        float(december.predicted_rate.max()),
    ]
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "data_audit.json", audit(data, validation))
    inputs = [
        args.data_dir / name
        for name in [
            "train_test.csv",
            "validation.csv",
            "validation_predictions_template.csv",
        ]
    ]
    save_json(
        output_dir / "input_sha256.json",
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*inputs, Path("score.py")]
        },
    )
    print("Saved model, diagnostics and both prediction files.", flush=True)


if __name__ == "__main__":
    main()
