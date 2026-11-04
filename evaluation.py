"""Metrics and diagnostics for chronological validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    relative_error = np.abs(actual - predicted) / actual
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_pct": float(np.mean(relative_error) * 100),
        "within_10pct_pct": float(np.mean(relative_error <= 0.10) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def split_before(frame: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame.date, errors="raise")
    if dates.isna().any():
        raise ValueError("Split dates must be present")
    earlier = dates < pd.Timestamp(cutoff)
    if earlier.all() or not earlier.any():
        raise ValueError(
            f"Cutoff {cutoff} must leave nonempty training and evaluation sets"
        )
    return frame.loc[earlier].copy(), frame.loc[~earlier].copy()


def day_bootstrap_interval(frame: pd.DataFrame, predictions) -> list[float]:
    return _day_interval(frame, np.abs(frame.posted_rate.to_numpy() - predictions))


def paired_mae_interval(frame, original, revised) -> list[float]:
    actual = frame.posted_rate.to_numpy()
    improvement = np.abs(actual - original) - np.abs(actual - revised)
    return _day_interval(frame, improvement)


def _day_interval(frame, values):
    errors = pd.DataFrame({"date": frame.date.to_numpy(), "error": values})
    by_day = errors.groupby("date").error.agg(["sum", "count"])
    indices = np.random.default_rng(42).integers(
        0, len(by_day), size=(2000, len(by_day))
    )
    values = by_day["sum"].to_numpy()[indices].sum(axis=1) / by_day["count"].to_numpy()[
        indices
    ].sum(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()


def error_slices(residuals: pd.DataFrame) -> pd.DataFrame:
    groups = [
        (f"equipment={name}", group) for name, group in residuals.groupby("equipment")
    ]
    distance = residuals.distance
    groups += [
        ("distance<=500", residuals[distance.le(500)]),
        ("500<distance<=1500", residuals[distance.gt(500) & distance.le(1500)]),
        ("distance>1500", residuals[distance.gt(1500)]),
    ]
    return pd.DataFrame(
        [
            {
                "slice": label,
                "n": len(group),
                **metrics(group.posted_rate, group.predicted_rate),
            }
            for label, group in groups
            if not group.empty
        ]
    )


def audit(data: pd.DataFrame, validation: pd.DataFrame) -> dict:
    result = {}
    for name, frame in [("labeled", data), ("validation", validation)]:
        result[name] = {
            "rows": len(frame),
            "columns": list(frame),
            "date_min": frame.date.min(),
            "date_max": frame.date.max(),
            "missing": {
                column: int(count) for column, count in frame.isna().sum().items()
            },
            "negative_weights": int(frame.weight.lt(0).sum()),
            "duplicate_ids": int(frame.load_id.duplicated().sum()),
            "duplicate_rows_without_id": int(
                frame.drop(columns="load_id").duplicated().sum()
            ),
        }
    result["labeled"]["nonpositive_target"] = int(data.posted_rate.le(0).sum())
    result["labeled"]["target_quantiles"] = {
        str(q): float(value)
        for q, value in data.posted_rate.quantile([0, 0.01, 0.5, 0.99, 1]).items()
    }
    return result
