"""Train-only preprocessing and models for load prices in dollars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CORE = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
ITERATIONS = 1600
SEED = 42


def validate_features(frame: pd.DataFrame, label: str = "loads") -> None:
    missing = set(CORE) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label}: no rows supplied")
    for column in ["pickup", "delivery", "equipment"]:
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{label}: {column} must be present")
    distance = pd.to_numeric(frame.distance, errors="coerce")
    if not np.isfinite(distance).all() or not distance.gt(0).all():
        raise ValueError(f"{label}: distance must be finite and positive")
    if pd.to_datetime(frame.date, format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError(f"{label}: date must be valid")


class RateFeatures:
    def fit(self, frame: pd.DataFrame) -> RateFeatures:
        validate_features(frame, "training")
        valid = pd.to_numeric(frame.weight, errors="coerce").where(
            lambda x: x.between(1, 80000)
        )
        self.weight_median = float(valid.median())
        self.equipment_medians = valid.groupby(frame.equipment).median().to_dict()
        self.cities = {}
        for side in ["pickup", "delivery"]:
            for city, group in frame.groupby(side):
                self.cities[city] = (
                    float(group[f"{side}_lat"].median()),
                    float(group[f"{side}_lon"].median()),
                )
        return self

    def transform(self, frame: pd.DataFrame, mode: str = "core") -> pd.DataFrame:
        validate_features(frame)
        x = frame[CORE].copy()
        x["distance"] = pd.to_numeric(x.distance)
        date = pd.to_datetime(x.pop("date"), errors="raise")
        x["lane"] = x.pickup + " -> " + x.delivery
        weight = pd.to_numeric(x.weight, errors="coerce")
        x["weight_invalid"] = (~weight.between(1, 80000)).astype(int)
        x["weight"] = (
            weight.where(weight.between(1, 80000))
            .fillna(x.equipment.map(self.equipment_medians))
            .fillna(self.weight_median)
        )
        x["weekday"] = date.dt.dayofweek.astype(str)
        x["dayofyear"] = date.dt.dayofyear
        x["month"] = date.dt.month
        x["dayofmonth"] = date.dt.day
        x["weekend"] = date.dt.dayofweek.ge(5).astype(int)
        for k in [1, 2]:
            angle = 2 * np.pi * k * (date.dt.dayofyear - 1) / 365.25
            x[f"annual_sin_{k}"] = np.sin(angle)
            x[f"annual_cos_{k}"] = np.cos(angle)
        for side in ["pickup", "delivery"]:
            x[f"{side}_lat"] = x[side].map(
                {city: point[0] for city, point in self.cities.items()}
            )
            x[f"{side}_lon"] = x[side].map(
                {city: point[1] for city, point in self.cities.items()}
            )
        x["lat_delta"] = x.delivery_lat - x.pickup_lat
        x["lon_delta"] = x.delivery_lon - x.pickup_lon
        x["log_distance"] = np.log(x.distance.clip(lower=1))
        x["inverse_distance"] = 1000 / x.distance.clip(lower=1)
        if mode in ["market", "quote"]:
            x["market_index"] = frame["market_index"]
        if mode == "quote":
            x["quote_signal"] = frame["quote_signal"]
        for column in x.select_dtypes("number"):
            x[column] = x[column].replace([np.inf, -np.inf], np.nan).fillna(-999)
        return x


class TreeRateModel:
    def __init__(self, mode="core", iterations=ITERATIONS, parameters=None):
        if mode not in {"core", "market", "quote"}:
            raise ValueError(f"Unknown feature mode: {mode}")
        self.mode, self.iterations = mode, iterations
        self.parameters = parameters or {}

    def fit(self, frame: pd.DataFrame) -> TreeRateModel:
        self.features = RateFeatures().fit(frame)
        x = self.features.transform(frame, self.mode)
        settings = {
            "iterations": self.iterations,
            "depth": 6,
            "learning_rate": 0.045,
            "loss_function": "MAE",
            "l2_leaf_reg": 8,
            "random_seed": SEED,
            "thread_count": 4,
            "verbose": False,
            "allow_writing_files": False,
        }
        self.model = CatBoostRegressor(**(settings | self.parameters))
        # Distance weights align rate-per-mile loss with dollar MAE.
        self.model.fit(
            x,
            frame.posted_rate / frame.distance,
            sample_weight=frame.distance / frame.distance.mean(),
            cat_features=list(x.select_dtypes("object").columns),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.features.transform(frame, self.mode)
        return np.maximum(
            1.0, self.model.predict(features) * features.distance.to_numpy()
        )


class AdditiveRateModel:
    """Regularized Huber regression on rate/mile with periodic calendar features."""

    def __init__(self, harmonics=1):
        self.harmonics = harmonics

    def _raw(self, frame):
        x = self.features.transform(frame)
        categories = x[["pickup", "delivery", "equipment", "weekday"]].copy()
        categories["pickup_equipment"] = x.pickup + ":" + x.equipment
        categories["delivery_equipment"] = x.delivery + ":" + x.equipment
        numeric = pd.DataFrame(
            {
                "distance": x.distance / 1000,
                "log_distance": np.log(x.distance / 1000),
                "inverse_distance": 1000 / x.distance,
                "weight": (x.weight - 32000) / 10000,
                "weight_invalid": x.weight_invalid,
            }
        )
        for k in range(1, self.harmonics + 1):
            for trig in ["sin", "cos"]:
                numeric[f"annual_{trig}_{k}"] = x[f"annual_{trig}_{k}"]
        return categories, numeric

    def _matrix(self, frame, fit=False):
        categories, numeric = self._raw(frame)
        if fit:
            self.encoder = OneHotEncoder(handle_unknown="ignore", dtype=float)
            self.scaler = StandardScaler()
            return sparse.hstack(
                [
                    self.encoder.fit_transform(categories),
                    self.scaler.fit_transform(numeric),
                ],
                format="csr",
            )
        return sparse.hstack(
            [self.encoder.transform(categories), self.scaler.transform(numeric)],
            format="csr",
        )

    def fit(self, frame):
        self.features = RateFeatures().fit(frame)
        x = self._matrix(frame, fit=True)
        y = (frame.posted_rate / frame.distance).to_numpy()
        base_weights = frame.distance.to_numpy() / frame.distance.mean()
        weights = base_weights.copy()
        # Downweight large residuals without deleting rows.
        for _ in range(20):
            self.model = Ridge(alpha=10, solver="lsqr", tol=1e-7)
            self.model.fit(x, y, sample_weight=weights)
            error = np.abs(y - self.model.predict(x))
            weights = base_weights * np.minimum(1.0, 0.08 / np.maximum(error, 1e-8))
        return self

    def predict(self, frame):
        return np.maximum(1.0, self.model.predict(self._matrix(frame)) * frame.distance)


class ResidualRateModel(TreeRateModel):
    """Fit smooth pricing effects first, then learn their remaining errors."""

    def __init__(self, depth=4):
        super().__init__(iterations=2400, parameters={"depth": depth})

    def fit(self, frame):
        self.baseline = AdditiveRateModel(harmonics=0).fit(frame)
        centered = frame.assign(
            posted_rate=frame.posted_rate - self.baseline.predict(frame)
        )
        super().fit(centered)
        return self

    def predict(self, frame):
        features = self.features.transform(frame)
        correction = self.model.predict(features) * features.distance.to_numpy()
        return np.maximum(1.0, self.baseline.predict(frame) + correction)
