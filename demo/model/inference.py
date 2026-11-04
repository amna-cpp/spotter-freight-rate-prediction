"""Apply the exported model using only the Python standard library."""

import json
import math
import struct
from datetime import date
from pathlib import Path

from .exported import apply_catboost_model, catboost_model

METADATA = json.loads(Path(__file__).with_name("metadata.json").read_text())
# The code export rounds thresholds; keep the native split precision.
catboost_model.float_feature_borders = METADATA["float_feature_borders"]


def baseline_rate(features, distance):
    model = METADATA.get("baseline")
    if model is None:
        return 0.0
    categories = {
        **features,
        "pickup_equipment": features["pickup"] + ":" + features["equipment"],
        "delivery_equipment": features["delivery"] + ":" + features["equipment"],
    }
    numeric = {
        "distance": distance / 1000,
        "log_distance": math.log(distance / 1000),
        "inverse_distance": 1000 / distance,
        "weight": (features["weight"] - 32000) / 10000,
        "weight_invalid": features["weight_invalid"],
    }
    value = model["intercept"]
    value += sum(
        weights.get(categories[name], 0)
        for name, weights in model["categories"].items()
    )
    value += sum(weight * numeric[name] for name, weight in model["numeric"].items())
    return max(1.0, value * distance)


def estimate(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Send one load as a JSON object.")
    pickup = payload.get("pickup")
    delivery = payload.get("delivery")
    equipment = payload.get("equipment")
    for city in [pickup, delivery]:
        if not isinstance(city, str) or city not in METADATA["cities"]:
            raise ValueError("Choose a pickup and delivery from the city list.")
    if equipment not in METADATA["equipment_medians"]:
        raise ValueError("Choose Dry Van, Reefer, or Flatbed.")
    try:
        distance = float(payload.get("distance"))
        pickup_date = date.fromisoformat(payload.get("date", ""))
    except (ValueError, TypeError):
        raise ValueError("Enter a valid distance and pickup date.") from None
    if not math.isfinite(distance) or not 70 <= distance <= 3500:
        raise ValueError("Enter a road distance between 70 and 3,500 miles.")
    if not date(2025, 11, 1) <= pickup_date <= date(2025, 12, 31):
        raise ValueError("This study covers pickup dates in November–December 2025.")
    try:
        weight = float(payload.get("weight"))
    except (ValueError, TypeError):
        weight = math.nan
    invalid_weight = not math.isfinite(weight) or not 1 <= weight <= 80000
    if invalid_weight:
        weight = METADATA["equipment_medians"].get(equipment, METADATA["weight_median"])
    pickup_lat, pickup_lon = METADATA["cities"][pickup]
    delivery_lat, delivery_lon = METADATA["cities"][delivery]
    day = pickup_date.timetuple().tm_yday
    features = {
        "pickup": pickup,
        "delivery": delivery,
        "equipment": equipment,
        "distance": distance,
        "weight": weight,
        "weight_invalid": int(invalid_weight),
        "lane": f"{pickup} -> {delivery}",
        "weekday": str(pickup_date.weekday()),
        "dayofyear": day,
        "month": pickup_date.month,
        "dayofmonth": pickup_date.day,
        "weekend": int(pickup_date.weekday() >= 5),
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "delivery_lat": delivery_lat,
        "delivery_lon": delivery_lon,
        "lat_delta": delivery_lat - pickup_lat,
        "lon_delta": delivery_lon - pickup_lon,
        "log_distance": math.log(distance),
        "inverse_distance": 1000 / distance,
    }
    for harmonic in [1, 2]:
        angle = 2 * math.pi * harmonic * (day - 1) / 365.25
        features[f"annual_sin_{harmonic}"] = math.sin(angle)
        features[f"annual_cos_{harmonic}"] = math.cos(angle)
    rate_per_mile = apply_catboost_model(
        # Match CatBoost's float32 input precision.
        [
            struct.unpack("f", struct.pack("f", features[name]))[0]
            for name in METADATA["numeric_features"]
        ],
        [features[name] for name in METADATA["categorical_features"]],
    )
    rate = max(1.0, baseline_rate(features, distance) + rate_per_mile * distance)
    return {
        "predicted_rate": round(rate, 2),
        "rate_per_mile": round(rate / distance, 2),
        "weight_used": weight,
        "weight_imputed": invalid_weight,
    }
