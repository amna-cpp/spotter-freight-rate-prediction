"""Small fixtures check leakage guards and submission contracts."""

import unittest

import numpy as np
import pandas as pd

from evaluation import split_before
from freight_model import CORE, RateFeatures, validate_features
from predict import predict_files
from score import validate_predictions


class DistanceModel:
    def predict(self, frame):
        return frame.distance.to_numpy() * 2


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "pickup": ["A", "B", "A", "B"],
                "delivery": ["B", "A", "B", "A"],
                "distance": [100.0, 200.0, 300.0, 400.0],
                "equipment": ["Dry Van", "Dry Van", "Reefer", "Reefer"],
                "weight": [20000.0, 30000.0, 40000.0, 44000.0],
                "date": ["2025-07-30", "2025-07-31", "2025-08-01", "2025-08-01"],
                "pickup_lat": [30.0, 40.0, 30.0, 40.0],
                "pickup_lon": [-80.0, -90.0, -80.0, -90.0],
                "delivery_lat": [40.0, 30.0, 40.0, 30.0],
                "delivery_lon": [-90.0, -80.0, -90.0, -80.0],
            }
        )
        self.validation = self.train.loc[:2, CORE].assign(
            load_id=["TE-000001", "TE-000002", "TE-000003"],
            date="2025-11-01",
        )
        self.template = self.validation[["load_id"]].assign(predicted_rate=np.nan)
        self.december = pd.DataFrame(
            {
                "pickup": "Lexington",
                "delivery": "Fort Wayne",
                "distance": 360,
                "equipment": "Dry Van",
                "weight": 32000,
                "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"),
                "predicted_rate": np.nan,
            }
        )

    def test_template_joins_by_id_and_preserves_december_inputs(self):
        template = self.template.iloc[::-1]
        output, december = predict_files(
            DistanceModel(), self.validation, template, self.december
        )
        self.assertEqual(output.load_id.tolist(), template.load_id.tolist())
        np.testing.assert_allclose(
            output.predicted_rate, self.validation.distance.iloc[::-1] * 2
        )
        pd.testing.assert_frame_equal(december[CORE], self.december[CORE])

    def test_duplicate_and_mismatched_ids_fail(self):
        for replacement in ["TE-000001", "TE-999999"]:
            template = self.template.copy()
            template.loc[1, "load_id"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                predict_files(DistanceModel(), self.validation, template, self.december)

    def test_preprocessing_ignores_auxiliary_signals_and_labels(self):
        features = RateFeatures().fit(self.train)
        changed = self.validation.assign(
            quote_signal=1e9, market_index=-1e9, posted_rate=-1e9
        )
        pd.testing.assert_frame_equal(
            features.transform(self.validation), features.transform(changed)
        )

    def test_invalid_weights_use_training_medians(self):
        features = RateFeatures().fit(self.train)
        invalid = self.validation.copy()
        invalid["weight"] = [np.nan, -30000.0, np.inf]
        transformed = features.transform(invalid)
        np.testing.assert_allclose(transformed.weight, [25000.0, 25000.0, 42000.0])
        self.assertTrue(transformed.weight_invalid.eq(1).all())

    def test_unknown_equipment_uses_global_weight_median(self):
        features = RateFeatures().fit(self.train)
        unseen = self.validation.assign(equipment="New equipment", weight=np.nan)
        np.testing.assert_allclose(features.transform(unseen).weight, 35000.0)

    def test_invalid_distances_and_dates_fail(self):
        for distance in [0.0, -1.0, np.nan, np.inf]:
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                validate_features(self.validation.assign(distance=distance))
        with self.assertRaises(ValueError):
            validate_features(self.validation.assign(date="invalid"))

    def test_nonfinite_model_predictions_fail(self):
        class InvalidModel:
            def predict(self, frame):
                return np.full(len(frame), np.inf)

        with self.assertRaises(ValueError):
            predict_files(InvalidModel(), self.validation, self.template, self.december)

    def test_time_split_keeps_dates_together(self):
        train, holdout = split_before(
            self.train.sample(frac=1, random_state=42), "2025-08-01"
        )
        self.assertEqual((len(train), len(holdout)), (2, 2))
        self.assertLess(train.date.max(), holdout.date.min())
        self.assertTrue(set(train.date).isdisjoint(holdout.date))

    def test_empty_time_split_fails(self):
        with self.assertRaises(ValueError):
            split_before(self.train, "2026-01-01")

    def test_original_scorer_rejects_incomplete_output(self):
        with self.assertRaises(SystemExit):
            validate_predictions(
                pd.DataFrame({"load_id": ["TE-000001"], "predicted_rate": [100.0]})
            )


if __name__ == "__main__":
    unittest.main()
