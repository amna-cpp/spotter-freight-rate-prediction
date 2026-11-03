"""Check the lightweight demo against the submitted December predictions."""

import csv
import json
import math
import unittest
from pathlib import Path

from demo.model.inference import METADATA, estimate

ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.load = {
            "pickup": "Lexington",
            "delivery": "Fort Wayne",
            "distance": 360,
            "equipment": "Dry Van",
            "weight": 32000,
            "date": "2025-12-04",
        }

    def test_every_december_prediction_matches_submission(self):
        with (ROOT / "data/december_chart_inputs.csv").open() as source:
            for row in csv.DictReader(source):
                with self.subTest(date=row["date"]):
                    self.assertEqual(
                        estimate(row)["predicted_rate"], float(row["predicted_rate"])
                    )

    def test_blank_weight_uses_the_training_equipment_median(self):
        result = estimate({**self.load, "weight": ""})
        self.assertTrue(result["weight_imputed"])
        self.assertEqual(
            result["weight_used"], METADATA["equipment_medians"]["Dry Van"]
        )
        self.assertTrue(math.isfinite(result["predicted_rate"]))

    def test_numeric_edges_match_native_predictions(self):
        cases = json.loads((ROOT / "demo/model/parity_cases.json").read_text())
        for case in cases:
            with self.subTest(load=case["load"]):
                self.assertEqual(
                    estimate(case["load"])["predicted_rate"], case["predicted_rate"]
                )

    def test_invalid_loads_are_rejected(self):
        cases = [
            {"distance": 0},
            {"distance": math.inf},
            {"pickup": "Unknown city"},
            {"equipment": "Unknown equipment"},
            {"date": "2026-01-01"},
        ]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                estimate({**self.load, **change})


if __name__ == "__main__":
    unittest.main()
