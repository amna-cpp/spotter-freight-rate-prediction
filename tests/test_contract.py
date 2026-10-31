"""Checks for prediction alignment, signal independence and fail-closed output handling."""
import unittest
import numpy as np
import pandas as pd
from freight_model import RateFeatures
from predict import predict_files
from score import validate_predictions


class DistanceModel:
    def predict(self, frame):
        return frame.distance.to_numpy() * 2


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv('data/train_test.csv').head(1000)
        cls.validation = pd.read_csv('data/validation.csv').head(3)
        cls.december = pd.read_csv('data/december_chart_inputs.csv')

    def test_template_is_joined_by_id_not_position(self):
        template = self.validation[['load_id']].iloc[::-1].assign(predicted_rate=np.nan)
        output, december = predict_files(DistanceModel(), self.validation, template, self.december)
        self.assertEqual(output.load_id.tolist(), template.load_id.tolist())
        np.testing.assert_allclose(output.predicted_rate, self.validation.distance.iloc[::-1]*2)
        pd.testing.assert_frame_equal(december.drop(columns='predicted_rate'), self.december.drop(columns='predicted_rate'))

    def test_duplicate_ids_fail(self):
        template = self.validation[['load_id']].assign(predicted_rate=np.nan)
        template.loc[1, 'load_id'] = template.loc[0, 'load_id']
        with self.assertRaises(ValueError):
            predict_files(DistanceModel(), self.validation, template, self.december)

    def test_preprocessing_is_train_only_and_ignores_auxiliary_columns(self):
        features = RateFeatures().fit(self.train)
        original = features.transform(self.validation)
        changed = self.validation.assign(quote_signal=1e9, market_index=-1e9, posted_rate=-1e9)
        pd.testing.assert_frame_equal(original, features.transform(changed))
        invalid = self.validation.copy()
        invalid['weight'] = -100
        transformed = features.transform(invalid)
        self.assertTrue(transformed.weight_invalid.eq(1).all())
        expected = invalid.equipment.map(features.equipment_medians)
        np.testing.assert_allclose(transformed.weight, expected)

    def test_scorer_rejects_incomplete_output(self):
        with self.assertRaises(SystemExit):
            validate_predictions(pd.DataFrame({'load_id':['TE-000001'], 'predicted_rate':[100.]}))


if __name__ == '__main__':
    unittest.main()
