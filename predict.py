"""Load a saved model and fill both submission CSVs without changing input features."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from freight_model import CORE


def predict_files(model, validation, template, december):
    for label, frame in [('validation', validation), ('December', december)]:
        missing = set(CORE) - set(frame)
        if missing:
            raise ValueError(f'{label}: missing columns {sorted(missing)}')
        if frame[CORE].drop(columns='weight').isna().any().any():
            raise ValueError(f'{label}: required feature missing')
        if not pd.to_numeric(frame.distance, errors='coerce').gt(0).all():
            raise ValueError(f'{label}: distance must be positive')
    if list(template.columns) != ['load_id', 'predicted_rate']:
        raise ValueError('Template columns must be load_id,predicted_rate')
    for label, frame in [('validation', validation), ('template', template)]:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError(f'{label}: load_id must be nonmissing and unique')
    if set(validation.load_id) != set(template.load_id):
        raise ValueError('Validation and template load_id sets differ')
    rates = pd.Series(np.asarray(model.predict(validation)), index=validation.load_id)
    output = template.copy()
    output['predicted_rate'] = output.load_id.map(rates)
    completed_december = december.copy()
    completed_december['predicted_rate'] = np.asarray(model.predict(december))
    for frame in [output, completed_december]:
        if not np.isfinite(frame.predicted_rate).all() or not frame.predicted_rate.gt(0).all():
            raise ValueError('Model produced invalid/nonpositive predictions')
    return output, completed_december


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/freight.joblib')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('validation_predictions.csv'))
    args = parser.parse_args()
    model = joblib.load(args.model)
    output, december = predict_files(model,
        pd.read_csv(args.data_dir/'validation.csv'),
        pd.read_csv(args.data_dir/'validation_predictions_template.csv'),
        pd.read_csv(args.data_dir/'december_chart_inputs.csv'))
    output.to_csv(args.output, index=False, float_format='%.2f')
    december.to_csv(args.data_dir/'december_chart_inputs.csv', index=False, float_format='%.2f')
    print(f'Wrote {len(output):,} validation and {len(december)} December predictions.')


if __name__ == '__main__':
    main()
