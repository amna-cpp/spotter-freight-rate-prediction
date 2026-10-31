"""Development comparison only: January-July -> August-September."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from freight_model import AdditiveRateModel, TreeRateModel


def metrics(y, p):
    return {'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'mape_pct': float(np.mean(np.abs(y-p)/y)*100),
            'r2': float(r2_score(y,p))}


if __name__ == '__main__':
    data = pd.read_csv('data/train_test.csv')
    train = data[data.date < '2025-08-01']
    development = data[data.date.between('2025-08-01','2025-09-30')]
    results = []
    candidates = {
        'distance_equipment_median': None,
        'quote_times_distance': 'quote',
        'additive_1': AdditiveRateModel(1),
        'additive_2': AdditiveRateModel(2),
        'tree_core': TreeRateModel('core'),
        'tree_market': TreeRateModel('market'),
        'tree_quote': TreeRateModel('quote'),
    }
    for name, model in candidates.items():
        print('Fitting', name, flush=True)
        if model is None:
            medians = (train.posted_rate/train.distance).groupby(train.equipment).median()
            predictions = development.distance * development.equipment.map(medians)
        elif isinstance(model, str):
            predictions = development.distance * development.quote_signal
        else:
            model.fit(train)
            predictions = model.predict(development)
        predictions = np.asarray(predictions)
        for month in ['all', '2025-08', '2025-09']:
            mask = np.ones(len(development), dtype=bool) if month == 'all' else development.date.str.startswith(month).to_numpy()
            row = {'model':name, 'period':month, 'n':int(mask.sum()), **metrics(development.posted_rate.to_numpy()[mask],predictions[mask])}
            print(json.dumps(row), flush=True)
            results.append(row)
    Path('results').mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv('results/development_comparison.csv',index=False)
