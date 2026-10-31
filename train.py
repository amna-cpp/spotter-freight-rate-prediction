"""Evaluate the locked model on October; refit all labeled data and predict."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from experiment import metrics
from freight_model import TreeRateModel
from predict import predict_files


def day_bootstrap_interval(frame, predictions):
    errors = pd.DataFrame({'date':frame.date.to_numpy(),
                           'error':np.abs(frame.posted_rate.to_numpy()-predictions)})
    by_day = errors.groupby('date').error.agg(['sum','count'])
    rng = np.random.default_rng(42)
    indices = rng.integers(0,len(by_day),size=(2000,len(by_day)))
    values = by_day['sum'].to_numpy()[indices].sum(axis=1)/by_day['count'].to_numpy()[indices].sum(axis=1)
    return np.quantile(values,[.025,.975]).tolist()


def audit(data, validation):
    result = {}
    for name, frame in [('labeled',data),('validation',validation)]:
        result[name] = {'rows':len(frame),'columns':list(frame),
            'date_min':frame.date.min(),'date_max':frame.date.max(),
            'missing':{k:int(v) for k,v in frame.isna().sum().items()},
            'negative_weights':int(frame.weight.lt(0).sum()),
            'duplicate_ids':int(frame.load_id.duplicated().sum()),
            'duplicate_rows_without_id':int(frame.drop(columns='load_id').duplicated().sum())}
    result['labeled']['nonpositive_target'] = int(data.posted_rate.le(0).sum())
    result['labeled']['target_quantiles'] = {str(q):float(v) for q,v in data.posted_rate.quantile([0,.01,.5,.99,1]).items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--skip-horizon-stress', action='store_true', help='Skip the additional two-month October check')
    args = parser.parse_args()
    output = Path('results'); output.mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    data = pd.read_csv(args.data_dir/'train_test.csv')
    validation = pd.read_csv(args.data_dir/'validation.csv')
    if not data.load_id.is_unique or not validation.load_id.is_unique:
        raise ValueError('Duplicate IDs in supplied datasets')
    if not set(data.load_id).isdisjoint(validation.load_id):
        raise ValueError('Labeled and prediction IDs overlap')
    if not np.isfinite(data.posted_rate).all() or not data.posted_rate.gt(0).all():
        raise ValueError('Labeled target must be finite and positive')
    if not data.date.max() < validation.date.min():
        raise ValueError('This validation strategy requires future-only prediction data')
    train = data[data.date < '2025-10-01']
    holdout = data[data.date >= '2025-10-01']
    print('Evaluating fixed core model on October...', flush=True)
    model = TreeRateModel('core').fit(train)
    predictions = np.asarray(model.predict(holdout))
    baselines = (train.posted_rate/train.distance).groupby(train.equipment).median()
    baseline = np.asarray(holdout.distance * holdout.equipment.map(baselines))
    summary = {'selection':{'criterion':'lowest August-September MAE',
                           'model':'tree_core','iterations':1600,'seed':42},
        'split':{'fit_start':train.date.min(),'fit_end':train.date.max(),'fit_n':len(train),
                 'holdout_start':holdout.date.min(),'holdout_end':holdout.date.max(),'holdout_n':len(holdout)},
        'october':metrics(holdout.posted_rate.to_numpy(),predictions),
        'october_baseline':metrics(holdout.posted_rate.to_numpy(),baseline),
        'october_mae_day_bootstrap_95pct':day_bootstrap_interval(holdout,predictions)}
    print(json.dumps(summary,indent=2),flush=True)
    residuals = holdout[['load_id','date','equipment','distance','posted_rate']].copy()
    residuals['predicted_rate'] = predictions
    residuals['baseline_rate'] = baseline
    residuals['absolute_error'] = np.abs(residuals.posted_rate-predictions)
    residuals.to_csv(output/'october_holdout_predictions.csv',index=False)
    rows = []
    for name, frame in residuals.groupby('equipment'):
        rows.append({'slice':f'equipment={name}','n':len(frame),**metrics(frame.posted_rate,frame.predicted_rate)})
    for label, mask in [('distance<=500',residuals.distance.le(500)),
                        ('500<distance<=1500',residuals.distance.gt(500)&residuals.distance.le(1500)),
                        ('distance>1500',residuals.distance.gt(1500))]:
        frame = residuals[mask]
        rows.append({'slice':label,'n':len(frame),**metrics(frame.posted_rate,frame.predicted_rate)})
    pd.DataFrame(rows).to_csv(output/'october_slices.csv',index=False)
    summary['october']['median_absolute_error'] = float(np.median(residuals.absolute_error))
    summary['october']['p95_absolute_error'] = float(residuals.absolute_error.quantile(.95))
    squared = np.sort((residuals.posted_rate-predictions).to_numpy()**2)
    summary['october']['largest_1pct_errors_share_squared_error'] = float(squared[-int(np.ceil(.01*len(squared))):].sum()/squared.sum())
    if not args.skip_horizon_stress:
        print('Two-month horizon stress check: fit through August, predict October...',flush=True)
        stress = TreeRateModel('core').fit(data[data.date < '2025-09-01'])
        summary['october_two_month_horizon'] = metrics(holdout.posted_rate.to_numpy(),np.asarray(stress.predict(holdout)))
    print('Refitting locked model on all 48,000 labeled loads...',flush=True)
    model = TreeRateModel('core').fit(data)
    joblib.dump(model,'models/freight.joblib')
    importance = pd.Series(model.model.feature_importances_,index=model.model.feature_names_).sort_values(ascending=False)
    importance.rename_axis('feature').reset_index(name='importance').to_csv(output/'feature_importance.csv',index=False)
    template = pd.read_csv(args.data_dir/'validation_predictions_template.csv')
    december_inputs = pd.read_csv(args.data_dir/'december_chart_inputs.csv')
    predictions, december = predict_files(model,validation,template,december_inputs)
    predictions.to_csv('validation_predictions.csv',index=False,float_format='%.2f')
    december.to_csv(args.data_dir/'december_chart_inputs.csv',index=False,float_format='%.2f')
    summary['final_fit_rows'] = len(data)
    summary['output_rows'] = {'validation':len(predictions),'december':len(december)}
    summary['december_range'] = [float(december.predicted_rate.min()),float(december.predicted_rate.max())]
    (output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'data_audit.json').write_text(json.dumps(audit(data,validation),indent=2)+'\n')
    paths = [args.data_dir/'train_test.csv',args.data_dir/'validation.csv',
             args.data_dir/'validation_predictions_template.csv',Path('score.py')]
    manifest = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output/'input_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved model, diagnostics and both completed submission files.',flush=True)


if __name__ == '__main__':
    main()
