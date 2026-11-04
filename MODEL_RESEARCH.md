# Model improvement

The revision keeps the same six inputs and compares candidates using dollar MAE.
It does not use final November–December labels, which were not supplied.

## Research and decisions

- [CatBoost's parameter-tuning guide](https://catboost.ai/docs/en/concepts/parameter-tuning)
  recommends checking tree count, learning rate, depth and categorical handling.
  A small initial August–September screen tested shallower trees, more iterations,
  built-in one-hot encoding, fewer calendar inputs, direct dollar targets and route
  interactions. These changes alone did not clearly beat the original model.
- [CatBoost's regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression)
  distinguish MAE from squared-error objectives. Keep MAE: a small number of extreme
  posted rates dominate RMSE. All rows remain in training and evaluation.
- [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
  requires learning preprocessing from training data only and keeping test data out
  of model selection. The revision shortlist uses only dates before October.

A smooth additive baseline followed by residual trees was the most useful change.
The baseline captures broad distance, weight, city, equipment and weekday effects.
CatBoost learns the remaining signed errors, including nonlinear and calendar
patterns. This is sequential residual fitting, not an ensemble selected using
October labels. No new dependency or external data source is needed.

## Reproduce selection

Run `python tune.py`. It compares five candidates on two expanding-time windows:

| Fit | Check | Purpose |
| --- | --- | --- |
| January–May | June–July | Earlier two-month forecast |
| January–July | August–September | Later two-month forecast |

Each candidate fits its own preprocessing inside the training window. Selection
minimizes the mean of the two window MAEs. The selected configuration is a Huber
additive baseline with no annual harmonics, followed by 2,400 depth-4 CatBoost
trees at learning rate 0.045. L2 is 8 and the random seed is 42.

`results/tuning_comparison.csv` records every shortlist result, and
`results/model_selection.json` records the choice. The original signal-ablation
experiment is still available in `experiment.py`.

## Evaluate and export

`python run.py` evaluates the fixed revision on October, reproduces the original
model on the same rows, checks the two-month horizon, refits all labeled data,
writes both prediction files, runs the unchanged scorer and builds the PDF.

`results/metrics.json` includes a paired day-bootstrap interval for the reduction
in dollar error and the percentage of predictions within 10% of actual prices.
That percentage has an explicit tolerance; it is not classification accuracy.
MAPE is average percentage error and must not be converted into “100 minus MAPE
accuracy.”

October was already inspected in the original assessment. The revision was chosen
using earlier windows, but October is a reused historical test, not a fresh blind
test. No claim can be made about hidden November–December accuracy.

`python export_demo.py` exports both the additive coefficients and residual trees.
The demo adds both components and uses the same $1 floor as native inference.
Export tests cover all December predictions and numeric split boundaries. Raw
assessment rows stay local.
