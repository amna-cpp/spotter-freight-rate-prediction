# Freight rate prediction

[**Open the live demo →**](https://spotter-rate-desk.vercel.app)

A reproducible submission for Spotter's Machine Learning Engineer assessment.
The final estimator uses pickup, delivery, distance, equipment, weight and date.
It predicts all **12,000** future loads and the **31** fixed December inputs with
the same fitted model.

The [interactive demo](demo/README.md) runs the actual fitted model. Enter a load,
explore the December curve, and read the short validation walkthrough. It uses
plain HTML/CSS/JavaScript and a lightweight Python model export on Vercel.
The 64 supported cities and date range match the demo's historical study scope.

To preview locally without installing the training packages:

```bash
python demo/serve.py
```

Then open `http://127.0.0.1:8765`. Rebuild the export after retraining with
`python export_demo.py` in the assessment environment.

## Results

The original model was selected on **August-September** after fitting January-July.
CatBoost using the core inputs achieved **$90.25 MAE**, versus **$235.21** for
the equipment-median rate-per-mile baseline. Adding `market_index` gave $97.90;
adding both auxiliary signals gave $134.80. These comparisons use identical
CatBoost settings and the same chronological split.

The revision fits a smooth additive baseline, then uses CatBoost to correct its
remaining errors. Five candidates were compared on **June-July** and
**August-September**, using only earlier training months in each window. Mean
window MAE fell from **$109.73 to $104.37**. See [research and decisions](MODEL_RESEARCH.md).

Fitting January-September and checking **October's 4,853 loads** gives
**$99.29 MAE**, **$645.89 RMSE**, **4.66% MAPE** and **0.821 R2**.
**98.33% of predictions are within 10% of the actual price.** This explicit
tolerance is not classification accuracy; MAPE is not an accuracy percentage.
The original model's MAE was $100.26: the revision improves MAE by **0.97%**,
while RMSE is slightly worse ($645.76 originally). The paired day-bootstrap 95%
interval for the MAE improvement is **$0.07-$1.93** per load. The simple baseline's
MAE is $231.30, a **57.1% reduction** for the revision.
These are local labeled holdout results; final November-December accuracy is unknown.
The supplied scorer checks file validity and produces a chart; it does not calculate accuracy.

## Run

Use **Python 3.12**. The broad requirements satisfy the supplied scorer's ranges;
`requirements.lock.txt` records the exact tested environment on macOS ARM64.
Place the assessment's supplied raw files under `data/` using these names. Raw
training/validation inputs and the row-level October label ledger stay local and
are excluded from this public repository.

| Supplied file | Local path |
| --- | --- |
| `train-test.csv` | `data/train_test.csv` |
| `validation.csv` | `data/validation.csv` |
| `validation-predictions-template.csv` | `data/validation_predictions_template.csv` |

The completed `data/december_chart_inputs.csv` is included. Training overwrites
only its prediction column, so it also works as the inference input.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Train, score both CSVs, and build the report.
python run.py

# Tests use small fixtures; no assessment data needed.
python -m unittest discover -s tests -v
```

`run.py` runs `train.py`, the unchanged `score.py`, and `make_report.py` in order,
stopping on any failure. Training includes the October evaluation, a two-month
horizon check, and the final refit. To reproduce the original model comparison,
run `python experiment.py`; it uses only January-September. Run `python tune.py`
to reproduce the five-candidate revision comparison on both earlier windows.

The provided scorer can also be run directly:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

For the exact tested versions, install `requirements.lock.txt` instead. On a different
operating system, pip selects its compatible wheels. CatBoost training uses four CPU
threads, a fixed seed and 2,400 residual-tree iterations; small platform-level numeric differences
are possible. No external services, external datasets or API credentials are needed.
The generated model is ignored by Git and recreated by `train.py`.

For development checks:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
```

The formatter and linter exclude the supplied `score.py` to preserve it exactly.

To predict again without retraining:

```bash
python predict.py --model models/freight.joblib
```

## Validation and modeling decisions

- **Forward validation:** January-May (24,023 rows) -> June-July (9,695), and
  January-July (33,718) -> August-September (9,429), select the revision using
  mean window dollar MAE. January-September (43,147) -> October (4,853) checks
  the fixed choice. A second fixed-model check trains
  through August and evaluates October to test a two-month forecast horizon.
  Finally, January-October (48,000) is used to predict November-December.
  Splits keep whole dates together. No random split mixes future and past loads.
- **Selection discipline:** Revision comparisons, preprocessing and settings use
  only pre-October data; no October early stopping is used. October was already
  inspected during the original assessment, so it is a reused historical check,
  not a fresh blind test. The two-month stress check does not change the model.
- **Dirty inputs:** 300 labeled weights are missing and 292 are negative. Negative
  weights are treated as invalid rather than assumed to be sign errors. Invalid or
  missing weights use equipment medians fitted only on the current training fold,
  with a global median fallback and an invalid/missing flag.
- **Target tails:** Posted rates reach $25,533. Large rates are not automatically
  treated as mistakes or deleted. All labels remain in training and evaluation;
  MAE training reduces their influence. Raw RMSE is reported alongside MAE.
- **Proxy stability:** `quote_signal * distance` is exceptionally close to the target
  in some months but unreliable in others. This suggests a proxy/availability risk,
  not proof of how the column was generated. Controlled future-month ablations favor
  excluding both auxiliary signals. The December table also lacks those columns.
- **Features:** Directional lane and city categories, equipment, cleaned weight,
  distance/log/inverse distance, weekday, weekend, day/month/day-of-year and two
  annual sine/cosine harmonics. City coordinates are looked up from the fitting
  fold, so the December table needs no new coordinate values. Raw coordinates on
  prediction rows are not used. No `load_id`, label or manually computed target-derived
  aggregate is used as an input. CatBoost learns its categorical statistics internally
  within each training fit.
- **Model:** A Huber additive baseline learns smooth distance/weight and categorical
  effects without annual harmonics. CatBoost predicts its signed remaining error
  per mile with distance-proportional sample weights. This loss is equivalent,
  up to a constant, to dollar MAE. Prediction adds the correction to the baseline
  and applies a $1 floor to the total. Depth 4, learning rate 0.045, L2 8, seed 42,
  2,400 iterations; no October early stopping. The original estimators remain
  available in `experiment.py` for comparison.
- **Output safety:** Validation rates are joined to template IDs with uniqueness
  and exact-ID-set checks. Input row order and December's original six input
  columns are preserved. Predictions are finite, positive and written to cents.

## Files to submit

- `validation_predictions.csv` - exactly `load_id,predicted_rate` for 12,000 loads.
- `output/pdf/freight_rate_report.pdf` - validation, methodology and the required chart.
- This repository URL.
- Your recorded **2-3 minute Loom URL with camera on**. Use `LOOM_SCRIPT.md`
  as the timed speaking script. A recording/link must still be created by the candidate.

Other useful files: `data/december_chart_inputs.csv` contains the completed December
predictions; `scorer_results/candidate_december.png` is the original scorer's chart;
`results/metrics.json`, `results/development_comparison.csv`, `results/tuning_comparison.csv`,
`results/october_slices.csv` preserve aggregate evidence. Running `train.py` also
writes a local-only October prediction ledger, excluded from the public repository.
`results/input_sha256.json` fingerprints immutable model inputs and the original scorer.

## Code map

| File | Purpose |
| --- | --- |
| `run.py` | Complete submission workflow |
| `freight_model.py` | Input checks, train-only features, estimators |
| `evaluation.py` | Metrics, date splits, error summaries |
| `train.py` | Fixed-model evaluation and final training |
| `predict.py` | Saved-model inference and ID matching |
| `experiment.py` | Development comparisons |
| `tune.py` | Revision comparison using two earlier forecast windows |
| `make_report.py` | Report from saved results and scorer chart |

Test fixtures cover ID matching, invalid inputs, training-only imputation, signal exclusion,
time boundaries, and scorer rejection. Public CI runs lint, formatting and tests.

## Limits

Only ten labeled months of one year are available. Annual features do not establish
reliable December seasonality, holiday peaks or future market shocks. The fixed-lane
December chart is a model forecast with no December labels for that lane. Category
overlap is intentional for repeat-lane forecasting; this experiment does not estimate
performance in unseen cities or a new year. About 12.2% of future loads have directional
lanes absent from training. There is no external lane-pricing or holiday data.
No claim is made about Spotter's hidden final score.
