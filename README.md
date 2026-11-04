# Freight rate prediction

[Live demo](https://spotter-rate-desk.vercel.app) · [Report](output/pdf/freight_rate_report.pdf) · [Predictions](validation_predictions.csv)

Predict freight prices from pickup, delivery, distance, equipment, weight and date.
The completed outputs contain 12,000 load predictions and 31 fixed December predictions.

## Results

The model fits broad pricing effects with an additive model, then uses CatBoost
to predict the remaining errors. Five candidates were compared on June–July and
August–September, training only on earlier months in each window.

After fitting January–September, results on October's 4,853 loads were:

| Measure | Result |
| --- | ---: |
| Mean absolute error | $99.29 |
| Root mean squared error | $645.89 |
| Mean absolute percentage error | 4.66% |
| Predictions within 10% of actual price | 98.33% |
| Equipment/distance baseline MAE | $231.30 |

October was previously inspected when evaluating the original model. It was not
used to select the revised model. Final November–December accuracy is unknown;
the supplied scorer checks the output files and creates the December chart.

Missing or invalid weights use equipment medians learned from training data.
The model excludes `quote_signal` and `market_index` after comparisons on later
months showed higher error with those inputs. All labeled rows remain in training
and evaluation, including extreme prices.

## Setup and run

Use Python 3.12. Place the supplied files at these paths; raw datasets are excluded
from the repository.

| Supplied file | Local path |
| --- | --- |
| `train-test.csv` | `data/train_test.csv` |
| `validation.csv` | `data/validation.csv` |
| `validation-predictions-template.csv` | `data/validation_predictions_template.csv` |

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

This trains and evaluates the model, writes both prediction files, runs the
unchanged scorer and builds the PDF report. `requirements.lock.txt` contains the
exact tested package versions. Training uses a fixed seed and four CPU threads.

To run the scorer separately:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Demo

```bash
python demo/serve.py
```

Open `http://127.0.0.1:8765`. The included model export runs with the Python
standard library. It supports the 64 training cities and November–December 2025.
After retraining, refresh it with `python export_demo.py`. Deployment instructions
are in [demo/README.md](demo/README.md).

## Checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

Tests cover input validation, ID matching, training-only cleaning, residual
corrections and agreement between the native model and demo export. CI runs these
checks on each push.

## Files

| Path | Purpose |
| --- | --- |
| `freight_model.py` | Features and model definitions |
| `train.py` | Evaluation and final training |
| `predict.py` | Saved-model inference and template matching |
| `evaluation.py` | Metrics and error summaries |
| `experiment.py` | Original model and input comparisons |
| `tune.py` | Model comparison on two earlier forecast windows |
| `run.py` | Training, scoring and report generation |
| `make_report.py` | PDF report generation |
| `validation_predictions.csv` | Completed 12,000-load prediction file |
| `data/december_chart_inputs.csv` | Completed December inputs |
| `scorer_results/candidate_december.png` | Chart produced by the supplied scorer |
| `output/pdf/freight_rate_report.pdf` | Method, split, results and December chart |
| `results/` | Aggregate metrics and comparisons |
| `demo/` | Web interface and exported model |

Only ten labeled months from one year are available. December seasonality,
unseen cities and future market changes have not been validated. Large errors
remain and are included in the reported metrics.
