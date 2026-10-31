# Submission checklist

| Requirement | Evidence |
| --- | --- |
| Train and validate labeled loads | `train.py`, `experiment.py`, `results/metrics.json` |
| Explain split and validation | `README.md`, PDF report |
| Predict every validation load | `validation_predictions.csv`: 12,000 unique IDs |
| Fill the matching template | ID join in `predict.py`; checked by unit tests |
| Predict all fixed December inputs | `data/december_chart_inputs.csv`: 31 days |
| Use the supplied scorer | Unmodified `score.py`; `scorer_results/validation_log.txt` |
| Include the December chart in the report | `output/pdf/freight_rate_report.pdf` |
| Accessible code, dependencies and instructions | Public repository, requirements, `README.md` |
| Efficient, repeatable workflow | `python run.py`; fixed seed and CPU thread limit |
| Clean code and concise comments | Ruff checks, portable tests, public CI |
| 2-3 minute camera-on Loom | **Candidate action remaining**; `LOOM_WALKTHROUGH.md` is the guide |

The scorer validates files; it does not calculate hidden-label accuracy. The
reported October metrics use only labeled development data. Raw assessment
datasets and the row-level labeled holdout ledger remain local.
