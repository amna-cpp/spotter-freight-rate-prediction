# Camera-on walkthrough guide

Use [the final timed script](LOOM_SCRIPT.md) for recording. The outline below is
extra preparation material; do not read both scripts in the video.

Record this yourself in Loom with camera on. Aim for 2-3 minutes. Replace the prose
with your own explanation and make sure you can explain each decision. Do not
submit this guide as a recording link.

## 0:00-0:30 - Data and findings

Show `results/data_audit.json` and `results/development_comparison.csv`.

"The labeled data has 48,000 loads from January through October. The 12,000 final
loads are in November and December, so this is a future prediction task. I found
missing and negative weights and a long tail of posted rates. The quote signal
looks extremely predictive in some months but is unstable across time."

## 0:30-1:05 - Validation and model choice

Show the comparison table in the report.

"I fitted January through July and compared models on August and September. The
core-input CatBoost model had about 90 dollars MAE, compared with 235 for a simple
equipment rate-per-mile baseline. Adding the auxiliary signals worsened the
future-month result. I then locked the settings, fitted through September and
evaluated October: about 100 dollars MAE and 4.7 percent MAPE. The high RMSE reflects
a small number of large errors; I report those rather than dropping them."

## 1:05-1:45 - Important code

Show `freight_model.py`, then `train.py`.

"Preprocessing is fitted inside each time split. Invalid weights use a training-only
equipment median and a missingness flag. Features describe equipment, the directional
lane, distance, weight and calendar patterns. The model predicts rate per mile, with
distance weights that make its MAE objective proportional to dollar error. The
training script evaluates the fixed model, checks the longer horizon, then refits
on all labeled loads. Load IDs are never features."

## 1:45-2:30 - Outputs, chart and limitations

Show `predict.py`, the scorer output and `scorer_results/candidate_december.png`.

"Predictions are matched by ID to the supplied template and checked for missing or
duplicate IDs and invalid prices. The same trained model generates the 31 December
points from the six fixed inputs. The unchanged scorer validates both CSVs and
creates this chart; it does not report final accuracy. The report explains the
limited seasonal history and outlier risk. The README has the complete commands
to recreate the predictions and report."

Before submitting, verify the Loom link opens for someone outside your account and
paste that actual link into your submission message.
