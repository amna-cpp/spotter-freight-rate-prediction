# Camera-on video script

Aim for 2:40-2:55, including screen changes. Keep your camera on throughout.
The italic directions are screen cues, not spoken lines.

## 0:00-0:15 — Opening

*Show the demo, with Lexington to Fort Wayne selected.*

Hi, I'm Amna. For this assessment, I built a freight-rate prediction pipeline
and a working demo. My focus was reliable future-month evaluation and a solution
another engineer can reproduce.

## 0:15-0:45 — Data and validation

*Show the split table on page one of the report.*

The data contains 48,000 labeled loads from January through October. The final
loads are in November and December, so I validated forward in time.

I trained on January through July, compared models on August and September,
then locked the choice before evaluating October.

I also found missing and negative weights. I used equipment medians fitted
only on each training fold, with a missingness flag.

## 0:45-1:15 — The key finding and model choice

*Point to the model comparison table.*

One important finding was that the quote signal was extremely predictive in
some months but unreliable in others. I tested it under identical model settings.
Including the auxiliary signals increased development MAE from about 90 dollars
to 135 dollars, so I excluded them.

CatBoost performed best on that split and captured interactions between lane,
equipment, distance, and calendar features.

## 1:15-1:40 — Results and honest error analysis

*Point to October MAE and the baseline.*

On October's 4,853 loads, the selected model achieved a mean absolute error of
100 dollars and 26 cents, compared with 231 dollars for the baseline—a 56.7 percent
reduction.

I kept extreme posted rates in evaluation. They explain the much higher RMSE,
which I report alongside MAE.

## 1:40-2:15 — Code walkthrough

*Show the weight-imputation and loss setup in `freight_model.py`, then the ID join
in `predict.py`. Have both files open beforehand.*

Here, preprocessing is fitted within each training split. The model predicts
rate per mile, with distance weights that align its loss with dollar error.

Here, predictions are matched to the template by load ID, with uniqueness and
coverage checks. One command recreates training, both output files, the scorer
chart, and the report.

## 2:15-2:50 — Demo, limitation and closing

*Return to the demo and show the December curve. Keep it brief.*

The demo makes the estimate and validation approach easy to inspect. The supplied
scorer validates all 12,000 predictions and 31 December rows; it doesn't measure
hidden-label accuracy.

With only ten labeled months, December seasonality remains uncertain. With new
labels, I'd monitor error by equipment, distance, and time.

This project shows how I approach ML work: test assumptions, justify decisions
with evidence, and make the result reproducible and usable.

## Recording notes

- Speak at a comfortable pace; the screen cues are approximate, not deadlines.
- Keep the camera bubble away from the table and code you point to.
- Use a readable editor zoom and pre-open the report, code and demo.
- Pause after the 56.7% result. Explain it; do not call it "56.7% accuracy."
- Say "suspected proxy risk" if asked about leakage. The data does not prove how
  the quote signal was generated.
- Explain the choices in your own words and be ready to answer follow-up questions.

## Why this emphasis

Spotter's public ML role emphasizes data preparation, experimentation, clean
maintainable Python, production integration and communication. These are public
role expectations, not a disclosed assessment scoring rubric.

- [Spotter's official ML role](https://careers.spotter.ai/jobs/671154-remote-machine-learning-engineer)
- [MIT Communication Lab: structure around one message and supporting evidence](https://mitcommlab.mit.edu/nse/commkit/structuring-a-slide-presentation/)
