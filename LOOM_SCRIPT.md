# Camera-on video script

Aim for 2:40-2:55, including screen changes. Keep your camera on throughout.
The italic directions are screen cues, not spoken lines.

## 0:00-0:15 — Opening

*Show the demo, with Lexington to Fort Wayne selected.*

Hi, I'm Amna. I built a system that predicts freight prices, with a working demo.
My main question was: how well will it perform on loads from later months?

## 0:15-0:45 — Data and validation

*Show the split table on page one of the report.*

The data contains 48,000 loads with known prices, from January through October.
The final predictions are for November and December, so I tested on later months.

I trained on January through July, compared models on August and September,
then locked the choice before evaluating October.

Some weights were missing or negative. I filled those using the typical weight
for that equipment type, calculated from training data only, and marked them
as filled in.

## 0:45-1:15 — The key finding and model choice

*Point to the model comparison table.*

One useful finding was that an extra input called quote signal looked very
helpful in some months, but was less reliable in others. In a controlled comparison,
adding it and the market index increased average error from about 90 dollars
to 135 dollars. I left both out.

I chose CatBoost, a tree-based model. It performed best in these comparisons
and can learn how route, distance, equipment and date affect prices together.

## 1:15-1:40 — Results and honest error analysis

*Point to October MAE and the baseline.*

On October's 4,853 loads, the average prediction error was about 100 dollars
per load, compared with 231 dollars for a simple comparison model.
That's about 57 percent lower error.

A few large mistakes remain. I kept the unusual prices in the test and included
those errors in the report.

## 1:40-2:15 — Code walkthrough

*Show the weight-imputation and loss setup in `freight_model.py`, then the ID join
in `predict.py`. Have both files open beforehand.*

Here, the cleaning step learns from training data only. The model learns price
per mile, with training weights chosen to measure mistakes in dollars per load.

Here, each prediction is matched to the correct load ID, with checks for missing
or duplicate IDs. One command runs training and creates both prediction files,
the required chart, and the report.

## 2:15-2:50 — Demo, limitation and closing

*Return to the demo and show the December curve. Keep it brief.*

The demo lets you enter a load and see a real model estimate. The supplied scorer
checks all 12,000 predictions and 31 December rows. Their actual prices are
unavailable, so I cannot claim a final accuracy score.

With only ten months of known prices, December's seasonal changes remain uncertain.
My next step would be to track errors as new prices arrive.

My approach is simple: check the data, test each choice, explain the results,
and make the work easy for another engineer to run.

## Recording notes

- Speak at a comfortable pace; the screen cues are approximate, not deadlines.
- Keep the camera bubble away from the table and code you point to.
- Use a readable editor zoom and pre-open the report, code and demo.
- Pause after the improvement. "About 57 percent lower error" rounds the measured
  56.7% reduction; it is not an accuracy percentage.
- Say "suspected proxy risk" if asked about leakage. The data does not prove how
  the quote signal was generated.
- Explain the choices in your own words and be ready to answer follow-up questions.

## Why this emphasis

Spotter's public ML role emphasizes data preparation, experimentation, clean
maintainable Python, production integration and communication. These are public
role expectations, not a disclosed assessment scoring rubric.

- [Spotter's official ML role](https://careers.spotter.ai/jobs/671154-remote-machine-learning-engineer)
- [MIT Communication Lab: structure around one message and supporting evidence](https://mitcommlab.mit.edu/nse/commkit/structuring-a-slide-presentation/)
