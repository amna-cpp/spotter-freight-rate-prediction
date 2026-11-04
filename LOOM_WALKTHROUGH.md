# Recording preparation

Use `LOOM_SCRIPT.md` for the one-paragraph script. Keep your camera on and aim
for 2–3 minutes. Only two screens are needed: the demo, then one code file.

1. Start on the demo and click **Estimate rate**.
2. Scroll to **Checked against the future** to explain the two earlier comparison
   windows and October results. Stay on the same page.
3. Switch once to VS Code and show `freight_model.py`, around lines 36–65.
   Explain training-only weight cleaning. Finish speaking on this screen.

Remember the story: load details → estimate → later-month tests → results → code.
The model first learns broad pricing effects, then uses CatBoost to correct the
remaining errors. On October, average error is $99.29 per load, versus $231.30
for the simple equipment/distance model. About 98.33% of predictions are within
10% of the actual price. Some large errors remain.

The revision was selected using pre-October data. October was previously inspected
for the original model, so it is a reused check. The final November–December prices
are unknown. Do not call 100 minus MAPE an accuracy score.

After recording, check that someone outside your account can open the Loom link.
The actual camera-on recording and link are still your submission step.
