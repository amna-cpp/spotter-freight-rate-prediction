"""Build the assessment PDF from actual saved metrics and the original scorer chart."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
TEAL = colors.HexColor("#064A56")
INK = colors.HexColor("#193139")
MUTED = colors.HexColor("#52676E")
PALE = colors.HexColor("#EDF4F4")


def main():
    result = json.loads((ROOT / "results/metrics.json").read_text())
    comparison = pd.read_csv(ROOT / "results/tuning_comparison.csv")
    top = pd.read_csv(ROOT / "results/feature_importance.csv").head(10).iloc[::-1]
    target = ROOT / "output/pdf/freight_rate_report.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = ROOT / "tmp/pdfs"
    temp.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.1, 2.3), dpi=180)
    ax.barh(top.feature, top.importance, color="#087C88", height=0.65)
    ax.set_xlabel(
        "Residual-tree prediction-change importance (relative units)", fontsize=9
    )
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    importance_path = temp / "importance.png"
    fig.savefig(importance_path, bbox_inches="tight")
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=TEAL,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Deck", fontSize=11, leading=15, textColor=MUTED, spaceAfter=13
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=TEAL,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Copy", fontSize=9.4, leading=13.3, textColor=INK, spaceAfter=8
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCopy", fontSize=8, leading=11, textColor=MUTED, spaceAfter=7
        )
    )
    styles.add(ParagraphStyle(name="Cell", fontSize=8.1, leading=10.8, textColor=INK))
    story = []

    def para(text, style="Copy"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths, header=True):
        wrapped = [[Paragraph(str(v), styles["Cell"]) for v in row] for row in rows]
        t = Table(
            wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0
        )
        commands = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#CCDADB")),
        ]
        if header:
            commands += [
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
            ]
        t.setStyle(TableStyle(commands))
        story.append(t)

    para("Freight rate prediction", "ReportTitle")
    para(
        "Model, validation and December forecast<br/>Amna Hameed",
        "Deck",
    )
    m = result["october"]
    baseline = result["october_baseline"]
    improvement = 100 * (1 - m["mae"] / baseline["mae"])
    table(
        [
            ["October MAE", "October MAPE", "Baseline MAE", "MAE reduction"],
            [
                f"<b>${m['mae']:,.2f}</b>",
                f"<b>{m['mape_pct']:.2f}%</b>",
                f"${baseline['mae']:,.2f}",
                f"<b>{improvement:.1f}%</b>",
            ],
        ],
        [126] * 4,
    )
    para("Evaluation design", "Section")
    para(
        "The final inputs cover November-December 2025, after the January-October labeled data. "
        "I therefore validated forward in time, keeping every date in one partition. "
        "The revision selects by mean dollar MAE across two pre-October forecast windows."
    )
    table(
        [
            ["Purpose", "Fit period / rows", "Evaluate or predict / rows"],
            ["Earlier comparison", "Jan-May / 24,023", "Jun-Jul / 9,695"],
            ["Later comparison", "Jan-Jul / 33,718", "Aug-Sep / 9,429"],
            ["Fixed-model check", "Jan-Sep / 43,147", "October / 4,853"],
            ["Two-month stress check", "Jan-Aug / 38,477", "October / 4,853"],
            ["Final fit", "Jan-Oct / 48,000", "Nov-Dec / 12,000; fixed chart / 31"],
        ],
        [116, 175, 213],
    )
    para("Candidate comparison", "Section")
    names = {
        "original": "Original CatBoost",
        "shallow": "CatBoost, depth 4 / faster learning",
        "medium": "CatBoost, depth 5 / faster learning",
        "residual_4": "Smooth baseline + depth-4 trees (selected)",
        "residual_6": "Smooth baseline + depth-6 trees",
    }
    rows = [["Model / dollar MAE", "Jun-Jul", "Aug-Sep", "Mean"]]
    for name, group in comparison.groupby("model", sort=False):
        values = group.sort_values("check_start").mae.tolist()
        rows.append(
            [names[name], *[f"{v:,.2f}" for v in values], f"{sum(values) / 2:,.2f}"]
        )
    table(rows, [258, 82, 82, 82])
    para(
        "Each preprocessing fit uses only that window's training rows. The chosen revision gives "
        "October RMSE $%.2f and R2 %.3f. The day-block bootstrap 95%% MAE interval is $%.2f-$%.2f "
        "(2,000 resamples; conditional on this month). October was inspected in the original assessment; "
        "revision selection uses only earlier months. This is a reused chronological check, not a fresh blind test."
        % (m["rmse"], m["r2"], *result["october_mae_day_bootstrap_95pct"]),
        "SmallCopy",
    )
    para("Error behavior", "Section")
    revision = result["revision"]
    para(
        "The original model's October MAE was $%.2f; the revision reduces it by %.2f%%. "
        "The paired day-bootstrap 95%% interval for the MAE improvement is $%.2f-$%.2f. "
        "%.2f%% of October predictions are within 10%% of the actual price. This is a tolerance-based "
        "success rate, not classification accuracy."
        % (
            revision["original_october"]["mae"],
            revision["mae_reduction_pct"],
            *revision["paired_mae_improvement_day_bootstrap_95pct"],
            m["within_10pct_pct"],
        )
    )
    para(
        "October median absolute error is $%.2f and its 95th percentile is $%.2f. "
        "The largest 1%% of absolute residuals account for %.1f%% of squared error. "
        "No evaluation rows were removed. The fixed two-month horizon check gives "
        "$%.2f MAE and %.2f%% MAPE; it did not change the model."
        % (
            m["median_absolute_error"],
            m["p95_absolute_error"],
            100 * m["largest_1pct_errors_share_squared_error"],
            result["october_two_month_horizon"]["mae"],
            result["october_two_month_horizon"]["mape_pct"],
        )
    )

    story.append(PageBreak())
    para("Data quality and model", "ReportTitle")
    para("Input checks, missing weights and model features", "Deck")
    table(
        [
            ["Issue", "Labeled data", "Final prediction data", "Treatment"],
            [
                "Missing weight",
                "300",
                "165",
                "Training-fold equipment median; global fallback",
            ],
            [
                "Negative weight",
                "292",
                "145",
                "Treat as invalid; same imputation; add flag",
            ],
            ["Missing market index", "374", "249", "Exclude from selected model"],
            [
                "Duplicate IDs / rows",
                "0 / 0",
                "0 / 0",
                "Fail on duplicate prediction IDs",
            ],
            [
                "Long target tail",
                "Max $25,533",
                "No labels supplied",
                "Retain all labeled rows; robust MAE loss",
            ],
        ],
        [113, 89, 104, 198],
    )
    para("Why exclude the auxiliary signals?", "Section")
    para(
        "The quote-derived baseline has $317.26 MAE in August but $71.75 in September. "
        "That changing reliability suggests a proxy or availability risk; it does not prove leakage. "
        "The matched CatBoost ablations confirm that including the signals hurts the two-month "
        "development result. Both are also absent from the fixed December table. The final model "
        "uses the same six-input contract for every prediction."
    )
    para("Estimator and feature engineering", "Section")
    para(
        "A regularized Huber additive model first learns smooth distance/weight effects and city, "
        "equipment and weekday effects, without annual harmonics. CatBoost then predicts its signed "
        "remaining error per mile. Distance-proportional weights align this loss with dollar MAE. "
        "The correction is added to the baseline and the total is floored at $1. "
        "Tree settings: 2,400 iterations, depth 4, learning rate 0.045, L2 8, seed 42, four CPU threads."
    )
    para(
        "Features include directional city/lane categories, equipment, cleaned weight and its flag, "
        "distance/log/inverse distance, weekday/weekend, day, month, day of year and two annual "
        "sine/cosine harmonics. City coordinates are derived from fitting-fold lookups. "
        "No load ID, manually computed target aggregate, validation label or auxiliary signal enters "
        "the feature table. CatBoost learns its categorical statistics inside each training fit. "
        "The tree count is fixed; October is not used for early stopping."
    )
    story.append(Image(str(importance_path), width=504, height=163))
    para(
        "Importance describes only the residual trees, not the additive component or causal effects. "
        "Correlated calendar/location features can share importance.",
        "SmallCopy",
    )

    story.append(PageBreak())
    para("Fixed December forecast", "ReportTitle")
    para("Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb", "Deck")
    para(
        "This is the unmodified image produced by the supplied <b>score.py</b>. "
        "Only the date varies across the 31 rows. The same final model, fitted on all 48,000 labeled "
        "loads, produced this curve and the 12,000 load-level predictions."
    )
    chart = ROOT / "scorer_results/candidate_december.png"
    from PIL import Image as PILImage

    with PILImage.open(chart) as im:
        w, h = im.size
    story.append(Image(str(chart), width=504, height=504 * h / w))
    para(
        "Daily predicted rates range from $%.2f to $%.2f. The scorer validates both CSVs and "
        "generates this chart; Spotter calculates hidden-label accuracy after submission."
        % tuple(result["december_range"]),
        "SmallCopy",
    )
    para("Run instructions and output checks", "Section")
    para(
        "The repository includes source, input setup instructions, the original scorer, requirements and an exact "
        "environment lock. Raw assessment datasets stay local. <b>tune.py</b> recreates revision selection; <b>train.py</b> evaluates the fixed "
        "model, saves diagnostics and refits; <b>predict.py</b> matches predictions to template IDs. "
        "It preserves December input columns and validates one-to-one ID coverage and finite positive rates. "
        "Both CSV outputs are written to cents. Portable contract and model-export tests check the workflow."
    )
    table(
        [
            ["Run after placing the supplied inputs and installing requirements.txt"],
            ["python run.py  (train, validate CSVs, build PDF)"],
            ["python -m unittest discover -s tests -v"],
        ],
        [504],
    )
    para("Limits and next steps", "Section")
    para(
        "Only ten labeled months from one year are available. Seasonal features cannot establish "
        "December holiday peaks or future market shocks. The chart has no observed December prices "
        "for this fixed lane and should not be interpreted as validated daily seasonality. "
        "Repeat cities and lanes are allowed across time splits, matching the forecasting use case. "
        "About 12.2% of final directional lanes were unseen in training; unseen-city or new-year "
        "performance was not validated. Monitor future dollar error and drift once new labels arrive."
    )

    def footer(canvas, doc):
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(0.6)
        canvas.line(54, 42, 558, 42)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(54, 29, "AMNA HAMEED  /  FREIGHT RATE PREDICTION")
        canvas.drawRightString(558, 29, str(doc.page))

    doc = SimpleDocTemplate(
        str(target),
        pagesize=(612, 792),
        rightMargin=54,
        leftMargin=54,
        topMargin=43,
        bottomMargin=56,
        title="Freight Rate Prediction - Validation and Model Report",
        author="Amna Hameed",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    main()
