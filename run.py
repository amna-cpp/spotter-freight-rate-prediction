"""Train, validate both CSVs, and build the report."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    steps = [
        ["train.py"],
        [
            "score.py",
            "--predictions",
            "validation_predictions.csv",
            "--december-predictions",
            "data/december_chart_inputs.csv",
        ],
        ["make_report.py"],
    ]
    for step in steps:
        print(f"Running {step[0]}...", flush=True)
        subprocess.run([sys.executable, *step], cwd=root, check=True)


if __name__ == "__main__":
    main()
