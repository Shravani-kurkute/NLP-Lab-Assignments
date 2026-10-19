#!/usr/bin/env python3
"""Render the captured console output into PNG files for the report.

The PNGs are produced from the real log files in results/logs/ that were
captured while running BuildDatasets.py and country_prediction.py. They are
text renderings of those logs; replace them with real terminal screenshots if
you prefer (see screenshots/SCREENSHOTS.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "results" / "logs"
SHOTS_DIR = BASE_DIR / "screenshots"

FONT_SIZE = 9
LINE_HEIGHT = 0.175
FIG_WIDTH = 13.0
MAX_LINES = 70

CAPTURES = [
    (
        "build_datasets.log",
        "01_dataset_generation.png",
        "python3 BuildDatasets.py",
        None,
    ),
    (
        "predict_words.log",
        "02_single_prediction.png",
        "python3 country_prediction.py --words athens greece baghdad",
        None,
    ),
    (
        "demo.log",
        "03_demo_test15.png",
        "python3 country_prediction.py --demo",
        None,
    ),
    (
        "evaluation.log",
        "04_evaluation_summary.png",
        "python3 country_prediction.py --evaluate",
        "[evaluate]",
    ),
    (
        "missing_word.log",
        "05_missing_word_error.png",
        "python3 country_prediction.py --words athens greece zzznotaword",
        None,
    ),
]


def line_colour(line: str) -> str:
    if "WRONG" in line or "Error" in line or "Incorrect" in line:
        return "#f48771"
    if "CORRECT" in line or "correct" in line or "passed" in line:
        return "#89d185"
    if "=====" in line or line.startswith("ACCURACY") or line.startswith("Assignment No."):
        return "#e5c07b"
    if line.startswith("[evaluate]") or line.startswith("[model]") or line.startswith("[data]"):
        return "#75beff"
    if line.startswith("  top-1") or "Demo accuracy" in line:
        return "#dcdcaa"
    return "#d4d4d4"


def collect_lines(log_name: str, start_marker: str | None) -> list[str]:
    path = LOG_DIR / log_name
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if start_marker:
        for index, line in enumerate(lines):
            if line.startswith(start_marker):
                lines = lines[index:]
                break

    lines = [line.rstrip() for line in lines if line.strip()]
    if len(lines) > MAX_LINES:
        omitted = len(lines) - MAX_LINES
        lines = [f"... ({omitted} earlier lines omitted) ..."] + lines[-MAX_LINES:]
    return lines


def render(log_name: str, output_name: str, command: str, start_marker) -> Path:
    lines = collect_lines(log_name, start_marker)
    top_margin = 0.42
    bottom_margin = 0.18
    height = top_margin + len(lines) * LINE_HEIGHT + bottom_margin

    fig = plt.figure(figsize=(FIG_WIDTH, height), dpi=140)
    fig.patch.set_facecolor("#1e1e1e")
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.add_patch(
        Rectangle((0, 1 - top_margin / height), 1, top_margin / height,
                  color="#3c3c3c", transform=axis.transAxes)
    )
    axis.text(
        0.008,
        1 - top_margin / height / 2,
        f"shravani@fedora: ~/NLP/Assignment 4 $ {command}",
        family="DejaVu Sans Mono",
        fontsize=FONT_SIZE,
        color="#eeeeee",
        va="center",
    )

    for index, line in enumerate(lines):
        y = (height - top_margin - index * LINE_HEIGHT - 0.14) / height
        axis.text(
            0.008,
            y,
            line,
            family="DejaVu Sans Mono",
            fontsize=FONT_SIZE,
            color=line_colour(line),
            va="top",
            transform=axis.transAxes,
        )

    SHOTS_DIR.mkdir(exist_ok=True)
    path = SHOTS_DIR / output_name
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    for log_name, output_name, command, start_marker in CAPTURES:
        path = render(log_name, output_name, command, start_marker)
        print(f"saved {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
