#!/usr/bin/env python3
"""Create the charts used in the notebook and the assignment report.

Reads the CSV/JSON files produced by country_prediction.py --evaluate and
writes PNG files into visualizations/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualizations"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "top1": "#1f77b4",
    "top5": "#ff7f0e",
    "correct": "#2ca02c",
    "incorrect": "#d62728",
    "neutral": "#7f7f7f",
}


def load_summary() -> dict:
    return json.loads((RESULTS_DIR / "accuracy_summary.json").read_text())


def load_predictions(dataset: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"predictions_{dataset}_{mode}.csv")


def chart_accuracy_comparison(summary: dict) -> Path:
    labels = [
        "Test-15\ncountry-filtered",
        "Test-15\nfull vocabulary",
        "1,000\ncountry-filtered",
        "1,000\nfull vocabulary",
    ]
    top1 = [
        summary["results"]["test15"]["country_filtered"]["top1_accuracy_pct"],
        summary["results"]["test15"]["full_vocabulary"]["top1_accuracy_pct"],
        summary["results"]["large1000"]["country_filtered"]["top1_accuracy_pct"],
        summary["results"]["large1000"]["full_vocabulary"]["top1_accuracy_pct"],
    ]
    top5 = [
        summary["results"]["test15"]["country_filtered"]["top5_accuracy_pct"],
        summary["results"]["test15"]["full_vocabulary"]["top5_accuracy_pct"],
        summary["results"]["large1000"]["country_filtered"]["top5_accuracy_pct"],
        summary["results"]["large1000"]["full_vocabulary"]["top5_accuracy_pct"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars1 = axis.bar(x - width / 2, top1, width, label="Top-1 accuracy", color=COLORS["top1"])
    bars2 = axis.bar(x + width / 2, top5, width, label="Top-5 accuracy", color=COLORS["top5"])

    for bars in (bars1, bars2):
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 108)
    axis.set_xticks(x, labels)
    axis.set_title("Top-1 and Top-5 accuracy: country filtering vs full vocabulary")
    axis.legend(loc="lower right")
    axis.axhline(100, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
    fig.tight_layout()

    path = VIS_DIR / "accuracy_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_similarity_distribution() -> Path:
    rows = load_predictions("large1000", "country_filtered")
    correct = rows.loc[rows["correct"] == "Correct", "predicted_score"]
    incorrect = rows.loc[rows["correct"] == "Incorrect", "predicted_score"]

    fig, axis = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.35, 1.0, 26)
    axis.hist(
        correct,
        bins=bins,
        alpha=0.75,
        label=f"Correct predictions (n={len(correct)})",
        color=COLORS["correct"],
    )
    axis.hist(
        incorrect,
        bins=bins,
        alpha=0.75,
        label=f"Incorrect predictions (n={len(incorrect)})",
        color=COLORS["incorrect"],
    )
    axis.axvline(
        correct.mean(),
        color=COLORS["correct"],
        linestyle="--",
        linewidth=1.5,
        label=f"mean (correct) = {correct.mean():.3f}",
    )
    axis.axvline(
        incorrect.mean(),
        color=COLORS["incorrect"],
        linestyle="--",
        linewidth=1.5,
        label=f"mean (incorrect) = {incorrect.mean():.3f}",
    )
    axis.set_xlabel("Cosine similarity of the top-ranked candidate")
    axis.set_ylabel("Number of analogies")
    axis.set_title("Top-1 cosine similarity for 1,000 analogies (country-filtered)")
    axis.legend(fontsize=9)
    fig.tight_layout()

    path = VIS_DIR / "similarity_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_error_countries(summary: dict) -> Path:
    breakdown = summary.get("error_breakdown", [])[:10]
    labels = [item["country"].title() for item in breakdown][::-1]
    values = [item["errors"] for item in breakdown][::-1]

    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(labels, values, color=COLORS["incorrect"], alpha=0.85)
    for bar, value in zip(bars, values):
        axis.annotate(
            str(value),
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    axis.set_xlabel("Incorrect top-1 predictions")
    axis.set_title("Actual countries that were most often predicted wrongly\n"
                   "(1,000 analogies, country-filtered)")
    fig.tight_layout()

    path = VIS_DIR / "errors_by_country.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_rank_distribution() -> Path:
    rows = load_predictions("large1000", "country_filtered")
    ranks = rows["actual_rank"].fillna(99).astype(int)
    buckets = {
        "rank 1": int((ranks == 1).sum()),
        "rank 2": int((ranks == 2).sum()),
        "rank 3": int((ranks == 3).sum()),
        "rank 4": int((ranks == 4).sum()),
        "rank 5": int((ranks == 5).sum()),
        "rank > 5": int((ranks > 5).sum()),
    }

    fig, axis = plt.subplots(figsize=(9, 5))
    colors = [COLORS["top1"], COLORS["top1"], COLORS["top1"], COLORS["top1"],
              COLORS["top1"], COLORS["incorrect"]]
    bars = axis.bar(buckets.keys(), buckets.values(), color=colors, alpha=0.9)
    for bar, value in zip(bars, buckets.values()):
        axis.annotate(
            f"{value}\n({value / len(rows):.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel("Number of analogies")
    axis.set_ylim(0, max(buckets.values()) * 1.22)
    axis.set_title("Rank of the correct country in the filtered candidate list "
                   "(1,000 analogies)")
    fig.tight_layout()

    path = VIS_DIR / "rank_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_test15_confidence() -> Path:
    rows = load_predictions("test15", "country_filtered")
    labels = [
        f"{i}. {r.capital1} : {r.country1} :: {r.capital2}"
        for i, r in enumerate(rows.itertuples(), 1)
    ]
    scores = rows["predicted_score"].to_numpy()
    correct = rows["correct"].eq("Correct").to_numpy()
    margins = rows["margin"].fillna(0).to_numpy()

    y = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(9, 6.5))
    colors = [COLORS["correct"] if ok else COLORS["incorrect"] for ok in correct]
    axis.barh(y, scores, color=colors, alpha=0.85)
    axis.set_yticks(y, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Cosine similarity of the top-ranked candidate")
    for index, (score, margin) in enumerate(zip(scores, margins)):
        axis.annotate(
            f"{score:.3f}  (margin {margin:.3f})",
            (score, index),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axis.set_title("Top-1 confidence and runner-up margin on the 15 test analogies\n"
                   "green = correct, red = wrong")
    fig.tight_layout()

    path = VIS_DIR / "test15_confidence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_analogy_pca() -> Path:
    """Project the required analogy into two dimensions with PCA."""
    from sklearn.decomposition import PCA

    from country_prediction import load_embeddings

    vectors, _ = load_embeddings()
    normed = np.asarray(vectors.get_normed_vectors(), dtype=np.float32)

    words = [
        "athens", "greece", "baghdad", "iraq",
        "paris", "france", "berlin", "germany",
        "vienna", "austria",
    ]
    indices = [vectors.key_to_index[word] for word in words]
    word_vectors = normed[indices]

    query = normed[vectors.key_to_index["greece"]]
    query = query - normed[vectors.key_to_index["athens"]]
    query = query + normed[vectors.key_to_index["baghdad"]]
    query = query / np.linalg.norm(query)
    query_score = float(query @ normed[vectors.key_to_index["iraq"]])

    projection = PCA(n_components=2)
    projection.fit(np.vstack([word_vectors, query]))
    points = projection.transform(word_vectors)
    query_point = projection.transform(query.reshape(1, -1))[0]

    fig, axis = plt.subplots(figsize=(9.5, 6.5))
    for start, end in ((0, 1), (2, 3), (4, 5)):
        axis.annotate(
            "",
            xy=points[end],
            xytext=points[start],
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.6),
        )
    axis.annotate(
        "",
        xy=query_point,
        xytext=points[1],
        arrowprops=dict(arrowstyle="->", color=COLORS["top5"], lw=2.2),
    )

    for word, point in zip(words, points):
        is_country = word in {
            "greece", "iraq", "france", "germany", "austria",
        }
        axis.scatter(
            *point,
            s=90,
            color=COLORS["top1"] if is_country else COLORS["neutral"],
            zorder=3,
        )
        axis.annotate(
            word.title(),
            point,
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=10,
        )
    axis.scatter(
        *query_point,
        s=260,
        marker="*",
        color=COLORS["top5"],
        zorder=4,
        label="q = Greece - Athens + Baghdad",
    )
    axis.annotate(
        f"q ({query_score:.3f} cosine to Iraq)",
        query_point,
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=10,
        color=COLORS["top5"],
    )

    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.set_title("2-D PCA of selected GloVe vectors and the analogy vector")
    axis.legend(loc="best", fontsize=10)
    fig.tight_layout()

    path = VIS_DIR / "analogy_vector_pca.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    VIS_DIR.mkdir(exist_ok=True)
    summary = load_summary()

    charts = [
        chart_accuracy_comparison(summary),
        chart_similarity_distribution(),
        chart_error_countries(summary),
        chart_rank_distribution(),
        chart_test15_confidence(),
        chart_analogy_pca(),
    ]
    for path in charts:
        print(f"saved {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    sys.exit(main())
