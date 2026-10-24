#!/usr/bin/env python3
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# PATH CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "bbc-text.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
VISUALIZATION_DIR = ARTIFACT_DIR / "visualization"

MODEL_PATH = ARTIFACT_DIR / "news_classifier.joblib"
TEST_DATA_PATH = ARTIFACT_DIR / "test_data.csv"
METADATA_PATH = ARTIFACT_DIR / "training_metadata.json"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_preprocess_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place bbc-text.csv inside the dataset folder."
        )

    df = pd.read_csv(path)

    original_rows = len(df)

    # Standardize column names.
    df.columns = [str(column).strip().lower() for column in df.columns]

    required_columns = {"text", "category"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[["text", "category"]].copy()

    # Remove missing and empty values.
    df = df.dropna(subset=["text", "category"])

    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    df = df[(df["text"] != "") & (df["category"] != "")]

    rows_after_missing_cleanup = len(df)

    # Remove articles whose identical text has conflicting labels.
    label_counts_per_text = df.groupby("text")["category"].nunique()

    ambiguous_texts = label_counts_per_text[label_counts_per_text > 1].index

    ambiguous_rows_removed = int(df["text"].isin(ambiguous_texts).sum())

    if len(ambiguous_texts) > 0:
        df = df[~df["text"].isin(ambiguous_texts)]

    # Remove duplicate articles.
    rows_before_deduplication = len(df)

    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    duplicate_rows_removed = rows_before_deduplication - len(df)

    # Validate the dataset for stratified splitting.
    if df["category"].nunique() < 2:
        raise ValueError("At least two categories are required to train a classifier.")

    if df["category"].value_counts().min() < 2:
        raise ValueError("Every category must have at least two rows for a stratified split.")

    cleaning_stats = {
        "original_rows": int(original_rows),
        "rows_after_missing_or_empty_cleanup": int(rows_after_missing_cleanup),
        "ambiguous_text_rows_removed": int(ambiguous_rows_removed),
        "duplicate_text_rows_removed": int(duplicate_rows_removed),
        "cleaned_rows": int(len(df)),
    }

    return df, cleaning_stats


def save_cleaning_summary(cleaning_stats: dict, output_dir: Path) -> None:
    labels = [
        "Original Rows",
        "After Missing /\nEmpty Cleanup",
        "Ambiguous Rows\nRemoved",
        "Duplicate Rows\nRemoved",
        "Cleaned Rows",
    ]

    values = [
        cleaning_stats["original_rows"],
        cleaning_stats["rows_after_missing_or_empty_cleanup"],
        cleaning_stats["ambiguous_text_rows_removed"],
        cleaning_stats["duplicate_text_rows_removed"],
        cleaning_stats["cleaned_rows"],
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(labels, values)

    ax.set_title("BBC News Dataset Cleaning Summary", fontsize=16, fontweight="bold")
    ax.set_ylabel("Number of Rows")
    ax.set_xlabel("Cleaning Stage")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.bar_label(bars, fmt="%d", padding=4)

    ax.set_ylim(0, max(values, default=1) * 1.15)

    plt.tight_layout()

    output_path = output_dir / "cleaning_summary.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved cleaning summary chart: {output_path}")


def save_category_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    category_counts = df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(category_counts.index, category_counts.values)

    ax.set_title("BBC News Category Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("News Category")
    ax.set_ylabel("Number of Articles")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.bar_label(bars, fmt="%d", padding=4)

    ax.set_ylim(0, category_counts.max() * 1.15)
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    output_path = output_dir / "category_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved category distribution chart: {output_path}")


def save_category_pie_chart(df: pd.DataFrame, output_dir: Path) -> None:
    category_counts = df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
    )

    ax.set_title("BBC News Category Proportions", fontsize=16, fontweight="bold")

    plt.tight_layout()

    output_path = output_dir / "category_distribution_pie.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved category pie chart: {output_path}")


def save_dataset_overview(cleaning_stats: dict, df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot - cleaning summary
    labels = [
        "Original",
        "After Cleanup",
        "Ambiguous\nRemoved",
        "Duplicates\nRemoved",
        "Cleaned",
    ]
    values = [
        cleaning_stats["original_rows"],
        cleaning_stats["rows_after_missing_or_empty_cleanup"],
        cleaning_stats["ambiguous_text_rows_removed"],
        cleaning_stats["duplicate_text_rows_removed"],
        cleaning_stats["cleaned_rows"],
    ]
    bars1 = axes[0].bar(labels, values)
    axes[0].set_title("Dataset Cleaning Overview", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Rows")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].bar_label(bars1, fmt="%d", padding=4)
    axes[0].set_ylim(0, max(values, default=1) * 1.15)

    # Right plot - category distribution
    category_counts = df["category"].value_counts()
    bars2 = axes[1].bar(category_counts.index, category_counts.values)
    axes[1].set_title("Cleaned Dataset Category Distribution", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Articles")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].bar_label(bars2, fmt="%d", padding=4)
    axes[1].set_ylim(0, category_counts.max() * 1.15)
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()

    output_path = output_dir / "dataset_overview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved dataset overview chart: {output_path}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing dataset...")
    df, cleaning_stats = load_and_preprocess_dataset(DATASET_PATH)

    print("\n=== Dataset Cleaning Statistics ===")
    for key, value in cleaning_stats.items():
        print(f"{key}: {value}")

    print("\n=== Cleaned Dataset Category Distribution ===")
    print(df["category"].value_counts())

    # Save visualizations
    print("\nGenerating visualizations...")
    save_cleaning_summary(cleaning_stats, VISUALIZATION_DIR)
    save_category_distribution(df, VISUALIZATION_DIR)
    save_category_pie_chart(df, VISUALIZATION_DIR)
    save_dataset_overview(cleaning_stats, df, VISUALIZATION_DIR)

    # Train-test split
    print("\nPerforming stratified train-test split (80:20, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Build pipeline
    print("\nBuilding and training pipeline (TF-IDF + MultinomialNB)...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=20000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("nb", MultinomialNB(alpha=0.5)),
        ]
    )

    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    # Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    test_df = pd.DataFrame({"text": X_test, "category": y_test})
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved test data: {TEST_DATA_PATH}")

    metadata = {
        "dataset_path": str(DATASET_PATH),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cleaning_stats": cleaning_stats,
        "category_distribution_cleaned": df["category"].value_counts().to_dict(),
        "train_size": int(len(X_train)),
        "test_size_actual": int(len(X_test)),
        "pipeline": {
            "tfidf": {
                "stop_words": "english",
                "max_features": 20000,
                "ngram_range": [1, 2],
                "sublinear_tf": True,
            },
            "classifier": {"name": "MultinomialNB", "alpha": 0.5},
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved metadata: {METADATA_PATH}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
