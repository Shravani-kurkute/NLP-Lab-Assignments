#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Laboratory - Assignment 05 - PART A
=======================================
Sentiment Analysis on Amazon Product Reviews.

Student : Shravani Balasaheb Kurkute
Roll No.: 421129          PRN: 22311926
Class   : BTechA          Academic Year: 2026-2027
Faculty : Dr. Kapil N. Vhatkar

Pipeline
--------
  1. Dataset loading + inspection (missing values, duplicates, class balance)
  2. Cleaning, tokenisation, negation marking (neg_), stop-word removal, stemming
  3. TF-IDF feature extraction
  4. Model comparison (Naive Bayes, Logistic Regression, Linear SVM, Random Forest)
     with stratified 5-fold cross-validation
  5. Hyper-parameter tuning with GridSearchCV
  6. Full evaluation: accuracy, precision, recall, F1, ROC-AUC, MCC,
     confusion matrix and classification report
  7. Repeated stratified cross-validation on all cleaned reviews
  8. Error analysis, demonstration predictions and saved artifacts

Dataset
-------
The reference assignment uses the Kaggle *Amazon Product Reviews Dataset*
(https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset, file
``7817_1.csv``).  If that file is present in ``data/`` this script uses it.
Because Kaggle downloads require an authenticated account and the exact file was
not available locally, the script falls back to a genuine, openly accessible
Datafiniti Amazon product-review CSV that carries the same schema
(``name``, ``reviews.title``, ``reviews.text``, ``reviews.rating``).  The source
actually used is always printed and recorded in ``results/part_a_metrics.json``.
No results in this file are hard-coded.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.request

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import PorterStemmer

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "visualizations")

# Preferred (exact reference) dataset, then the documented fallback.
DATASET_CANDIDATES = [
    ("data/7817_1.csv", "Kaggle yasserh/amazon-product-reviews-dataset (exact reference file)"),
    (
        "data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
        "Datafiniti Amazon Consumer Reviews (schema-compatible fallback)",
    ),
]

# Public mirror of the genuine Datafiniti file (used only if nothing is local).
FALLBACK_URL = (
    "https://raw.githubusercontent.com/gyaneshpandey09/amazonreviews/"
    "master/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
)

# Reviews needed as positive (>=4 stars) / negative (<=2 stars) labels.
POSITIVE_MIN = 4
NEGATIVE_MAX = 2


# --------------------------------------------------------------------------- #
# Dataset resolution
# --------------------------------------------------------------------------- #
def _download(url: str, dest: str) -> bool:
    """Download ``url`` to ``dest`` with a small progress report."""
    try:
        print(f"[info] downloading fallback dataset ...\n       {url}")
        tmp = dest + ".part"
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
        print(f"[info] saved {os.path.getsize(dest) / 1e6:.1f} MB to {dest}")
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[warn] download failed: {exc}")
        return False


def resolve_dataset(allow_download: bool = True):
    """Return ``(path, description, df)`` for the first available dataset."""
    for rel, desc in DATASET_CANDIDATES:
        path = os.path.join(HERE, rel)
        if os.path.isfile(path):
            return path, desc, pd.read_csv(path)

    # Nothing local: try to fetch the genuine fallback.
    fallback_path = os.path.join(DATA_DIR, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
    if allow_download and _download(FALLBACK_URL, fallback_path):
        return fallback_path, DATASET_CANDIDATES[1][1], pd.read_csv(fallback_path)

    print(
        "\n[ERROR] No dataset found.\n"
        "  Obtain the exact reference dataset from Kaggle:\n"
        "    https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset\n"
        "  Download 7817_1.csv and place it at data/7817_1.csv, then re-run.\n"
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
STEMMER = PorterStemmer()
NEGATIONS = {
    "not", "no", "never", "nor", "cannot", "cant", "wont", "dont", "doesnt", "didnt",
    "isnt", "wasnt", "arent", "werent", "wouldnt", "couldnt", "shouldnt", "hasnt",
    "havent", "hadnt", "neither", "nothing", "nobody", "none", "hardly", "barely",
}
# Keep negations and polarity-bearing connectives, drop the rest of the stop words.
STOP = set(ENGLISH_STOP_WORDS) - NEGATIONS - {"but", "however", "against", "very", "too"}

EMO_MAP = {
    ":)": " emopos ", ":-)": " emopos ", ":d": " emopos ", "<3": " emopos ",
    "\U0001f44d": " emopos ", "\u263a": " emopos ", "\u2639": " emoneg ",
    ":(": " emoneg ", ":-(": " emoneg ", ":/": " emoneg ", "\U0001f44e": " emoneg ",
    "\U0001f621": " emoneg ", "\U0001f61e": " emoneg ",
}
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TAG_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
ELONG_RE = re.compile(r"(.)\1{2,}")            # coooool -> cool
TOKEN_RE = re.compile(r"[a-z]+|[.!?;,]")        # words + clause-boundary punctuation


def clean_text(text: str) -> str:
    """Normalise raw review text (without tokenising)."""
    text = html.unescape(str(text)).lower()
    for k, v in EMO_MAP.items():
        text = text.replace(k, v)
    text = URL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASH_RE.sub(r" \1 ", text)
    text = re.sub(r"n't\b", " not", text)                 # isn't -> is not
    text = re.sub(r"'(s|re|ve|ll|d|m)\b", "", text)       # drop clitics
    text = re.sub(r"['\u2019]", "", text)
    text = ELONG_RE.sub(r"\1", text)
    return text


def preprocess(text: str) -> str:
    """clean -> tokenise -> negation scope marking -> stop words -> stemming."""
    tokens = TOKEN_RE.findall(clean_text(text))
    out, negated = [], False
    for tok in tokens:
        if tok in ".!?;,":
            negated = False                    # negation scope ends at punctuation
            continue
        if tok in NEGATIONS:
            negated = True
            out.append("not")
            continue
        if tok in STOP or len(tok) < 2:
            continue
        stem = STEMMER.stem(tok)
        out.append(("neg_" + stem) if negated else stem)   # Pang et al. (2002) NOT_ trick
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 5 - Part A sentiment analysis")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional cap on cleaned reviews (for a quick run).")
    parser.add_argument("--no-download", action="store_true",
                        help="Never download; fail if no local dataset is present.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 74)
    print("NLP ASSIGNMENT 05 - PART A : AMAZON REVIEW SENTIMENT ANALYSIS")
    print("Student: Shravani Kurkute | Roll No: 421129 | PRN: 22311926")
    print("=" * 74)

    # ---------------------------------------------------------------- 1. LOAD
    path, source_desc, raw_df = resolve_dataset(allow_download=not args.no_download)
    print("\n" + "=" * 74 + "\n1. DATASET INSPECTION\n" + "=" * 74)
    print("Dataset file       :", os.path.relpath(path, HERE))
    print("Dataset source     :", source_desc)
    print("Raw shape          :", raw_df.shape)

    required = {"reviews.text", "reviews.rating"}
    if not required.issubset(raw_df.columns):
        sys.exit(f"[ERROR] dataset must contain columns {required}; got {list(raw_df.columns)}")

    print("Products (unique)  :", raw_df["name"].nunique() if "name" in raw_df else "n/a")
    print("Rating counts      :\n",
          raw_df["reviews.rating"].value_counts(dropna=False).sort_index().to_string())
    print("Missing review text:", int(raw_df["reviews.text"].isna().sum()))
    print("Missing rating     :", int(raw_df["reviews.rating"].isna().sum()))
    print("Duplicate review text:", int(raw_df.duplicated(subset=["reviews.text"]).sum()))

    cols = [c for c in ["reviews.title", "reviews.text", "reviews.rating", "name"] if c in raw_df]
    df = raw_df[cols].copy()
    n_raw = len(df)

    df = df.dropna(subset=["reviews.text", "reviews.rating"])
    n_missing = n_raw - len(df)

    df = df[(df["reviews.rating"] <= NEGATIVE_MAX) | (df["reviews.rating"] >= POSITIVE_MIN)]
    n_neutral = (n_raw - n_missing) - len(df)

    n_before = len(df)
    df = df.drop_duplicates(subset=["reviews.text"]).reset_index(drop=True)
    n_dups = n_before - len(df)

    title = df["reviews.title"] if "reviews.title" in df else pd.Series("", index=df.index)
    df["review"] = title.fillna("").astype(str) + " . " + df["reviews.text"].astype(str)
    df["sentiment"] = np.where(df["reviews.rating"] >= POSITIVE_MIN, "positive", "negative")

    print(f"\nRemoved {n_missing} rows with missing text/rating, {n_neutral} neutral rows, "
          f"{n_dups} duplicate reviews -> {len(df)} records remain")
    print("Class distribution :\n", df["sentiment"].value_counts().to_string())
    print(f"Imbalance: {df['sentiment'].value_counts(normalize=True)['positive']:.1%} positive")

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=SEED).reset_index(drop=True)
        print(f"[info] capped to {len(df)} reviews via --max-rows (seed {SEED})")

    # ------------------------------------------------------------ 2. PREPROCESS
    print("\n" + "=" * 74 + "\n2. PREPROCESSING\n" + "=" * 74)
    df["clean"] = df["review"].apply(preprocess)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    df["n_tokens"] = df["clean"].str.split().str.len()

    for i in range(min(3, len(df))):
        print(f"\nRAW   : {df['review'][i][:280]}")
        print(f"CLEAN : {df['clean'][i][:280]}")
    print(f"\nAvg tokens/review after cleaning: {df['n_tokens'].mean():.2f}")

    # ---------------------------------------------------------------- 3. SPLIT
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        df["clean"], df["label"], df["review"],
        test_size=0.20, stratify=df["label"], random_state=SEED,
    )
    print(f"\nTrain = {len(X_train)}   Test = {len(X_test)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ------------------------------------------------- 4. FEATURES + COMPARE
    def tfidf(ngram=(1, 1)):
        return TfidfVectorizer(ngram_range=ngram, min_df=2, max_df=0.95,
                               max_features=20000, sublinear_tf=True)

    candidates = {
        "Naive Bayes": ComplementNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000,
                                                  class_weight="balanced", random_state=SEED),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
    }

    print("\n" + "=" * 74)
    print("3. MODEL COMPARISON (5-fold stratified CV, TF-IDF unigrams)")
    print("=" * 74)
    cv_rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro")
        cv_rows.append((name, scores.mean(), scores.std()))
        print(f"{name:22s} macro-F1 = {scores.mean():.4f} +/- {scores.std():.4f}")
    cv_df = pd.DataFrame(cv_rows, columns=["model", "cv_f1_mean", "cv_f1_std"])
    cv_df.to_csv(os.path.join(RESULTS_DIR, "part_a_cv_comparison.csv"), index=False)

    # --------------------------------------------- 5. TUNE LOGISTIC REGRESSION
    print("\nGrid-search tuning Logistic Regression ...")
    pipe = Pipeline([("tfidf", tfidf()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                random_state=SEED))])
    grid = GridSearchCV(
        pipe,
        {"clf__C": [0.5, 1, 5, 10, 50],
         "tfidf__ngram_range": [(1, 1), (1, 2)],
         "tfidf__min_df": [2, 3]},
        cv=skf, scoring="f1_macro", n_jobs=1,   # avoid loky shutdown noise; fast enough
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params :", grid.best_params_)
    print("CV macro-F1 :", round(grid.best_score_, 4))

    # ------------------------------------------------------------- 6. EVALUATE
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_pos": precision_score(y_test, y_pred, zero_division=0),
        "recall_pos": recall_score(y_test, y_pred, zero_division=0),
        "f1_pos": f1_score(y_test, y_pred, zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_neg": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall_neg": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1_neg": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"], digits=4)

    print("\n" + "=" * 74 + "\n4. TEST-SET EVALUATION (tuned Logistic Regression)\n" + "=" * 74)
    for k, v in metrics.items():
        print(f"{k:16s}: {v:.4f}")
    print("\nConfusion matrix [rows=true, cols=pred] (neg, pos):\n", cm)
    print("\n" + report)

    pd.DataFrame([metrics]).T.rename(columns={0: "value"}).to_csv(
        os.path.join(RESULTS_DIR, "part_a_test_metrics.csv"))
    with open(os.path.join(RESULTS_DIR, "part_a_classification_report.txt"), "w") as fh:
        fh.write(report)

    # Repeated stratified CV on the full cleaned corpus (stable estimate).
    print("\nRepeated 5-fold stratified CV on all", len(df), "reviews (3 repeats):")
    rep_scores = {"accuracy": [], "f1_macro": [], "recall_neg": [], "precision_neg": []}
    for rep in range(3):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        pred_cv = cross_val_predict(clone(best), df["clean"], df["label"], cv=kf)
        rep_scores["accuracy"].append(accuracy_score(df["label"], pred_cv))
        rep_scores["f1_macro"].append(f1_score(df["label"], pred_cv, average="macro", zero_division=0))
        rep_scores["recall_neg"].append(recall_score(df["label"], pred_cv, pos_label=0, zero_division=0))
        rep_scores["precision_neg"].append(precision_score(df["label"], pred_cv, pos_label=0, zero_division=0))
    cv_full = {k: float(np.mean(v)) for k, v in rep_scores.items()}
    for k, v in cv_full.items():
        print(f"  {k:14s}: {v:.4f}")

    # ------------------------------------------------------------- 7. PLOTS
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "part_a_confusion_matrix.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    df["sentiment"].value_counts().plot(kind="bar", color=["#4C9F70", "#D9534F"])
    plt.title("Class distribution (Amazon reviews, cleaned)"); plt.ylabel("Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "part_a_class_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(cv_df["model"], cv_df["cv_f1_mean"], yerr=cv_df["cv_f1_std"],
            color="#4A7BC8", capsize=4)
    plt.ylim(max(0.0, cv_df["cv_f1_mean"].min() - 0.08), 1.0)
    plt.ylabel("5-fold CV macro-F1"); plt.xticks(rotation=20)
    plt.title("Model comparison (cross-validation)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "part_a_model_comparison.png"), dpi=150)
    plt.close()

    vec, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    top_pos, top_neg = np.argsort(coef)[-15:], np.argsort(coef)[:15]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].barh(names[top_pos], coef[top_pos], color="#4C9F70")
    ax[0].set_title("Top positive features")
    ax[1].barh(names[top_neg], coef[top_neg], color="#D9534F")
    ax[1].set_title("Top negative features")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "part_a_top_features.png"), dpi=150)
    plt.close()
    print("\nTop positive features:", list(names[top_pos][::-1][:10]))
    print("Top negative features:", list(names[top_neg][:10]))

    # -------------------------------------------------------- 8. ERROR ANALYSIS
    err = pd.DataFrame({"review": raw_test.values, "cleaned": X_test.values,
                        "actual": y_test.values, "predicted": y_pred,
                        "p_positive": y_prob.round(3)})
    err = err[err["actual"] != err["predicted"]].copy()
    err["error_type"] = np.where(err["actual"] == 1, "False Negative", "False Positive")
    err["has_negation"] = err["cleaned"].str.contains(r"\bnot\b|neg_", regex=True)
    err["is_mixed"] = err["review"].str.lower().str.contains(r"\bbut\b|however", regex=True)
    err.to_csv(os.path.join(RESULTS_DIR, "part_a_misclassified.csv"), index=False)

    print("\n" + "=" * 74 + "\n5. ERROR ANALYSIS\n" + "=" * 74)
    print("Total errors :", len(err), f"({len(err) / max(len(y_test),1):.2%} of test set)")
    print(err["error_type"].value_counts().to_string() if len(err) else "none")
    print("Errors containing negation :", int(err["has_negation"].sum()))
    print("Errors in mixed reviews    :", int(err["is_mixed"].sum()))
    conf_wrong = int(((err["p_positive"] > 0.85) | (err["p_positive"] < 0.15)).sum()) if len(err) else 0
    print("High-confidence errors (p<0.15 or >0.85):", conf_wrong)
    if len(err):
        print("Avg tokens of errors:", round(err["cleaned"].str.split().str.len().mean(), 1),
              "vs all reviews:", round(df["n_tokens"].mean(), 1))

    # ---------------------------------------------------------- 9. SAVE + DEMO
    joblib.dump(best, os.path.join(RESULTS_DIR, "part_a_sentiment_model.joblib"))
    demo = [
        "Love this tablet, the screen is gorgeous and the battery lasts all day!",
        "Terrible product. Stopped working after two weeks, total waste of money.",
        "The sound is not bad at all, quite good for the price.",
        "The speaker is not good and the Bluetooth keeps disconnecting.",
        "The design is nice but overall the remote is awful and I regret buying it.",
    ]
    print("\n" + "=" * 74 + "\n6. DEMO PREDICTIONS\n" + "=" * 74)
    demo_rows = []
    for s in demo:
        p = float(best.predict_proba([preprocess(s)])[0, 1])
        lab = "positive" if p >= 0.5 else "negative"
        demo_rows.append({"sentence": s, "prediction": lab, "p_positive": round(p, 3)})
        print(f"[{lab:8s} p={p:.3f}] {s}")
    pd.DataFrame(demo_rows).to_csv(os.path.join(RESULTS_DIR, "part_a_demo_predictions.csv"), index=False)

    summary = {
        "student": {"name": "Shravani Balasaheb Kurkute", "roll_no": "421129", "prn": "22311926"},
        "dataset": {"file": os.path.relpath(path, HERE), "source": source_desc,
                    "n_raw": int(n_raw), "n_clean": int(len(df))},
        "cleaning": {"missing_removed": int(n_missing), "neutral_removed": int(n_neutral),
                     "duplicates_removed": int(n_dups)},
        "split": {"n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "class_counts_clean": df["sentiment"].value_counts().to_dict(),
        "avg_tokens": round(float(df["n_tokens"].mean()), 2),
        "best_params": {k: (str(v) if not isinstance(v, (int, float, str)) else v)
                        for k, v in grid.best_params_.items()},
        "cv_f1_best": float(grid.best_score_),
        "cv_comparison": cv_df.to_dict("records"),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "cv_full_data": cv_full,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_errors": int(len(err)),
        "errors_negation": int(err["has_negation"].sum()),
        "errors_mixed": int(err["is_mixed"].sum()),
        "errors_confident_wrong": conf_wrong,
        "n_features": int(len(names)),
        "top_pos": [str(x) for x in names[top_pos][::-1][:10]],
        "top_neg": [str(x) for x in names[top_neg][:10]],
        "demo": demo_rows,
    }
    with open(os.path.join(RESULTS_DIR, "part_a_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\nSaved results/part_a_*.json/.csv, model and visualizations.")


if __name__ == "__main__":
    main()
