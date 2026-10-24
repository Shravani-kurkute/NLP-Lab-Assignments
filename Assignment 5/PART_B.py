#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Laboratory - Assignment 05 - PART B
=======================================
Mini Project: Study & reproducible implementation of

    Pang, B., Lee, L., & Vaithyanathan, S. (2002).
    "Thumbs up? Sentiment Classification using Machine Learning Techniques."
    EMNLP 2002, pp. 79-86.  https://aclanthology.org/W02-1011/

Student : Shravani Balasaheb Kurkute
Roll No.: 421129          PRN: 22311926
Class   : BTechA          Academic Year: 2026-2027
Faculty : Dr. Kapil N. Vhatkar

Implemented components
----------------------
  * NOT_ negation tagging (words between a negation and the next punctuation)
  * Unigram frequency and unigram presence features
  * Unigram + bigram presence and bigram-presence features
  * Top-N unigram presence features
  * Naive Bayes, Maximum Entropy (multinomial logistic regression) and Linear SVM
  * Stratified 3-fold cross-validation over reproducible 700/700 subsamples,
    repeated across several random seeds
  * Human word-list baseline (paper Table 1 "Human 1")
  * Negation-tagging ablation
  * Comparison against the values published in Pang et al. (2002), Table 3
    (published values are clearly labelled as such and are never mixed with our
    own measured accuracies)

Dataset: the NLTK ``movie_reviews`` corpus (Pang & Lee polarity data v1.0,
1,000 positive + 1,000 negative).  It matches the paper's corpus family.  The
paper used an earlier 1,400-review release, so our randomised 700/700 subsamples
are not the identical documents.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
N_PER_CLASS = 700
SEEDS = [0, 1, 2, 3, 4]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "visualizations")
NLTK_DATA = os.path.join(HERE, "data", "movie_reviews")   # optional local copy

NEGATIONS = {
    "not", "no", "never", "nor", "cannot", "cant", "wont", "dont", "doesnt", "didnt",
    "isnt", "wasnt", "arent", "werent", "wouldnt", "couldnt", "shouldnt", "hasnt",
    "havent", "hadnt", "neither", "nothing", "nobody", "none",
}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.!?;,:]")

# Published results, Pang et al. (2002) Table 3 (3-fold CV, 700 pos / 700 neg).
# NOTE: these are *published* numbers reproduced from the paper for comparison;
# they are NOT produced by this script.
PAPER_TABLE_3 = {
    "(1) unigrams, frequency":        {"Naive Bayes": 78.7, "Max Entropy": np.nan, "SVM": 72.8},
    "(2) unigrams, presence":         {"Naive Bayes": 81.0, "Max Entropy": 80.4, "SVM": 82.9},
    "(3) unigrams+bigrams, presence": {"Naive Bayes": 80.6, "Max Entropy": 80.8, "SVM": 82.7},
    "(4) bigrams, presence":          {"Naive Bayes": 77.3, "Max Entropy": 77.4, "SVM": 77.1},
    "(7) top 2633 unigrams, presence": {"Naive Bayes": 80.3, "Max Entropy": 81.0, "SVM": 81.4},
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_corpus():
    """Load NLTK movie_reviews (pos/neg) -> (docs, labels)."""
    from nltk.corpus import movie_reviews

    try:
        fileids = movie_reviews.fileids("pos")
        assert movie_reviews.fileids("neg")
    except LookupError:
        import nltk
        print("[info] downloading NLTK movie_reviews corpus ...")
        nltk.download("movie_reviews")

    docs, labels = [], []
    for lab, folder in ((1, "pos"), (0, "neg")):
        for fid in sorted(movie_reviews.fileids(folder)):
            docs.append(movie_reviews.raw(fid))
            labels.append(lab)
    return docs, np.array(labels)


def load_corpus_from_disk():
    """Fallback: load a local data/movie_reviews/{pos,neg}/*.txt corpus."""
    docs, labels = [], []
    for lab, folder in ((1, "pos"), (0, "neg")):
        for fp in sorted(glob.glob(os.path.join(NLTK_DATA, folder, "*.txt"))):
            with open(fp, encoding="latin-1") as fh:
                docs.append(fh.read())
            labels.append(lab)
    return docs, np.array(labels)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def tag_negation(text: str) -> str:
    """Pang et al.: prefix NOT_ to every word from a negation to the next punctuation."""
    out, neg = [], False
    for tok in TOKEN_RE.findall(text.lower()):
        tok = tok.replace("'", "")
        if tok in ".!?;,:":
            neg = False
            continue
        if tok in NEGATIONS or tok.endswith("nt"):
            neg = True
            out.append(tok)
            continue
        out.append("NOT_" + tok if neg else tok)
    return " ".join(out)


def build_vocab(docs, ngram, min_count=None, top_k=None):
    """Select features by corpus frequency, like the paper."""
    cv = CountVectorizer(ngram_range=ngram, token_pattern=r"\S+", lowercase=False)
    X = cv.fit_transform(docs)
    total = np.asarray(X.sum(axis=0)).ravel()
    names = np.array(cv.get_feature_names_out())
    order = np.argsort(-total)
    if min_count is not None:
        order = order[total[order] >= min_count]
    if top_k is not None:
        order = order[:top_k]
    return list(names[order])


def run_cv(docs, y, vocab, ngram, binary, clf, seed):
    """Mean accuracy (%) of 3-fold stratified CV."""
    cv = CountVectorizer(ngram_range=ngram, token_pattern=r"\S+", lowercase=False,
                         vocabulary=vocab, binary=binary)
    X = cv.transform(docs)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=skf, scoring="accuracy").mean() * 100


def human_baseline(raw_docs, y):
    """Paper Table 1 'Human 1': 5 positive + 5 negative hand-picked words."""
    pos_w = {"dazzling", "brilliant", "phenomenal", "excellent", "fantastic"}
    neg_w = {"suck", "terrible", "awful", "unwatchable", "hideous"}
    correct, ties = 0.0, 0
    for d, lab in zip(raw_docs, y):
        toks = re.findall(r"[a-z]+", d.lower())
        p, n = sum(t in pos_w for t in toks), sum(t in neg_w for t in toks)
        if p == n:
            ties += 1
            correct += 0.5
        else:
            correct += int((p > n) == bool(lab))
    return correct / len(y) * 100, ties / len(y) * 100


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 74)
    print("NLP ASSIGNMENT 05 - PART B : PANG, LEE & VAITHYANATHAN (2002) REPLICATION")
    print("Student: Shravani Kurkute | Roll No: 421129 | PRN: 22311926")
    print("=" * 74)

    try:
        raw, y_all = load_corpus()
    except Exception as exc:  # pragma: no cover - offline fallback
        print(f"[warn] NLTK corpus unavailable ({exc}); trying {NLTK_DATA}")
        raw, y_all = load_corpus_from_disk()
    if not raw:
        sys.exit("[ERROR] movie_reviews corpus not found. Run: python3 -c \"import nltk; nltk.download('movie_reviews')\"")

    print(f"Loaded {len(raw)} reviews ({int(y_all.sum())} pos / {int((1 - y_all).sum())} neg)")
    tagged_all = [tag_negation(d) for d in raw]

    feature_sets = {
        "(1) unigrams, frequency":        dict(ngram=(1, 1), binary=False, min_count=4, top_k=None),
        "(2) unigrams, presence":         dict(ngram=(1, 1), binary=True,  min_count=4, top_k=None),
        "(3) unigrams+bigrams, presence": dict(ngram=(1, 2), binary=True,  min_count=None, top_k=None),
        "(4) bigrams, presence":          dict(ngram=(2, 2), binary=True,  min_count=None, top_k=16165),
        "(7) top 2633 unigrams, presence": dict(ngram=(1, 1), binary=True, min_count=None, top_k=2633),
    }
    classifiers = {
        "Naive Bayes": lambda: MultinomialNB(alpha=1.0),              # add-1 smoothing
        "Max Entropy": lambda: LogisticRegression(C=10, max_iter=2000, random_state=0),
        "SVM":         lambda: LinearSVC(C=0.1, max_iter=10000, random_state=0),
    }

    rows, hb = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pos = rng.choice(np.where(y_all == 1)[0], N_PER_CLASS, replace=False)
        neg = rng.choice(np.where(y_all == 0)[0], N_PER_CLASS, replace=False)
        idx = np.concatenate([pos, neg])
        docs = [tagged_all[i] for i in idx]
        y = y_all[idx]
        hb.append(human_baseline([raw[i] for i in idx], y))
        for fname, cfg in feature_sets.items():
            if cfg["ngram"] == (1, 2):
                vocab = build_vocab(docs, (1, 1), min_count=4) + build_vocab(docs, (2, 2), top_k=16165)
            else:
                vocab = build_vocab(docs, cfg["ngram"], cfg["min_count"], cfg["top_k"])
            for cname, mk in classifiers.items():
                acc = run_cv(docs, y, vocab, cfg["ngram"], cfg["binary"], mk(), seed)
                rows.append(dict(seed=seed, features=fname, n_features=len(vocab),
                                 classifier=cname, accuracy=acc))
        print(f"seed {seed} done")

    # ---- our results: mean/std over the 5 subsamples --------------------
    df = pd.DataFrame(rows)
    summary = (df.groupby(["features", "classifier"])
                 .agg(n_features=("n_features", "first"),
                      acc_mean=("accuracy", "mean"),
                      acc_std=("accuracy", "std"))
                 .reset_index())
    pivot = summary.pivot(index="features", columns="classifier", values="acc_mean").round(1)
    pivot_std = summary.pivot(index="features", columns="classifier", values="acc_std").round(1)
    nf = summary.groupby("features")["n_features"].first()

    order = list(feature_sets.keys())
    pivot = pivot.reindex(order)
    pivot_std = pivot_std.reindex(order)
    pivot.insert(0, "n_features", nf.reindex(order))

    paper_df = pd.DataFrame(PAPER_TABLE_3).T[["Naive Bayes", "Max Entropy", "SVM"]]
    paper_df = paper_df.reindex(order)
    diff = (pivot[["Naive Bayes", "Max Entropy", "SVM"]] - paper_df).round(1)

    print("\n=== OUR RESULTS: mean 3-fold CV accuracy (%) over 5 random 700/700 subsamples ===")
    print(pivot.to_string())
    print("\n=== std-dev across subsamples (percentage points) ===")
    print(pivot_std.to_string())
    print("\n=== PUBLISHED: Pang et al. (2002), Table 3 (NOT our output) ===")
    print(paper_df.to_string())
    print("\n=== DIFFERENCE (ours - paper), percentage points ===")
    print(diff.to_string())

    hb_acc = float(np.mean([h[0] for h in hb]))
    hb_tie = float(np.mean([h[1] for h in hb]))
    print(f"\nHuman-1 word-list baseline: accuracy={hb_acc:.1f}%  ties={hb_tie:.1f}%  "
          f"(paper: 58% acc / 75% ties)")

    # ---- extra: full 2000-review corpus, unigram presence, 10-fold CV ----
    vocab = build_vocab(tagged_all, (1, 1), min_count=4)
    cvz = CountVectorizer(token_pattern=r"\S+", lowercase=False, vocabulary=vocab, binary=True)
    X = cvz.transform(tagged_all)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    acc_full, cm_full = {}, None
    for cname, mk in classifiers.items():
        pred = cross_val_predict(mk(), X, y_all, cv=skf)
        acc_full[cname] = float(accuracy_score(y_all, pred) * 100)
        if cname == "SVM":
            cm_full = confusion_matrix(y_all, pred)
    print("\n=== Extra: full 2,000 reviews, unigram presence, 10-fold CV accuracy (%) ===")
    print({k: round(v, 1) for k, v in acc_full.items()})
    print("SVM confusion matrix [neg,pos]:\n", cm_full)

    # ---- ablation: effect of NOT_ negation tagging (seed 0, SVM) --------
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y_all == 1)[0], N_PER_CLASS, replace=False),
                          rng.choice(np.where(y_all == 0)[0], N_PER_CLASS, replace=False)])
    y0 = y_all[idx]
    plain = [" ".join(TOKEN_RE.findall(raw[i].lower())) for i in idx]
    ablation = {}
    for name, docs in (("without negation tagging", plain),
                       ("with NOT_ negation tagging", [tagged_all[i] for i in idx])):
        v = build_vocab(docs, (1, 1), min_count=4)
        ablation[name] = run_cv(docs, y0, v, (1, 1), True, LinearSVC(C=0.1, max_iter=10000), 0)
    print("\n=== Ablation (unigram presence, SVM, seed 0) ===")
    for k, v in ablation.items():
        print(f"  {k:30s}: {v:.1f}%")

    # ---- most informative features (SVM on the seed-0 1400-doc sample) --
    v = build_vocab([tagged_all[i] for i in idx], (1, 1), min_count=4)
    cvz0 = CountVectorizer(token_pattern=r"\S+", lowercase=False, vocabulary=v, binary=True)
    Xs = cvz0.transform([tagged_all[i] for i in idx])
    svm = LinearSVC(C=0.1, max_iter=10000).fit(Xs, y0)
    names = np.array(cvz0.get_feature_names_out())
    top_pos = list(names[np.argsort(svm.coef_[0])[-12:]][::-1])
    top_neg = list(names[np.argsort(svm.coef_[0])[:12]])
    print("\nTop positive:", top_pos)
    print("Top negative:", top_neg)

    # ---- save -----------------------------------------------------------
    summary.to_csv(os.path.join(RESULTS_DIR, "part_b_results_full.csv"), index=False)
    pivot.to_csv(os.path.join(RESULTS_DIR, "part_b_ours_mean.csv"))
    pivot_std.to_csv(os.path.join(RESULTS_DIR, "part_b_ours_std.csv"))
    paper_df.to_csv(os.path.join(RESULTS_DIR, "part_b_paper_published.csv"))
    diff.to_csv(os.path.join(RESULTS_DIR, "part_b_difference.csv"))

    with open(os.path.join(RESULTS_DIR, "part_b_metrics.json"), "w") as fh:
        json.dump({
            "student": {"name": "Shravani Balasaheb Kurkute", "roll_no": "421129", "prn": "22311926"},
            "corpus": {"name": "NLTK movie_reviews", "n_docs": len(raw),
                       "n_pos": int(y_all.sum()), "n_neg": int((1 - y_all).sum())},
            "protocol": {"n_per_class": N_PER_CLASS, "seeds": SEEDS, "folds": 3},
            "ours_mean": pivot.reset_index().to_dict("records"),
            "ours_std": pivot_std.reset_index().to_dict("records"),
            "paper": paper_df.reset_index().to_dict("records"),
            "paper_values_source": "Pang et al. (2002), Table 3 - published values, not our output",
            "difference": diff.reset_index().to_dict("records"),
            "human_baseline": {"acc": hb_acc, "ties": hb_tie,
                               "paper": {"acc": 58.0, "ties": 75.0}},
            "full_corpus_10fold": acc_full,
            "ablation": ablation,
            "top_pos": top_pos, "top_neg": top_neg,
            "svm_full_cm": cm_full.tolist(),
        }, fh, indent=2, default=float)

    # ---- plot: paper vs ours -------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    w = 0.13
    x = np.arange(len(pivot))
    for i, c in enumerate(["Naive Bayes", "Max Entropy", "SVM"]):
        ax.bar(x + (i - 1.5) * w, pivot[c], w, label=f"{c} (ours)")
        ax.bar(x + (i + 1.5) * w, paper_df.loc[pivot.index, c], w,
               label=f"{c} (paper)", alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split(",")[0] + "\n" + s.split(",")[-1] for s in pivot.index], fontsize=8)
    ax.set_ylim(60, 90); ax.set_ylabel("3-fold CV accuracy (%)")
    ax.legend(ncol=3, fontsize=7)
    ax.set_title("Pang, Lee & Vaithyanathan (2002): paper vs. our replication")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "part_b_comparison.png"), dpi=150)
    plt.close()

    # ---- plot: ablation ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(list(ablation.keys()), list(ablation.values()), color=["#D9534F", "#4C9F70"])
    ax.set_ylim(70, 86); ax.set_ylabel("3-fold CV accuracy (%)")
    ax.set_title("Ablation: effect of NOT_ negation tagging (SVM, seed 0)")
    plt.xticks(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "part_b_ablation.png"), dpi=150)
    plt.close()

    print("\nSaved results/part_b_*.csv/.json and visualizations.")


if __name__ == "__main__":
    main()
