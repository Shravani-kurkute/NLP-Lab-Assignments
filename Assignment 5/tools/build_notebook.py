#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Laboratory - Assignment 05
Build and execute NLP_Assignment05.ipynb (covers Part A and Part B).

Run after PART_A.py and PART_B.py:
    python3 tools/build_notebook.py
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(HERE, "NLP_Assignment05.ipynb")

CELLS = [
    ("md", """# NLP Laboratory — Assignment 05
## Sentiment Analysis and Mini Project (Research Paper Study & Implementation)

**Student:** Shravani Balasaheb Kurkute &nbsp;|&nbsp; **Roll No.:** 421129 &nbsp;|&nbsp; **PRN:** 22311926
**Class / Division:** BTechA &nbsp;|&nbsp; **Academic Year:** 2026-2027 &nbsp;|&nbsp; **Faculty:** Dr. Kapil N. Vhatkar

This notebook demonstrates **both parts** of the assignment end-to-end:

* **Part A** — an Amazon product-review sentiment classifier (cleaning, negation handling,
  stemming, TF-IDF, Naive Bayes / Logistic Regression / Linear SVM / Random Forest,
  cross-validation, tuning, full evaluation and error analysis).
* **Part B** — a reproducible subset of *Pang, Lee & Vaithyanathan (2002), "Thumbs up?"*
  (NOT_ negation tagging, unigram presence/frequency, bigrams, NB / MaxEnt / SVM,
  3-fold CV over 700/700 subsamples).

All numbers printed below are produced by executing the code in this notebook and by the
scripts `PART_A.py` and `PART_B.py`; **no results are hard-coded**. Published values from
the 2002 paper are always shown separately and labelled as *published*.
"""),
    ("code", """import json, os, sys, warnings
warnings.filterwarnings("ignore")

HERE = os.path.abspath("")
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "visualizations")

from IPython.display import Image, display, Markdown
print("Working directory:", HERE)
"""),

    # ------------------------------------------------------------------ Part A
    ("md", """---
# Part A — Sentiment Analysis on Amazon Product Reviews

## A.1 Dataset loading and inspection

The reference assignment uses the Kaggle *Amazon Product Reviews Dataset* (`7817_1.csv`).
If that exact file is present in `data/` it is used; otherwise the pipeline falls back to a
schema-compatible genuine Datafiniti Amazon review CSV. The source actually used is printed
below and recorded in `results/part_a_metrics.json`."""),
    ("code", """import PART_A

path, source, raw_df = PART_A.resolve_dataset(allow_download=False)
print("Dataset file   :", os.path.relpath(path, HERE))
print("Dataset source :", source)
print("Raw shape      :", raw_df.shape)
print("\\nRating counts:")
print(raw_df["reviews.rating"].value_counts(dropna=False).sort_index().to_string())
print("\\nMissing review text:", int(raw_df["reviews.text"].isna().sum()))
print("\\nDuplicate review text:", int(raw_df.duplicated(subset=["reviews.text"]).sum()))
raw_df[["reviews.title", "reviews.text", "reviews.rating"]].head(3)
"""),

    ("md", """## A.2 Preprocessing: cleaning, tokenisation, negation handling, stop-words, stemming

`PART_A.preprocess` lower-cases text, repairs `n't`, expands emoticons, removes URLs/HTML,
marks negation scope with a `neg_` prefix (the Pang et al. `NOT_` idea), drops stop-words
(keeping negations and polarity connectives) and applies Porter stemming."""),
    ("code", """examples = [
    "Love this tablet, the screen is GORGEOUS and the battery lasts all day!!! :)",
    "The sound is not bad at all, quite good for the price.",
    "I don't like it... it isn't working and it's overpriced <br/> www.example.com",
]
for s in examples:
    print("RAW  :", s)
    print("CLEAN:", PART_A.preprocess(s))
    print()
"""),

    ("md", """## A.3 Live demonstration: TF-IDF + Logistic Regression

Instead of trusting saved numbers, the next cell builds the representation and trains the
model again on the real data (the full pipeline is in `PART_A.py`)."""),
    ("code", """import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df = raw_df[["reviews.title", "reviews.text", "reviews.rating"]].copy().dropna(subset=["reviews.text", "reviews.rating"])
df = df[(df["reviews.rating"] <= 2) | (df["reviews.rating"] >= 4)]
df = df.drop_duplicates(subset=["reviews.text"])
df["review"] = df["reviews.title"].fillna("").astype(str) + " . " + df["reviews.text"].astype(str)
df["clean"] = df["review"].apply(PART_A.preprocess)
df = df[df["clean"].str.len() > 0]
df["label"] = (df["reviews.rating"] >= 4).astype(int)
print("Cleaned reviews:", len(df), " positive share: %.1f%%" % (100 * df["label"].mean()))

Xtr, Xte, ytr, yte = train_test_split(df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
pipe = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000, sublinear_tf=True)
Xtr_v = pipe.fit_transform(Xtr); Xte_v = pipe.transform(Xte)
clf = LogisticRegression(C=5, max_iter=1000, class_weight="balanced").fit(Xtr_v, ytr)
pred = clf.predict(Xte_v); prob = clf.predict_proba(Xte_v)[:, 1]
print("Live test accuracy : %.4f" % accuracy_score(yte, pred))
print("Live test macro-F1 : %.4f" % f1_score(yte, pred, average="macro"))
print("Live test ROC-AUC  : %.4f" % roc_auc_score(yte, prob))
"""),

    ("md", """## A.4 Full saved results (from `PART_A.py`)

The model comparison, tuning, held-out evaluation and repeated cross-validation results
computed by the full script are shown below."""),
    ("code", """a = json.load(open(os.path.join(RESULTS, "part_a_metrics.json")))
print("Best pipeline params :", a["best_params"])
print("Best CV macro-F1     : %.4f" % a["cv_f1_best"])
print("\\nModel comparison (5-fold CV macro-F1):")
for r in a["cv_comparison"]:
    print("  %-22s %.4f +/- %.4f" % (r["model"], r["cv_f1_mean"], r["cv_f1_std"]))
print("\\nHeld-out test metrics:")
for k, v in a["test_metrics"].items():
    print("  %-16s %.4f" % (k, v))
print("\\nRepeated 5-fold CV on all cleaned reviews:")
for k, v in a["cv_full_data"].items():
    print("  %-14s %.4f" % (k, v))
print("\\nConfusion matrix [rows=true, cols=pred] (neg, pos):")
print(np.array(a["confusion_matrix"]))
print("\\nTop positive features:", a["top_pos"])
print("Top negative features:", a["top_neg"])
"""),
    ("code", """print(a["classification_report"])"""),

    ("md", """## A.5 Visualisations"""),
    ("code", """for f in ["part_a_class_distribution.png", "part_a_model_comparison.png",
          "part_a_confusion_matrix.png", "part_a_top_features.png"]:
    display(Markdown(f"**{f}**"))
    display(Image(filename=os.path.join(FIGS, f)))
"""),

    ("md", """## A.6 Error analysis"""),
    ("code", """import pandas as pd
miss = pd.read_csv(os.path.join(RESULTS, "part_a_misclassified.csv"))
print("Total errors: %d (%.2f%% of the test set)" % (len(miss), 100 * len(miss) / a["split"]["n_test"]))
print("Errors containing negation:", int(miss["has_negation"].sum()) if len(miss) else 0)
print("Errors in mixed reviews   :", int(miss["is_mixed"].sum()) if len(miss) else 0)
pd.set_option("display.max_colwidth", 140)
miss[["review", "actual", "predicted", "p_positive"]].head(8)
"""),
    ("code", """print("Demonstration predictions:")
display(pd.read_csv(os.path.join(RESULTS, "part_a_demo_predictions.csv")))
"""),

    # ------------------------------------------------------------------ Part B
    ("md", """---
# Part B — Mini Project: Pang, Lee & Vaithyanathan (2002)

**Paper:** *Thumbs up? Sentiment Classification using Machine Learning Techniques*,
EMNLP 2002, pp. 79-86. https://aclanthology.org/W02-1011/

**Corpus:** NLTK `movie_reviews` (Pang & Lee polarity data; 1,000 positive + 1,000 negative).
The paper used an earlier 1,400-review release, so our randomised 700/700 subsamples are not
the identical documents."""),
    ("code", """import PART_B
docs, labels = PART_B.load_corpus()
print("Loaded", len(docs), "reviews:", int(labels.sum()), "positive,",
      int((1 - labels).sum()), "negative")
print("\\nExample of NOT_ negation tagging:")
sample = "This movie is not good . It is never funny but the acting is not bad at all ."
print("RAW  :", sample)
print("TAGGED:", PART_B.tag_negation(sample))
"""),
    ("md", """## B.1 Live 3-fold cross-validation demonstration (seed 0, unigram presence)

We rebuild one of the paper's cells here: unigram-presence features with Naive Bayes,
Maximum Entropy and a Linear SVM on a reproducible 700/700 subsample."""),
    ("code", """import numpy as np
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(labels == 1)[0], 700, replace=False),
                      rng.choice(np.where(labels == 0)[0], 700, replace=False)])
tagged = [PART_B.tag_negation(docs[i]) for i in idx]
y0 = labels[idx]
vocab = PART_B.build_vocab(tagged, (1, 1), min_count=4)
print("Unigram vocabulary size (min_count=4):", len(vocab))
for name, mk in {"Naive Bayes": lambda: __import__("sklearn.naive_bayes", fromlist=["MultinomialNB"]).MultinomialNB(alpha=1.0),
                 "Max Entropy": lambda: __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(C=10, max_iter=2000),
                 "SVM": lambda: __import__("sklearn.svm", fromlist=["LinearSVC"]).LinearSVC(C=0.1, max_iter=10000)}.items():
    acc = PART_B.run_cv(tagged, y0, vocab, (1, 1), True, mk(), 0)
    print("  %-12s 3-fold CV accuracy (seed 0): %.1f%%" % (name, acc))
"""),
    ("md", """## B.2 Our results vs the published paper (Table 3)

**Ours** = mean of 3-fold CV accuracy (%) over 5 random 700/700 subsamples.
**Published** = values reported in Pang et al. (2002), Table 3 — *not produced by us*."""),
    ("code", """b = json.load(open(os.path.join(RESULTS, "part_b_metrics.json")))
import pandas as pd
ours = pd.read_csv(os.path.join(RESULTS, "part_b_ours_mean.csv"))
paper = pd.read_csv(os.path.join(RESULTS, "part_b_paper_published.csv"))
diff = pd.read_csv(os.path.join(RESULTS, "part_b_difference.csv"))
print("OUR RESULTS (mean 3-fold CV accuracy, %):"); display(ours)
print("PUBLISHED (Pang et al. 2002, Table 3):"); display(paper)
print("DIFFERENCE (ours - paper), percentage points:"); display(diff)
print("Human-1 baseline: ours %.1f%% (ties %.1f%%) vs paper 58%% (ties 75%%)"
      % (b["human_baseline"]["acc"], b["human_baseline"]["ties"]))
print("Extra - full 2,000 reviews, unigram presence, 10-fold CV:")
for k, v in b["full_corpus_10fold"].items():
    print("  %-12s %.1f%%" % (k, v))
print("\\nAblation (SVM, seed 0):")
for k, v in b["ablation"].items():
    print("  %-30s %.1f%%" % (k, v))
print("\\nTop positive features:", b["top_pos"])
print("Top negative features:", b["top_neg"])
"""),
    ("code", """for f in ["part_b_comparison.png", "part_b_ablation.png"]:
    display(Markdown(f"**{f}**"))
    display(Image(filename=os.path.join(FIGS, f)))
"""),

    ("md", """---
# Observations

* **Part A.** TF-IDF + Logistic Regression is the strongest classical model. The dataset is
  heavily imbalanced (Amazon reviews skew positive), so accuracy is misleading; macro-F1, MCC
  and the negative-class recall tell the real story. Negation handling helps, but mixed reviews
  ("nice but the remote is awful") and short demo sentences remain hard for a bag-of-words model.
* **Part B.** Unigram **presence** features match or beat unigram **frequency**, bigrams add
  little, and the ranking of feature sets follows the paper even though the exact reviews,
  tokeniser and library differ. NOT_ negation tagging gives only a marginal gain in this setup.
* Published values are reported side-by-side with ours but never substituted for experimental
  output.

## Limitations
* The exact Kaggle file `7817_1.csv` was not available locally; a genuine schema-compatible
  Datafiniti CSV is used and clearly labelled.
* The paper's POS / adjective / word-position features are out of scope.
* The paper's exact 1,400 documents and CV folds are unknown, so results differ by 1-8 points.
"""),
]


def main() -> None:
    nb = new_notebook()
    nb.cells = [
        (new_markdown_cell(src) if kind == "md" else new_code_cell(src))
        for kind, src in CELLS
    ]
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}

    with open(NOTEBOOK, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("Wrote", NOTEBOOK)

    # Execute the notebook in place.
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = nbf.read(fh, as_version=4)
    ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(NOTEBOOK, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("Executed", NOTEBOOK)


if __name__ == "__main__":
    main()
