# NLP Laboratory Assignment 05 — Sentiment Analysis and Mini Project

End-to-end **Amazon product-review sentiment analysis** (Part A) and a reproducible
implementation of **Pang, Lee & Vaithyanathan (2002), "Thumbs up? Sentiment Classification
using Machine Learning Techniques"** (Part B).

**Student:** Shravani Balasaheb Kurkute | **Roll No.:** 421129 | **PRN:** 22311926
**Class:** BTechA | **Academic Year:** 2026-2027 | **Faculty:** Dr. Kapil N. Vhatkar

---

## 1. What the project does

### Part A — Sentiment analysis (`PART_A.py`)
1. Loads and inspects the Amazon product reviews dataset (missing values, duplicates, ratings).
2. Cleans text, tokenises, marks negation scope (`neg_` prefix), removes stop-words (keeping
   negations) and applies Porter stemming.
3. Extracts **TF-IDF** features.
4. Compares **Naive Bayes, Logistic Regression, Linear SVM and Random Forest** with stratified
   5-fold cross-validation (macro-F1).
5. Tunes the best model (Logistic Regression) with `GridSearchCV`.
6. Evaluates on a held-out test set: accuracy, precision, recall, F1, ROC-AUC, MCC, confusion
   matrix and classification report; plus repeated 5-fold CV on all cleaned reviews.
7. Performs error analysis, prints demonstration predictions and saves all artifacts.

### Part B — Mini project (`PART_B.py`)
1. **NOT_** negation tagging (words between a negation and the next punctuation).
2. Unigram **frequency** and unigram **presence** features.
3. Unigram+bigram presence and bigram-presence features; top-N unigram presence.
4. **Naive Bayes, Maximum Entropy (multinomial logistic regression), Linear SVM**.
5. Stratified **3-fold cross-validation** over reproducible **700/700** subsamples with
   **5 random seeds** (0–4).
6. Comparison with the published values of Pang et al. (2002), Table 3 — published values are
   always labelled and never substituted for our results.
7. Human word-list baseline, a **negation-tagging ablation** and informative plots.

No results are hard-coded anywhere; every number in the outputs/report is produced by running
the scripts.

---

## 2. Requirements and installation (Fedora)

* Fedora Linux, Python 3.10+ (developed with Python 3.13.9)
* ~200 MB free disk (the downloaded review CSV) plus a few MB for outputs

```bash
cd "/home/shravanikurkute/BTech/4th Year/NLP/Assignment 5"

# (optional but recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Part B needs the NLTK movie_reviews corpus (one-time, ~4 MB download)
python3 -c "import nltk; nltk.download('movie_reviews')"
```

System tools used to build the report/PDF on Fedora:

```bash
sudo dnf install libreoffice      # provides soffice for DOCX -> PDF
```

---

## 3. Dataset

**Reference dataset:** `Amazon Product Reviews Dataset`, file `7817_1.csv`
(Kaggle: <https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset>).

`PART_A.py` resolves the dataset in this order:

1. `data/7817_1.csv` — the **exact reference file**, if you have placed it there.
2. `data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` — a genuine, openly
   accessible **Datafiniti Amazon product-review CSV with the same schema**
   (`name`, `reviews.title`, `reviews.text`, `reviews.rating`). This is committed/used as the
   fallback when the Kaggle file is unavailable.
3. If neither exists, the script **downloads** the Datafiniti fallback automatically (or exits
   with instructions if run with `--no-download`).

> **Why a fallback?** The Kaggle file requires an authenticated account and was not available
> locally. The fallback is a real dataset from the same source family — no data was fabricated.
> To reproduce with the exact reference file, download `7817_1.csv` from Kaggle and place it in
> `data/`; the script will then prefer it. The dataset actually used is printed and recorded in
> `results/part_a_metrics.json` under `dataset`.

**Part B corpus:** NLTK `movie_reviews` (Pang & Lee polarity data, 1,000 positive + 1,000
negative). The paper used an earlier 1,400-review release, so the randomised 700/700 subsamples
are not the identical documents.

---

## 4. Running the assignment (Fedora terminal commands)

```bash
cd "/home/shravanikurkute/BTech/4th Year/NLP/Assignment 5"

# --- Part A: Amazon sentiment pipeline -------------------------------------
python3 PART_A.py                      # full run (~15 s)
python3 PART_A.py --max-rows 1500      # optional quick run on a subsample

# --- Part B: Pang et al. (2002) replication --------------------------------
python3 PART_B.py                      # full run (~60 s)

# --- Excel workbook of metrics and experiment comparisons -------------------
python3 tools/generate_workbook.py     # -> evaluation_results.xlsx

# --- Executed notebook (covers Part A and Part B) --------------------------
python3 tools/build_notebook.py        # -> NLP_Assignment05.ipynb (built + executed)

# --- Academic report (DOCX + PDF) ------------------------------------------
python3 tools/generate_report.py       # -> report/*.docx and report/*.pdf
```

Notes:

* Run `PART_A.py` and `PART_B.py` **before** the three `tools/` generators, because those
  read the JSON/CSV outputs in `results/`.
* `tools/generate_report.py` converts DOCX → PDF using LibreOffice (`libreoffice --headless`);
  install LibreOffice or export the PDF manually if it is not available.

---

## 5. Results produced on this machine

### Part A (tuned Logistic Regression, held-out test set)

| Metric | Value |
|---|---|
| Accuracy | 0.9869 |
| Precision (positive) | 0.9903 |
| Recall (positive) | 0.9964 |
| F1 (positive) | 0.9933 |
| Precision / Recall / F1 (macro) | 0.8880 / 0.7877 / 0.8300 |
| Precision / Recall / F1 (negative) | 0.7857 / 0.5789 / 0.6667 |
| ROC-AUC | 0.9861 |
| MCC | 0.6682 |

Repeated 5-fold CV on all cleaned reviews (3 repeats): accuracy 0.9826, macro-F1 0.7702,
negative recall 0.4702, negative precision 0.6602. The data is heavily imbalanced, so macro-F1,
MCC and negative-class recall are reported alongside accuracy.

### Part B (mean 3-fold CV accuracy over 5 random 700/700 subsamples)

| Feature set | Naive Bayes | Max Entropy | SVM |
|---|---|---|---|
| (1) unigrams, frequency | 80.3 | 81.9 | 81.1 |
| (2) unigrams, presence | 81.4 | 83.4 | 82.6 |
| (3) unigrams+bigrams, presence | 83.8 | 84.2 | 83.6 |
| (4) bigrams, presence | 82.2 | 79.3 | 78.4 |
| (7) top 2633 unigrams, presence | 81.6 | 81.3 | 79.4 |

Human-1 word-list baseline: 58.3% accuracy / 72.2% ties (paper: 58% / 75%).
Full 2,000-review corpus, unigram presence, 10-fold CV: NB 82.8%, MaxEnt 86.0%, SVM 85.1%.
Negation ablation (SVM, seed 0): 82.8% without vs 82.9% with NOT_ tagging.

Published values (Pang et al. 2002, Table 3) are stored separately in
`results/part_b_paper_published.csv` and shown side-by-side in the report; the ranking of feature
sets matches the paper even though the exact reviews, tokeniser and library differ.

---

## 6. Project structure

```
Assignment 5/
├── PART_A.py                                   Amazon sentiment pipeline
├── PART_B.py                                   Pang et al. (2002) replication
├── NLP_Assignment05.ipynb                      executed notebook (both parts)
├── evaluation_results.xlsx                     Excel workbook (15 sheets)
├── requirements.txt
├── README.md
├── SCREENSHOT_GUIDE.md                         how to capture terminal/output screenshots
├── data/
│   ├── 7817_1.csv                              (optional) exact Kaggle reference file
│   └── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv   genuine fallback
├── results/
│   ├── part_a_metrics.json / part_a_test_metrics.csv
│   ├── part_a_cv_comparison.csv / part_a_classification_report.txt
│   ├── part_a_misclassified.csv / part_a_demo_predictions.csv
│   ├── part_a_sentiment_model.joblib
│   ├── part_b_metrics.json / part_b_results_full.csv
│   ├── part_b_ours_mean.csv / part_b_ours_std.csv
│   ├── part_b_paper_published.csv / part_b_difference.csv
├── visualizations/                             six figures (Part A + Part B)
├── report/
│   ├── 421129_Shravani Balasaheb Kurkute_NLP_Assignment05.docx
│   └── 421129_Shravani Balasaheb Kurkute_NLP_Assignment05.pdf
├── tools/
│   ├── generate_workbook.py
│   ├── build_notebook.py
│   └── generate_report.py
└── (the two reference PDFs supplied with the assignment)
```

---

## 7. Reproducibility and limitations

* **Exact reference dataset not used.** The Kaggle `7817_1.csv` was unavailable; a genuine
  schema-compatible Datafiniti CSV is used and clearly labelled in the console, JSON, workbook
  and report.
* **Class imbalance.** The Amazon data is strongly positive-skewed, so negative-class recall is
  low; macro-F1 / MCC are the fairer metrics. Class weights are used.
* **Part B differences.** Different corpus version, tokeniser, vocabulary cut-offs and
  scikit-learn defaults explain 1–8 point gaps versus the published numbers. POS, adjective-only
  and word-position features were not implemented.
* **Seeds and folds** are fixed (`SEED = 42`; Part B seeds 0–4), so runs are deterministic.

---

## 8. Troubleshooting

* **`ModuleNotFoundError`** — run `python3 -m pip install -r requirements.txt`.
* **NLTK `LookupError: movie_reviews`** — run
  `python3 -c "import nltk; nltk.download('movie_reviews')"`.
* **No dataset found** — either let `PART_A.py` download the fallback, or place the exact
  `7817_1.csv` in `data/`.
* **PDF not generated** — install LibreOffice (`sudo dnf install libreoffice`) or export the
  DOCX manually.
* **Run from the correct folder** — the scripts use paths relative to the `Assignment 5`
  directory; run them from there.
