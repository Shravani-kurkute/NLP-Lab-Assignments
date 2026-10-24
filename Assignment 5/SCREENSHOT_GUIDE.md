# Screenshot Guide — NLP Assignment 05

This guide lists the **genuine** terminal/output screenshots to capture for submission.
Do **not** reuse the reference student's screenshots — run the code on your own machine and
capture your own output. Every command below is safe to run from the `Assignment 5` folder.

---

## 0. One-time setup

```bash
cd "/home/shravanikurkute/BTech/4th Year/NLP/Assignment 5"
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -c "import nltk; nltk.download('movie_reviews')"
```

Screenshot tools on Fedora (any one is fine):

```bash
sudo dnf install flameshot            # region + full-screen capture (recommended)
# alternatives already present on many Fedora systems:
gnome-screenshot -a                   # select an area
gnome-screenshot -f shot.png          # full screen
```

Tip: run each command in a clean terminal, keep the window wide, and capture the **prompt +
command + complete output**. Save the files into `screenshots/`.

---

## 1. Environment proof

```bash
python3 --version
python3 -m pip show scikit-learn nltk pandas | grep -E "Name|Version"
```

Save as `screenshots/01_environment.png`.

---

## 2. Part A — dataset inspection and preprocessing

```bash
python3 PART_A.py | tee screenshots/part_a_console.txt
```

Capture the top of the output (sections **1. DATASET INSPECTION** and **2. PREPROCESSING**,
which show the raw shape, rating counts, missing/duplicate counts, cleaned class distribution and
the RAW → CLEAN examples). Save as `screenshots/02_part_a_dataset_preprocessing.png`.

---

## 3. Part A — model comparison, tuning and evaluation

From the same console output (or re-run), capture:

* **3. MODEL COMPARISON** — naive-bayes / logistic-regression / linear-svm / random-forest
  macro-F1 values.
* **Grid-search** best parameters and CV macro-F1.
* **4. TEST-SET EVALUATION** — all metrics, confusion matrix and classification report.

Save as `screenshots/03_part_a_models_eval.png`.
Also capture **5. ERROR ANALYSIS** and **6. DEMO PREDICTIONS** as
`screenshots/04_part_a_errors_demo.png`.

---

## 4. Part A — visualizations

```bash
ls visualizations/
# open each figure, e.g.:
xdg-open visualizations/part_a_confusion_matrix.png
xdg-open visualizations/part_a_top_features.png
```

Capture:
* `screenshots/05_part_a_class_distribution.png`
* `screenshots/06_part_a_model_comparison.png`
* `screenshots/07_part_a_confusion_matrix.png`
* `screenshots/08_part_a_top_features.png`

---

## 5. Part B — replication output

```bash
python3 PART_B.py | tee screenshots/part_b_console.txt
```

Capture:
* Corpus summary and the **NOT_ negation tagging** example.
* **OUR RESULTS** table, **PUBLISHED (Pang et al. 2002)** table and the **DIFFERENCE** table.
* Human-1 baseline, full-corpus 10-fold results and the **ablation**.

Save as `screenshots/09_part_b_results.png` and `screenshots/10_part_b_ablation.png`.

Part B figures:

```bash
xdg-open visualizations/part_b_comparison.png
xdg-open visualizations/part_b_ablation.png
```

Save as `screenshots/11_part_b_comparison.png` and `screenshots/12_part_b_ablation.png`.

---

## 6. Deliverables proof (workbook, notebook, report)

```bash
python3 tools/generate_workbook.py
python3 tools/build_notebook.py
python3 tools/generate_report.py
ls -R results visualizations report evaluation_results.xlsx
```

* Open `evaluation_results.xlsx` (LibreOffice Calc) and capture a sheet showing the metrics —
  `screenshots/13_workbook.png`.
* Open `NLP_Assignment05.ipynb` in Jupyter and capture executed cells with outputs —
  `screenshots/14_notebook.png`.
* Open the report PDF and capture the title page and the results tables —
  `screenshots/15_report.png`.

---

## 7. Submission checklist

- [ ] All screenshots show your own prompt/hostname and are taken on your machine.
- [ ] No reference-student name, roll number, PRN or screenshots are reused.
- [ ] Console captures include the command that produced them.
- [ ] Figures match the values in `results/*.json` and the report.
- [ ] Screenshots saved under `Assignment 5/screenshots/`.
