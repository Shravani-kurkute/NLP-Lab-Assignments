# NLP Laboratory Assignment 04 - Country Prediction using Word Embeddings

Predict a country from a capital-city analogy with pre-trained GloVe vectors.

```
Athens : Greece :: Baghdad : ?
                -> Iraq
```

**Student:** Shravani Balasaheb Kurkute | **Roll No.:** 421129 | **PRN:** 22311926
**Class:** BTechA | **Academic Year:** 2026-2027 | **Faculty:** Dr. Kapil N. Vhatkar

---

## 1. What the project does

1. Loads the pre-trained **glove-wiki-gigaword-100** model (400,000 words, 100 dimensions).
2. Builds the analogy vector `vector(Country 1) - vector(Capital 1) + vector(Capital 2)`.
3. Ranks candidates by **cosine similarity** (inputs removed from the results).
4. Predicts the country either from a curated list of 131 countries
   (country-filtered) or from the full 400,000-word vocabulary (for comparison).
5. Reproducibly generates datasets (seeds 42 and 2026), evaluates 15 and 1,000
   analogies, saves predictions, error analysis and a JSON summary.

## 2. Requirements and installation

* Fedora Linux, Python 3.10+ (developed with Python 3.13.9)
* About 420 MB of free disk space (128 MB GloVe model + results)

```bash
cd "/home/shravanikurkute/BTech/4th Year/NLP/Assignment 4"
python3 -m pip install -r requirements.txt
```

The first model load downloads `glove-wiki-gigaword-100` (~128 MB) into
`~/gensim-data`; later runs reuse the cached file automatically. To use a local
copy instead, pass `--model-path /path/to/glove-wiki-gigaword-100.gz`
(a gensim `.kv` file or a `.txt`/`.bin` word2vec file also works).

## 3. Running the assignment

```bash
# 1. Generate the datasets (needs the model; ~78 valid pairs, 131 candidates)
python3 BuildDatasets.py

# 2. Predict a single analogy (accepts --model-path too)
python3 country_prediction.py --words athens greece baghdad

# 3. Required example + the 15-example test set
python3 country_prediction.py --demo

# 4. Full evaluation: 15 and 1,000 analogies, both search modes
python3 country_prediction.py --evaluate

# 5. Interactive mode (no arguments)
python3 country_prediction.py
```

Notes:

* `BuildDatasets.py` must be run once before `--demo`, `--evaluate` or
  `--words`, because those commands read `datasets/candidate_countries.csv`.
* Everything is case-insensitive. Out-of-vocabulary words raise a clear error
  instead of returning a wrong prediction.

## 4. Tests

```bash
python3 -m pytest tests -q
```

`tests/test_datasets.py` checks the CSV structure and reproducibility,
`tests/test_predictor.py` checks the vector maths and ranking logic, and
`tests/test_integration.py` checks the real model (required example, test set,
missing words).

## 5. Regenerating the report, notebook, workbook and charts

```bash
python3 tools/generate_visualizations.py     # visualizations/*.png
python3 tools/generate_workbook.py           # results/Assignment4_Results.xlsx
python3 tools/render_console_captures.py     # screenshots/01..05_*.png
python3 tools/build_notebook.py              # builds + executes the notebook
python3 tools/generate_report.py             # report/*.docx and report/*.pdf
```

`tools/build_notebook.py` and `tools/generate_report.py` need the evaluation
outputs, so run `country_prediction.py --evaluate` first. `generate_report.py`
converts the DOCX to PDF with LibreOffice (`soffice`); remove that step if
LibreOffice is not installed.

## 6. Results produced on this machine

| Dataset | Search mode | Total | Correct | Top-1 | Top-5 |
|---|---|---|---|---|---|
| Test (15) | country-filtered | 15 | 14 | 93.33% | 100.00% |
| Test (15) | full vocabulary | 15 | 13 | 86.67% | 100.00% |
| Evaluation (1,000) | country-filtered | 1,000 | 944 | 94.40% | 98.90% |
| Evaluation (1,000) | full vocabulary | 1,000 | 922 | 92.20% | 97.70% |

The required example `athens : greece :: baghdad` predicts **iraq** with cosine
similarity 0.790. Country filtering improves top-1 accuracy by **+6.66**
percentage points on the test set and **+2.20** points on the 1,000-analogy set.

**Comparison with the reference report.** The reference reports 14/15 (93.33%)
and 95.0% top-1 / 98.9% top-5. This project reproduces the dataset counts
(78 pairs, 131 candidates), the 14/15 test accuracy and the 98.90% top-5
accuracy exactly. Its 1,000-analogy set uses a different random seed (2026
instead of the reference seed), so the individual analogies and the top-1 value
(94.40%) differ slightly; the values were not tuned to match the reference.

## 7. Project structure

```
Assignment 4/
├── BuildDatasets.py                     dataset generation (validate + CSV)
├── country_prediction.py                predictor + CLI + evaluation
├── requirements.txt
├── README.md
├── Country_Prediction_Word_Embeddings.ipynb   executed notebook
├── data/
│   ├── capital_country_pairs.csv        seed list (78 pairs)
│   └── additional_countries.csv         extra candidate countries (53)
├── datasets/
│   ├── capitals_countries.csv           validated pairs
│   ├── candidate_countries.csv          131 candidate countries
│   ├── test_analogies.csv               15 test analogies
│   ├── eval_analogies_large.csv         1,000 evaluation analogies
│   └── dataset_metadata.json
├── results/
│   ├── predictions_*.csv                detailed predictions (4 files)
│   ├── error_analysis_*.csv             error tables (4 files)
│   ├── accuracy_summary.json            accuracy + metadata
│   ├── Assignment4_Results.xlsx         Excel workbook
│   └── logs/                            captured console output
├── visualizations/                      six presentation charts
├── screenshots/                         console captures + screenshot guide
├── tests/                               pytest unit + integration tests
├── tools/                               generators for notebook/report/figures
├── report/
│   ├── 421129_Shravani_Kurkute_NLP_Assignment04.docx
│   └── 421129_Shravani_Kurkute_NLP_Assignment04.pdf
└── (reference PDFs supplied with the assignment)
```

## 8. Acceptance checks

- Required example gives **Iraq**.
- Test set: 14/15 correct (93.33% top-1, 100% top-5).
- 1,000 analogies: 94.40% top-1, 98.90% top-5 (country-filtered).
- Country filtering beats the full-vocabulary baseline in every setting.
- `pytest tests -q` passes (25 tests).
- The notebook runs from top to bottom without errors.

## 9. Troubleshooting

* **`ModuleNotFoundError: gensim`** - install with
  `python3 -m pip install -r requirements.txt`.
* **Slow first run** - the 128 MB GloVe model is downloaded once.
* **`No module named country_prediction`** - run the commands from the
  `Assignment 4` folder (the scripts import each other by module name).
* **PDF not regenerated** - install LibreOffice or export the DOCX manually.
