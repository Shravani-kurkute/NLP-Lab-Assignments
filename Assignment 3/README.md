# NLP Laboratory Assignment 3 — POS Tagging with an HMM

Reproducible English part-of-speech tagging using a supervised **Hidden Markov Model**
with **Viterbi decoding**, trained on the NLTK Penn Treebank and evaluated with the
NLTK Universal tagset.

## Project directory

```
/home/shravanikurkute/BTech/4th Year/NLP/Assignment 3
```

## Quick start

```bash
cd "/home/shravanikurkute/BTech/4th Year/NLP/Assignment 3"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 pos_tagging_hmm.py
python3 -m pytest tests -v
```

`python3 pos_tagging_hmm.py` runs the complete pipeline and writes:

- `results.json` — all configuration, statistics, metrics, and predictions
- `evaluation_results.csv` — overall and per-tag metrics for the baseline and HMM
- `sample_tagged_output.csv` — gold vs. predicted tags for held-out test sentences
- `error_analysis.csv` — every HMM error token with its surrounding context
- `custom_predictions.csv` — token-level results for the 25 custom sentences
- `train_sentences.txt`, `test_sentences.txt`, `custom_test_sentences.txt`
- `visualizations/*.png` — five charts computed from the actual run
- `POS_Tagging_Results.xlsx` — nine formatted worksheets
- `POS_Tagging_Assignment_03_Report.docx` — editable laboratory report
- `outputs/terminal_output.txt` — console log of the run

## Method summary

| Item | Setting |
|---|---|
| Corpus | NLTK Penn Treebank (`treebank.tagged_sents()`) |
| Cleaning | Remove `-NONE-` trace tokens; drop exact duplicate sentences |
| Tagset | NLTK Universal tagset (12 tags; `.` = punctuation) |
| Split | 80:20 sentence-level, seed 42, zero sentence overlap |
| Model | Supervised first-order HMM |
| Transitions | Add-k smoothing, `k = 0.001`, START/END states, log space |
| Emissions | Maximum likelihood for observed words |
| OOV | Digit-shape prior + suffixes (1–4 chars) of rare words + tag prior |
| Decoder | Viterbi with backpointers, deterministic tie-breaking |
| Baseline | Most frequent tag per word; global most frequent tag for OOV |

## Actual results (seed 42)

| Metric | Baseline | HMM |
|---|---|---|
| Accuracy | 0.932594 | **0.953475** |
| Macro F1 | 0.856378 | **0.916128** |
| Weighted F1 | 0.931416 | **0.953004** |
| OOV accuracy | 0.536469 | **0.755965** |
| Known-token accuracy | 0.966359 | 0.970310 |

Training: 3,122 sentences / 75,349 tokens. Test: 781 sentences / 18,678 tokens
(1,467 OOV = 7.85%). HMM token errors on the test set: 869.

## Project structure

```
Assignment 3/
├── pos_tagging_hmm.py                     # main implementation (HMM, Viterbi, OOV, evaluation, charts)
├── generate_workbook.py                   # builds POS_Tagging_Results.xlsx
├── generate_report.py                     # builds the DOCX report
├── build_notebook.py                      # builds + executes the Jupyter notebook
├── requirements.txt
├── README.md
├── RUN_COMMANDS.txt
├── SCREENSHOT_GUIDE.md
├── .gitignore
├── results.json
├── evaluation_results.csv
├── sample_tagged_output.csv
├── error_analysis.csv
├── custom_predictions.csv
├── train_sentences.txt
├── test_sentences.txt
├── custom_test_sentences.txt
├── POS_Tagging_Assignment_03_Report.docx
├── POS_Tagging_Assignment_03.ipynb
├── POS_Tagging_Results.xlsx
├── outputs/
│   ├── terminal_output.txt
│   └── POS_Tagging_Assignment_03_executed.ipynb
├── visualizations/
│   ├── confusion_matrix.png
│   ├── per_tag_scores.png
│   ├── model_comparison.png
│   ├── oov_analysis.png
│   └── tag_distribution.png
├── tests/
│   └── test_pos_tagging.py
└── screenshots/                           # captured by the student (see SCREENSHOT_GUIDE.md)
```

## NLTK data

The pipeline needs `treebank`, `universal_tagset`, `punkt`, and `punkt_tab`.
If they are missing, it downloads them once and prints the remedial command
(`python3 -m nltk.downloader treebank universal_tagset punkt punkt_tab`)
if the download fails. It never substitutes fabricated training data.

## Tests

```bash
python3 -m pytest tests -v
```

Covers empty input, single/multiword sentences, output length, valid tags,
reproducible splits, Viterbi decoding, unknown words, numeric tokens, metric
calculation, baseline comparison, and output-file generation. Current status:
**24 passed, 0 failed**.

## Notebook

```bash
python3 build_notebook.py     # rebuild + execute
jupyter notebook POS_Tagging_Assignment_03.ipynb
```

## Opening the report and workbook

```bash
libreoffice --writer POS_Tagging_Assignment_03_Report.docx &
libreoffice --calc  POS_Tagging_Results.xlsx &
```
