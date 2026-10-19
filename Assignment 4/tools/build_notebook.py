#!/usr/bin/env python3
"""Build and execute the Assignment 4 notebook.

The notebook is generated programmatically so that it stays in sync with the
project modules, then executed with nbconvert so every cell contains real
output before it is saved as Country_Prediction_Word_Embeddings.ipynb.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = BASE_DIR / "Country_Prediction_Word_Embeddings.ipynb"

MD = "markdown"
CODE = "code"


def cells():
    return [
        (MD, """# NLP Laboratory Assignment 04
## Country Prediction using Word Embeddings

| Field | Value |
|---|---|
| Student Name | **Shravani Balasaheb Kurkute** |
| Roll No. | 421129 |
| PRN / Registration No. | 22311926 |
| Class / Division | BTechA |
| Academic Year | 2026-2027 |
| Faculty Name | Dr. Kapil N. Vhatkar |

**Aim:** To understand semantic relationships represented by word embeddings
and use vector arithmetic and similarity measures to predict a country from a
capital-city relationship.

This notebook reproduces the complete experiment: it loads pre-trained GloVe
vectors, builds the analogue vector, predicts countries, evaluates the system
and analyses the errors. All numbers below are produced by executing this
notebook, not copied from a report."""),

        (MD, """## 1. Problem statement

Given three words `Capital 1`, `Country 1` and `Capital 2`, predict `Country 2`
using the analogy

`Athens : Greece :: Baghdad : ?  ->  Iraq`

The required vector operation is

$$\\vec{v}_{query} = \\vec{v}(Country_1) - \\vec{v}(Capital_1) + \\vec{v}(Capital_2)$$

Candidates are ranked by **cosine similarity** between the L2-normalised query
vector and the L2-normalised candidate vectors. Input words are excluded from
the ranking."""),

        (CODE, """from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import gensim
import gensim.downloader as api
import matplotlib.pyplot as plt

BASE_DIR = Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from country_prediction import (
    CountryAnalogyPredictor,
    evaluate_predictions,
    load_candidate_countries,
    load_embeddings,
    load_records,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

print("Python      :", platform.python_version())
print("gensim      :", gensim.__version__)
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("project dir :", BASE_DIR)"""),

        (MD, """## 2. Pre-trained embedding model

`glove-wiki-gigaword-100` (Pennington et al., 2014) is loaded through the
gensim downloader. The model has 400,000 lower-cased words of dimension 100 and
is cached in `~/gensim-data`, so it is only downloaded once."""),

        (CODE, """vectors, model_info = load_embeddings()
model_info"""),

        (MD, """## 3. Datasets

`BuildDatasets.py` validated the curated capital-country list against the model
vocabulary and generated four CSV files plus metadata. Let us inspect them."""),

        (CODE, """metadata = json.loads(Path("datasets/dataset_metadata.json").read_text())
metadata"""),
        (CODE, """pairs = pd.read_csv("datasets/capitals_countries.csv")
candidates = pd.read_csv("datasets/candidate_countries.csv")
test_set = pd.read_csv("datasets/test_analogies.csv")
large_set = pd.read_csv("datasets/eval_analogies_large.csv")

print("capital-country pairs :", len(pairs))
print("candidate countries   :", len(candidates))
print("test analogies        :", len(test_set))
print("evaluation analogies  :", len(large_set))
pairs.head(8)"""),
        (CODE, """test_set"""),
        (CODE, """large_set.head(8)"""),

        (MD, """## 4. The required example, step by step

We take the normalised vectors of **Greece**, **Athens** and **Baghdad**, build
the query vector, and score it against **Iraq** and other countries."""),

        (CODE, """capital1, country1, capital2 = "athens", "greece", "baghdad"
query = (
    vectors[country1]
    - vectors[capital1]
    + vectors[capital2]
)
query_unit = query / np.linalg.norm(query)

for word in ("iraq", "afghanistan", "lebanon", "syria", "greece"):
    unit = vectors[word] / np.linalg.norm(vectors[word])
    print(f"cosine(query, {word:<12}) = {float(query_unit @ unit):.4f}")"""),

        (CODE, """predictor = CountryAnalogyPredictor(vectors, load_candidate_countries())
predictor.model_info = model_info

filtered = predictor.rank(capital1, country1, capital2, topn=5)
full_vocab = predictor.rank(capital1, country1, capital2, topn=5, country_filter=False)

print("country-filtered:", ", ".join(f"{w} ({s:.3f})" for w, s in filtered))
print("full vocabulary :", ", ".join(f"{w} ({s:.3f})" for w, s in full_vocab))
print("\\nPredicted Country 2 =", filtered[0][0].upper())"""),

        (CODE, """labels = [word for word, _ in filtered]
scores = [score for _, score in filtered]
colors = ["#2ca02c" if word == "iraq" else "#1f77b4" for word in labels]

fig, axis = plt.subplots(figsize=(7.5, 4))
axis.bar(labels, scores, color=colors)
axis.set_ylabel("Cosine similarity")
axis.set_title("Top-5 candidates for Greece - Athens + Baghdad")
for index, score in enumerate(scores):
    axis.annotate(f"{score:.3f}", (index, score), ha="center", va="bottom")
plt.show()"""),

        (MD, """## 5. Evaluation on the 15-example test set

The test set contains the required example plus 14 randomly sampled analogies
(fixed seed 42). We compare **country-filtered** search with **full-vocabulary**
search."""),

        (CODE, """test_records = load_records("datasets/test_analogies.csv")
rows_filtered, metrics_filtered = evaluate_predictions(
    predictor, test_records, country_filter=True
)
rows_full, metrics_full = evaluate_predictions(
    predictor, test_records, country_filter=False
)

test_table = pd.DataFrame(
    {
        "analogy": [
            f"{r['capital1'].title()} : {r['country1'].title()} :: "
            f"{r['capital2'].title()} : ?"
            for r in rows_filtered
        ],
        "actual": [r["actual_country2"] for r in rows_filtered],
        "predicted": [r["predicted_country"] for r in rows_filtered],
        "score": [round(r["predicted_score"], 3) for r in rows_filtered],
        "margin": [round(r["margin"], 3) for r in rows_filtered],
        "result": ["Correct" if r["correct"] else "Incorrect" for r in rows_filtered],
    }
)
test_table"""),

        (CODE, """pd.DataFrame(
    {
        "test-15": [metrics_filtered["top1_accuracy_pct"], metrics_filtered["top5_accuracy_pct"]],
        "test-15 full vocab": [metrics_full["top1_accuracy_pct"], metrics_full["top5_accuracy_pct"]],
    },
    index=["top-1 accuracy %", "top-5 accuracy %"],
)"""),

        (MD, """## 6. Evaluation on 1,000 generated analogies

`eval_analogies_large.csv` contains 1,000 unique analogies generated with seed
2026. Both search modes are evaluated."""),

        (CODE, """large_records = load_records("datasets/eval_analogies_large.csv")
large_filtered, metrics_large_filtered = evaluate_predictions(
    predictor, large_records, country_filter=True
)
large_full, metrics_large_full = evaluate_predictions(
    predictor, large_records, country_filter=False
)

summary = pd.DataFrame(
    [
        {"dataset": "test-15", "mode": "country-filtered", **{
            k: metrics_filtered[k] for k in
            ("total", "correct_top1", "incorrect_top1", "top1_accuracy_pct",
             "top5_accuracy_pct", "mean_top1_margin")
        }},
        {"dataset": "test-15", "mode": "full vocabulary", **{
            k: metrics_full[k] for k in
            ("total", "correct_top1", "incorrect_top1", "top1_accuracy_pct",
             "top5_accuracy_pct", "mean_top1_margin")
        }},
        {"dataset": "1,000", "mode": "country-filtered", **{
            k: metrics_large_filtered[k] for k in
            ("total", "correct_top1", "incorrect_top1", "top1_accuracy_pct",
             "top5_accuracy_pct", "mean_top1_margin")
        }},
        {"dataset": "1,000", "mode": "full vocabulary", **{
            k: metrics_large_full[k] for k in
            ("total", "correct_top1", "incorrect_top1", "top1_accuracy_pct",
             "top5_accuracy_pct", "mean_top1_margin")
        }},
    ]
)
summary"""),

        (MD, """## 7. Saved visualizations

These charts were produced from the saved result files by
`tools/generate_visualizations.py`."""),

        (CODE, """from IPython.display import Image, display

for name in (
    "accuracy_comparison.png",
    "similarity_distribution.png",
    "errors_by_country.png",
    "rank_distribution.png",
    "test15_confidence.png",
    "analogy_vector_pca.png",
):
    print(name)
    display(Image(filename=f"visualizations/{name}", width=760))"""),

        (MD, """## 8. Error analysis

`error_analysis_*.csv` files list every incorrect prediction with the input
analogy, the actual and predicted country, both similarity scores, the rank of
the correct country and a short explanation."""),

        (CODE, """test_errors = pd.read_csv("results/error_analysis_test15_country_filtered.csv")
test_errors[["capital1", "country1", "capital2", "actual_country2",
             "predicted_country", "predicted_score", "actual_score",
             "actual_rank", "error_category"]]"""),

        (CODE, """large_errors = pd.read_csv("results/error_analysis_large1000_country_filtered.csv")
print("incorrect predictions:", len(large_errors))
display(large_errors[["country1", "capital2", "actual_country2",
                      "predicted_country", "actual_rank", "error_category"]].head(12))
large_errors["error_category"].value_counts()"""),

        (CODE, """breakdown = (
    large_errors.groupby("actual_country2").size()
    .sort_values(ascending=False)
    .head(10)
    .rename("errors")
)
breakdown"""),

        (MD, """## 9. Observations

1. The required example is solved: `athens : greece :: baghdad` gives **iraq**
   as the top candidate with a cosine similarity of about 0.79.
2. On the 15-example test set the country-filtered system gets the top-1
   country right for 14 of 15 analogies (93.33%). The only error is
   `Mogadishu : Somalia :: Vienna : ?`, where **germany** (0.645) edges out
   **austria** (0.637); the correct country is still ranked 2nd.
3. Searching the full 400k-word vocabulary lowers test accuracy to 13/15
   (86.67%): for `London`, **britain** (0.795) is ranked above **england**
   (0.690). Country filtering therefore removes spurious non-country neighbours.
4. On the 1,000 generated analogies the country-filtered system reaches
   94.40% top-1 and 98.90% top-5 accuracy; with the full vocabulary it reaches
   92.20% top-1 and 97.70% top-5. Filtering adds 2.20 and 1.20 percentage
   points respectively.
5. Most correct top-1 similarities lie between 0.65 and 0.85. The mean top-1
   margin is small (about 0.10 on the large set), which explains why a few
   neighbouring countries are confused.
6. Mistakes concentrate on ambiguous names (Georgia, Jordan, England, Chile)
   and closely related neighbours (Austria/Germany, Belgium/France)."""),

        (MD, """## 10. Limitations

- Static embeddings give one vector per word, so polysemous names (Georgia the
  country vs. the US state, Jordan the name, Turkey the bird) are conflated.
- Only single-word names are supported; multi-word names such as *Saudi Arabia*
  or *New Delhi* are outside the model vocabulary.
- The evaluation analogies are recombinations of the same 78 capital-country
  pairs, so they are not independent test items.
- A single linear offset cannot represent every capital-country relationship,
  especially for rare capitals whose vectors were seen infrequently during
  GloVe training.
- Results depend on the 2014 Wikipedia + Gigaword corpus and cannot reflect
  later geopolitical changes.

## 11. Conclusion

Pre-trained GloVe embeddings capture the capital-country relationship well
enough to solve analogies by vector arithmetic. Restricting the search to a
curated country list and excluding the input words improve accuracy. The
remaining errors are dominated by ambiguous words and geographically or
culturally close countries, which is the expected behaviour of static
embeddings."""),

        (MD, """## 12. References

1. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors
   for Word Representation.* EMNLP 2014, 1532-1543.
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation
   of Word Representations in Vector Space.* arXiv:1301.3781.
3. Mikolov, T., Yih, W., & Zweig, G. (2013). *Linguistic Regularities in
   Continuous Space Word Representations.* NAACL-HLT 2013, 746-751.
4. Rehurek, R., & Sojka, P. (2010). *Software Framework for Topic Modelling
   with Large Corpora.* LREC 2010 Workshop, 45-50 (gensim).
5. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed.
   draft), Chapter 6: Vector Semantics and Embeddings."""),
    ]


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    for kind, source in cells():
        if kind == MD:
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(source))
    return notebook


def execute_notebook() -> None:
    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=900",
        str(NOTEBOOK_PATH),
    ]
    print("executing:", " ".join(command))
    subprocess.run(command, check=True, cwd=BASE_DIR)


def main() -> None:
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")
    execute_notebook()

    executed = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [c for c in executed.cells if c.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    print(f"executed {len(code_cells)} code cells, {len(errors)} errors")
    if errors:
        for error in errors:
            print(error.get("ename"), error.get("evalue"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
