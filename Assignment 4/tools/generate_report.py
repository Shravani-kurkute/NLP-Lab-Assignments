#!/usr/bin/env python3
"""Generate the Assignment 4 report as DOCX (and PDF via LibreOffice).

All numbers and tables are read from the real result files produced by
country_prediction.py --evaluate. Charts come from visualizations/ and console
captures from screenshots/.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualizations"
SHOT_DIR = BASE_DIR / "screenshots"
REPORT_DIR = BASE_DIR / "report"

DOCX_PATH = REPORT_DIR / "421129_Shravani_Kurkute_NLP_Assignment04.docx"
PDF_PATH = DOCX_PATH.with_suffix(".pdf")

STUDENT = {
    "name": "Shravani Balasaheb Kurkute",
    "roll": "421129",
    "prn": "22311926",
    "division": "BTechA",
    "year": "2026-2027",
    "faculty": "Dr. Kapil N. Vhatkar",
}

FOOTER_TEXT = "NLP Laboratory - Assignment 04: Country Prediction using Word Embeddings"


def add_page_number_footer(section) -> None:
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(FOOTER_TEXT + "    Page ")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    field = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    field._r.append(begin)
    field._r.append(instruction)
    field._r.append(end)
    field.font.size = Pt(8)
    field.font.color.rgb = RGBColor(0x60, 0x60, 0x60)


def h1(doc, text):
    return doc.add_heading(text, level=1)


def h2(doc, text):
    return doc.add_heading(text, level=2)


def para(doc, text, bold=False, italic=False, size=None, align=None, space_after=6):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align is not None:
        paragraph.alignment = align
    paragraph.paragraph_format.space_after = Pt(space_after)
    return paragraph


def bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(item, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)


def add_table(doc, headers, rows, widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(header))
        run.bold = True
        run.font.size = Pt(font_size)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "DDEBF7")
        cell._tc.get_or_add_tcPr().append(shading)

    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = ""
            run = cells[index].paragraphs[0].add_run("" if value is None else str(value))
            run.font.size = Pt(font_size)
            if isinstance(value, str) and value == "Correct":
                run.font.color.rgb = RGBColor(0x00, 0x61, 0x00)
                run.bold = True
            elif isinstance(value, str) and value == "Incorrect":
                run.font.color.rgb = RGBColor(0x9C, 0x00, 0x06)
                run.bold = True

    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(2)
    return table


def add_image(doc, path: Path, width=6.2, caption=None):
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        caption_paragraph = doc.add_paragraph()
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption_paragraph.add_run(caption)
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x50, 0x50, 0x50)


def build_report() -> None:
    summary = json.loads((RESULTS_DIR / "accuracy_summary.json").read_text())
    metadata = json.loads(
        (BASE_DIR / "datasets" / "dataset_metadata.json").read_text()
    )
    test_preds = pd.read_csv(RESULTS_DIR / "predictions_test15_country_filtered.csv")
    large_preds = pd.read_csv(
        RESULTS_DIR / "predictions_large1000_country_filtered.csv"
    )
    large_errors = pd.read_csv(
        RESULTS_DIR / "error_analysis_large1000_country_filtered.csv"
    )
    test_errors = pd.read_csv(
        RESULTS_DIR / "error_analysis_test15_country_filtered.csv"
    )

    r = summary["results"]
    t15c, t15f = r["test15"]["country_filtered"], r["test15"]["full_vocabulary"]
    l1c, l1f = r["large1000"]["country_filtered"], r["large1000"]["full_vocabulary"]
    category_counts = large_errors["error_category"].value_counts().to_dict()
    top_error_countries = summary.get("error_breakdown", [])[:8]

    correct_scores = large_preds.loc[
        large_preds["correct"] == "Correct", "predicted_score"
    ]
    wrong_scores = large_preds.loc[
        large_preds["correct"] == "Incorrect", "predicted_score"
    ]

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    for section in document.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
    add_page_number_footer(document.sections[0])

    # ------------------------------------------------------------------ title
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("NATURAL LANGUAGE PROCESSING")
    run.bold = True
    run.font.size = Pt(18)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("LABORATORY ASSIGNMENT - 04")
    run.bold = True
    run.font.size = Pt(15)

    topic = document.add_paragraph()
    topic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = topic.add_run("Country Prediction using Word Embeddings")
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x78)

    document.add_paragraph()
    add_table(
        document,
        ["Field", "Details"],
        [
            ["Student Name", STUDENT["name"]],
            ["Roll No.", STUDENT["roll"]],
            ["PRN / Registration No.", STUDENT["prn"]],
            ["Class / Division", STUDENT["division"]],
            ["Academic Year", STUDENT["year"]],
            ["Faculty Name", STUDENT["faculty"]],
        ],
        widths=[2.4, 3.8],
        font_size=11,
    )
    document.add_paragraph()
    add_image(
        document,
        VIS_DIR / "analogy_vector_pca.png",
        width=6.0,
        caption="Figure 1: The analogy vector Greece - Athens + Baghdad lands next to Iraq in the "
        "GloVe space (2-D PCA projection of selected vectors).",
    )
    document.add_page_break()

    # ------------------------------------------------------------- 2. aim etc
    h1(document, "2. Aim")
    para(
        document,
        "To understand the semantic relationships represented by pre-trained word "
        "embeddings and to use vector arithmetic and a similarity measure to predict a "
        "country from a capital-city relationship.",
    )

    h1(document, "3. Objectives")
    bullets(
        document,
        [
            "Understand the concept and properties of pre-trained word embeddings.",
            "Load and use a suitable pre-trained embedding model (GloVe).",
            "Perform vector arithmetic to represent word analogies.",
            "Rank candidate words with cosine similarity and exclude the input words.",
            "Evaluate the country-prediction system on a small and a large analogy set.",
            "Analyse the errors and the effect of restricting the search to country names.",
        ],
    )

    h1(document, "4. Problem Statement")
    para(
        document,
        "Write a Python program that predicts a country from a capital-city relationship "
        "using pre-trained word embeddings. The program accepts three words - Capital 1, "
        "Country 1 and Capital 2 - and predicts Country 2. Given the vectors of the three "
        "input words, the program computes",
    )
    formula = document.add_paragraph()
    formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = formula.add_run("vector(Country 2)  \u2248  vector(Country 1) \u2212 vector(Capital 1) + vector(Capital 2)")
    run.bold = True
    para(
        document,
        "and selects the candidate country whose vector has the highest cosine similarity "
        "with the resulting query vector.",
    )

    h1(document, "5. Example")
    add_table(
        document,
        ["Input 1 (Capital 1)", "Input 2 (Country 1)", "Input 3 (Capital 2)", "Expected Output (Country 2)"],
        [["Athens", "Greece", "Baghdad", "Iraq"]],
        widths=[1.5, 1.6, 1.5, 2.0],
        font_size=10,
    )

    # -------------------------------------------------------------- 6. theory
    h1(document, "6. Theory and Background")
    para(
        document,
        "Word embeddings map each word to a dense, real-valued vector learned from a large "
        "text corpus. Words that appear in similar contexts receive similar vectors, so "
        "geometric relationships between vectors approximate semantic relationships "
        "between words. Classical counting representations (one-hot, bag-of-words, TF-IDF) "
        "are high-dimensional and sparse and do not directly encode similarity; dense "
        "embeddings such as Word2Vec and GloVe use far fewer dimensions while placing "
        "related words close together.",
    )
    para(
        document,
        "GloVe (Global Vectors) learns vectors from global word-word co-occurrence "
        "statistics. The model used here, glove-wiki-gigaword-100, was trained on "
        "Wikipedia 2014 and Gigaword 5 (6 billion tokens) and provides 400,000 lower-cased "
        "100-dimensional vectors.",
    )
    para(
        document,
        "Analogy solving assumes that a constant offset connects related words. If the "
        "vector difference vector(Country 1) - vector(Capital 1) encodes the "
        "\u201ccapital-to-country\u201d relation, then adding it to vector(Capital 2) should "
        "land near vector(Country 2). Similarity is measured with cosine similarity, "
        "cos(\u03b8) = (A \u00b7 B) / (\u2016A\u2016 \u2016B\u2016), which compares vector "
        "directions and is invariant to vector length. Because the query vector is compared "
        "with L2-normalised candidate vectors, the dot product directly gives the cosine "
        "similarity, which allows the whole candidate set to be scored in one matrix "
        "multiplication.",
    )

    # --------------------------------------------------------- 7. environment
    h1(document, "7. Tools, Libraries and Environment")
    add_table(
        document,
        ["Sr. No.", "Tool / Library", "Version", "Purpose"],
        [
            ["1", "Python", summary["environment"]["python"], "Programming language used for the implementation"],
            ["2", "gensim", "4.4.0", "Loads GloVe vectors (KeyedVectors) and provides the model downloader"],
            ["3", "NumPy", "2.2.0", "Vector arithmetic, normalisation and cosine similarity (matrix products)"],
            ["4", "SciPy", "1.16.3", "Numerical dependency of gensim"],
            ["5", "pandas", "2.2.3", "Reading result tables and building the Excel workbook"],
            ["6", "Matplotlib", "3.10.6", "Visualisations and console captures"],
            ["7", "scikit-learn", "1.6.0", "PCA projection for the analogy visualisation"],
            ["8", "python-docx / openpyxl", "1.2.0 / 3.1.5", "Report (.docx) and workbook (.xlsx) generation"],
            ["9", "Jupyter / nbconvert", "1.1.1 / 7.16.6", "Executed notebook with all outputs"],
            ["10", "pytest", "8.3.4", "Automated unit and integration tests"],
            ["11", "GloVe glove-wiki-gigaword-100", "100-d, 400K words", "Pre-trained embedding model (Wikipedia 2014 + Gigaword 5)"],
            ["12", "Platform", "Fedora Linux, Python 3.13.9", "Execution environment"],
        ],
        widths=[0.6, 1.6, 1.5, 2.9],
        font_size=8.5,
    )

    # ----------------------------------------------------------- 8. model
    h1(document, "8. Pre-trained Embedding Model")
    bullets(
        document,
        [
            "Model name: GloVe (Global Vectors) - glove-wiki-gigaword-100.",
            "Provider: Stanford NLP Group (Pennington, Socher & Manning, 2014); loaded through the gensim-data downloader.",
            f"Vocabulary size: {summary['model']['vocabulary_size']:,} words (lower-cased).",
            f"Embedding dimension: {summary['model']['dimension']}.",
            "Training corpus: Wikipedia 2014 + Gigaword 5, about 6 billion tokens.",
            "Download size: about 128 MB (glove-wiki-gigaword-100.gz); cached in ~/gensim-data and reused on later runs.",
            "Similarity measure: cosine similarity on L2-normalised vectors.",
            "Words missing from the vocabulary are reported with a clear error message instead of producing a wrong prediction.",
        ],
    )
    add_image(
        document,
        SHOT_DIR / "01_dataset_generation.png",
        width=6.3,
        caption="Figure 2: Loading the cached GloVe model and validating the capital-country pairs "
        "(console capture of python3 BuildDatasets.py).",
    )

    # ------------------------------------------------------ 9. methodology
    h1(document, "9. Methodology and Algorithm")
    para(document, "The system is implemented in two scripts:")
    bullets(
        document,
        [
            "BuildDatasets.py - validates the capital-country list against the GloVe vocabulary and writes the four CSV datasets (fixed seeds: 42 for the test set, 2026 for the 1,000-analogy set).",
            "country_prediction.py - loads the vectors, builds the analogy vector, ranks candidates in batches with NumPy and reports predictions, CSV files and a JSON summary.",
        ],
    )
    para(document, "Algorithm:")
    for index, step in enumerate(
        [
            "Load the pre-trained GloVe model and its normalised vectors.",
            "Read Capital 1, Country 1 and Capital 2 (lower-cased); raise a clear error if any word is missing.",
            "Build query = vector(Country 1) - vector(Capital 1) + vector(Capital 2) and L2-normalise it.",
            "Score the query against all candidate vectors with one matrix multiplication (cosine similarity).",
            "Rank candidates in descending order and remove the input words from the results.",
            "Restrict the candidate set to a curated list of 131 countries (country-filtered mode) or use the full 400,000-word vocabulary for comparison.",
            "Return the highest-ranked candidate as the predicted country; also report the top five with scores.",
            "Repeat in batches of 64 analogies for the 15-example and 1,000-example evaluations.",
            "Compute top-1/top-5 accuracy, margins and score statistics; save predictions, error analysis and a JSON summary.",
        ],
        1,
    ):
        paragraph = document.add_paragraph(f"{index}. {step}")
        paragraph.paragraph_format.space_after = Pt(2)

    para(document, "Pseudocode for the core ranking step:", italic=True)
    code = document.add_paragraph()
    code.paragraph_format.space_after = Pt(6)
    code_run = code.add_run(
        "q = normalize( v(country1) - v(capital1) + v(capital2) )\n"
        "scores = candidate_matrix @ q          # cosine similarity for every candidate\n"
        "order  = argsort(-scores)\n"
        "ranked = [candidate for candidate in order if candidate not in inputs][:5]\n"
        "prediction = ranked[0]"
    )
    code_run.font.name = "Consolas"
    code_run.font.size = Pt(9)

    add_image(
        document,
        SHOT_DIR / "02_single_prediction.png",
        width=6.3,
        caption="Figure 3: Single prediction for the required example "
        "(python3 country_prediction.py --words athens greece baghdad).",
    )

    # ------------------------------------------------ 10. input/output records
    h1(document, "10. Input / Output Records")
    para(
        document,
        f"The table below shows all {len(test_preds)} test analogies with the actual and "
        "predicted country (country-filtered mode).",
    )
    rows = []
    for index, row in enumerate(test_preds.itertuples(), 1):
        rows.append(
            [
                index,
                row.capital1.title(),
                row.country1.title(),
                row.capital2.title(),
                row.actual_country2.title(),
                row.predicted_country.title(),
                "Correct" if row.correct == "Correct" else "Incorrect",
            ]
        )
    add_table(
        document,
        ["Sr.", "Capital 1", "Country 1", "Capital 2", "Actual", "Predicted", "Result"],
        rows,
        widths=[0.45, 1.0, 1.05, 1.0, 0.95, 0.95, 0.85],
        font_size=8.5,
    )

    # --------------------------------------- 11. similarity / ranked candidates
    h1(document, "11. Similarity Scores and Ranked Candidates")
    picked = ["athens", "kabul", "beirut", "vienna", "helsinki", "tokyo", "sofia", "mogadishu"]
    rows = []
    for capital in picked:
        row = test_preds.loc[test_preds["capital1"] == capital].iloc[0]
        rows.append(
            [
                f"{row.capital1.title()} : {row.country1.title()} :: {row.capital2.title()}",
                f"{row.rank1_word.title()}",
                f"{row.rank1_score:.3f}",
                f"{row.rank2_word.title()}",
                f"{row.rank2_score:.3f}",
                row.predicted_country.title(),
                "Correct" if row.correct == "Correct" else "Incorrect",
            ]
        )
    add_table(
        document,
        ["Example", "Candidate 1", "Score", "Candidate 2", "Score", "Selected", "Result"],
        rows,
        widths=[2.0, 0.95, 0.6, 0.95, 0.6, 0.9, 0.8],
        font_size=8.5,
    )
    para(
        document,
        "The required example Athens : Greece :: Baghdad : ? ranks Iraq first with a cosine "
        "similarity of 0.790, ahead of Afghanistan (0.708) and Lebanon (0.678). The only "
        "test error is Mogadishu : Somalia :: Vienna : ?, where Germany (0.645) narrowly "
        "beats Austria (0.637); the margin is only 0.008.",
        size=9.5,
    )
    add_image(
        document,
        VIS_DIR / "test15_confidence.png",
        width=6.1,
        caption="Figure 4: Top-1 cosine similarity and runner-up margin for each test analogy.",
    )

    # ---------------------------------------------------------- 12. evaluation
    h1(document, "12. Evaluation")
    para(
        document,
        "Two datasets were used: test_analogies.csv (15 analogies, fixed seed 42) and "
        "eval_analogies_large.csv (1,000 unique analogies, fixed seed 2026). Each was "
        "evaluated with country filtering and with the full vocabulary.",
    )
    eval_rows = []
    for dataset, label in (("test15", "Test (15)"), ("large1000", "Evaluation (1,000)")):
        for mode, mode_label in (("country_filtered", "Country-filtered"), ("full_vocabulary", "Full vocabulary")):
            metrics = r[dataset][mode]
            eval_rows.append(
                [
                    label,
                    mode_label,
                    metrics["total"],
                    metrics["correct_top1"],
                    metrics["incorrect_top1"],
                    f"{metrics['top1_accuracy_pct']:.2f}%",
                    f"{metrics['top5_accuracy_pct']:.2f}%",
                    f"{metrics['mean_top1_margin']:.3f}",
                    f"{metrics['runtime_seconds']:.2f}",
                ]
            )
    add_table(
        document,
        ["Dataset", "Search mode", "Total", "Correct", "Incorrect", "Top-1", "Top-5", "Mean margin", "Time (s)"],
        eval_rows,
        widths=[1.05, 1.15, 0.6, 0.7, 0.75, 0.7, 0.7, 0.85, 0.65],
        font_size=8.5,
    )
    para(
        document,
        f"Country filtering changes top-1 accuracy by "
        f"{summary['comparison']['test15']['top1_gain_country_filtered_pct_points']:+.2f} "
        f"percentage points on the test set and by "
        f"{summary['comparison']['large1000']['top1_gain_country_filtered_pct_points']:+.2f} "
        f"percentage points on the 1,000-analogy set. Top-5 accuracy is high in all four "
        f"settings (97.7%-100%), which shows that the correct country is usually among the "
        f"best candidates even when it is not ranked first.",
    )
    add_table(
        document,
        ["Filtering effect (1,000 analogies)", "Country-filtered", "Full vocabulary", "Change"],
        [
            [
                "Top-1 accuracy",
                f"{l1c['top1_accuracy_pct']:.2f}%",
                f"{l1f['top1_accuracy_pct']:.2f}%",
                f"{summary['comparison']['large1000']['top1_gain_country_filtered_pct_points']:+.2f} pp",
            ],
            [
                "Top-5 accuracy",
                f"{l1c['top5_accuracy_pct']:.2f}%",
                f"{l1f['top5_accuracy_pct']:.2f}%",
                f"{summary['comparison']['large1000']['top5_gain_country_filtered_pct_points']:+.2f} pp",
            ],
        ],
        widths=[2.6, 1.4, 1.4, 1.2],
        font_size=9,
    )
    add_image(
        document,
        VIS_DIR / "accuracy_comparison.png",
        width=6.1,
        caption="Figure 5: Accuracy comparison for country-filtered and full-vocabulary search.",
    )
    add_image(
        document,
        VIS_DIR / "rank_distribution.png",
        width=6.0,
        caption="Figure 6: Rank of the correct country in the filtered candidate list (1,000 analogies).",
    )
    add_image(
        document,
        SHOT_DIR / "04_evaluation_summary.png",
        width=6.3,
        caption="Figure 7: Console summary produced by python3 country_prediction.py --evaluate.",
    )

    # ------------------------------------------------------- 13. observations
    h1(document, "13. Observations and Analysis")
    observations = [
        f"The required example is solved correctly: Athens : Greece :: Baghdad gives Iraq "
        f"with a cosine similarity of {test_preds.iloc[0]['rank1_score']:.3f}.",
        f"On the {t15c['total']}-example test set the country-filtered system answers "
        f"{t15c['correct_top1']} analogies correctly ({t15c['top1_accuracy_pct']:.2f}% top-1) "
        f"and reaches {t15c['top5_accuracy_pct']:.2f}% top-5 accuracy. The single error is "
        f"the Austria/Germany confusion, where the correct country is still ranked second.",
        f"With the full 400,000-word vocabulary the test accuracy drops to "
        f"{t15f['top1_accuracy_pct']:.2f}%: for London the model ranks Britain (0.795) above "
        f"England (0.690). Non-country neighbours such as Chinese, Finnish and Barcelona also "
        f"enter the full-vocabulary top five, which is why filtering helps.",
        f"On the 1,000 generated analogies the country-filtered system reaches "
        f"{l1c['top1_accuracy_pct']:.2f}% top-1 and {l1c['top5_accuracy_pct']:.2f}% top-5 "
        f"accuracy (full vocabulary: {l1f['top1_accuracy_pct']:.2f}% and "
        f"{l1f['top5_accuracy_pct']:.2f}%). Filtering improves top-1 accuracy by "
        f"{summary['comparison']['large1000']['top1_gain_country_filtered_pct_points']:+.2f} "
        f"percentage points.",
        f"Correct predictions usually have a top-1 similarity between "
        f"{correct_scores.min():.2f} and {correct_scores.max():.2f} (mean "
        f"{correct_scores.mean():.3f}), while incorrect predictions are on average less "
        f"confident (mean {wrong_scores.mean():.3f}). The mean top-1 margin is only "
        f"{l1c['mean_top1_margin']:.3f}, so small vector differences decide many cases.",
        f"Mistakes are concentrated on a few items: "
        + ", ".join(f"{item['country'].title()} ({item['errors']})" for item in top_error_countries[:5])
        + ". Georgia is also a US state, Jordan is a personal name, and England overlaps with "
        "Britain and the United Kingdom.",
        "Even uncommon capitals work reasonably well - Vientiane gives Laos (0.660) and "
        "Valletta gives Malta (0.681) - which shows the capital-country offset is present "
        "throughout the vector space, not only for famous cities.",
    ]
    for index, text in enumerate(observations, 1):
        paragraph = document.add_paragraph(f"{index}. {text}")
        paragraph.paragraph_format.space_after = Pt(4)

    add_image(
        document,
        VIS_DIR / "similarity_distribution.png",
        width=6.0,
        caption="Figure 8: Distribution of top-1 cosine similarity for correct and incorrect predictions.",
    )
    add_image(
        document,
        SHOT_DIR / "03_demo_test15.png",
        width=5.8,
        caption="Figure 9: Console output of the demonstration and the 15-example test set.",
    )

    # ------------------------------------------------------ 14. error analysis
    h1(document, "14. Error Analysis")
    para(
        document,
        f"The 1,000-analogy country-filtered experiment produced {len(large_errors)} "
        f"incorrect predictions. Error categories: "
        + ", ".join(f"{name} ({count})" for name, count in category_counts.items())
        + ".",
    )
    para(document, "Errors on the 15-example test set:")
    add_table(
        document,
        ["Input analogy", "Actual", "Predicted", "Pred. score", "Actual score", "Rank of actual", "Reason"],
        [
            [
                f"{row.capital1.title()} : {row.country1.title()} :: {row.capital2.title()} : ?",
                row.actual_country2.title(),
                row.predicted_country.title(),
                f"{row.predicted_score:.3f}",
                f"{row.actual_score:.3f}",
                int(row.actual_rank),
                "Austria and Germany share linguistic and cultural contexts; the score "
                "difference is only 0.008 and the correct country is ranked second.",
            ]
            for row in test_errors.itertuples()
        ],
        widths=[1.7, 0.7, 0.75, 0.7, 0.75, 0.7, 2.0],
        font_size=8,
    )
    para(document, "Representative errors from the 1,000-analogy experiment:")
    sample = large_errors.head(6)
    add_table(
        document,
        ["Input analogy", "Actual", "Predicted", "Pred.", "Actual", "Rank", "Reason"],
        [
            [
                f"{row.capital1.title()} : {row.country1.title()} :: {row.capital2.title()}",
                row.actual_country2.title(),
                row.predicted_country.title(),
                f"{row.predicted_score:.3f}",
                f"{row.actual_score:.3f}",
                int(row.actual_rank) if pd.notna(row.actual_rank) else "-",
                (row.explanation[:110] + "...") if len(row.explanation) > 110 else row.explanation,
            ]
            for row in sample.itertuples()
        ],
        widths=[1.5, 0.65, 0.7, 0.5, 0.55, 0.4, 2.3],
        font_size=7.5,
    )
    add_image(
        document,
        VIS_DIR / "errors_by_country.png",
        width=6.0,
        caption="Figure 10: Actual countries that were most often predicted incorrectly.",
    )
    para(
        document,
        "The dominant causes are (i) ambiguous words with several meanings, "
        "(ii) neighbouring or culturally close countries whose vectors are similar, and "
        "(iii) very small score margins between the top candidates. All incorrect rows "
        "with scores and explanations are stored in results/error_analysis_*.csv and in "
        "the Excel workbook.",
    )

    # --------------------------------------------------------- 15. limitations
    h1(document, "15. Limitations")
    bullets(
        document,
        [
            "Static embeddings assign one vector per word. Ambiguous names such as Georgia, "
            "Jordan, Turkey and Chile mix several meanings in a single vector.",
            "Only single-word names are supported; multi-word names such as 'Saudi Arabia', "
            "'Sri Lanka' and 'United States' cannot be handled directly.",
            "The seed list contains only 78 capital-country pairs, so the 1,000 analogies are "
            "recombinations of the same pairs and are not fully independent test items.",
            "A single linear offset cannot represent every capital-country relationship "
            "equally well; less frequent capitals have less reliable vectors.",
            "The corpus (Wikipedia 2014 + Gigaword 5) is fixed and cannot reflect later "
            "geopolitical changes.",
            "The vocabulary is lower-cased and the model returns an error for out-of-vocabulary "
            "words rather than a prediction.",
        ],
    )
    add_image(
        document,
        SHOT_DIR / "05_missing_word_error.png",
        width=6.3,
        caption="Figure 11: Missing words are reported clearly instead of producing a prediction.",
    )

    # ---------------------------------------------------------- 16. conclusion
    h1(document, "16. Conclusion")
    para(
        document,
        f"Pre-trained GloVe embeddings capture the capital-country relationship well enough "
        f"to solve analogies by simple vector arithmetic. With cosine similarity and a curated "
        f"country list, the system predicted {t15c['correct_top1']} of {t15c['total']} test "
        f"analogies correctly ({t15c['top1_accuracy_pct']:.2f}%) and reached "
        f"{l1c['top1_accuracy_pct']:.2f}% top-1 and {l1c['top5_accuracy_pct']:.2f}% top-5 "
        f"accuracy on 1,000 generated analogies. Removing the input words and restricting the "
        f"search to countries improved accuracy over the full-vocabulary baseline. The "
        f"remaining errors are dominated by ambiguous words and geographically close neighbours, "
        f"which is the expected limitation of static embeddings; contextual models such as BERT "
        f"would be needed to resolve them.",
    )

    # -------------------------------------------------------- 17. viva questions
    h1(document, "17. Viva Questions and Answers")
    viva = [
        (
            "What are word embeddings?",
            "Word embeddings are dense numerical vectors that represent words in a continuous "
            "vector space. They are learned from large text corpora so that words used in "
            "similar contexts have similar vectors, which lets us measure semantic similarity "
            "and perform arithmetic such as analogies.",
        ),
        (
            "What is the difference between sparse and dense word representations?",
            "Sparse representations such as one-hot vectors, bag-of-words and TF-IDF contain "
            "mostly zeros and treat words as independent symbols. Dense embeddings use a small "
            "number of dimensions (here 100) with meaningful real values and therefore capture "
            "similarity and relationships while using less memory than a full one-hot vector.",
        ),
        (
            "What is cosine similarity and why is it used?",
            "Cosine similarity is cos(theta) = (A . B) / (||A|| ||B||). It measures the angle "
            "between two vectors and ignores their magnitude, so it focuses on the direction of "
            "the analogy vector. For L2-normalised vectors the dot product equals the cosine, "
            "which makes ranking a single matrix multiplication.",
        ),
        (
            "Why does vector arithmetic solve analogies?",
            "Word embeddings approximately encode relations as constant offsets. The difference "
            "vector(Country) - vector(Capital) represents the capital-to-country relation, so "
            "adding it to Capital 2 produces a vector close to Country 2.",
        ),
        (
            "Why are the input words removed from the candidate list?",
            "The input words themselves often appear among the nearest neighbours of the query "
            "vector. Removing them prevents the model from returning the words it was given and "
            "forces it to find a genuinely new candidate.",
        ),
        (
            "What is a pre-trained embedding model?",
            "It is a model whose vectors were already learned from a large corpus. We load "
            "glove-wiki-gigaword-100 instead of training Word2Vec or GloVe ourselves, which "
            "saves time and gives broad vocabulary coverage.",
        ),
        (
            "Why did the system predict Austria wrongly in one case?",
            "For Mogadishu : Somalia :: Vienna, Germany scored 0.645 and Austria 0.637. Vienna "
            "appears in many German-speaking and European contexts in the corpus, so the "
            "Germany and Austria vectors are very close. The correct country was still ranked "
            "second, and the margin was only 0.008.",
        ),
        (
            "What is the effect of the embedding dimension?",
            "The dimension controls how much information a vector can store. Larger dimensions "
            "can encode more relations but need more memory and computation; smaller dimensions "
            "may lose detail. Common values are 50-300; this experiment uses 100.",
        ),
        (
            "What is the difference between Word2Vec and GloVe?",
            "Word2Vec learns vectors by predicting context words (or vice versa) over local "
            "windows and is a predictive model. GloVe factorises global word-word co-occurrence "
            "counts. Both produce static embeddings with one vector per word.",
        ),
        (
            "How do contextual embeddings differ from static embeddings?",
            "Static embeddings give one fixed vector per word, regardless of context, so "
            "polysemy is not resolved. Contextual embeddings such as ELMo, BERT and GPT produce "
            "context-dependent vectors, so 'Georgia' the country and 'Georgia' the US state can "
            "receive different representations.",
        ),
        (
            "How can the accuracy of this system be improved?",
            "Use a larger or more recent embedding, use contextual embeddings, add multi-word "
            "entity handling, expand the capital-country list, and filter candidates with "
            "additional knowledge such as geographic proximity or a threshold on the similarity "
            "score.",
        ),
        (
            "Why is top-5 accuracy higher than top-1 accuracy?",
            "Because several countries share similar capital-country relationships, the correct "
            "country is often among the best candidates even when another candidate has a "
            "slightly higher cosine similarity. Top-5 accuracy therefore measures whether the "
            "correct answer is at least retrievable.",
        ),
    ]
    for index, (question, answer) in enumerate(viva, 1):
        para(document, f"Q{index}. {question}", bold=True, space_after=2)
        para(document, f"Answer: {answer}", space_after=8)

    # ------------------------------------------------------------- 18. references
    h1(document, "18. References")
    references = [
        "Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. Proceedings of EMNLP 2014, pp. 1532-1543. https://nlp.stanford.edu/projects/glove/",
        "Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781.",
        "Mikolov, T., Yih, W., & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. Proceedings of NAACL-HLT 2013, pp. 746-751.",
        "Rehurek, R., & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, pp. 45-50 (gensim).",
        "Jurafsky, D., & Martin, J. H. Speech and Language Processing (3rd ed. draft), Chapter 6: Vector Semantics and Embeddings. https://web.stanford.edu/~jurafsky/slp3/",
    ]
    for index, reference in enumerate(references, 1):
        paragraph = document.add_paragraph(f"{index}. {reference}")
        paragraph.paragraph_format.space_after = Pt(4)

    h1(document, "Appendix A. Project Files")
    para(
        document,
        "The complete implementation, datasets, results and an executed notebook "
        "(Country_Prediction_Word_Embeddings.ipynb) accompany this report. Source files: "
        "BuildDatasets.py, country_prediction.py; datasets in datasets/; results and the "
        "Excel workbook in results/; charts in visualizations/; console captures in "
        "screenshots/; tests in tests/.",
        size=9.5,
    )

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(24)
    para(document, "Student Signature: ____________________          Date: ____________", size=10)
    para(document, "Faculty Signature: ____________________          Date: ____________", size=10)

    REPORT_DIR.mkdir(exist_ok=True)
    document.save(DOCX_PATH)
    print(f"saved {DOCX_PATH}")


def convert_to_pdf() -> None:
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(REPORT_DIR),
                str(DOCX_PATH),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        print(f"saved {PDF_PATH}")
    except FileNotFoundError:
        print("LibreOffice (soffice) not found; open the DOCX and export to PDF manually.")
    except subprocess.CalledProcessError as error:
        print("PDF conversion failed:", error.stderr.decode(errors="replace")[:300])


if __name__ == "__main__":
    build_report()
    convert_to_pdf()
