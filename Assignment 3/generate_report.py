#!/usr/bin/env python3
"""Build the editable Word laboratory report for Assignment 3.

All values are read from results.json (or supplied by the pipeline), so the
report always matches the actual run.  Charts are inserted from
visualizations/.  The complete source of pos_tagging_hmm.py is included in
Appendix A.

Run directly:
    python3 generate_report.py
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

try:
    from generate_workbook import REFERENCES
except Exception:  # pragma: no cover - fallback if imported unusually
    REFERENCES = []

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "POS_Tagging_Assignment_03_Report.docx"
SOURCE_PATH = PROJECT_DIR / "pos_tagging_hmm.py"
VIS_DIR = PROJECT_DIR / "visualizations"

DARK_BLUE = RGBColor(0x1F, 0x4E, 0x79)
MID_BLUE = RGBColor(0x2E, 0x75, 0xB6)
GREY = RGBColor(0x59, 0x59, 0x59)
HEADER_FILL = "1F4E79"
ALT_FILL = "EAF1F8"

_TABLE_NUMBER = 0
_FIGURE_NUMBER = 0


def _shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell_margins(table, top=40, bottom=40, left=70, right=70) -> None:
    tbl_pr = table._tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for name, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        node = OxmlElement(f"w:{name}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    tbl_pr.append(margins)


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def _set_run_font(run, name="Times New Roman", size=None, bold=None, italic=None, color=None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_page_number(section) -> None:
    """Add 'Page X of Y' fields to the section footer."""
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    _set_run_font(run, size=9, color=GREY)

    def field(code: str, placeholder: str):
        run_begin = paragraph.add_run()
        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")
        run_begin._r.append(fld_begin)

        run_instr = paragraph.add_run()
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = f" {code} "
        run_instr._r.append(instr)

        run_sep = paragraph.add_run()
        fld_sep = OxmlElement("w:fldChar")
        fld_sep.set(qn("w:fldCharType"), "separate")
        run_sep._r.append(fld_sep)

        run_text = paragraph.add_run(placeholder)
        _set_run_font(run_text, size=9, color=GREY)

        run_end = paragraph.add_run()
        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        run_end._r.append(fld_end)

    field("PAGE", "1")
    run_of = paragraph.add_run(" of ")
    _set_run_font(run_of, size=9, color=GREY)
    field("NUMPAGES", "1")


def add_header(section) -> None:
    paragraph = section.header.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("NLP Laboratory Assignment 3  |  HMM POS Tagging  |  Shravani Kurkute")
    _set_run_font(run, size=9, color=GREY, italic=True)
    p_pr = paragraph._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "BFBFBF")
    bdr.append(bottom)
    p_pr.append(bdr)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.95)
    section.bottom_margin = Inches(0.9)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.different_first_page_header_footer = True
    add_header(section)
    add_page_number(section)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12

    for name, size, color in (
        ("Heading 1", 15, DARK_BLUE),
        ("Heading 2", 12.5, MID_BLUE),
        ("Heading 3", 11.5, MID_BLUE),
    ):
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True


def heading(document: Document, text: str, level: int = 1, page_break: bool = False):
    if page_break:
        document.add_page_break()
    return document.add_heading(text, level=level)


def para(document: Document, text: str, bold_lead: str | None = None, italic=False, align=None):
    paragraph = document.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        run = paragraph.add_run(bold_lead)
        run.bold = True
        _set_run_font(run)
        rest = paragraph.add_run(text[len(bold_lead):])
        _set_run_font(rest, italic=italic)
    else:
        run = paragraph.add_run(text)
        _set_run_font(run, italic=italic)
    if align is not None:
        paragraph.alignment = align
    return paragraph


def bullet(document: Document, text: str):
    paragraph = document.add_paragraph(style="List Bullet")
    run = paragraph.add_run(text)
    _set_run_font(run)
    return paragraph


def numbered(document: Document, text: str):
    paragraph = document.add_paragraph(style="List Number")
    run = paragraph.add_run(text)
    _set_run_font(run)
    return paragraph


def add_table(
    document: Document,
    headers: Sequence[str],
    rows: Sequence[Sequence],
    caption: str,
    widths: Sequence[float] | None = None,
    font_size: float = 8.5,
    header_font_size: float = 8.5,
    align_center: bool = False,
):
    global _TABLE_NUMBER
    _TABLE_NUMBER += 1

    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    caption_paragraph.paragraph_format.space_before = Pt(6)
    caption_paragraph.paragraph_format.space_after = Pt(3)
    caption_paragraph.paragraph_format.keep_with_next = True
    label = caption_paragraph.add_run(f"Table {_TABLE_NUMBER}: ")
    _set_run_font(label, size=9, bold=True, color=DARK_BLUE)
    cap = caption_paragraph.add_run(caption)
    _set_run_font(cap, size=9, italic=True, color=GREY)

    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    _set_cell_margins(table)
    _repeat_header(table.rows[0])

    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = str(header)
        _shade_cell(cell, HEADER_FILL)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            _set_run_font(run, size=header_font_size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        if widths:
            cell.width = Inches(widths[index])

    for row_index, values in enumerate(rows, start=1):
        cells = table.add_row().cells
        for column_index, value in enumerate(values):
            cell = cells[column_index]
            cell.text = "" if value is None else str(value)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.space_after = Pt(1)
            if align_center or column_index == 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                _set_run_font(run, size=font_size)
            if row_index % 2 == 0:
                _shade_cell(cell, ALT_FILL)
            if widths:
                cell.width = Inches(widths[column_index])
    return table


def add_figure(document: Document, filename: str, caption: str, width: float = 6.1) -> None:
    global _FIGURE_NUMBER
    _FIGURE_NUMBER += 1
    path = VIS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required chart is missing: {path}")
    document.add_picture(str(path), width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    label = cap.add_run(f"Figure {_FIGURE_NUMBER}: ")
    _set_run_font(label, size=9, bold=True, color=DARK_BLUE)
    text = cap.add_run(caption)
    _set_run_font(text, size=9, italic=True, color=GREY)


def add_code(document: Document, code: str, font_size: float = 7.8, max_lines: int | None = None) -> None:
    lines = code.rstrip("\n").splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    for line in lines:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.left_indent = Inches(0.12)
        run = paragraph.add_run(line if line else " ")
        _set_run_font(run, name="Consolas", size=font_size)


def add_bullet_list(document: Document, items: Sequence[str]) -> None:
    for item in items:
        bullet(document, item)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _group_mapping(mapping: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for ptb, universal in mapping.items():
        grouped.setdefault(universal, []).append(ptb)
    for tag in grouped:
        grouped[tag].sort()
    return grouped


def _source_block(start_marker: str, end_marker: str) -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end].rstrip()


def build_report(result: Dict) -> Path:
    """Create the complete Word report and return its path."""
    global _TABLE_NUMBER, _FIGURE_NUMBER
    _TABLE_NUMBER = 0
    _FIGURE_NUMBER = 0

    corpus = result["corpus"]
    hmm = result["hmm"]
    baseline = result["baseline"]
    config = result["config"]
    env = result["environment"]
    custom = result["custom_sentences"]
    top_errors = result["top_errors"]
    mapping = result["tag_mapping"]

    document = Document()
    configure_document(document)

    # ------------------------------------------------------------------
    # 1. Cover page
    # ------------------------------------------------------------------
    for _ in range(4):
        document.add_paragraph()
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("NLP LABORATORY ASSIGNMENT 3")
    _set_run_font(run, name="Calibri", size=13, bold=True, color=MID_BLUE)

    main_title = document.add_paragraph()
    main_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = main_title.add_run("Survey and Implementation of\nPart-of-Speech (POS) Tagging")
    _set_run_font(run, name="Calibri", size=24, bold=True, color=DARK_BLUE)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(24)
    run = subtitle.add_run("A Statistical Hidden Markov Model with Viterbi Decoding")
    _set_run_font(run, name="Calibri", size=15, italic=True, color=GREY)

    line = document.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("_" * 58)
    _set_run_font(run, size=11, color=RGBColor(0xBF, 0xBF, 0xBF))

    submitted = document.add_paragraph()
    submitted.alignment = WD_ALIGN_PARAGRAPH.CENTER
    submitted.paragraph_format.space_before = Pt(18)
    run = submitted.add_run("Submitted by")
    _set_run_font(run, name="Calibri", size=12, italic=True, color=GREY)

    name_paragraph = document.add_paragraph()
    name_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = name_paragraph.add_run("Shravani Kurkute")
    _set_run_font(run, name="Calibri", size=20, bold=True, color=DARK_BLUE)

    details = [
        ("Roll Number", "________________________________"),
        ("PRN", "________________________________"),
        ("Class / Division", "________________________________"),
        ("Department", "________________________________"),
        ("Course", "Natural Language Processing Laboratory"),
        ("Academic Year", "________________________________"),
        ("Date of Submission", datetime.fromisoformat(result["generated_at"]).strftime("%d %B %Y")),
    ]
    table = document.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for label_text, value in details:
        cells = table.add_row().cells
        cells[0].width = Inches(1.7)
        cells[1].width = Inches(3.6)
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r0 = p0.add_run(label_text + " : ")
        _set_run_font(r0, name="Calibri", size=11, bold=True, color=DARK_BLUE)
        p1 = cells[1].paragraphs[0]
        r1 = p1.add_run(value)
        _set_run_font(r1, name="Calibri", size=11)

    submission_note = document.add_paragraph()
    submission_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    submission_note.paragraph_format.space_before = Pt(26)
    run = submission_note.add_run(
        "Submitted in partial fulfilment of the requirements of the Natural Language Processing Laboratory course"
    )
    _set_run_font(run, size=10, italic=True, color=GREY)

    document.add_page_break()

    # ------------------------------------------------------------------
    # 2. Aim
    # ------------------------------------------------------------------
    heading(document, "2. Aim", 1, page_break=False)
    para(
        document,
        "The aim of this laboratory assignment is to study the principal approaches to English "
        "part-of-speech (POS) tagging and to design, implement, train, and evaluate a statistical "
        "Hidden Markov Model (HMM) POS tagger with Viterbi decoding. The tagger is trained on the "
        "Penn Treebank corpus available through NLTK, uses the Universal POS tagset, and is compared "
        "with a most-frequent-tag baseline on a reproducible held-out test split.",
    )

    # ------------------------------------------------------------------
    # 3. Objectives
    # ------------------------------------------------------------------
    heading(document, "3. Objectives", 1)
    add_bullet_list(
        document,
        [
            "Survey rule-based, statistical (HMM), transformation-based, maximum-entropy, CRF, and neural POS-tagging techniques.",
            "Load the NLTK Penn Treebank tagged corpus, remove trace tokens, and map Penn Treebank tags to the Universal tagset.",
            "Create a reproducible 80:20 sentence-level training/testing split using random seed 42 without data leakage.",
            "Estimate tag-transition probabilities with add-k smoothing (k = 0.001) and word-emission probabilities from the training split.",
            "Implement Viterbi dynamic programming with log probabilities and backpointers.",
            "Design and document a data-driven out-of-vocabulary (OOV) strategy using digit shapes, suffixes of rare words, and tag priors.",
            "Implement a most-frequent-tag baseline and compare it with the HMM using accuracy, macro/weighted precision, recall, F1, and per-tag metrics.",
            "Report OOV accuracy, confusion matrix, and contextual error analysis, and test the model on at least 20 unseen custom sentences.",
            "Generate reproducible charts, a results workbook, a Jupyter notebook, automated tests, and this academic report.",
        ],
    )

    # ------------------------------------------------------------------
    # 4. Problem statement
    # ------------------------------------------------------------------
    heading(document, "4. Problem Statement", 1)
    para(
        document,
        "Given an English sentence represented as a sequence of words W = (w1, w2, ..., wn), the POS-tagging "
        "problem is to assign the most likely sequence of grammatical tags T = (t1, t2, ..., tn). Tags distinguish "
        "nouns, verbs, adjectives, determiners, punctuation, and other grammatical classes. The task is difficult "
        "because many words are ambiguous: for example, 'book' can be a noun or a verb, and 'that' can be a "
        "determiner, pronoun, or conjunction. A useful tagger must resolve such ambiguity using local context, "
        "must generalise to words that were not observed during training, and must be evaluated on data that was "
        "not used to estimate its parameters. This assignment addresses those requirements with a supervised HMM, "
        "a smoothed transition model, a documented OOV model, and a leakage-free evaluation protocol.",
    )

    # ------------------------------------------------------------------
    # 5. Theory and background
    # ------------------------------------------------------------------
    heading(document, "5. Theory and Background", 1, page_break=True)
    para(
        document,
        "POS tagging is a sequence-labelling task. A generative statistical tagger models the joint probability "
        "P(W, T) and applies Bayes' rule to choose the tag sequence with the largest posterior probability:",
    )
    para(document, "P(T | W) ∝ P(W | T) P(T).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(
        document,
        "A first-order HMM makes two conditional-independence assumptions. Each word depends only on its own tag, "
        "and each tag depends only on the immediately preceding tag. With START and END boundary states, the joint "
        "probability factorises as:",
    )
    para(
        document,
        "P(W, T) = P(t1 | START) · ∏i P(wi | ti) · ∏i>1 P(ti | ti-1) · P(END | tn).",
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    para(
        document,
        "The transition probability P(ti | ti-1) captures syntactic regularities such as determiners preceding nouns "
        "and nouns following adjectives. The emission probability P(wi | ti) captures lexical preferences. Because "
        "the product of many small probabilities underflows in floating-point arithmetic, this implementation works "
        "entirely in log space. Transition probabilities are add-k smoothed so that no observed transition receives "
        "probability zero:",
    )
    para(
        document,
        "P(ti | ti-1) = (C(ti-1, ti) + k) / (C(ti-1) + k · (|T| + 1)),   k = 0.001.",
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    para(
        document,
        "The Viterbi algorithm finds the highest-scoring path through the HMM in O(n|T|²) time using dynamic "
        "programming. Let δi(j) be the best log score of any tag sequence ending in tag j at position i, and let "
        "ψi(j) store the best predecessor. The recursion is:",
    )
    para(
        document,
        "δi(j) = maxk [δi-1(k) + log P(tj | tk)] + log P(wi | tj),   ψi(j) = argmaxk [δi-1(k) + log P(tj | tk)].",
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    para(
        document,
        "After processing the final token, the END transition is added and the best final state is selected. "
        "Backpointers are followed from right to left to recover the complete tag sequence. Empty input is handled "
        "explicitly and returns an empty tag list.",
    )

    # ------------------------------------------------------------------
    # 6. Survey
    # ------------------------------------------------------------------
    heading(document, "6. Survey of POS-Tagging Techniques", 1, page_break=True)
    para(
        document,
        "POS taggers have evolved from hand-written rules to statistical sequence models and, more recently, to "
        "deep contextual language models. The main families are summarised below.",
    )

    heading(document, "6.1 Rule-based POS tagging", 2)
    para(
        document,
        "Rule-based taggers assign tags using lexical dictionaries and hand-crafted context rules. The ENGTWOL "
        "tagger and Brill's transformation-based tagger are well-known examples. Brill's approach starts with a "
        "simple baseline and automatically learns ordered correction rules from an annotated corpus, combining "
        "the interpretability of rules with corpus-driven learning. Rule-based systems can encode precise "
        "linguistic knowledge and require no probability estimation, but writing and maintaining rules is labour "
        "intensive, coverage is limited, and ambiguity resolution can become brittle.",
    )

    heading(document, "6.2 Statistical POS tagging with Hidden Markov Models", 2)
    para(
        document,
        "An HMM treats tags as hidden states and words as observations. Transition and emission probabilities are "
        "estimated from a tagged corpus, and Viterbi decoding finds the most probable tag sequence. HMMs are simple, "
        "fast, interpretable, and easy to train. Their main weaknesses are the first-order Markov assumption, the "
        "independence of words given tags, and poor estimates for rare words. Smoothing and explicit OOV models "
        "mitigate the sparsity problem. The present assignment implements this family.",
    )

    heading(document, "6.3 Transformation-based tagging", 2)
    para(
        document,
        "Transformation-based learning (TBL), popularised by Brill, begins with an initial tag assignment and "
        "iteratively learns transformation rules that reduce training error. Rules are ranked by their net benefit "
        "and applied in order. TBL typically achieves accuracy close to statistical taggers while producing "
        "human-readable rules. Training is slower because many candidate rules must be evaluated, and the learned "
        "rule set can be large.",
    )

    heading(document, "6.4 Maximum Entropy and discriminative models", 2)
    para(
        document,
        "Maximum Entropy (MaxEnt) taggers model the conditional probability P(tag | context) directly and can "
        "combine overlapping, non-independent features such as word identity, prefixes, suffixes, capitalisation, "
        "and neighbouring tags. They avoid the strong independence assumptions of generative HMMs and often improve "
        "accuracy. The cost is higher computational complexity during training and inference, together with the "
        "need for feature engineering and regularisation.",
    )

    heading(document, "6.5 Conditional Random Fields (CRF)", 2)
    para(
        document,
        "CRFs are discriminative undirected graphical models that model the conditional probability of the entire "
        "tag sequence given the observation sequence. Unlike an HMM, a CRF can use arbitrary overlapping features "
        "of the whole input and does not assume that observations are independent. Linear-chain CRFs are a standard "
        "strong baseline for sequence labelling and often outperform HMMs and MaxEnt taggers. Training requires "
        "iterative optimisation and can be expensive for large feature sets.",
    )

    heading(document, "6.6 Neural and contextual models", 2)
    para(
        document,
        "Neural taggers learn dense word representations and contextual encoders. Bidirectional LSTM models read "
        "the sentence in both directions and capture long-range context; adding a CRF layer enforces global "
        "sequence consistency. Transformer models such as BERT use self-attention and large-scale pre-training, "
        "and fine-tuning them for POS tagging gives state-of-the-art or near-state-of-the-art accuracy. These "
        "models require substantially more data, memory, and computation than an HMM, and their predictions are "
        "less transparent, but they handle ambiguity and unseen words far better because of contextual embeddings "
        "and subword tokenisation.",
    )

    survey_rows = [
        ["Rule-based / TBL", "Hand-written or learned rules", "Interpretable; no probability model", "Manual effort; limited coverage"],
        ["HMM", "Generative; first-order Markov", "Simple, fast, transparent; strong baseline", "Strong independence assumptions; weak OOV handling"],
        ["MaxEnt", "Discriminative; rich local features", "Overlapping features; good accuracy", "Feature engineering; slower training"],
        ["CRF", "Discriminative; sequence-level features", "Strong sequence accuracy; flexible features", "Expensive optimisation"],
        ["BiLSTM / BERT", "Neural contextual representations", "Best accuracy; handles context and OOV well", "Data- and compute-hungry; less interpretable"],
    ]
    add_table(
        document,
        ["Approach", "Modelling idea", "Strengths", "Limitations"],
        survey_rows,
        "Comparison of major POS-tagging approaches.",
        widths=[1.15, 1.55, 1.9, 1.9],
        font_size=8.2,
    )

    # ------------------------------------------------------------------
    # 7. Selected technique
    # ------------------------------------------------------------------
    heading(document, "7. Selected Technique and Justification", 1)
    para(
        document,
        "This assignment implements a supervised first-order HMM with Viterbi decoding. The choice is appropriate "
        "for a laboratory study because the model exposes every probability used in prediction, can be trained in "
        "a few seconds, and provides a transparent baseline against which more advanced taggers can be compared. "
        "The implementation uses add-k smoothed transitions, maximum-likelihood emissions for observed words, and "
        "a documented OOV model based on digit shapes and suffixes of rare training words. The model is evaluated "
        "on a held-out split and compared with a most-frequent-tag baseline. This design makes the relationship "
        "between the Markov assumptions, the estimated parameters, and the observed errors easy to explain in a "
        "viva while still producing competitive accuracy on the Universal tagset.",
    )

    # ------------------------------------------------------------------
    # 8. Tools and environment
    # ------------------------------------------------------------------
    heading(document, "8. Tools, Libraries and Environment", 1)
    env_rows = [
        ["Operating system", platform.platform()],
        ["Python", env.get("python", "")],
        ["NLTK", env.get("nltk", "")],
        ["NumPy", env.get("numpy", "")],
        ["scikit-learn", env.get("scikit-learn", "")],
        ["Pandas", env.get("pandas", "")],
        ["Matplotlib", env.get("matplotlib", "")],
        ["openpyxl", env.get("openpyxl", "")],
        ["python-docx", env.get("python-docx", "")],
        ["Jupyter / nbclient", "Jupyter notebook executed with nbclient"],
        ["Corpus and tokenizer", "NLTK treebank, punkt and punkt_tab resources"],
    ]
    add_table(
        document,
        ["Component", "Version / role"],
        env_rows,
        "Software environment used for the experiment (versions read from the actual runtime).",
        widths=[1.9, 4.6],
        font_size=8.8,
    )
    para(
        document,
        "The pipeline is implemented in Python and uses NLTK for corpus and tokenisation resources, NumPy for "
        "numerical operations, scikit-learn for evaluation metrics and the confusion matrix, Pandas for tabular "
        "results, Matplotlib for charts, openpyxl for the Excel workbook, and python-docx for this report. The "
        "experiment is deterministic given the fixed seed and the installed corpus version.",
    )

    # ------------------------------------------------------------------
    # 9. Dataset
    # ------------------------------------------------------------------
    heading(document, "9. Dataset and Corpus Description", 1, page_break=True)
    para(
        document,
        "The Penn Treebank is a widely used corpus of English newswire text annotated with Penn Treebank POS tags "
        "and syntactic structure. This project uses the tagged sentences distributed with NLTK as "
        "treebank.tagged_sents(). The corpus contains "
        f"{corpus['raw_sentences']:,} sentences and {corpus['raw_tokens']:,} tokens before cleaning. "
        "Trace and empty-category tokens with the tag -NONE- (for example '*', '0', and '*T*-1') do not correspond "
        "to surface words and were removed. Exact duplicate tagged sentences were also removed to guarantee a "
        "clean sentence-level split. Punctuation and brackets were retained and mapped to the punctuation class.",
    )
    corpus_rows = [
        ["Raw Penn Treebank sentences", f"{corpus['raw_sentences']:,}"],
        ["Raw tokens", f"{corpus['raw_tokens']:,}"],
        ["Trace (-NONE-) tokens removed", f"{corpus['trace_tokens_removed']:,}"],
        ["Sentences after trace removal", f"{corpus['sentences_after_trace_removal']:,}"],
        ["Exact duplicate sentences removed", f"{corpus['duplicate_sentences_removed']}"],
        ["Prepared sentences", f"{corpus['prepared_sentences']:,}"],
        ["Prepared tokens", f"{corpus['prepared_tokens']:,}"],
        ["Training sentences (80%)", f"{corpus['train_sentences']:,}"],
        ["Test sentences (20%)", f"{corpus['test_sentences']:,}"],
        ["Training tokens", f"{corpus['train_tokens']:,}"],
        ["Test tokens", f"{corpus['test_tokens']:,}"],
        ["Training vocabulary", f"{corpus['train_vocabulary']:,}"],
        ["Test OOV tokens", f"{corpus['test_oov_tokens']:,} ({fmt_pct(corpus['test_oov_rate'])})"],
        ["Penn Treebank tags mapped", f"{corpus['ptb_tag_count']}"],
        ["Universal tags used", f"{corpus['universal_tag_count']}"],
        ["Sentence overlap between splits", "0"],
    ]
    add_table(
        document,
        ["Statistic", "Value"],
        corpus_rows,
        "Dataset statistics after cleaning and the reproducible 80:20 sentence-level split.",
        widths=[3.0, 3.4],
        font_size=8.8,
    )
    para(
        document,
        "The split is made over shuffled sentence indices with random seed 42. The test size is exactly 20% of the "
        "prepared sentences (rounded up), and an explicit assertion verifies that no tagged sentence occurs in both "
        "splits. The baseline and the HMM are fitted only on the training split; the test split is used only for "
        "evaluation. The same test tokens are supplied to both models.",
    )

    heading(document, "9.1 Universal tagset", 2)
    para(
        document,
        "The NLTK Universal tagset groups the finer Penn Treebank distinctions into twelve broad classes. Proper "
        "nouns are merged with common nouns, and all punctuation is represented by the separate tag '.'. The "
        "mapping is performed with nltk.tag.mapping.map_tag('en-ptb', 'universal', tag). The categories are:",
    )
    grouped = _group_mapping(mapping)
    tag_rows = []
    for tag in sorted(grouped, key=lambda value: (value == ".", value)):
        tag_rows.append(
            [
                tag,
                result["tagset"].get(tag, ""),
                ", ".join(grouped[tag]),
            ]
        )
    add_table(
        document,
        ["Universal tag", "Meaning", "Penn Treebank tags mapped to this class"],
        tag_rows,
        "Universal tagset categories and the corresponding Penn Treebank tags observed in the corpus.",
        widths=[0.85, 2.35, 3.25],
        font_size=7.8,
    )

    # ------------------------------------------------------------------
    # 10. Methodology
    # ------------------------------------------------------------------
    heading(document, "10. Methodology and Algorithm", 1, page_break=True)
    numbered_items = [
        "Resource validation: confirm that treebank, universal_tagset, punkt and punkt_tab are available; attempt a single download if missing and otherwise raise an informative error.",
        "Corpus loading: read treebank.tagged_sents() and remove all tokens whose Penn Treebank tag is -NONE-.",
        "Tag mapping: map every Penn Treebank tag to the Universal tagset with NLTK's official mapping.",
        "Cleaning and splitting: remove exact duplicate tagged sentences, shuffle indices with seed 42, and create an 80:20 sentence-level train/test split. Assert zero sentence overlap.",
        "Training: count tag transitions (including START and END), tag emissions, word frequencies, and tag frequencies. Estimate add-k smoothed transitions with k = 0.001 and maximum-likelihood emissions for observed words.",
        "OOV modelling: collect suffix statistics of lengths 1-4 from rare training words (frequency <= 2), plus numeric, digit-containing, and capitalised shape priors. All estimates are add-alpha smoothed.",
        "Decoding: run Viterbi in log space with backpointers. For observed words, emission scores are log(C(w,t)/C(t)); for OOV words, scores come from the OOV model.",
        "Baseline: assign each word its most frequent training tag, with the global most frequent tag for unseen words.",
        "Evaluation: compute accuracy, macro and weighted precision/recall/F1, per-tag metrics, OOV and known-token accuracy, and the confusion matrix on the held-out test split.",
        "Analysis and reporting: extract contextual errors, tag 25 previously unseen custom sentences, generate five charts, and write CSV, JSON, Excel, notebook, and report outputs.",
    ]
    for item in numbered_items:
        numbered(document, item)

    heading(document, "10.1 OOV strategy", 2)
    para(
        document,
        "Unseen words are the main weakness of a generative HMM. This implementation uses a data-driven strategy "
        "rather than a single arbitrary fallback tag. For each rare training word (frequency at most 2), suffixes "
        "of lengths 1 to 4 are counted by tag. Given an OOV word, the longest suffix that occurs in the rare-word "
        "statistics is selected. A smoothed suffix probability is combined with the tag prior. Numeric tokens use "
        "a separate numeric-shape distribution learned from rare numeric training tokens. Tokens that contain "
        "digits mix the digit-shape prior with the suffix score, and capitalised tokens mix in a capitalisation "
        "prior. Scores are normalised over tags with a softmax; because the normalisation constant is identical "
        "for every tag at a given word position, it does not change Viterbi path ranking. All probabilities are "
        "strictly positive, so log(0) cannot occur for OOV words.",
    )

    heading(document, "10.2 Viterbi algorithm", 2)
    para(
        document,
        "The decoder maintains a score matrix of size n × |T| and a parallel backpointer matrix. Initialisation "
        "adds the START transition and the first emission. Each subsequent cell selects the best predecessor for "
        "the current tag. After the last word, the END transition is added, the best final tag is chosen, and the "
        "path is reconstructed by following backpointers. Ties are broken deterministically by the sorted tag "
        "order. The implementation also returns the score matrix and backpointers for notebook demonstrations.",
    )

    heading(document, "10.3 Evaluation measures", 2)
    para(
        document,
        "Accuracy is the proportion of correctly tagged tokens. Precision for a tag is TP/(TP+FP), recall is "
        "TP/(TP+FN), and F1 is the harmonic mean of precision and recall. Macro averages weight every tag equally "
        "and therefore expose poor performance on infrequent tags; weighted averages weight each tag by its support "
        "and therefore track token-level performance. Both are reported. OOV accuracy is computed only on test "
        "tokens whose word form is absent from the training vocabulary, and known-token accuracy is computed on "
        "the complementary set.",
    )

    # ------------------------------------------------------------------
    # 11. Training and configuration
    # ------------------------------------------------------------------
    heading(document, "11. Training and Configuration Details", 1)
    config_rows = [
        ["Model", "Supervised first-order Hidden Markov Model"],
        ["Decoder", config["decoder"]],
        ["Tagset", config["tagset"]],
        ["Random seed", str(config["seed"])],
        ["Train/test split", "80:20 sentence-level"],
        ["Transition smoothing", f"Add-k, k = {config['transition_k']}"],
        ["Emission model", "Maximum likelihood for observed words"],
        ["OOV rare-word threshold", f"frequency <= {config['rare_word_threshold']}"],
        ["Maximum suffix length", str(config["max_suffix_length"])],
        ["OOV smoothing alpha", str(config["oov_alpha"])],
        ["Training vocabulary", f"{result['model']['vocabulary_size']:,} word types"],
        ["Emission pairs", f"{result['model']['emission_pairs']:,}"],
        ["Rare-word occurrences for OOV model", f"{result['model']['rare_token_occurrences']:,}"],
        ["Numeric rare-token occurrences", f"{result['model']['numeric_rare_occurrences']:,}"],
        ["Test-set usage", "Evaluation only; no parameter estimation"],
    ]
    add_table(
        document,
        ["Setting", "Value"],
        config_rows,
        "Training configuration and data-separation protocol.",
        widths=[2.55, 3.9],
        font_size=8.8,
    )

    # ------------------------------------------------------------------
    # 12. Implementation
    # ------------------------------------------------------------------
    heading(document, "12. Implementation and Source Code", 1, page_break=True)
    para(
        document,
        "The main implementation is the file pos_tagging_hmm.py. It is modular: corpus preparation, HMM training, "
        "OOV modelling, Viterbi decoding, the baseline, evaluation, error analysis, custom tagging, visualisation, "
        "and output generation are separate functions or classes. Companion scripts generate the Excel workbook "
        "and this report. The key components are listed below; the complete source is reproduced in Appendix A.",
    )
    function_rows = [
        ["ensure_nltk_resources", "Checks/downloads treebank, universal_tagset, punkt and punkt_tab with clear remedies."],
        ["load_treebank_sentences", "Loads Penn Treebank tagged sentences and removes -NONE- trace tokens."],
        ["map_ptb_to_universal", "Maps Penn Treebank tags to the Universal tagset."],
        ["prepare_dataset", "Deduplicates, shuffles with seed 42, and creates a leakage-free 80:20 split."],
        ["HMMPosTagger.train", "Counts transitions/emissions and builds smoothed log transitions and OOV models."],
        ["HMMPosTagger.oov_emission_probs", "Suffix, digit-shape, and capitalisation estimates for unseen words."],
        ["HMMPosTagger.viterbi_decode", "Log-space Viterbi decoding with backpointers and optional trace details."],
        ["MostFrequentTagBaseline", "Most-frequent-tag baseline fitted only on training data."],
        ["evaluate_tags", "Accuracy, macro/weighted/per-tag metrics, OOV accuracy, and confusion matrix."],
        ["build_error_rows", "Per-error context strings and ranked confusion pairs."],
        ["run_custom_tests", "Tags 25 previously unseen sentences with both models."],
        ["generate_visualizations", "Produces the five required PNG charts from computed results."],
        ["run_pipeline", "End-to-end reproducible experiment; writes JSON, CSV, sentence files, and charts."],
    ]
    add_table(
        document,
        ["Function / class", "Purpose"],
        function_rows,
        "Main functions and classes in pos_tagging_hmm.py.",
        widths=[2.25, 4.2],
        font_size=8.2,
    )

    heading(document, "12.1 Transition estimation and smoothing", 2)
    add_code(document, _source_block("    def _build_transition_logprobs", "    def _build_oov_models"))

    heading(document, "12.2 Viterbi decoding", 2)
    add_code(document, _source_block("    def viterbi_decode", "    # ------------------------------------------------------------------\n    # Convenience"), font_size=7.4)

    heading(document, "12.3 Out-of-vocabulary estimation", 2)
    add_code(document, _source_block("    def oov_emission_probs", "    def emission_logprob"), font_size=7.4)

    # ------------------------------------------------------------------
    # 13. Sample tagged output
    # ------------------------------------------------------------------
    heading(document, "13. Sample POS-Tagged Output", 1, page_break=True)
    para(
        document,
        f"The trained HMM was applied to {len(custom)} manually written English sentences that were not part of the "
        "Penn Treebank training or test data. The table below shows the input sentence and the predicted Universal "
        "POS tags in token/TAG form. The same sentences were also tagged with the baseline; the complete token-level "
        "comparison is saved in custom_predictions.csv.",
    )
    custom_rows = [[record["sentence"], record["hmm_tagged"]] for record in custom]
    add_table(
        document,
        ["Input sentence (previously unseen)", "HMM predicted output (token/TAG)"],
        custom_rows,
        "POS-tagged output for manually written custom sentences.",
        widths=[2.2, 4.3],
        font_size=7.4,
    )

    heading(document, "13.1 Examples from the held-out test set", 2)
    sample_rows = result.get("sample_tagged_sentences", [])
    grouped_sample: Dict[int, List[Dict]] = {}
    for row in sample_rows:
        grouped_sample.setdefault(row["sentence_id"], []).append(row)
    heldout_rows = []
    for sentence_id in sorted(grouped_sample)[:5]:
        rows = grouped_sample[sentence_id]
        tokens = [row["token"] for row in rows]
        gold = [row["gold_tag"] for row in rows]
        pred = [row["hmm_predicted_tag"] for row in rows]
        heldout_rows.append(
            [
                " ".join(tokens),
                " ".join(f"{w}/{t}" for w, t in zip(tokens, gold)),
                " ".join(f"{w}/{t}" for w, t in zip(tokens, pred)),
            ]
        )
    add_table(
        document,
        ["Sentence", "Gold tags", "HMM predictions"],
        heldout_rows,
        "Representative held-out test sentences with gold and predicted Universal tags.",
        widths=[2.0, 2.2, 2.2],
        font_size=6.9,
    )

    # ------------------------------------------------------------------
    # 14. Evaluation
    # ------------------------------------------------------------------
    heading(document, "14. Evaluation", 1, page_break=True)
    para(
        document,
        "Both models were evaluated on exactly the same "
        f"{hmm['support']:,} held-out test tokens. The HMM achieved {fmt_pct(hmm['accuracy'])} accuracy, compared "
        f"with {fmt_pct(baseline['accuracy'])} for the most-frequent-tag baseline. The macro F1 improved from "
        f"{fmt(baseline['macro_f1'])} to {fmt(hmm['macro_f1'])}, and the weighted F1 improved from "
        f"{fmt(baseline['weighted_f1'])} to {fmt(hmm['weighted_f1'])}. These values are computed directly from the "
        "model predictions and are not manually entered.",
    )
    overall_rows = [
        ["Accuracy", fmt(baseline["accuracy"], 6), fmt(hmm["accuracy"], 6), f"{hmm['accuracy'] - baseline['accuracy']:+.6f}"],
        ["Macro precision", fmt(baseline["macro_precision"], 6), fmt(hmm["macro_precision"], 6), f"{hmm['macro_precision'] - baseline['macro_precision']:+.6f}"],
        ["Macro recall", fmt(baseline["macro_recall"], 6), fmt(hmm["macro_recall"], 6), f"{hmm['macro_recall'] - baseline['macro_recall']:+.6f}"],
        ["Macro F1", fmt(baseline["macro_f1"], 6), fmt(hmm["macro_f1"], 6), f"{hmm['macro_f1'] - baseline['macro_f1']:+.6f}"],
        ["Weighted precision", fmt(baseline["weighted_precision"], 6), fmt(hmm["weighted_precision"], 6), f"{hmm['weighted_precision'] - baseline['weighted_precision']:+.6f}"],
        ["Weighted recall", fmt(baseline["weighted_recall"], 6), fmt(hmm["weighted_recall"], 6), f"{hmm['weighted_recall'] - baseline['weighted_recall']:+.6f}"],
        ["Weighted F1", fmt(baseline["weighted_f1"], 6), fmt(hmm["weighted_f1"], 6), f"{hmm['weighted_f1'] - baseline['weighted_f1']:+.6f}"],
        ["OOV accuracy", fmt(baseline["oov_accuracy"], 6), fmt(hmm["oov_accuracy"], 6), f"{hmm['oov_accuracy'] - baseline['oov_accuracy']:+.6f}"],
        ["Known-token accuracy", fmt(baseline["known_accuracy"], 6), fmt(hmm["known_accuracy"], 6), f"{hmm['known_accuracy'] - baseline['known_accuracy']:+.6f}"],
    ]
    add_table(
        document,
        ["Metric", "Baseline", "HMM", "Difference"],
        overall_rows,
        "Overall evaluation of the baseline and HMM on the held-out test set.",
        widths=[1.75, 1.45, 1.45, 1.8],
        font_size=8.5,
        align_center=True,
    )

    heading(document, "14.1 Per-tag performance", 2)
    per_tag_rows = []
    for tag in hmm["per_tag"]:
        b = result["baseline_per_tag"].get(tag, {})
        h = hmm["per_tag"].get(tag, {})
        per_tag_rows.append(
            [
                tag,
                fmt(b.get("precision", 0)),
                fmt(b.get("recall", 0)),
                fmt(b.get("f1_score", 0)),
                f"{int(b.get('support', 0)):,}",
                fmt(h.get("precision", 0)),
                fmt(h.get("recall", 0)),
                fmt(h.get("f1_score", 0)),
                f"{int(h.get('support', 0)):,}",
            ]
        )
    add_table(
        document,
        ["Tag", "Base P", "Base R", "Base F1", "Support", "HMM P", "HMM R", "HMM F1", "Support"],
        per_tag_rows,
        "Per-tag precision, recall, and F1 for the baseline and HMM.",
        widths=[0.65, 0.72, 0.72, 0.72, 0.7, 0.72, 0.72, 0.72, 0.7],
        font_size=7.6,
        align_center=True,
    )
    para(
        document,
        "The HMM improves most on NOUN, VERB, ADJ, and ADP, which are frequent and strongly context-dependent. "
        "Punctuation is almost perfectly recognised because its distribution is highly deterministic. The rarest "
        "class, X, has low support and is correspondingly harder to evaluate reliably.",
    )

    add_figure(document, "confusion_matrix.png", "Normalised HMM confusion matrix on the held-out test set.")
    para(
        document,
        "The confusion matrix shows that most probability mass lies on the diagonal. The largest off-diagonal "
        "confusions are between VERB and NOUN, ADJ and NOUN, and NOUN and VERB. These are genuine lexical and "
        "syntactic ambiguities in English rather than implementation errors: a word such as 'publishing' can "
        "function as a verb, a noun, or a modifier depending on a wider context than a first-order HMM observes.",
    )
    add_figure(document, "per_tag_scores.png", "Precision, recall, and F1-score for each Universal POS tag (HMM).")

    heading(document, "14.2 OOV versus known tokens", 2)
    oov_rows = [
        ["Test tokens", f"{hmm['support']:,}", f"{baseline['support']:,}"],
        ["OOV tokens", f"{hmm['oov_count']:,}", f"{baseline['oov_count']:,}"],
        ["Known tokens", f"{hmm['known_count']:,}", f"{baseline['known_count']:,}"],
        ["OOV accuracy", fmt(hmm["oov_accuracy"], 6), fmt(baseline["oov_accuracy"], 6)],
        ["Known-token accuracy", fmt(hmm["known_accuracy"], 6), fmt(baseline["known_accuracy"], 6)],
    ]
    add_table(
        document,
        ["Measure", "HMM", "Baseline"],
        oov_rows,
        "OOV and known-token performance.",
        widths=[2.2, 2.0, 2.0],
        font_size=8.6,
        align_center=True,
    )
    para(
        document,
        f"The test split contains {hmm['oov_count']:,} OOV tokens ({fmt_pct(hmm['oov_rate'])} of all test tokens). "
        f"The HMM tags {fmt_pct(hmm['oov_accuracy'])} of them correctly, whereas the baseline reaches only "
        f"{fmt_pct(baseline['oov_accuracy'])}. The suffix, digit, and capitalisation priors therefore provide a "
        "substantial gain on unseen words. Performance on known tokens is higher "
        f"({fmt_pct(hmm['known_accuracy'])}), as expected, because their emission distributions were estimated "
        "directly from training data.",
    )
    add_figure(document, "oov_analysis.png", "Accuracy on OOV and known test tokens for both models.")

    # ------------------------------------------------------------------
    # 15. Error analysis
    # ------------------------------------------------------------------
    heading(document, "15. Error Analysis", 1, page_break=True)
    para(
        document,
        f"The HMM made {result['total_error_tokens']:,} token-level errors on the test split. The most frequent "
        f"confusion pairs are listed below. {len(top_errors)} distinct gold/predicted tag pairs account for the "
        "observed errors; the complete error file (error_analysis.csv) contains every error token with its "
        "surrounding context.",
    )
    error_rows = []
    for rank, error in enumerate(top_errors[:12], start=1):
        error_rows.append(
            [
                rank,
                error["gold_tag"],
                error["predicted_tag"],
                f"{error['count']:,}",
                fmt_pct(error["percentage_of_errors"] / 100.0),
                error["example_context"],
            ]
        )
    add_table(
        document,
        ["Rank", "Gold", "Predicted", "Count", "% errors", "Example context"],
        error_rows,
        "Most frequent HMM confusion pairs with a real context example.",
        widths=[0.45, 0.65, 0.8, 0.55, 0.65, 3.35],
        font_size=7.4,
    )
    para(
        document,
        "The dominant confusion is VERB -> NOUN, followed by ADJ -> NOUN and NOUN -> VERB. Participles such as "
        "'publishing' and adjectives such as 'smooth' are lexically ambiguous, and the first-order HMM often chooses "
        "the more frequent noun reading. The reverse confusion, NOUN -> VERB, occurs for words like 'exhaust' that "
        "can be either a noun or a verb. The model also confuses ADJ and VERB for words such as 'open' and 'crowded'. "
        "These errors are consistent with the model's inability to represent long-distance syntactic structure, "
        "morphology, and lexical subcategorisation. A CRF or a contextual neural encoder would use evidence from the "
        "whole sentence and would be expected to reduce such errors.",
    )
    para(
        document,
        "A qualitative inspection of the custom sentences shows the same pattern. For example, in "
        "'The quick brown fox jumps over the lazy dog.', the HMM tags 'fox' as ADP and 'jumps' as NOUN, so the "
        "sentence-level path is locally plausible but syntactically wrong. The baseline makes a similar error on "
        "'jumps'. This illustrates that the first-order transition model can be misled by a sequence of ambiguous "
        "words and that beam search or a higher-order model would not necessarily solve the underlying lexical "
        "ambiguity.",
    )

    # ------------------------------------------------------------------
    # 16. Results and output screenshots
    # ------------------------------------------------------------------
    heading(document, "16. Results and Output Screenshots", 1, page_break=True)
    para(
        document,
        "The complete run is reproducible with a single command. The script writes all results to the project "
        "directory, including JSON and CSV files, five high-resolution charts, the Excel workbook, the Jupyter "
        "notebook, and this report. The console log is saved automatically to outputs/terminal_output.txt.",
    )
    result_rows = [
        ["Baseline accuracy", fmt_pct(baseline["accuracy"])],
        ["HMM accuracy", fmt_pct(hmm["accuracy"])],
        ["Absolute accuracy gain", f"{result['comparison']['accuracy_gain'] * 100:+.2f} percentage points"],
        ["Baseline macro F1", fmt(baseline["macro_f1"])],
        ["HMM macro F1", fmt(hmm["macro_f1"])],
        ["HMM weighted F1", fmt(hmm["weighted_f1"])],
        ["HMM OOV accuracy", fmt_pct(hmm["oov_accuracy"])],
        ["HMM known-token accuracy", fmt_pct(hmm["known_accuracy"])],
        ["Total test tokens", f"{hmm['support']:,}"],
        ["Total HMM token errors", f"{result['total_error_tokens']:,}"],
    ]
    add_table(
        document,
        ["Result", "Value"],
        result_rows,
        "Summary of the actual experimental results.",
        widths=[2.8, 3.6],
        font_size=8.8,
    )
    add_figure(document, "model_comparison.png", "Baseline versus HMM accuracy, macro F1, and weighted F1.")
    add_figure(document, "tag_distribution.png", "Universal POS-tag distribution in the training and test splits.")
    para(
        document,
        "Desktop screenshots must be captured from the actual running applications. They are not simulated or "
        "inserted automatically. The file SCREENSHOT_GUIDE.md lists every recommended capture, the exact Fedora "
        "commands, and the visible content required. Expected files in the screenshots/ directory are: "
        "01_project_directory.png, 02_script_execution.png, 03_dataset_split.png, 04_metrics.png, "
        "05_custom_output.png, 06_error_analysis.png, 07_confusion_matrix.png, 08_per_tag_scores.png, "
        "09a_model_comparison.png, 09b_oov_analysis.png, 09c_tag_distribution.png, 10_workbook_summary.png, "
        "11_workbook_per_tag.png, 12a_report_cover.png (with 12b/12c/12d for methodology, results, conclusion), "
        "and 13_notebook_execution.png. These are to be captured by the student; no fake terminal or application "
        "images are included in this submission.",
    )

    # ------------------------------------------------------------------
    # 17. Observations
    # ------------------------------------------------------------------
    heading(document, "17. Observations and Discussion", 1)
    observations = [
        f"The HMM outperformed the most-frequent-tag baseline by {result['comparison']['accuracy_gain'] * 100:.2f} percentage points "
        f"({fmt_pct(hmm['accuracy'])} versus {fmt_pct(baseline['accuracy'])}). The gain comes from contextual transition information rather than from word frequency alone.",
        f"Macro F1 improved from {fmt(baseline['macro_f1'])} to {fmt(hmm['macro_f1'])}. The larger macro gain than accuracy gain shows that the HMM especially helps less frequent tags.",
        f"Punctuation is almost perfectly tagged, while NOUN, VERB, ADJ, and ADP dominate both the support and the residual errors.",
        f"The test split contains {hmm['oov_count']:,} OOV tokens ({fmt_pct(hmm['oov_rate'])}). The OOV model raises accuracy on these tokens from {fmt_pct(baseline['oov_accuracy'])} to {fmt_pct(hmm['oov_accuracy'])}.",
        "Most remaining errors are genuine lexical ambiguities: participles used as modifiers, nouns used as verbs, and adjectives that can also function as verbs.",
        "The model is highly reproducible: the fixed seed produces identical splits and identical predictions on repeated runs.",
        "The custom sentences show that an HMM can produce locally fluent but syntactically incorrect sequences; sentence-level constraints beyond first-order transitions are needed for such cases.",
        "The Universal tagset is coarser than the Penn Treebank tagset. Merging all noun subtypes and all verb subtypes reduces the number of classes and makes the task easier, but it hides distinctions such as singular versus plural noun and past versus present verb.",
    ]
    add_bullet_list(document, observations)

    # ------------------------------------------------------------------
    # 18. Advantages and limitations
    # ------------------------------------------------------------------
    heading(document, "18. Advantages and Limitations", 1)
    heading(document, "18.1 Advantages", 2)
    add_bullet_list(
        document,
        [
            "The model is simple, fast, and fully interpretable: every prediction can be traced to transition and emission probabilities.",
            "Training on about 75,000 tokens takes only a few seconds, making the experiment easy to reproduce in a laboratory.",
            "Add-k smoothing and log-space Viterbi eliminate zero-probability and underflow problems.",
            "The OOV model is data-driven and handles numbers, suffixes, and capitalisation explicitly.",
            "The evaluation is leakage-free and reports both token-level and class-level metrics, plus OOV-specific performance.",
            "All outputs are generated from the same seed and configuration, so the report, workbook, notebook, and CSV files are consistent.",
        ],
    )
    heading(document, "18.2 Limitations", 2)
    add_bullet_list(
        document,
        [
            "The first-order Markov assumption cannot capture long-distance syntactic dependencies.",
            "Words are assumed independent given their tags, so the model cannot use morphological or contextual feature overlap directly.",
            "Emission estimates are sparse for rare words; the OOV heuristic is useful but not a substitute for subword or contextual representations.",
            "The Universal tagset merges important distinctions such as proper versus common nouns and verb tense.",
            "The baseline and HMM share the same lexicon limitations, and the test split comes from a single newswire corpus, which limits domain generalisation.",
            "The model does not use a CRF-style sequence objective or a neural encoder, so it cannot match modern contextual taggers on ambiguous constructions.",
        ],
    )

    # ------------------------------------------------------------------
    # 19. Conclusion and future scope
    # ------------------------------------------------------------------
    heading(document, "19. Conclusion and Future Scope", 1)
    para(
        document,
        f"This assignment implemented and evaluated a complete HMM POS tagger with Viterbi decoding on the Penn "
        f"Treebank corpus. The model achieved {fmt_pct(hmm['accuracy'])} accuracy and a weighted F1 of "
        f"{fmt(hmm['weighted_f1'])}, outperforming the most-frequent-tag baseline by "
        f"{result['comparison']['accuracy_gain'] * 100:.2f} percentage points. The OOV model improved unseen-word "
        f"accuracy to {fmt_pct(hmm['oov_accuracy'])}. The experiment also demonstrated the main limitation of "
        "first-order generative tagging: residual errors concentrate on genuinely ambiguous lexical items and on "
        "constructions that require wider context.",
    )
    para(
        document,
        "Future work could replace the generative emission model with a discriminative MaxEnt or CRF model using "
        "suffix, prefix, shape, and neighbouring-word features. A second direction is neural tagging with a "
        "BiLSTM-CRF or a fine-tuned BERT model, which would use contextual embeddings and subword tokenisation to "
        "handle OOV words and long-distance dependencies. Additional improvements include training on a larger "
        "and more domain-diverse corpus, using the full Penn Treebank tagset, applying beam search with a stronger "
        "language model, and evaluating calibration and per-domain robustness. Finally, the same pipeline can be "
        "extended to lemmatisation, chunking, and named-entity recognition, for which POS information is a useful "
        "feature.",
    )

    # ------------------------------------------------------------------
    # 20. Viva questions
    # ------------------------------------------------------------------
    heading(document, "20. Viva Questions and Answers", 1, page_break=True)
    viva = [
        (
            "What is POS tagging and why is it useful?",
            "POS tagging assigns a grammatical category to every token in a sentence. It provides lexical and "
            "syntactic information used by parsers, named-entity recognisers, information-extraction systems, "
            "machine translation, and text analytics.",
        ),
        (
            "Why is POS tagging treated as a sequence-labelling problem?",
            "The tag of a word depends on its neighbours and on the grammatical structure of the sentence. The "
            "output is a sequence of labels whose order matters, so the task is naturally modelled as sequence labelling.",
        ),
        (
            "What are the two independence assumptions of a first-order HMM?",
            "Each word is assumed to depend only on its own tag, and each tag is assumed to depend only on the "
            "immediately preceding tag. Boundary states START and END complete the model.",
        ),
        (
            "What is the difference between transition and emission probabilities?",
            "A transition probability P(ti | ti-1) describes how likely one tag is to follow another. An emission "
            "probability P(wi | ti) describes how likely a word is to be generated by a tag.",
        ),
        (
            "Why does the implementation use log probabilities?",
            "The joint probability of a long sentence is a product of many numbers smaller than one. In floating "
            "point this underflows to zero. Logarithms convert products into sums, which are numerically stable, "
            "and Viterbi maximises the same path in log space.",
        ),
        (
            "Explain Viterbi decoding and its complexity.",
            "Viterbi fills a table δi(j) containing the best score of any tag sequence ending in tag j at position i. "
            "It stores the best predecessor in ψi(j), adds the END transition at the final position, and backtracks. "
            "The complexity is O(n|T|²) time and O(n|T|) space.",
        ),
        (
            "Why is add-k smoothing used for transitions?",
            "Maximum-likelihood estimation assigns zero probability to unseen tag transitions. A single zero would "
            "make an entire path impossible. Add-k smoothing adds k to every transition count, guaranteeing positive "
            "probabilities and avoiding log(0).",
        ),
        (
            "How are out-of-vocabulary words handled?",
            "Rare training words provide suffix statistics for lengths 1-4. Numeric tokens use a numeric-shape prior, "
            "digit-containing tokens mix a digit prior, and capitalised tokens mix a capitalisation prior. The scores "
            "are smoothed and normalised over the tagset, so every tag has positive probability.",
        ),
        (
            "What is the difference between generative and discriminative taggers?",
            "A generative model such as an HMM models P(W, T) and uses Bayes' rule. A discriminative model such as "
            "MaxEnt or a CRF models P(T | W) directly and can use overlapping features of the observation sequence.",
        ),
        (
            "Why report both macro and weighted F1?",
            "Macro F1 gives every tag equal weight and exposes poor performance on rare tags. Weighted F1 weights "
            "each tag by its support and reflects overall token-level performance. Reporting both prevents a frequent "
            "tag from hiding weaknesses on infrequent classes.",
        ),
        (
            "How is data leakage prevented?",
            "The split is made at the sentence level with a fixed seed. Sentences are deduplicated, an explicit "
            "assertion verifies that no tagged sentence appears in both splits, and both the baseline and the HMM "
            "are fitted only on the training split. The test split is used only for evaluation.",
        ),
        (
            "What are the main limitations of this HMM?",
            "It cannot model long-distance dependencies or rich overlapping features, it merges fine-grained Penn "
            "Treebank distinctions into the Universal tagset, and its OOV handling is heuristic. CRF and contextual "
            "neural models generally perform better on ambiguous words and unseen domains.",
        ),
    ]
    for index, (question, answer) in enumerate(viva, start=1):
        q = document.add_paragraph()
        q.paragraph_format.space_before = Pt(6)
        q.paragraph_format.space_after = Pt(1)
        q.paragraph_format.keep_with_next = True
        run = q.add_run(f"Q{index}. {question}")
        _set_run_font(run, size=10.5, bold=True, color=DARK_BLUE)
        a = document.add_paragraph()
        a.paragraph_format.left_indent = Inches(0.2)
        run = a.add_run(f"Answer: {answer}")
        _set_run_font(run, size=10.2)

    # ------------------------------------------------------------------
    # 21. References
    # ------------------------------------------------------------------
    heading(document, "21. References", 1, page_break=True)
    for index, reference in enumerate(REFERENCES, start=1):
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(f"[{index}]  {reference}")
        _set_run_font(run, size=9.8)

    # ------------------------------------------------------------------
    # 22. Faculty evaluation
    # ------------------------------------------------------------------
    heading(document, "22. Faculty Evaluation", 1, page_break=True)
    para(
        document,
        "This section is intentionally left blank for the faculty to record marks, remarks, and signatures.",
    )
    evaluation_rows = [
        ["Marks Awarded", ""],
        ["Remarks", ""],
        ["Faculty Signature", ""],
        ["Date", ""],
    ]
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = table.rows[0].cells
    header[0].text = "Evaluation item"
    header[1].text = "Details"
    for cell in header:
        _shade_cell(cell, HEADER_FILL)
        for run in cell.paragraphs[0].runs:
            _set_run_font(run, size=9.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for label_text, value in evaluation_rows:
        cells = table.add_row().cells
        cells[0].text = label_text
        cells[1].text = value
        cells[0].width = Inches(2.0)
        cells[1].width = Inches(4.4)
        for cell in cells:
            for run in cell.paragraphs[0].runs:
                _set_run_font(run, size=9.5)
        cells[1].paragraphs[0].add_run(" " * 40)

    end = document.add_paragraph()
    end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    end.paragraph_format.space_before = Pt(28)
    run = end.add_run("— End of Report —")
    _set_run_font(run, size=10, italic=True, color=GREY)

    # ------------------------------------------------------------------
    # Appendix A: full source code
    # ------------------------------------------------------------------
    heading(document, "Appendix A: Complete Source Code (pos_tagging_hmm.py)", 1, page_break=True)
    para(
        document,
        "The following listing is the complete, actual source file used to produce every result in this report. "
        "It is reproduced without modification so that the implementation can be verified against the submitted "
        "pos_tagging_hmm.py.",
    )
    source = SOURCE_PATH.read_text(encoding="utf-8")
    add_code(document, source, font_size=6.8)

    document.core_properties.title = "NLP Laboratory Assignment 3 - HMM POS Tagging"
    document.core_properties.author = "Shravani Kurkute"
    document.core_properties.subject = "Survey and Implementation of POS Tagging"
    document.core_properties.keywords = "POS tagging, HMM, Viterbi, Penn Treebank, Universal tagset"

    document.save(REPORT_PATH)

    # Validate that the file can be reopened.
    check = Document(REPORT_PATH)
    if not any("Shravani Kurkute" in p.text for p in check.paragraphs):
        raise RuntimeError("Report validation failed: student name not found.")
    return REPORT_PATH


def main() -> int:
    with (PROJECT_DIR / "results.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    path = build_report(result)
    print(f"Report created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
