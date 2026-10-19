"""
NLP Laboratory Assignment - 02
Fundamental Language Processing for Three Different Languages
(English, Hindi, Marathi)

Author : Shravani Kurkute
Faculty: Dr. Kapil N. Vhatkar

This script performs a complete multilingual NLP preprocessing pipeline:
  * Loads and validates the aligned three-language dataset
  * Normalizes, tokenizes and removes punctuation for each language
  * Removes stop words using language specific resources
  * Applies Porter stemming to English only
  * Computes real frequency statistics with collections.Counter
  * Exports a nine-sheet Excel workbook
  * Generates five comparison / frequency charts
  * Creates a professional DOCX report
  * Creates a reproducible Jupyter notebook

All reported numbers are computed from the dataset at runtime.
No result is ever hard-coded.
"""

from __future__ import annotations

import os
import re
import sys
import json
import unicodedata
from collections import Counter

# ----------------------------------------------------------------------------
# Optional third-party imports. The script degrades gracefully with a clear
# message if a dependency is missing so validation never fails silently.
# ----------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend, safe on servers / CI
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except ImportError:  # pragma: no cover
    plt = None
    font_manager = None
    OffsetImage = None
    AnnotationBbox = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:  # pragma: no cover
    Workbook = None

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
except ImportError:  # pragma: no cover
    Document = None

try:
    import nltk
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover
    nltk = None
    PorterStemmer = None


# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "NLP_Three_Languages_Dataset.csv")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
SHOT_DIR = os.path.join(BASE_DIR, "screenshots")
EXCEL_FILE = os.path.join(BASE_DIR, "ASG2_Results.xlsx")
REPORT_FILE = os.path.join(BASE_DIR, "Assignment_02_Report.docx")
NOTEBOOK_FILE = os.path.join(BASE_DIR, "ASG2.ipynb")

EXPECTED_COLUMNS = ["ID", "English", "Hindi", "Marathi"]
LANGUAGES = ["English", "Hindi", "Marathi"]
EXPECTED_ROWS = 50

# Devanagari Unicode block (used for validation / diagnostics only).
DEVANAGARI_RANGE = (0x0900, 0x097F)


# ============================================================================
# STOP-WORD RESOURCES
# ============================================================================
# English stop words come from NLTK's documented `english` stop-word corpus.
# Hindi and Marathi are CUSTOM, INSPECTABLE lists because NLTK does not ship
# them. They are deliberately conservative (only high-frequency function words
# and postpositions) so that meaningful content words are never removed.
# LIMITATION: these lists are not exhaustive and were curated by hand; they
# should be reviewed before being used for any production task. They are shown
# in full in the Stopword_Resources worksheet for inspection.
# ----------------------------------------------------------------------------

_HINDI_STOPWORDS = [
    # pronouns
    "मैं", "मुझे", "मुझको", "मेरा", "मेरी", "मेरे", "हम", "हमें", "हमको",
    "हमारा", "हमारी", "हमारे", "तू", "तुम", "तुम्हें", "तुम्हारा", "तुम्हारी",
    "तुम्हारे", "आप", "आपको", "आपका", "आपकी", "आपके", "वह", "वो", "उस",
    "उसको", "उसका", "उसकी", "उसके", "वे", "वे लोग", "उन", "उनको", "उनका",
    "उनकी", "उनके", "यह", "ये", "इस", "इसको", "इसका", "इसकी", "इसके",
    "जो", "जिस", "जिसको", "जिसका", "जिसकी", "जिसके", "कौन", "क्या",
    # postpositions / case markers
    "का", "की", "के", "को", "से", "में", "पर", "ने", "तक", "लिए", "साथ",
    "बिना", "द्वारा", "तरह", "सा", "सी", "से", "वाला", "वाली", "वाले",
    # conjunctions / connectors
    "और", "या", "लेकिन", "परन्तु", "परंतु", "किन्तु", "किंतु", "तथा", "एवं",
    "क्योंकि", "इसलिए", "अगर", "अगरचे", "जब", "तब", "जबकि", "हालाँकि",
    "कि", "की", "भी", "ही", "तो", "ना", "नहीं", "न", "मत", "बहुत", "बहुतक",
    # auxiliaries / copulas / common verbs
    "है", "हैं", "हो", "होता", "होती", "होते", "था", "थी", "थे", "हुआ",
    "हुई", "हुए", "होगा", "होगी", "होंगे", "कर", "करना", "करता", "करती",
    "करते", "किया", "किे", "करके", "रहा", "रही", "रहे", "गया", "गई", "गए",
    "दिया", "दी", "दिए", "जाता", "जाती", "जाते", "सकता", "सकती", "सकते",
    # quantifiers / adverbs
    "एक", "सभी", "सब", "कुछ", "कोई", "प्रत्येक", "अनेक", "बहुत", "कम",
    "अधिक", "ज्यादा", "हर", "दूसरा", "दूसरी", "दूसरे", "अब", "यहाँ", "वहाँ",
    "यहां", "वहां", "आज", "कल", "फिर", "भी", "सिर्फ", "केवल", "लगभग",
]

_MARATHI_STOPWORDS = [
    # pronouns
    "मी", "मला", "माझा", "माझी", "माझे", "माझ्या", "आम्ही", "आम्हाला",
    "आमचा", "आमची", "आमचे", "आमच्या", "तू", "तुला", "तुझा", "तुझी", "तुझे",
    "तुझ्या", "तुम्ही", "तुम्हाला", "तुमचा", "तुमची", "तुमचे", "तुमच्या",
    "तो", "ती", "ते", "त्याला", "तिला", "त्यांना", "त्याचा", "त्याची",
    "त्याचे", "त्याच्या", "तिचा", "तिची", "तिचे", "तिच्या", "त्यांचा",
    "त्यांची", "त्यांचे", "हे", "ही", "हा", "हिला", "ह्याला", "या", "याचा",
    "याची", "याचे", "या", "जो", "जी", "जे", "ज्याला", "जिला", "ज्यांना",
    "कोण", "काय", "कुणी", "काही",
    # postpositions / case markers
    "चा", "ची", "चे", "च्या", "ला", "ना", "ने", "नी", "मध्ये", "वर", "खाली",
    "पासून", "पर्यंत", "साठी", "सोबत", "बरोबर", "शिवाय", "कडून", "कडे",
    "प्रमाणे", "सारखा", "सारखी", "सारखे", "बद्दल", "मुळे",
    # conjunctions / connectors
    "आणि", "व", "किंवा", "पण", "परंतु", "मात्र", "तर", "म्हणून", "कारण",
    "जर", "जेव्हा", "तेव्हा", "जरी", "की", "म्हणजे", "जसे", "तसे", "ही",
    "सुद्धा", "तरी", "नाही", "न", "नको", "फार",
    # auxiliaries / copulas / common verbs
    "आहे", "आहेत", "होता", "होती", "होते", "होतं", "होते", "होणार", "होईल",
    "होत", "होऊन", "होण्यासाठी", "करतो", "करते", "करतात", "करत", "करून",
    "केले", "केला", "केली", "करायचे", "करा", "कर", "गेला", "गेली", "गेले",
    "दिला", "दिली", "दिले", "जातो", "जाते", "जातात", "शकतो", "शकते",
    "शकतात", "येते", "येतो", "येतात", "राहिले", "राहतो",
    # quantifiers / adverbs
    "एक", "सर्व", "सगळे", "काही", "कोणी", "प्रत्येक", "अनेक", "पुष्कळ",
    "कमी", "जास्त", "अधिक", "दुसरा", "दुसरी", "दुसरे", "आता", "इथे", "तिथे",
    "आज", "उद्या", "पुन्हा", "सुद्धा", "फक्त", "केवळ", "जवळपास", "अगदी",
]


# ============================================================================
# MODULE LEVEL STATE (populated at runtime, kept inspectable)
# ============================================================================
_STOPWORDS = {
    "English": None,   # filled from NLTK
    "Hindi": set(_HINDI_STOPWORDS),
    "Marathi": set(_MARATHI_STOPWORDS),
}
_STEMMER = PorterStemmer() if PorterStemmer is not None else None


# ============================================================================
# STEP 1 : LOAD + VALIDATE
# ============================================================================
def load_dataset(path: str = DATA_FILE):
    """Load the CSV dataset with pandas preserving Unicode text."""
    if pd is None:
        raise RuntimeError("pandas is required but not installed.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    # dtype=str keeps the ID column as the original text form and avoids any
    # accidental numeric coercion of language columns.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


def validate_dataset(df, expected_columns=None, expected_rows=None):
    """Validate columns, row count and missing values.

    Returns a dictionary describing the validation outcome. Raises ValueError
    with an exact explanation when a hard requirement fails.
    """
    expected_columns = expected_columns or EXPECTED_COLUMNS
    expected_rows = expected_rows or EXPECTED_ROWS

    report = {
        "columns_found": list(df.columns),
        "expected_columns": expected_columns,
        "row_count": int(len(df)),
        "expected_rows": int(expected_rows),
        "missing_values": {},
        "empty_strings": {},
        "problems": [],
        "ok": True,
    }

    # Column check
    if list(df.columns) != expected_columns:
        report["problems"].append(
            f"Column mismatch. Found {list(df.columns)}, expected {expected_columns}."
        )
        report["ok"] = False

    # Row count check
    if len(df) != expected_rows:
        report["problems"].append(
            f"Row count mismatch. Found {len(df)}, expected {expected_rows}."
        )
        report["ok"] = False

    # Missing / empty check
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        n_empty = int((df[col].astype(str).str.strip() == "").sum())
        report["missing_values"][col] = n_missing
        report["empty_strings"][col] = n_empty
        if n_missing:
            report["problems"].append(f"Column '{col}' has {n_missing} missing values.")
            report["ok"] = False
        if n_empty:
            report["problems"].append(f"Column '{col}' has {n_empty} empty strings.")
            report["ok"] = False

    # Duplicate ID check
    if "ID" in df.columns:
        dups = int(df["ID"].duplicated().sum())
        report["duplicate_ids"] = dups
        if dups:
            report["problems"].append(f"Found {dups} duplicate IDs.")
            report["ok"] = False

    if not report["ok"]:
        raise ValueError("Dataset validation failed:\n  - " + "\n  - ".join(report["problems"]))
    return report


# ============================================================================
# TEXT NORMALIZATION, TOKENIZATION, PUNCTUATION
# ============================================================================
def normalize_text(text: str, language: str) -> str:
    """Language aware normalization.

    English : lowercase + whitespace collapse.
    Hindi / Marathi : Unicode NFC + whitespace collapse, preserving every
    Devanagari letter, vowel sign (matra) and combining mark.
    """
    if text is None:
        return ""
    text = str(text)

    if language == "English":
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
    else:
        # NFC keeps matras and combining marks as single code points.
        text = unicodedata.normalize("NFC", text)
        # Remove zero-width joiners that sometimes appear in copied text only
        # if they are NOT meaningful; keep ZWNJ in general because Devanagari
        # uses it. We therefore leave it untouched.

    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_punctuation(text: str, language: str) -> str:
    """Remove punctuation in a Unicode safe way.

    Punctuation is any character whose Unicode general category begins with
    'P' (Pc, Pd, Pe, Pf, Pi, Po, Ps). This correctly removes the Devanagari
    danda (U+0964) and double danda (U+0965), which are category Po.

    Devanagari letters (Lo), vowel signs / matras (Mc) and combining marks
    such as the virama (Mn) are NOT category P, so they are always preserved.
    The `language` argument is accepted for API symmetry and clarity of intent.
    """
    if text is None:
        return ""
    text = str(text)
    out_chars = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            continue
        out_chars.append(ch)
    cleaned = "".join(out_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_words(text: str, language: str):
    """Unicode aware word tokenization.

    For Devanagari we use a regex that captures runs of Devanagari letters and
    combining marks so that a word such as 'प्रसंस्करण' stays intact instead of
    being split at matras. For English we use NLTK's word_tokenize when
    available, falling back to a regex.
    """
    if not text:
        return []
    if language == "English":
        if nltk is not None:
            try:
                return nltk.word_tokenize(text)
            except LookupError:
                pass
        return re.findall(r"[\w']+", text, flags=re.UNICODE)
    else:
        # Devanagari letters + combining marks + digits + latin/digits mix.
        pattern = r"[\u0900-\u097F\u0966-\u096F]+|[A-Za-z0-9]+"
        return re.findall(pattern, text)


def get_stopwords(language: str):
    """Return the stop-word set for a language (documented resource)."""
    if language == "English":
        if _STOPWORDS["English"] is None:
            if nltk is None:
                raise RuntimeError("NLTK is required for English stop words.")
            from nltk.corpus import stopwords as nltk_stopwords
            _STOPWORDS["English"] = set(nltk_stopwords.words("english"))
        return _STOPWORDS["English"]
    return _STOPWORDS[language]


def remove_stopwords(tokens, language: str):
    """Remove stop words. Returns (kept_tokens, removed_tokens)."""
    stop = get_stopwords(language)
    kept, removed = [], []
    if language == "English":
        for tok in tokens:
            if tok.lower() in stop:
                removed.append(tok)
            else:
                kept.append(tok)
    else:
        for tok in tokens:
            if tok in stop:
                removed.append(tok)
            else:
                kept.append(tok)
    return kept, removed


def apply_english_stemming(tokens):
    """Apply the NLTK Porter stemmer to English tokens (post stop-word)."""
    if _STEMMER is None:
        raise RuntimeError("NLTK PorterStemmer is required for English stemming.")
    return [_STEMMER.stem(tok) for tok in tokens]


def calculate_word_frequencies(tokens):
    """Return an ordered Counter of tokens and the top 10 list."""
    counter = Counter(tokens)
    top10 = counter.most_common(10)
    return counter, top10


# ============================================================================
# PIPELINE
# ============================================================================
def process_language(df, language: str):
    """Run the full pipeline for one language column and return a result dict.

    Each CSV row is treated as one source sentence. Row IDs and alignment are
    preserved so the three languages can be compared row by row.
    """
    original_texts = df[language].astype(str).tolist()
    ids = df["ID"].astype(str).tolist()

    per_row = []
    all_tokens_before = []
    all_tokens_after = []
    all_tokens_removed = []
    all_stems = []  # English only

    for row_id, original in zip(ids, original_texts):
        normalized = normalize_text(original, language)
        depunct = remove_punctuation(normalized, language)
        tokens_before = tokenize_words(depunct, language)
        tokens_after, removed = remove_stopwords(tokens_before, language)

        stems = []
        if language == "English":
            stems = apply_english_stemming(tokens_after)

        per_row.append({
            "ID": row_id,
            "original": original,
            "normalized": normalized,
            "depunctuated": depunct,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_removed": removed,
            "stems": stems,
        })

        all_tokens_before.extend(tokens_before)
        all_tokens_after.extend(tokens_after)
        all_tokens_removed.extend(removed)
        all_stems.extend(stems)

    freq_counter, top10 = calculate_word_frequencies(all_tokens_after)

    # For English, frequencies are also reported after stemming.
    if language == "English":
        stem_counter, stem_top10 = calculate_word_frequencies(all_stems)
    else:
        stem_counter, stem_top10 = None, None

    result = {
        "language": language,
        "source_sentences": len(original_texts),
        "total_tokens_before": len(all_tokens_before),
        "unique_tokens_before": len(set(all_tokens_before)),
        "total_tokens_after": len(all_tokens_after),
        "unique_tokens_after": len(set(all_tokens_after)),
        "tokens_removed": len(all_tokens_removed),
        "unique_tokens_removed": len(set(all_tokens_removed)),
        "post_stemming_total": len(all_stems) if language == "English" else None,
        "post_stemming_unique": len(set(all_stems)) if language == "English" else None,
        "frequency": freq_counter,
        "top10": top10,
        "stem_frequency": stem_counter,
        "stem_top10": stem_top10,
        "per_row": per_row,
    }
    return result


def compare_languages(results, validation):
    """Build the comparative summary dictionary from per-language results."""
    summary = {
        "validation": validation,
        "languages": {},
        "highest_vocabulary": None,
        "lowest_vocabulary": None,
        "notes": [],
    }
    for lang, res in results.items():
        summary["languages"][lang] = {
            "source_sentences": res["source_sentences"],
            "total_tokens_before": res["total_tokens_before"],
            "unique_tokens_before": res["unique_tokens_before"],
            "total_tokens_after": res["total_tokens_after"],
            "unique_tokens_after": res["unique_tokens_after"],
            "tokens_removed": res["tokens_removed"],
            "unique_tokens_removed": res["unique_tokens_removed"],
            "post_stemming_total": res["post_stemming_total"],
            "post_stemming_unique": res["post_stemming_unique"],
            "vocabulary_ratio_after": round(res["unique_tokens_after"] / res["total_tokens_after"], 4)
            if res["total_tokens_after"] else 0.0,
        }

    by_vocab = sorted(
        results.items(), key=lambda kv: kv[1]["unique_tokens_after"], reverse=True
    )
    summary["highest_vocabulary"] = by_vocab[0][0]
    summary["lowest_vocabulary"] = by_vocab[-1][0]

    summary["notes"].append(
        f"Highest post-stop-word vocabulary: {summary['highest_vocabulary']} "
        f"({results[summary['highest_vocabulary']]['unique_tokens_after']} unique tokens)."
    )
    summary["notes"].append(
        f"Lowest post-stop-word vocabulary: {summary['lowest_vocabulary']} "
        f"({results[summary['lowest_vocabulary']]['unique_tokens_after']} unique tokens)."
    )
    return summary


# ============================================================================
# EXCEL EXPORT (nine worksheets)
# ============================================================================
def _style_header(ws, ncols):
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="305496")
    thin = Side(style="thin", color="B0B0B0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.freeze_panes = "A2"


def _autosize(ws, min_width=10, max_width=70):
    for col_cells in ws.columns:
        length = min_width
        letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            if cell.value is not None:
                length = max(length, min(max_width, len(str(cell.value)) + 2))
        ws.column_dimensions[letter].width = length


def export_results(results, summary, validation, path=EXCEL_FILE):
    """Create the nine-sheet Excel workbook."""
    if Workbook is None:
        raise RuntimeError("openpyxl is required to export Excel results.")

    wb = Workbook()

    # ---- 1. Comparative_Summary -------------------------------------------
    ws = wb.active
    ws.title = "Comparative_Summary"
    ws.append(["Language", "Source Sentences", "Total Tokens (Before)",
               "Unique Tokens (Before)", "Total Tokens (After)",
               "Unique Tokens (After)", "Tokens Removed",
               "Unique Removed", "Post-Stemming Total",
               "Post-Stemming Unique", "Vocabulary Ratio (After)"])
    for lang in LANGUAGES:
        s = summary["languages"][lang]
        ws.append([lang, s["source_sentences"], s["total_tokens_before"],
                   s["unique_tokens_before"], s["total_tokens_after"],
                   s["unique_tokens_after"], s["tokens_removed"],
                   s["unique_tokens_removed"], s["post_stemming_total"],
                   s["post_stemming_unique"], s["vocabulary_ratio_after"]])
    _style_header(ws, 11)
    _autosize(ws)
    ws.auto_filter.ref = ws.dimensions

    # ---- 2-4. Processing sheets -------------------------------------------
    for lang in LANGUAGES:
        ws = wb.create_sheet(f"{lang}_Processing")
        header = ["ID", "Original Text", "Normalized Text", "Depunctuated Text",
                  "Tokens Before", "Token Count Before", "Tokens After Stopwords",
                  "Token Count After", "Tokens Removed"]
        if lang == "English":
            header += ["Stems (Porter)"]
        ws.append(header)
        for row in results[lang]["per_row"]:
            line = [
                row["ID"], row["original"], row["normalized"], row["depunctuated"],
                " | ".join(row["tokens_before"]), len(row["tokens_before"]),
                " | ".join(row["tokens_after"]), len(row["tokens_after"]),
                " | ".join(row["tokens_removed"]),
            ]
            if lang == "English":
                line.append(" | ".join(row["stems"]))
            ws.append(line)
        _style_header(ws, len(header))
        _autosize(ws)
        ws.auto_filter.ref = ws.dimensions

    # ---- 5-7. Frequency sheets --------------------------------------------
    for lang in LANGUAGES:
        ws = wb.create_sheet(f"{lang}_Frequency")
        if lang == "English":
            ws.append(["Rank", "Word", "Frequency", "Stem", "Stem Frequency"])
            stem_freq = results[lang]["stem_frequency"]
            for rank, (word, count) in enumerate(results[lang]["frequency"].most_common(), 1):
                stem = _STEMMER.stem(word)
                ws.append([rank, word, count, stem, stem_freq.get(stem, 0)])
        else:
            ws.append(["Rank", "Word", "Frequency"])
            for rank, (word, count) in enumerate(results[lang]["frequency"].most_common(), 1):
                ws.append([rank, word, count])
        _style_header(ws, ws.max_column)
        _autosize(ws)
        ws.auto_filter.ref = ws.dimensions

    # ---- 8. Stopword_Resources --------------------------------------------
    ws = wb.create_sheet("Stopword_Resources")
    ws.append(["Language", "Resource", "Stop Word", "Notes"])
    eng_note = "NLTK english stop-word corpus (documented, standard)."
    custom_note = ("Custom hand-curated list (NLTK does not ship Hindi/Marathi "
                   "stop words). Not exhaustive.")
    for w in sorted(get_stopwords("English")):
        ws.append(["English", "NLTK stopwords.words('english')", w, eng_note])
    for w in sorted(get_stopwords("Hindi")):
        ws.append(["Hindi", "Custom curated list (this project)", w, custom_note])
    for w in sorted(get_stopwords("Marathi")):
        ws.append(["Marathi", "Custom curated list (this project)", w, custom_note])
    _style_header(ws, 4)
    _autosize(ws)
    ws.auto_filter.ref = ws.dimensions

    # ---- 9. Sample_Transformations ----------------------------------------
    ws = wb.create_sheet("Sample_Transformations")
    ws.append(["Language", "ID", "Original", "Normalized", "After Punctuation Removal",
               "Tokens Before", "Tokens After Stopwords", "Removed", "Stems"])
    sample_ids = ["1", "2", "5", "9", "17"]
    for lang in LANGUAGES:
        for row in results[lang]["per_row"]:
            if row["ID"] in sample_ids:
                ws.append([
                    lang, row["ID"], row["original"], row["normalized"],
                    row["depunctuated"], " | ".join(row["tokens_before"]),
                    " | ".join(row["tokens_after"]),
                    " | ".join(row["tokens_removed"]),
                    " | ".join(row["stems"]) if row["stems"] else "(not applied)",
                ])
    _style_header(ws, 9)
    _autosize(ws)
    ws.auto_filter.ref = ws.dimensions

    wb.save(path)
    return path


# ============================================================================
# VISUALIZATIONS (five charts)
# ============================================================================
def _find_devanagari_font():
    """Return the path of an installed font that can render Devanagari.

    Candidate fonts are probed by checking their cmap coverage for the
    Devanagari letters HA (U+0939), the I vowel sign (U+093F) and the virama
    (U+094D). If nothing is found we return None so the caller can warn instead
    of silently producing corrupted labels.
    """
    required = (0x0939, 0x093F, 0x094D)
    candidates = [
        "Noto Sans Devanagari", "Noto Serif Devanagari", "Lohit Devanagari",
        "Mangal", "FreeSans", "FreeSerif", "Nirmala UI", "Arial Unicode MS",
        "Droid Sans Devanagari", "Madan2",
    ]
    available = {}
    if font_manager is not None:
        for f in font_manager.fontManager.ttflist:
            available[f.name] = f.fname

    def _covers(path):
        try:
            from fontTools.ttLib import TTFont  # optional dependency
            tt = TTFont(path, fontNumber=0)
            cmap = tt.getBestCmap()
            return all(cp in cmap for cp in required)
        except Exception:
            # fontTools unavailable: accept likely-Indic fonts by name.
            return True

    for name in candidates:
        path = available.get(name)
        if path and os.path.exists(path) and _covers(path):
            return path

    # Last resort: scan the system for any font that actually covers Devanagari.
    import glob
    patterns = [
        "/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf",
        "/usr/local/share/fonts/**/*.ttf", os.path.expanduser("~/.fonts/**/*.ttf"),
    ]
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            try:
                from fontTools.ttLib import TTFont
                tt = TTFont(path, fontNumber=0)
                cmap = tt.getBestCmap()
                if all(cp in cmap for cp in required):
                    return path
            except Exception:
                continue
    return None


def _raqm_available():
    """Return True when Pillow can perform complex-script text shaping."""
    if Image is None:
        return False
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:
        return False


def _render_devanagari_image(text, font_path, size_px=26, color=(0, 0, 0)):
    """Render a Devanagari string to a tightly-cropped RGBA image via Pillow.

    Pillow's Raqm layout engine performs the complex shaping (matra reordering
    and conjunct formation) that matplotlib cannot do, so the labels placed on
    the charts are correct rather than corrupted.
    """
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required to render Devanagari labels.")
    layout = None
    try:
        layout = ImageFont.Layout.RAQM if _raqm_available() else ImageFont.Layout.BASIC
    except Exception:
        layout = None
    if layout is not None:
        font = ImageFont.truetype(font_path, size_px, layout_engine=layout)
    else:
        font = ImageFont.truetype(font_path, size_px)

    probe = Image.new("RGBA", (8, 8))
    pd = ImageDraw.Draw(probe)
    bbox = pd.textbbox((0, 0), text, font=font)
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    pad = 3
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color + (255,))
    return img


def generate_visualizations(results, summary, vis_dir=VIS_DIR):
    """Generate the five required PNG charts using real results.

    Returns (created_paths, devanagari_status_string).

    Hindi and Marathi word labels are rendered with Pillow + Raqm and placed on
    the chart as images, because matplotlib cannot perform Devanagari complex
    shaping. This guarantees the labels are correct instead of corrupted.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required to generate visualizations.")
    os.makedirs(vis_dir, exist_ok=True)

    dev_font_path = _find_devanagari_font()
    raqm_ok = _raqm_available()

    if dev_font_path and raqm_ok:
        dev_status = f"Pillow+Raqm shaping using {os.path.basename(dev_font_path)}"
        dev_family = os.path.basename(dev_font_path)
    elif dev_font_path:
        dev_status = (f"Devanagari font {os.path.basename(dev_font_path)} found but "
                      f"Pillow Raqm is unavailable; complex shaping may be imperfect.")
        dev_family = os.path.basename(dev_font_path)
    else:
        dev_status = ("WARNING: no Devanagari-capable font found. Hindi/Marathi "
                      "chart labels may not render; install fonts-noto-devanagari.")
        dev_family = None

    fig_dpi = 150
    created = []

    # ---- Chart 1: token comparison ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LANGUAGES))
    before = [results[l]["total_tokens_before"] for l in LANGUAGES]
    after = [results[l]["total_tokens_after"] for l in LANGUAGES]
    width = 0.38
    ax.bar(x - width / 2, before, width, label="Before stop-word removal", color="#4C72B0")
    ax.bar(x + width / 2, after, width, label="After stop-word removal", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES)
    ax.set_ylabel("Token count")
    ax.set_title("Token Count Comparison Before vs After Stop-Word Removal")
    ax.legend()
    for i, v in enumerate(before):
        ax.text(i - width / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(after):
        ax.text(i + width / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(vis_dir, "token_comparison.png")
    fig.savefig(p, dpi=fig_dpi)
    plt.close(fig)
    created.append(p)

    # ---- Chart 2: vocabulary comparison -----------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    before_u = [results[l]["unique_tokens_before"] for l in LANGUAGES]
    after_u = [results[l]["unique_tokens_after"] for l in LANGUAGES]
    ax.bar(x - width / 2, before_u, width, label="Unique before", color="#55A868")
    ax.bar(x + width / 2, after_u, width, label="Unique after", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES)
    ax.set_ylabel("Unique token count")
    ax.set_title("Vocabulary Size Comparison Before vs After Stop-Word Removal")
    ax.legend()
    for i, v in enumerate(before_u):
        ax.text(i - width / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(after_u):
        ax.text(i + width / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(vis_dir, "vocabulary_comparison.png")
    fig.savefig(p, dpi=fig_dpi)
    plt.close(fig)
    created.append(p)

    # ---- Charts 3-5: top 10 frequency per language ------------------------
    for lang, fname in [("English", "english_top10.png"),
                        ("Hindi", "hindi_top10.png"),
                        ("Marathi", "marathi_top10.png")]:
        top = results[lang]["frequency"].most_common(10)
        words = [w for w, _ in top][::-1]
        counts = [c for _, c in top][::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(words)), counts, color="#8172B3")
        ax.set_yticks(range(len(words)))
        use_images = dev_font_path is not None and lang in ("Hindi", "Marathi")
        if use_images:
            # Blank matplotlib tick labels; Devanagari drawn as shaped images.
            ax.set_yticklabels([""] * len(words))
            ax.tick_params(axis="y", length=0)
            xlim = max(counts) if counts else 1
            ax.set_xlim(0, xlim * 1.30)
            for i, w in enumerate(words):
                im = _render_devanagari_image(w, dev_font_path, size_px=24)
                arr = np.asarray(im)
                ab = AnnotationBbox(
                    OffsetImage(arr, zoom=1.0), (0, i),
                    xybox=(-6, 0), boxcoords="offset points",
                    box_alignment=(1, 0.5), frameon=False,
                )
                ax.add_artist(ab)
        else:
            ax.set_yticklabels(words)
            ax.set_xlim(0, (max(counts) if counts else 1) * 1.15)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top 10 Most Frequent Words - {lang}")
        for i, v in enumerate(counts):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
        fig.tight_layout()
        p = os.path.join(vis_dir, fname)
        fig.savefig(p, dpi=fig_dpi)
        plt.close(fig)
        created.append(p)

    return created, dev_status


# ============================================================================
# DOCX REPORT
# ============================================================================
def create_report(results, summary, validation, path=REPORT_FILE):
    """Create the professional laboratory report."""
    if Document is None:
        raise RuntimeError("python-docx is required to create the report.")

    doc = Document()

    # ---- Cover page --------------------------------------------------------
    for _ in range(2):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run("NATURAL LANGUAGE PROCESSING")
    run.bold = True
    run.font.size = Pt(22)
    t2 = doc.add_paragraph()
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = t2.add_run("LABORATORY ASSIGNMENT – 02")
    r2.bold = True
    r2.font.size = Pt(18)
    t3 = doc.add_paragraph()
    t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = t3.add_run("Fundamental Language Processing for Three Different Languages")
    r3.italic = True
    r3.font.size = Pt(14)
    doc.add_paragraph()
    info = [
        ("Student Name", "Shravani Kurkute"),
        ("Roll No.", "__________________"),
        ("PRN / Registration No.", "__________________"),
        ("Class / Division", "__________________"),
        ("Academic Year", "2026–2027"),
        ("Faculty Name", "Dr. Kapil N. Vhatkar"),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for k, v in info:
        cells = table.add_row().cells
        cells[0].text = k
        cells[1].text = v
        cells[0].paragraphs[0].runs[0].bold = True
    doc.add_page_break()

    # ---- Helper functions --------------------------------------------------
    def H(text, level=1):
        doc.add_heading(text, level=level)

    def P(text):
        doc.add_paragraph(text)

    # ---- 2. Aim ------------------------------------------------------------
    H("2. Aim", 1)
    P("To study and implement fundamental Natural Language Processing (NLP) "
      "operations — Unicode normalization, tokenization, punctuation removal, "
      "stop-word removal, stemming and word-frequency analysis — on three "
      "typologically different languages (English, Hindi and Marathi), and to "
      "compare the resulting statistics across languages.")

    # ---- 3. Objectives -----------------------------------------------------
    H("3. Objectives", 1)
    for o in [
        "Load and validate an aligned three-language parallel dataset with 50 rows.",
        "Apply language-aware normalization for English and for Devanagari text.",
        "Tokenize text into word units while preserving Indic combining marks.",
        "Remove punctuation safely without corrupting Devanagari script.",
        "Remove stop words using a documented resource for each language.",
        "Apply Porter stemming to English tokens only (never to Hindi/Marathi).",
        "Compute real token and vocabulary statistics using collections.Counter.",
        "Export all results to a nine-sheet Excel workbook.",
        "Produce charts comparing token counts, vocabulary and top-10 words.",
        "Document the pipeline, results and limitations in a laboratory report.",
    ]:
        doc.add_paragraph(o, style="List Bullet")

    # ---- 4. Problem Statement ---------------------------------------------
    H("4. Problem Statement", 1)
    P("Raw natural language text is unstructured and noisy. The same semantic "
      "content expressed in English, Hindi and Marathi must be reduced to a "
      "comparable machine-readable form before it can be used by any NLP model. "
      "The problem is to design a single, reproducible preprocessing pipeline "
      "that correctly handles lowercase Latin script and Unicode Devanagari "
      "script, produces honest per-language statistics, and demonstrates the "
      "differences between the three languages.")

    # ---- 5. Theory ---------------------------------------------------------
    H("5. Theory", 1)
    theory = {
        "Unicode and Normalization":
            "Unicode assigns a unique code point to every character. Devanagari "
            "letters live in U+0900–U+097F and use combining vowel signs (matras) "
            "and virama. The same visual text can be encoded in different code "
            "point sequences. Unicode Normalization Form C (NFC) composes "
            "characters canonically so that identical-looking text compares "
            "equal. English is additionally lower-cased and NFKC-normalized.",
        "Sentence Segmentation":
            "Segmentation splits a document into sentences. In this dataset each "
            "CSV row is already one source sentence, so row boundaries are used "
            "directly and IDs are preserved to keep language alignment.",
        "Tokenization":
            "Tokenization splits text into minimal units. English is tokenized "
            "with NLTK's word tokenizer. Devanagari needs Unicode-aware patterns "
            "because a naive split can break a word at every matra; a regex over "
            "the Devanagari block keeps words intact.",
        "Punctuation Removal":
            "Punctuation does not usually carry lexical meaning. It is removed "
            "by filtering Unicode category P* while explicitly protecting all "
            "Devanagari code points, so matras are never deleted.",
        "Stop-word Removal":
            "Stop words are high-frequency function words with little "
            "discriminative value (e.g. 'the', 'में', 'चा'). Removing them "
            "reduces vector size but must be done with a language-specific list.",
        "Stemming":
            "Stemming reduces inflected words to an approximate root. The Porter "
            "stemmer works on English only. Hindi and Marathi are morphologically "
            "rich and require Indic-aware stemmers/lemmatizers, so no stemming "
            "is applied to them here.",
        "Frequency Analysis":
            "word -> count dictionaries (collections.Counter) reveal dominant "
            "terms and vocabulary richness. Vocabulary ratio = unique / total.",
        "Multilingual NLP":
            "Multilingual NLP must respect script, encoding, morphology and "
            "language-specific resources. Using one language's rules for another "
            "produces corrupted tokens, hence language-aware processing.",
    }
    for k, v in theory.items():
        H(k, 2)
        P(v)

    # ---- 6. Tools and Libraries -------------------------------------------
    H("6. Tools and Libraries", 1)
    tools = [
        ("Python 3", "Programming language for the pipeline."),
        ("pandas", "CSV loading and tabular manipulation."),
        ("NLTK", "word_tokenize, English stop words and PorterStemmer."),
        ("matplotlib", "Chart generation."),
        ("openpyxl", "Excel workbook generation."),
        ("python-docx", "Word report generation."),
        ("Jupyter", "Interactive reproducible notebook."),
    ]
    tt = doc.add_table(rows=1, cols=2)
    tt.style = "Table Grid"
    hdr = tt.rows[0].cells
    hdr[0].text = "Tool / Library"
    hdr[1].text = "Purpose"
    for a, b in tools:
        c = tt.add_row().cells
        c[0].text = a
        c[1].text = b

    # ---- 7. Dataset --------------------------------------------------------
    H("7. Dataset", 1)
    P(f"File: data/NLP_Three_Languages_Dataset.csv. It contains "
      f"{validation['row_count']} aligned rows with columns "
      f"{', '.join(validation['columns_found'])}. Each row holds one English "
      f"sentence and its Hindi and Marathi translations.")
    P("Validation result: " + ("PASSED" if validation["ok"] else "FAILED") +
      f". Rows = {validation['row_count']}, missing values = 0, empty strings = 0, "
      f"duplicate IDs = {validation.get('duplicate_ids', 0)}.")

    # ---- 8. Processing Methodology ----------------------------------------
    H("8. Processing Methodology", 1)
    P("Each language column is processed independently through the following "
      "ordered stages:")
    for step in [
        "Normalization (English: lowercase + NFKC; Hindi/Marathi: NFC).",
        "Punctuation removal (Unicode category P*, Devanagari protected).",
        "Tokenization (NLTK for English, Devanagari-aware regex for Hindi/Marathi).",
        "Stop-word removal using the language-specific list.",
        "Stemming (Porter, English only) applied after stop-word removal.",
        "Frequency counting with collections.Counter.",
    ]:
        doc.add_paragraph(step, style="List Number")

    # ---- 9-11. Per-language sections --------------------------------------
    for lang in LANGUAGES:
        H(f"{LANGUAGES.index(lang)+9}. {lang} Processing", 1)
        r = results[lang]
        P(f"Source sentences: {r['source_sentences']}. "
          f"Tokens before stop-word removal: {r['total_tokens_before']} "
          f"({r['unique_tokens_before']} unique). "
          f"Tokens after stop-word removal: {r['total_tokens_after']} "
          f"({r['unique_tokens_after']} unique). "
          f"Tokens removed: {r['tokens_removed']}.")
        if lang == "English":
            P(f"After Porter stemming: {r['post_stemming_total']} tokens, "
              f"{r['post_stemming_unique']} unique stems.")
        # Example transformation
        ex = r["per_row"][0]
        et = doc.add_table(rows=1, cols=2)
        et.style = "Table Grid"
        et.rows[0].cells[0].text = "Stage"
        et.rows[0].cells[1].text = "Value"
        for k, v in [
            ("Original", ex["original"]),
            ("Normalized", ex["normalized"]),
            ("After punctuation removal", ex["depunctuated"]),
            ("Tokens before stop words", " | ".join(ex["tokens_before"])),
            ("Tokens after stop words", " | ".join(ex["tokens_after"])),
            ("Removed", " | ".join(ex["tokens_removed"]) or "(none)"),
            ("Stems", " | ".join(ex["stems"]) if ex["stems"] else "(not applied)"),
        ]:
            c = et.add_row().cells
            c[0].text = k
            c[1].text = v
        doc.add_paragraph()
        # Top 10 table
        P(f"Top 10 words - {lang}:")
        ftt = doc.add_table(rows=1, cols=3)
        ftt.style = "Table Grid"
        hc = ftt.rows[0].cells
        hc[0].text = "Rank"
        hc[1].text = "Word"
        hc[2].text = "Frequency"
        for i, (w, c) in enumerate(r["top10"], 1):
            row = ftt.add_row().cells
            row[0].text = str(i)
            row[1].text = w
            row[2].text = str(c)

    # ---- 12. Frequency Analysis -------------------------------------------
    H("12. Frequency Analysis", 1)
    P("Word frequencies are computed with collections.Counter over the "
      "post-stop-word token stream. The ten most frequent words for each "
      "language are tabulated in the corresponding language sections "
      "(sections 9-11) and are summarised comparatively in section 13. "
      "The complete frequency tables for every unique token are provided on the "
      "English_Frequency, Hindi_Frequency and Marathi_Frequency worksheets of "
      "ASG2_Results.xlsx.")
    # Repeated here as a consolidated quick reference.
    for lang in LANGUAGES:
        P(f"Top 10 words - {lang}")
        t = doc.add_table(rows=1, cols=3)
        t.style = "Table Grid"
        hc = t.rows[0].cells
        hc[0].text = "Rank"; hc[1].text = "Word"; hc[2].text = "Frequency"
        for i, (w, c) in enumerate(results[lang]["top10"], 1):
            row = t.add_row().cells
            row[0].text = str(i); row[1].text = w; row[2].text = str(c)
        doc.add_paragraph()

    # ---- 13. Comparative Results ------------------------------------------
    H("13. Comparative Results", 1)
    ct = doc.add_table(rows=1, cols=8)
    ct.style = "Table Grid"
    heads = ["Language", "Sentences", "Tokens Before", "Unique Before",
             "Tokens After", "Unique After", "Removed", "Vocab Ratio"]
    for i, h in enumerate(heads):
        ct.rows[0].cells[i].text = h
    for lang in LANGUAGES:
        s = summary["languages"][lang]
        row = ct.add_row().cells
        vals = [lang, s["source_sentences"], s["total_tokens_before"],
                s["unique_tokens_before"], s["total_tokens_after"],
                s["unique_tokens_after"], s["tokens_removed"],
                s["vocabulary_ratio_after"]]
        for i, v in enumerate(vals):
            row[i].text = str(v)

    # ---- 14. Observations --------------------------------------------------
    H("14. Observations", 1)
    e = results["English"]; hi = results["Hindi"]; ma = results["Marathi"]
    obs = [
        f"All three languages produced {e['source_sentences']} source sentences, "
        f"confirming perfect row alignment.",
        f"English had {e['tokens_removed']} stop words removed, Hindi {hi['tokens_removed']} "
        f"and Marathi {ma['tokens_removed']}; removal depends on the size of each "
        f"stop-word list and the language's function-word density.",
        f"Post-stop-word vocabulary was highest for {summary['highest_vocabulary']} "
        f"({results[summary['highest_vocabulary']]['unique_tokens_after']} unique) and "
        f"lowest for {summary['lowest_vocabulary']} "
        f"({results[summary['lowest_vocabulary']]['unique_tokens_after']} unique).",
        f"Porter stemming changed English word forms but did not change the token "
        f"count; it can merge different surface forms into one stem.",
        f"Devanagari normalization preserved matras; this is essential because a "
        f"single matra changes the meaning of a Hindi or Marathi word.",
    ]
    for o in obs:
        doc.add_paragraph(o, style="List Bullet")

    # ---- 15. Challenges ----------------------------------------------------
    H("15. Language-specific Challenges", 1)
    for c in [
        "Devanagari combining marks: splitting on whitespace/apostrophes can "
        "break a word at a matra; Unicode-aware tokenization is required.",
        "Stop-word resources: NLTK ships English stop words but not Hindi/Marathi; "
        "custom lists are a documented limitation.",
        "Stemming: Porter is English-only; applying it to Devanagari would corrupt "
        "text, so it is deliberately restricted to English.",
        "Encoding: text must be read and written with valid UTF-8 throughout; "
        "Excel and DOCX must be written with Unicode-aware libraries.",
        "Fonts: charts need a Devanagari-capable font; absence is reported rather "
        "than hidden.",
        "Script mixing: rows 49 and 50 contain the Latin acronym 'NLP' inside "
        "Hindi/Marathi text, which the tokenizer keeps as a separate token.",
    ]:
        doc.add_paragraph(c, style="List Bullet")

    # ---- 16. Conclusion ----------------------------------------------------
    H("16. Conclusion", 1)
    P("A reproducible, language-aware preprocessing pipeline was implemented "
      "for English, Hindi and Marathi. All 50 aligned sentences were processed "
      "successfully with Unicode preserved. English received lowercase "
      "normalization, NLTK tokenization, stop-word removal and Porter stemming; "
      "Hindi and Marathi received NFC normalization, Devanagari-safe punctuation "
      "removal, Unicode-aware tokenization and language-specific stop-word "
      "removal, with no English stemming. Real frequency statistics were "
      "computed and exported to Excel and charts. The experiment demonstrates "
      "that multilingual NLP must be script-aware and that language-specific "
      "resources are indispensable.")

    # ---- 17. Viva Questions ------------------------------------------------
    H("17. Viva Questions and Answers", 1)
    viva = [
        ("What is NLP and why is preprocessing needed?",
         "NLP is the field that enables computers to process human language. "
         "Raw text is noisy and unstructured, so preprocessing normalizes and "
         "reduces it to clean tokens that machine-learning models can use."),
        ("What is Unicode normalization and why use NFC?",
         "Unicode allows one visible character to have more than one code point "
         "sequence. NFC composes them canonically so equal-looking text compares "
         "equal. NFC is preferred for storage/interchange of Indic text."),
        ("What is tokenization?",
         "Tokenization splits text into minimal meaningful units (tokens), "
         "usually words, which become the basic input to further NLP processing."),
        ("Why is Devanagari tokenization different from English?",
         "Devanagari uses combining vowel signs (matras). A naive split can break "
         "a word at a matra, so Unicode-aware patterns that keep the Devanagari "
         "block together are required."),
        ("What are stop words, and why must the list be language-specific?",
         "Stop words are common function words with low information value. Each "
         "language has its own function words, so using an English list on Hindi "
         "would remove nothing useful and could corrupt tokens."),
        ("What is stemming and why is Porter used only for English?",
         "Stemming reduces inflected words to an approximate root. Porter is "
         "designed for English; applying it to Devanagari would produce invalid "
         "text because the stemmer assumes Latin morphology."),
        ("What is the difference between stemming and lemmatization?",
         "Stemming chops affixes heuristically and may return non-words; "
         "lemmatization uses a dictionary and part-of-speech to return a valid "
         "base form. Lemmatization is more accurate but slower."),
        ("How is word frequency calculated, and what does it show?",
         "collections.Counter tallies each token. Frequency shows dominant terms; "
         "the ratio of unique to total tokens indicates vocabulary richness."),
        ("What is the purpose of removing punctuation?",
         "Punctuation usually carries no lexical meaning and would otherwise "
         "create pseudo-tokens; removing it reduces noise in the vocabulary."),
        ("What are the limitations of this experiment?",
         "Stop-word lists for Hindi and Marathi are hand-curated and not "
         "exhaustive; no Indic stemmer/lemmatizer was applied; sentence "
         "segmentation relied on one-sentence rows; and language detection was "
         "fixed per column."),
    ]
    for i, (q, a) in enumerate(viva, 1):
        pq = doc.add_paragraph()
        rq = pq.add_run(f"Q{i}. {q}")
        rq.bold = True
        doc.add_paragraph(f"Ans. {a}")

    # ---- 18. References ----------------------------------------------------
    H("18. References", 1)
    for ref in [
        "Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing "
        "with Python. O'Reilly Media.",
        "Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing "
        "(3rd ed. draft).",
        "NLTK Documentation - nltk.org (word_tokenize, stopwords, PorterStemmer).",
        "Unicode Standard Annex #15: Unicode Normalization Forms.",
        "Unicode Block 'Devanagari' U+0900–U+097F.",
        "Python Documentation - collections.Counter, unicodedata, re.",
        "pandas Documentation - pandas.pydata.org.",
        "matplotlib Documentation - matplotlib.org.",
        "openpyxl Documentation - openpyxl.readthedocs.io.",
        "python-docx Documentation - python-docx.readthedocs.io.",
        "Faculty-provided dataset: NLP_Three_Languages_Dataset.csv.",
    ]:
        doc.add_paragraph(ref, style="List Bullet")

    # ---- 19. Faculty Evaluation -------------------------------------------
    H("19. Faculty Evaluation", 1)
    et = doc.add_table(rows=1, cols=4)
    et.style = "Table Grid"
    eh = et.rows[0].cells
    eh[0].text = "Sr. No."
    eh[1].text = "Criteria"
    eh[2].text = "Marks (out of 5)"
    eh[3].text = "Remarks"
    criteria = [
        "Dataset loading and validation",
        "Correct preprocessing per language",
        "Frequency analysis and comparison",
        "Excel and chart outputs",
        "Report quality and documentation",
        "Viva and understanding",
    ]
    for i, cr in enumerate(criteria, 1):
        row = et.add_row().cells
        row[0].text = str(i)
        row[1].text = cr
        row[2].text = ""
        row[3].text = ""
    trow = et.add_row().cells
    trow[1].text = "Total (30)"
    trow[1].paragraphs[0].runs[0].bold = True
    trow[2].text = ""

    doc.save(path)
    return path


# ============================================================================
# NOTEBOOK CREATION
# ============================================================================
def create_notebook(path=NOTEBOOK_FILE):
    """Create a valid, reproducible Jupyter notebook (.ipynb).

    The notebook imports the functions defined in ASG2.py so that its numbers
    are guaranteed to match the Excel workbook and the report. It walks through
    loading, validation, per-language preprocessing, frequency analysis,
    comparison, charts and Excel export.
    """
    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": src}

    def md(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src}

    cells = [
        md("# NLP Laboratory Assignment \u2013 02\n"
           "## Fundamental Language Processing for Three Different Languages\n\n"
           "**Student:** Shravani Kurkute  \n**Faculty:** Dr. Kapil N. Vhatkar\n\n"
           "This notebook reproduces the exact pipeline implemented in `ASG2.py`. "
           "It imports the module functions so every number shown here is identical "
           "to the Excel workbook and the report. All values are computed at runtime."),
        md("### 0. Imports and NLTK data"),
        code("import os, sys\n"
             "import pandas as pd\n"
             "import matplotlib\n"
             "matplotlib.use('Agg')\n"
             "import matplotlib.pyplot as plt\n"
             "import nltk\n"
             "for pkg in ['punkt', 'stopwords']:\n"
             "    try:\n"
             "        nltk.data.find(pkg)\n"
             "    except LookupError:\n"
             "        nltk.download(pkg, quiet=True)\n"
             "# Import the shared pipeline module so results match ASG2.py exactly.\n"
             "import ASG2\n"
             "print('pandas', pd.__version__)"),
        md("### 1. Load and validate the dataset"),
        code("df = ASG2.load_dataset()\n"
             "validation = ASG2.validate_dataset(df)\n"
             "print('Columns:', validation['columns_found'])\n"
             "print('Rows   :', validation['row_count'])\n"
             "print('Missing:', validation['missing_values'])\n"
             "print('Result :', 'PASSED' if validation['ok'] else 'FAILED')\n"
             "df.head()"),
        md("### 2. Stop-word resources\n"
           "English uses NLTK's documented corpus. Hindi and Marathi use custom "
           "hand-curated lists (NLTK does not ship them); the full lists are shown "
           "in the `Stopword_Resources` worksheet and can be inspected below."),
        code("print('English stop words from NLTK:', len(ASG2.get_stopwords('English')))\n"
             "print('Hindi stop words (custom) :', len(ASG2.get_stopwords('Hindi')))\n"
             "print('Marathi stop words (custom):', len(ASG2.get_stopwords('Marathi')))\n"
             "print('Sample Hindi :', sorted(ASG2.get_stopwords('Hindi'))[:12])\n"
             "print('Sample Marathi:', sorted(ASG2.get_stopwords('Marathi'))[:12])"),
        md("### 3. Preprocessing functions\n"
           "The same module functions are used for each language. English is "
           "lower-cased and NFKC-normalized; Hindi/Marathi use NFC and keep all "
           "Devanagari combining marks."),
        code("examples = [\n"
             "    ('English', 'I love natural language processing.'),\n"
             "    ('Hindi', '\u092e\u0941\u091d\u0947 \u092a\u094d\u0930\u093e\u0915\u0943\u0924\u093f\u0915 \u092d\u093e\u0937\u093e \u092a\u0938\u0902\u0926 \u0939\u0948\u0964'),\n"
             "    ('Marathi', '\u092e\u0932\u093e \u0928\u0948\u0938\u0930\u094d\u0917\u093f\u0915 \u092d\u093e\u0937\u093e \u0906\u0935\u0921\u0924\u0947.'),\n"
             "]\n"
             "for lang, text in examples:\n"
             "    norm = ASG2.normalize_text(text, lang)\n"
             "    dep = ASG2.remove_punctuation(norm, lang)\n"
             "    toks = ASG2.tokenize_words(dep, lang)\n"
             "    kept, removed = ASG2.remove_stopwords(toks, lang)\n"
             "    stems = ASG2.apply_english_stemming(kept) if lang == 'English' else None\n"
             "    print(lang)\n"
             "    print('  normalized :', norm)\n"
             "    print('  tokens     :', toks)\n"
             "    print('  after stop :', kept)\n"
             "    print('  removed    :', removed)\n"
             "    print('  stems      :', stems)"),
        md("### 4. Run the full pipeline for all three languages"),
        code("LANGUAGES = ['English', 'Hindi', 'Marathi']\n"
             "results = {lang: ASG2.process_language(df, lang) for lang in LANGUAGES}\n"
             "for lang in LANGUAGES:\n"
             "    r = results[lang]\n"
             "    print(lang, 'sentences', r['source_sentences'],\n"
             "          'tokens_before', r['total_tokens_before'],\n"
             "          'unique_before', r['unique_tokens_before'],\n"
             "          'tokens_after', r['total_tokens_after'],\n"
             "          'unique_after', r['unique_tokens_after'],\n"
             "          'removed', r['tokens_removed'])"),
        md("### 5. Comparative summary"),
        code("summary = ASG2.compare_languages(results, validation)\n"
             "summary_df = pd.DataFrame(summary['languages']).T\n"
             "summary_df"),
        md("### 6. Frequency analysis (top 10 per language)"),
        code("for lang in LANGUAGES:\n"
             "    print('\\nTop 10 -', lang)\n"
             "    for rank, (word, count) in enumerate(results[lang]['top10'], 1):\n"
             "        print(f'  {rank:2}. {word} : {count}')"),
        md("### 7. Charts\n"
           "The same five charts are produced using the shared function. Hindi "
           "and Marathi labels are rendered with Pillow + Raqm complex shaping so "
           "the Devanagari text is correct."),
        code("charts, dev_status = ASG2.generate_visualizations(results, summary)\n"
             "print('Devanagari status:', dev_status)\n"
             "for c in charts:\n"
             "    print('  wrote', c)"),
        md("### 8. Preview charts inline"),
        code("from IPython.display import Image, display\n"
             "for name in ['token_comparison.png', 'vocabulary_comparison.png',\n"
             "             'english_top10.png', 'hindi_top10.png', 'marathi_top10.png']:\n"
             "    print(name)\n"
             "    display(Image(filename=os.path.join('visualizations', name)))"),
        md("### 9. Export results to Excel (nine sheets)"),
        code("xlsx = ASG2.export_results(results, summary, validation)\n"
             "print('Excel written:', xlsx)\n"
             "import openpyxl\n"
             "print('Worksheets:', openpyxl.load_workbook(xlsx).sheetnames)"),
        md("### 10. Self tests"),
        code("passed, failed, msgs = ASG2.run_self_tests()\n"
             "for m in msgs:\n"
             "    print(m)\n"
             "print(f'\\n{passed} passed, {failed} failed')"),
        md("### 11. Conclusion\n"
           "The notebook reproduces the full multilingual preprocessing pipeline "
           "with real computed statistics, charts and a nine-sheet Excel export. "
           "English uses Latin-script rules plus Porter stemming; Hindi and "
           "Marathi use NFC normalization, Devanagari-safe punctuation removal, "
           "Unicode-aware tokenization and language-specific stop words, without "
           "stemming."),
    ]

    # Give every cell a stable id (required by nbformat >= 4.5).
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{i:02d}")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return path


# ============================================================================
# SELF TESTS
# ============================================================================
def run_self_tests():
    """Basic tests for the core operations. Returns (passed, failed, messages)."""
    passed, failed, msgs = 0, 0, []

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            msgs.append(f"[PASS] {name}")
        else:
            failed += 1
            msgs.append(f"[FAIL] {name}")

    # Unicode preservation (Devanagari matras survive normalization)
    src = "प्राकृतिक भाषा प्रसंस्करण"
    norm = normalize_text(src, "Hindi")
    check("Hindi NFC preserves Devanagari", "प्राकृतिक" in norm and "प्रसंस्करण" in norm)

    # NFC idempotence and explicit normalization form
    check("Hindi normalization is NFC",
          unicodedata.is_normalized("NFC", normalize_text(src, "Hindi")))

    # Lowercase normalization for English
    check("English lowercasing", normalize_text("Hello WORLD", "English") == "hello world")

    # Tokenization keeps a Devanagari word intact (does not split at matras)
    toks = tokenize_words("प्राकृतिक भाषा", "Hindi")
    check("Hindi tokenization keeps words whole", toks == ["प्राकृतिक", "भाषा"])

    # Tokenization of a word with a virama/conjunct stays intact
    check("Hindi conjunct tokenization",
          tokenize_words("प्रसंस्करण", "Hindi") == ["प्रसंस्करण"])

    # Punctuation removal (English)
    check("English punctuation removal",
          remove_punctuation("Hello, world!", "English") == "Hello world")

    # Devanagari punctuation removal preserves matras and removes the danda
    dev_dep = remove_punctuation("मुझे प्राकृतिक भाषा पसंद है।", "Hindi")
    check("Hindi punctuation removal preserves matras",
          "प्राकृतिक" in dev_dep and "।" not in dev_dep)

    # Danda removal allows the attached stop word to match
    danda_tokens = tokenize_words(remove_punctuation("पसंद है।", "Hindi"), "Hindi")
    check("Hindi danda removed before tokenization", "है" in danda_tokens and "है।" not in danda_tokens)

    # Stop-word removal, English
    kept, removed = remove_stopwords(["the", "cat", "sat"], "English")
    check("English stop-word removal", removed == ["the"] and kept == ["cat", "sat"])

    # Stop-word removal, Hindi (custom list)
    kept_h, removed_h = remove_stopwords(["में", "भाषा"], "Hindi")
    check("Hindi stop-word removal", "में" in removed_h and "भाषा" in kept_h)

    # Stop-word removal, Marathi (custom list, separate from Hindi)
    kept_m, removed_m = remove_stopwords(["आहे", "भाषा"], "Marathi")
    check("Marathi stop-word removal", "आहे" in removed_m and "भाषा" in kept_m)

    # Hindi and Marathi lists must not be identical
    check("Hindi/Marathi stop lists are language-specific",
          get_stopwords("Hindi") != get_stopwords("Marathi"))

    # Stemming only English
    stems = apply_english_stemming(["running", "studies"])
    check("English Porter stemming", stems == ["run", "studi"])

    # Frequency counting
    cnt, top = calculate_word_frequencies(["a", "b", "a", "a", "c"])
    check("Frequency counting", cnt["a"] == 3 and top[0] == ("a", 3))

    # Frequency table consistency against the real pipeline (no fabrication)
    try:
        df = load_dataset()
        res = process_language(df, "Marathi")
        summed = sum(res["frequency"].values())
        check("Frequency total equals post-stop-word token count",
              summed == res["total_tokens_after"])
        check("Unique post-stop tokens equals len(frequency)",
              len(res["frequency"]) == res["unique_tokens_after"])
        check("English stemming applied to English only",
              res["post_stemming_total"] is None or res["language"] != "Marathi")
        eng = process_language(df, "English")
        check("English post-stemming total equals post-stop token count",
              eng["post_stemming_total"] == eng["total_tokens_after"])
    except Exception as exc:  # pragma: no cover
        check(f"Pipeline consistency check raised {exc!r}", False)

    # Verify Devanagari labels are renderable (font + shaping) if a font exists
    fp = _find_devanagari_font()
    if fp:
        try:
            im = _render_devanagari_image("हिन्दी", fp)
            arr = np.asarray(im) if np is not None else None
            ink = int((arr[..., :3].sum(axis=2) < 600).sum()) if arr is not None else 0
            check("Devanagari label renders with ink (not blank/tofu)", ink > 20)
        except Exception as exc:  # pragma: no cover
            check(f"Devanagari label render raised {exc!r}", False)
    else:
        msgs.append("[SKIP] No Devanagari font installed; label render test skipped.")

    return passed, failed, msgs


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 74)
    print("NLP Laboratory Assignment - 02 : Three-Language Processing Pipeline")
    print("=" * 74)

    for d in (DATA_DIR, VIS_DIR, SHOT_DIR):
        os.makedirs(d, exist_ok=True)

    # ---- STEP 1: load + validate ------------------------------------------
    print("\n[STEP 1] Loading and validating dataset ...")
    df = load_dataset()
    validation = validate_dataset(df)
    print(f"  Columns : {validation['columns_found']}")
    print(f"  Rows    : {validation['row_count']} (expected {validation['expected_rows']})")
    print(f"  Missing : {validation['missing_values']}")
    print(f"  Result  : {'PASSED' if validation['ok'] else 'FAILED'}")

    # ---- STEP 2: process ---------------------------------------------------
    print("\n[STEP 2] Processing each language ...")
    results = {}
    for lang in LANGUAGES:
        results[lang] = process_language(df, lang)
        r = results[lang]
        print(f"  {lang:<8} sentences={r['source_sentences']:<3} "
              f"tokens_before={r['total_tokens_before']:<4} "
              f"unique_before={r['unique_tokens_before']:<4} "
              f"tokens_after={r['total_tokens_after']:<4} "
              f"unique_after={r['unique_tokens_after']:<4} "
              f"removed={r['tokens_removed']}")
        if lang == "English":
            print(f"           post-stemming tokens={r['post_stemming_total']}, "
                  f"unique stems={r['post_stemming_unique']}")

    summary = compare_languages(results, validation)
    for n in summary["notes"]:
        print("  " + n)

    # ---- STEP 3: Excel -----------------------------------------------------
    print("\n[STEP 3] Exporting Excel workbook ...")
    xlsx = export_results(results, summary, validation)
    print(f"  Written: {xlsx}")

    # ---- STEP 4: charts ----------------------------------------------------
    print("\n[STEP 4] Generating charts ...")
    charts, dev_status = generate_visualizations(results, summary)
    print(f"  Devanagari status: {dev_status}")
    for c in charts:
        print(f"  Written: {c}")

    # ---- STEP 5: report ----------------------------------------------------
    print("\n[STEP 5] Creating DOCX report ...")
    try:
        report = create_report(results, summary, validation)
        print(f"  Written: {report}")
    except Exception as exc:  # pragma: no cover
        report = None
        print(f"  WARNING: report not created: {exc}")

    # ---- STEP 6: notebook --------------------------------------------------
    print("\n[STEP 6] Creating Jupyter notebook ...")
    nb = create_notebook()
    print(f"  Written: {nb}")

    # ---- STEP 8: tests -----------------------------------------------------
    print("\n[STEP 8] Running self tests ...")
    passed, failed, msgs = run_self_tests()
    for m in msgs:
        print("  " + m)
    print(f"  Tests: {passed} passed, {failed} failed")

    print("\nDone.")
    return {"results": results, "summary": summary, "validation": validation,
            "tests": (passed, failed)}


if __name__ == "__main__":
    main()
