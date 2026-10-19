from collections import Counter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


# -------------------- Training and testing data --------------------

training_sentences = [
    "I love natural language processing.",
    "I love machine learning.",
    "Natural language processing is interesting.",
    "Machine learning is interesting.",
    "Natural language processing is useful.",
    "Machine learning is useful.",
    "Students learn natural language processing.",
    "Students learn machine learning.",
    "Students study natural language processing.",
    "Students study machine learning.",
    "Natural language processing helps students.",
    "Machine learning helps students.",
    "I study natural language processing.",
    "I study machine learning.",
    "I enjoy learning natural language processing.",
    "I enjoy learning machine learning.",
    "Learning natural language processing is interesting.",
    "Learning machine learning is interesting.",
    "Natural language processing is a useful field.",
    "Machine learning is a useful field.",
    "Students find natural language processing interesting.",
    "Students find machine learning interesting.",
    "I find natural language processing useful.",
    "I find machine learning useful.",
    "Students love learning natural language processing.",
]

test_sentences = [
    "I love natural language processing.",
    "Students learn machine learning.",
    "Natural language processing helps students.",
    "I enjoy learning machine learning.",
    "Students find machine learning useful.",
]


# -------------------- Tokenizer --------------------

def tokenize(sentence):
    """Convert a sentence into tokens with start/end markers."""
    cleaned = sentence.lower().rstrip(".")
    return ["", *cleaned.split(), "孺"]


# -------------------- Ngram count builder --------------------

def build_ngram_counts(corpus):
    """Create unigram and bigram frequency tables."""
    unigram_counts = Counter()
    bigram_counts = Counter()
    for sentence in corpus:
        words = tokenize(sentence)
        unigram_counts.update(words)
        bigram_counts.update(
            (first, second) for first, second in zip(words, words[1:])
        )
    return unigram_counts, bigram_counts


# -------------------- Excel formatting helpers --------------------

FONT_NAME = "Arial"
HEADER_STYLE = Font(name=FONT_NAME, bold=True, color="FFFFFF")
HEADER_BACKGROUND = PatternFill("solid", fgColor="1F4E78")
NORMAL_STYLE = Font(name=FONT_NAME)
INPUT_STYLE = Font(name=FONT_NAME, color="0000FF")
EMPHASIS_STYLE = Font(name=FONT_NAME, bold=True)


def write_headers(sheet, row_number, headings):
    for column_number, heading in enumerate(headings, start=1):
        target = sheet.cell(row=row_number, column=column_number, value=heading)
        target.font = HEADER_STYLE
        target.fill = HEADER_BACKGROUND
        target.alignment = Alignment(horizontal="center", wrap_text=True)


def set_widths(sheet, widths):
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width


# -------------------- Sheet creators --------------------


def create_unigram_sheet(workbook, unigram_counts):
    sheet = workbook.active
    sheet.title = "Unigrams"
    write_headers(sheet, 1, ["Word", "Count"])
    ordered_words = sorted(
        unigram_counts.items(), key=lambda item: (-item[1], item[0])
    )
    row_number = 2
    for word, frequency in ordered_words:
        sheet.cell(row=row_number, column=1, value=word).font = NORMAL_STYLE
        sheet.cell(row=row_number, column=2, value=frequency).font = INPUT_STYLE
        row_number += 1
    final_row = row_number - 1
    sheet.cell(row=1, column=4, value="Vocabulary size V (incl.  and 孺)").font = EMPHASIS_STYLE
    sheet.cell(row=1, column=5, value="Total tokens").font = EMPHASIS_STYLE
    sheet.cell(row=2, column=4, value=f"=COUNTA(A2:A{final_row})").font = NORMAL_STYLE
    sheet.cell(row=2, column=5, value=f"=SUM(B2:B{final_row})").font = NORMAL_STYLE
    sheet.cell(row=1, column=4).font = EMPHASIS_STYLE
    sheet.cell(row=1, column=5).font = EMPHASIS_STYLE
    sheet.cell(row=2, column=4).font = NORMAL_STYLE
    sheet.cell(row=2, column=5).font = NORMAL_STYLE
    sheet.cell(row=3, column=4).font = NORMAL_STYLE
    sheet.cell(row=3, column=5).font = NORMAL_STYLE
    note = (
        "Note: counts come from the 25-sentence training corpus, "
        "lowercased, punctuation stripped, with /孺 added. "
        "V = Vocabulary size including  and 孺."
    )
    sheet.cell(row=4, column=4, value=note).font = NORMAL_STYLE
    set_widths(sheet, {"A": 16, "D": 40})
    return final_row


def create_bigram_sheet(workbook, bigram_counts, unigram_last_row):
    sheet = workbook.create_sheet("Bigrams")
    write_headers(
        sheet,
        1,
        ["w1", "w2", "C(w1,w2)", "C(w1)", "MLE P(w2|w1)", "Laplace P(w2|w1)"],
    )
    ordered_bigrams = sorted(
        bigram_counts.items(), key=lambda item: (-item[1], item[0])
    )
    row_number = 2
    for (first_word, second_word), frequency in ordered_bigrams:
        sheet.cell(row=row_number, column=1, value=first_word).font = NORMAL_STYLE
        sheet.cell(row=row_number, column=2, value=second_word).font = NORMAL_STYLE
        sheet.cell(row=row_number, column=3, value=frequency).font = INPUT_STYLE
        sheet.cell(
            row=row_number,
            column=4,
            value=f"=SUMPRODUCT((Unigrams!A2:A{unigram_last_row}=A{row_number})*(Unigrams!B2:B{unigram_last_row}))",
        ).font = NORMAL_STYLE
        sheet.cell(
            row=row_number,
            column=5,
            value=f"=C{row_number}/D{row_number}",
        ).font = NORMAL_STYLE
        sheet.cell(
            row=row_number,
            column=6,
            value=f"=(C{row_number}+1)/(D{row_number}+Unigrams!E1)",
        ).font = NORMAL_STYLE
        sheet.cell(row=row_number, column=5).number_format = "0.0000"
        sheet.cell(row=row_number, column=6).number_format = "0.0000"
        row_number += 1
    bigram_end = row_number - 1
    set_widths(
        sheet,
        {"A": 14, "B": 14, "C": 12, "D": 10, "E": 16, "F": 16},
    )
    return bigram_end


def create_test_calculation_sheet(workbook, test_data, bigram_last_row, unigram_last_row):
    sheet = workbook.create_sheet("Test_Calc")
    write_headers(
        sheet,
        1,
        ["Sentence #", "w1", "w2", "C(w1,w2)", "C(w1)", "MLE", "Laplace"],
    )
    current_row = 2
    sentence_ranges = {}
    for sentence_number, sentence in enumerate(test_data, start=1):
        words = tokenize(sentence)
        first_row = current_row
        for first_word, second_word in zip(words, words[1:]):
            sheet.cell(row=current_row, column=1, value=sentence_number).font = NORMAL_STYLE
            sheet.cell(row=current_row, column=2, value=first_word).font = NORMAL_STYLE
            sheet.cell(row=current_row, column=3, value=second_word).font = NORMAL_STYLE
            sheet.cell(
                row=current_row,
                column=4,
                value=f"=SUMPRODUCT((Bigrams!A2:A{bigram_last_row}=B{current_row})*(Bigrams!B2:B{bigram_last_row}=C{current_row})*(Bigrams!C2:C{bigram_last_row}))",
            ).font = NORMAL_STYLE
            sheet.cell(
                row=current_row,
                column=5,
                value=f"=SUMPRODUCT((Unigrams!A2:A{unigram_last_row}=B{current_row})*(Unigrams!B2:B{unigram_last_row}))",
            ).font = NORMAL_STYLE
            sheet.cell(
                row=current_row,
                column=6,
                value=f"=D{current_row}/E{current_row}",
            ).font = NORMAL_STYLE
            sheet.cell(
                row=current_row,
                column=7,
                value=f"=(D{current_row}+1)/(E{current_row}+Unigrams!E1)",
            ).font = NORMAL_STYLE
            sheet.cell(row=current_row, column=6).number_format = "0.0000"
            sheet.cell(row=current_row, column=7).number_format = "0.0000"
            current_row += 1
        sentence_ranges[sentence_number] = (first_row, current_row - 1)
    set_widths(
        sheet,
        {"A": 11, "B": 14, "C": 14, "D": 11, "E": 9, "F": 11, "G": 11},
    )
    return sentence_ranges


def create_summary_sheet(workbook, test_data, sentence_ranges):
    sheet = workbook.create_sheet("Summary", 0)
    write_headers(
        sheet,
        1,
        ["#", "Test sentence", "P (MLE)", "P (Laplace)", "log P (Laplace)"],
    )
    for sentence_number, sentence in enumerate(test_data, start=1):
        first_row, last_row = sentence_ranges[sentence_number]
        result_row = sentence_number + 1
        sheet.cell(row=result_row, column=1, value=sentence_number).font = NORMAL_STYLE
        sheet.cell(row=result_row, column=2, value=sentence).font = NORMAL_STYLE
        sheet.cell(
            row=result_row,
            column=3,
            value=f"=PRODUCT(Test_Calc!F{first_row}:F{last_row})",
        ).font = NORMAL_STYLE
        sheet.cell(
            row=result_row,
            column=4,
            value=f"=PRODUCT(Test_Calc!G{first_row}:G{last_row})",
        ).font = NORMAL_STYLE
        sheet.cell(
            row=result_row,
            column=5,
            value=f"=LN(D{result_row})",
        ).font = NORMAL_STYLE
        sheet.cell(row=result_row, column=3).number_format = "0.00000"
        sheet.cell(row=result_row, column=4).number_format = "0.00E+00"
        sheet.cell(row=result_row, column=5).number_format = "0.000"
        sheet.cell(row=result_row, column=3).font = NORMAL_STYLE
        sheet.cell(row=result_row, column=4).font = NORMAL_STYLE
        sheet.cell(row=result_row, column=5).font = NORMAL_STYLE
    summary_note = (
        "MLE: P(w2|w1)=C(w1,w2)/C(w1). "
        "Laplace: (C(w1,w2)+1)/(C(w1)+V), "
        "V from Unigrams!E1. Sentence probability = product of bigram probabilities."
    )
    sheet.cell(row=len(test_data) + 3, column=1, value=summary_note).font = NORMAL_STYLE
    set_widths(sheet, {"A": 5, "B": 50, "C": 16, "D": 16, "E": 16})
    return len(test_data) + 2


def build_workbook():
    unigram_table, bigram_table = build_ngram_counts(training_sentences)
    workbook = Workbook()
    unigram_end = create_unigram_sheet(workbook, unigram_table)
    bigram_end = create_bigram_sheet(workbook, bigram_table, unigram_end)
    ranges = create_test_calculation_sheet(
        workbook, test_sentences, bigram_end, unigram_end
    )
    create_summary_sheet(workbook, test_sentences, ranges)
    workbook.save("bigram_model.xlsx")
    print("Saved bigram_model.xlsx")


# -------------------- Main execution --------------------

if __name__ == "__main__":
    # Calculate counts for verification
    unigram_table, bigram_table = build_ngram_counts(training_sentences)

    # Verify reference values
    total_tokens = sum(unigram_table.values())
    vocabulary_size = len(unigram_table)
    distinct_bigrams = len(bigram_table)

    print(f"Training sentences: {len(training_sentences)}")
    print(f"Test sentences: {len(test_sentences)}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Total tokens: {total_tokens}")
    print(f"Distinct bigrams: {distinct_bigrams}")

    # Verify expected reference values
    expected_tokens = 176
    expected_vocab = 20
    expected_bigrams = 48

    tokens_ok = total_tokens == expected_tokens
    vocab_ok = vocabulary_size == expected_vocab
    bigrams_ok = distinct_bigrams == expected_bigrams

    print(f"\nVerification:")
    print(f"  Token count: {total_tokens} (expected {expected_tokens}) {'OK' if tokens_ok else 'DIFFER'}")
    print(f"  Vocabulary size: {vocabulary_size} (expected {expected_vocab}) {'OK' if vocab_ok else 'DIFFER'}")
    bg_status = "OK" if bigrams_ok else "DIFFER"
    print(f"  Distinct bigrams: {distinct_bigrams} (expected {expected_bigrams}) {bg_status}")

    if not (tokens_ok and vocab_ok and bigrams_ok):
        print("\nNote: Independently calculated values differ from reference - this will be reported in the assignment.")

    build_workbook()
    print("\nVerification info complete.")