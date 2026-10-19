"""Integration tests against the real glove-wiki-gigaword-100 model.

These tests load the 400k x 100 model (about 20 seconds) and verify the
assignment's required example and the small evaluation set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from country_prediction import (  # noqa: E402
    CountryAnalogyPredictor,
    MissingWordError,
    evaluate_predictions,
    load_candidate_countries,
    load_embeddings,
    load_records,
    DATASET_DIR,
)


@pytest.fixture(scope="module")
def predictor():
    vectors, info = load_embeddings()
    predictor = CountryAnalogyPredictor(vectors, load_candidate_countries())
    predictor.model_info = info
    return predictor


def test_model_properties(predictor):
    assert predictor.model_info["vocabulary_size"] == 400_000
    assert predictor.model_info["dimension"] == 100
    assert len(predictor.countries) == 131


def test_required_example_athens_greece_baghdad(predictor):
    assert predictor.predict("athens", "greece", "baghdad") == "iraq"


def test_required_example_top_five(predictor):
    ranked = predictor.rank("athens", "greece", "baghdad", topn=5)
    words = [word for word, _ in ranked]
    scores = [score for _, score in ranked]
    assert words[0] == "iraq"
    assert scores == sorted(scores, reverse=True)
    assert len(words) == 5


def test_small_test_set_is_mostly_correct(predictor):
    records = load_records(DATASET_DIR / "test_analogies.csv")
    rows, metrics = evaluate_predictions(predictor, records, country_filter=True)
    assert metrics["total"] == 15
    assert metrics["correct_top1"] >= 13
    assert metrics["top5_accuracy_pct"] == 100.0
    assert rows[0]["predicted_country"] == "iraq"


def test_missing_word_error_from_real_vocabulary(predictor):
    with pytest.raises(MissingWordError):
        predictor.predict("athens", "greece", "zzzznotaword")


def test_prediction_is_case_insensitive(predictor):
    assert predictor.predict("Athens", "GREECE", "Baghdad") == "iraq"
