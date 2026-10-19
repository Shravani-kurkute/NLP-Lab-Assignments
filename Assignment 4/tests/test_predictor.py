"""Unit tests for the predictor maths and ranking logic.

These tests use a tiny hand-crafted embedding, so they are fast and do not
require downloading GloVe. A separate integration test uses the real model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from gensim.models import KeyedVectors

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from country_prediction import (  # noqa: E402
    CountryAnalogyPredictor,
    MissingWordError,
    evaluate_predictions,
)


def make_tiny_vectors() -> KeyedVectors:
    words = ["capital1", "country1", "capital2", "country2", "unrelated"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.7071],
            [0.2, 0.1, 0.9],
        ],
        dtype=np.float32,
    )
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(words, vectors)
    return kv


def make_stub_predictor() -> CountryAnalogyPredictor:
    """Predictor with a fixed analogy query and hand-made candidate scores."""
    predictor = object.__new__(CountryAnalogyPredictor)
    predictor.countries = ["iraq", "afghanistan", "greece"]
    predictor.country_index = {
        country: index for index, country in enumerate(predictor.countries)
    }
    predictor.country_matrix = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
    )

    predictor.vocabulary = [
        "athens",
        "greece",
        "baghdad",
        "london",
        "iraq",
        "afghanistan",
        "france",
        "123",
    ]
    predictor.word_index = {
        word: index for index, word in enumerate(predictor.vocabulary)
    }
    predictor.normalized = np.array(
        [
            [1.0, 0.0],    # athens
            [0.0, 1.0],    # greece
            [0.6, 0.8],    # baghdad
            [0.0, 0.5],    # london
            [1.0, 0.0],    # iraq
            [0.8, 0.6],    # afghanistan
            [0.7, 0.7],    # france
            [0.3, 0.95],   # 123 (not alphabetic)
        ],
        dtype=np.float32,
    )
    predictor.alpha_mask = np.array(
        [word.isalpha() for word in predictor.vocabulary], dtype=bool
    )
    predictor.analogy_vector = lambda *args: np.array([1.0, 0.0], dtype=np.float32)
    return predictor


def record(actual: str, capital1="athens", country1="greece", capital2="baghdad"):
    return {
        "capital1": capital1,
        "country1": country1,
        "capital2": capital2,
        "country2": actual,
    }


def test_normalize_produces_unit_vector():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    unit = CountryAnalogyPredictor.normalize(vector)
    assert np.isclose(np.linalg.norm(unit), 1.0)
    assert np.allclose(unit, [0.6, 0.8])


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        CountryAnalogyPredictor.normalize(np.zeros(4, dtype=np.float32))


def test_analogy_vector_is_country_minus_capital_plus_capital():
    vectors = make_tiny_vectors()
    predictor = CountryAnalogyPredictor(vectors, ["country2"])
    normalized = np.asarray(vectors.get_normed_vectors(), dtype=np.float32)

    result = predictor.analogy_vector("capital1", "country1", "capital2")
    expected = (
        normalized[vectors.key_to_index["country1"]]
        - normalized[vectors.key_to_index["capital1"]]
        + normalized[vectors.key_to_index["capital2"]]
    )
    assert np.allclose(result, expected)


def test_candidates_outside_vocabulary_are_dropped():
    vectors = make_tiny_vectors()
    predictor = CountryAnalogyPredictor(vectors, ["country2", "atlantis"])
    assert predictor.countries == ["country2"]


def test_country_ranking_excludes_input_words():
    predictor = make_stub_predictor()
    ranked = predictor.rank("athens", "greece", "baghdad", topn=5)
    words = [word for word, _ in ranked]
    assert words == ["iraq", "afghanistan"]
    assert "greece" not in words


def test_full_vocabulary_ranking_filters_symbols_and_inputs():
    predictor = make_stub_predictor()
    ranked = predictor.rank("athens", "greece", "baghdad", topn=5, country_filter=False)
    words = [word for word, _ in ranked]
    assert words[0] == "iraq"
    assert "athens" not in words and "greece" not in words and "baghdad" not in words
    assert "123" not in words
    assert "france" in words


def test_batched_and_single_ranking_agree():
    predictor = make_stub_predictor()
    records = [record("iraq"), record("afghanistan"), record("france")]
    batched = predictor.run(records, topn=3, batch_size=3)
    singles = [
        predictor.run([row], topn=3, batch_size=1)[0] for row in records
    ]
    assert [item["ranked"] for item in batched] == [
        item["ranked"] for item in singles
    ]


def test_missing_word_raises_clear_error():
    predictor = make_stub_predictor()
    with pytest.raises(MissingWordError) as error:
        predictor.rank("athens", "greece", "atlantis")
    assert "atlantis" in str(error.value)


def test_evaluate_predictions_metrics():
    predictor = make_stub_predictor()
    records = [record("iraq"), record("afghanistan")]
    rows, metrics = evaluate_predictions(predictor, records, topk=5)

    assert metrics["total"] == 2
    assert metrics["correct_top1"] == 1
    assert metrics["incorrect_top1"] == 1
    assert metrics["top1_accuracy_pct"] == 50.0
    assert metrics["correct_top5"] == 2
    assert metrics["top5_accuracy_pct"] == 100.0
    assert rows[0]["actual_rank"] == 1
    assert rows[1]["actual_rank"] == 2
    assert rows[1]["predicted_country"] == "iraq"


def test_country_filtered_actual_rank_ignores_excluded_inputs():
    predictor = make_stub_predictor()
    _, metrics = evaluate_predictions(
        predictor, [record("afghanistan")], country_filter=True
    )
    assert metrics["correct_top5"] == 1
    assert metrics["correct_top1"] == 0
