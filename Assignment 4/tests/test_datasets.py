"""Structural tests for the generated CSV datasets (no model download needed)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from BuildDatasets import (  # noqa: E402
    DATA_DIR,
    DATASET_DIR,
    LARGE_SIZE,
    REQUIRED_EXAMPLE,
    TEST_SIZE,
    generate_analogies,
    read_capital_country_pairs,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def pairs():
    return read_capital_country_pairs(DATA_DIR / "capital_country_pairs.csv")


def test_seed_pairs_are_unique(pairs):
    assert len(pairs) == 78
    assert len(set(pairs)) == len(pairs)


def test_required_dataset_files_exist():
    for name in (
        "capitals_countries.csv",
        "candidate_countries.csv",
        "test_analogies.csv",
        "eval_analogies_large.csv",
    ):
        path = DATASET_DIR / name
        assert path.exists(), f"missing dataset: {path}"
        assert path.stat().st_size > 0, f"empty dataset: {path}"


def test_capitals_countries_match_seed_list(pairs):
    rows = read_csv(DATASET_DIR / "capitals_countries.csv")
    assert [(row["capital"], row["country"]) for row in rows] == pairs


def test_candidates_cover_dataset_countries(pairs):
    rows = read_csv(DATASET_DIR / "candidate_countries.csv")
    candidates = [row["country"] for row in rows]
    assert len(candidates) == len(set(candidates))
    dataset_countries = {country for _, country in pairs}
    assert dataset_countries.issubset(set(candidates))
    assert len(candidates) == 131


def test_test_set_size_and_required_example():
    rows = read_csv(DATASET_DIR / "test_analogies.csv")
    assert len(rows) == TEST_SIZE == 15
    first = rows[0]
    assert (
        first["capital1"],
        first["country1"],
        first["capital2"],
        first["country2"],
    ) == REQUIRED_EXAMPLE == ("athens", "greece", "baghdad", "iraq")

    records = [
        (r["capital1"], r["country1"], r["capital2"], r["country2"]) for r in rows
    ]
    assert len(set(records)) == len(records)


def test_large_set_is_unique_and_valid(pairs):
    rows = read_csv(DATASET_DIR / "eval_analogies_large.csv")
    assert len(rows) == LARGE_SIZE == 1000
    records = [
        (r["capital1"], r["country1"], r["capital2"], r["country2"]) for r in rows
    ]
    assert len(set(records)) == len(records)

    valid_pairs = set(pairs)
    for capital1, country1, capital2, country2 in records:
        assert (capital1, country1) in valid_pairs
        assert (capital2, country2) in valid_pairs
        assert country1 != country2
        assert capital1 != capital2


def test_generator_is_reproducible(pairs):
    first = generate_analogies(pairs, 50, seed=2026)
    second = generate_analogies(pairs, 50, seed=2026)
    assert first == second
    assert len(set(first)) == 50


def test_generator_rejects_impossible_request(pairs):
    with pytest.raises(ValueError):
        generate_analogies(pairs, len(pairs) * (len(pairs) - 1) + 1, seed=1)


def test_metadata_matches_files():
    metadata = json.loads(
        (DATASET_DIR / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["valid_pairs"] == 78
    assert metadata["candidate_countries"] == 131
    assert metadata["test_size"] == 15
    assert metadata["large_size"] == 1000
    assert metadata["test_seed"] == 42
    assert metadata["large_seed"] == 2026
    assert metadata["embedding_dimension"] == 100
