#!/usr/bin/env python3
"""Build the datasets used by the country-prediction experiment.

The script validates a hand-curated list of capital-country pairs against the
vocabulary of a pre-trained GloVe model and writes four CSV files:

    datasets/capitals_countries.csv   -- validated seed pairs
    datasets/candidate_countries.csv  -- countries the predictor may choose from
    datasets/test_analogies.csv       -- 15 hand-held test analogies
    datasets/eval_analogies_large.csv -- 1,000 reproducible evaluation analogies

Run:
    python3 BuildDatasets.py
    python3 BuildDatasets.py --model-path /path/to/glove-wiki-gigaword-100.gz
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "datasets"
DEFAULT_OUT_DIR = DATASET_DIR

TEST_SEED = 42
LARGE_SEED = 2026
TEST_SIZE = 15
LARGE_SIZE = 1000

REQUIRED_EXAMPLE = ("athens", "greece", "baghdad", "iraq")


def _has_word2vec_header(path: Path) -> bool:
    """Return True if the first line looks like a word2vec 'count dim' header."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    parts = first_line.split()
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def load_embeddings(model_path: str | None = None):
    """Load GloVe vectors from the gensim cache or from a local file."""
    from gensim.models import KeyedVectors

    start = time.time()
    print("[1/4] Loading GloVe embeddings ...", flush=True)

    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        print(f"      source: local file {path}")
        if path.suffix == ".kv":
            vectors = KeyedVectors.load(str(path))
        else:
            binary = path.suffix == ".bin"
            no_header = not binary and not _has_word2vec_header(path)
            vectors = KeyedVectors.load_word2vec_format(
                str(path),
                binary=binary,
                no_header=no_header,
            )
    else:
        import gensim.downloader as api

        print("      source: gensim-data cache (downloads once, ~128 MB)")
        vectors = api.load("glove-wiki-gigaword-100")

    print(
        f"      loaded {len(vectors.key_to_index):,} words "
        f"of dimension {vectors.vector_size} "
        f"in {time.time() - start:.1f}s",
        flush=True,
    )
    return vectors


def read_capital_country_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            (row["capital"].strip().lower(), row["country"].strip().lower())
            for row in reader
        ]


def read_additional_countries(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["country"].strip().lower() for row in reader]


def validate_pairs(pairs, vectors):
    """Keep only pairs whose capital and country are both in the vocabulary."""
    valid: list[tuple[str, str]] = []
    skipped: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for capital, country in pairs:
        if (capital, country) in seen:
            continue
        seen.add((capital, country))

        missing = [
            word for word in (capital, country) if word not in vectors.key_to_index
        ]
        if missing:
            skipped.append(
                {"capital": capital, "country": country, "missing": missing}
            )
        else:
            valid.append((capital, country))

    return valid, skipped


def build_candidates(valid_pairs, additional, vectors):
    """Union of dataset countries and extra countries present in the vocabulary."""
    dataset_countries = {country for _, country in valid_pairs}
    extra_in_vocab = {c for c in additional if c in vectors.key_to_index}
    extra_out_of_vocab = sorted(set(additional) - extra_in_vocab)
    return sorted(dataset_countries | extra_in_vocab), extra_out_of_vocab


def generate_analogies(valid_pairs, count: int, seed: int):
    """Generate unique (capital1, country1, capital2, country2) analogies.

    The two source pairs are always distinct, so the target country can never be
    identical to Country 1. A fixed seed makes the output reproducible.
    """
    maximum = len(valid_pairs) * (len(valid_pairs) - 1)
    if count > maximum:
        raise ValueError(
            f"Cannot generate {count} unique analogies from {len(valid_pairs)} "
            f"pairs; the maximum is {maximum}."
        )

    rng = random.Random(seed)
    used: set[tuple[str, str, str, str]] = set()
    records: list[tuple[str, str, str, str]] = []

    while len(records) < count:
        (capital1, country1), (capital2, country2) = rng.sample(valid_pairs, 2)
        record = (capital1, country1, capital2, country2)
        if record in used:
            continue
        used.add(record)
        records.append(record)

    return records


def build_test_set(valid_pairs, count: int, seed: int):
    """Required example followed by random analogies that share no words with it."""
    rng = random.Random(seed)
    forbidden = set(REQUIRED_EXAMPLE)
    records = [REQUIRED_EXAMPLE]
    used: set[tuple[str, str, str, str]] = set()

    while len(records) < count:
        (capital1, country1), (capital2, country2) = rng.sample(valid_pairs, 2)
        record = (capital1, country1, capital2, country2)
        if record in used or record == REQUIRED_EXAMPLE:
            continue
        if forbidden & set(record):
            continue
        used.add(record)
        records.append(record)

    return records


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate capital-country datasets for Assignment 4."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local GloVe .gz/.txt/.kv file. Default: gensim-data cache.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors = load_embeddings(args.model_path)

    print("[2/4] Validating capital-country pairs ...", flush=True)
    pairs = read_capital_country_pairs(DATA_DIR / "capital_country_pairs.csv")
    additional = read_additional_countries(DATA_DIR / "additional_countries.csv")
    valid_pairs, skipped = validate_pairs(pairs, vectors)

    for item in skipped:
        print(
            "      skipped "
            f"{item['capital']}/{item['country']} "
            f"(missing: {', '.join(item['missing'])})"
        )
    print(f"      valid pairs: {len(valid_pairs)} of {len(pairs)}")

    candidates, extra_out = build_candidates(valid_pairs, additional, vectors)
    for country in extra_out:
        print(f"      additional country not in vocabulary: {country}")
    print(f"      candidate countries: {len(candidates)}")

    write_csv(
        out_dir / "capitals_countries.csv",
        ["capital", "country"],
        valid_pairs,
    )
    write_csv(
        out_dir / "candidate_countries.csv",
        ["country"],
        [[country] for country in candidates],
    )

    print("[3/4] Generating test and evaluation analogies ...", flush=True)
    test_records = build_test_set(valid_pairs, TEST_SIZE, TEST_SEED)
    large_records = generate_analogies(valid_pairs, LARGE_SIZE, LARGE_SEED)

    write_csv(
        out_dir / "test_analogies.csv",
        ["capital1", "country1", "capital2", "country2"],
        test_records,
    )
    write_csv(
        out_dir / "eval_analogies_large.csv",
        ["capital1", "country1", "capital2", "country2"],
        large_records,
    )

    metadata = {
        "source_pairs": len(pairs),
        "valid_pairs": len(valid_pairs),
        "candidate_countries": len(candidates),
        "skipped_pairs": skipped,
        "additional_countries_out_of_vocabulary": extra_out,
        "test_size": len(test_records),
        "large_size": len(large_records),
        "test_seed": TEST_SEED,
        "large_seed": LARGE_SEED,
        "embedding_vocabulary": len(vectors.key_to_index),
        "embedding_dimension": vectors.vector_size,
    }
    with (out_dir / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("[4/4] Dataset generation complete.")
    print(f"      test analogies:       {len(test_records)}")
    print(f"      evaluation analogies: {len(large_records)}")
    print(f"      output directory:     {out_dir}")


if __name__ == "__main__":
    main()
