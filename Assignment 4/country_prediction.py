#!/usr/bin/env python3
"""Predict a country from a capital-country analogy using GloVe vectors.

The core relation is

    vector(Country 2) ~= vector(Country 1) - vector(Capital 1) + vector(Capital 2)

Candidates are ranked by cosine similarity. The search can be restricted to a
curated list of country names (default) or performed over the full 400k-word
vocabulary for comparison.

Examples:
    python3 country_prediction.py --words athens greece baghdad
    python3 country_prediction.py --demo
    python3 country_prediction.py --evaluate
    python3 country_prediction.py --model-path ~/glove-wiki-gigaword-100.gz
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = RESULTS_DIR / "logs"

MODEL_NAME = "glove-wiki-gigaword-100"
DEFAULT_TOP_N = 5
DEFAULT_BATCH_SIZE = 64
VOCAB_POOL_SIZE = 500

AMBIGUOUS_WORDS = {
    "georgia": "Georgia is also a US state and a common given name, so one "
    "static vector mixes several meanings.",
    "jordan": "Jordan is also a common personal name and a river, which "
    "dilutes the country sense of the vector.",
    "turkey": "Turkey also denotes the bird and a food term, so the embedding "
    "mixes unrelated senses.",
    "chile": "Chile is also a common word/name in English and Spanish, which "
    "makes its vector less country-specific.",
    "china": "China also means porcelain, adding a non-country sense to the "
    "same vector.",
    "england": "England appears mostly in sports, history and UK-related "
    "contexts, which overlap with Britain, the UK and Australia.",
    "ireland": "Ireland also names the island and appears in rugby/UK "
    "contexts, overlapping with Britain and England.",
    "cyprus": "Cyprus also names a tree and appears in regional political "
    "contexts, weakening the capital-country signal.",
    "niger": "Niger is also the name of a river and a language, so the vector "
    "mixes several senses.",
    "mali": "Mali is also a personal name and appears in non-country contexts.",
    "chad": "Chad is also a common personal name, which adds non-country "
    "context to the vector.",
}

NEIGHBOURING_COUNTRIES = {
    "austria": {"germany", "switzerland", "hungary", "slovakia", "slovenia", "italy"},
    "germany": {"austria", "switzerland", "poland", "belgium", "denmark", "france"},
    "england": {"britain", "uk", "scotland", "wales", "ireland", "australia"},
    "chile": {"argentina", "peru", "bolivia", "brazil", "uruguay"},
    "argentina": {"chile", "uruguay", "paraguay", "brazil", "bolivia"},
    "georgia": {"armenia", "azerbaijan", "russia", "turkey"},
    "armenia": {"georgia", "azerbaijan", "turkey", "iran"},
    "azerbaijan": {"armenia", "georgia", "turkey", "iran"},
    "belgium": {"france", "netherlands", "germany", "luxembourg"},
    "france": {"belgium", "germany", "italy", "spain", "switzerland", "luxembourg"},
    "portugal": {"spain", "brazil"},
    "spain": {"portugal", "france", "italy", "morocco"},
    "italy": {"france", "spain", "austria", "slovenia", "switzerland"},
    "switzerland": {"austria", "germany", "france", "italy"},
    "slovakia": {"czech", "austria", "hungary", "poland", "ukraine"},
    "hungary": {"austria", "slovakia", "romania", "serbia", "croatia", "slovenia"},
    "romania": {"hungary", "bulgaria", "serbia", "moldova", "ukraine"},
    "bulgaria": {"romania", "serbia", "macedonia", "greece", "turkey"},
    "serbia": {"croatia", "bosnia", "montenegro", "hungary", "romania", "bulgaria"},
    "croatia": {"slovenia", "serbia", "bosnia", "hungary"},
    "slovenia": {"croatia", "austria", "italy", "hungary"},
    "belarus": {"russia", "ukraine", "poland", "lithuania", "latvia"},
    "ukraine": {"russia", "belarus", "poland", "slovakia", "romania", "moldova"},
    "russia": {"ukraine", "belarus", "georgia", "kazakhstan", "latvia", "estonia"},
    "estonia": {"latvia", "lithuania", "russia", "finland"},
    "latvia": {"estonia", "lithuania", "russia", "belarus"},
    "lithuania": {"latvia", "estonia", "poland", "belarus", "russia"},
    "ireland": {"england", "britain", "uk", "scotland"},
    "egypt": {"libya", "sudan", "israel", "jordan"},
    "israel": {"lebanon", "syria", "jordan", "egypt"},
    "lebanon": {"syria", "israel", "jordan"},
    "syria": {"lebanon", "jordan", "iraq", "turkey", "israel"},
    "jordan": {"syria", "iraq", "israel", "lebanon", "egypt"},
    "iraq": {"iran", "syria", "jordan", "kuwait", "turkey", "saudi"},
    "iran": {"iraq", "afghanistan", "pakistan", "turkey", "armenia", "azerbaijan"},
    "afghanistan": {"pakistan", "iran", "tajikistan", "turkmenistan", "uzbekistan"},
    "pakistan": {"india", "afghanistan", "iran", "china"},
    "india": {"pakistan", "bangladesh", "nepal", "bhutan", "china", "myanmar"},
    "bangladesh": {"india", "myanmar", "nepal", "bhutan"},
    "nepal": {"india", "china", "bhutan"},
    "china": {"india", "russia", "mongolia", "vietnam", "laos", "myanmar", "kazakhstan"},
    "japan": {"china", "south", "korea", "russia", "taiwan"},
    "vietnam": {"china", "laos", "cambodia", "thailand"},
    "laos": {"vietnam", "thailand", "cambodia", "myanmar", "china"},
    "thailand": {"laos", "cambodia", "myanmar", "malaysia", "vietnam"},
    "cambodia": {"vietnam", "laos", "thailand"},
    "myanmar": {"thailand", "laos", "china", "india", "bangladesh"},
    "indonesia": {"malaysia", "singapore", "philippines", "australia"},
    "malaysia": {"indonesia", "singapore", "thailand", "brunei"},
    "philippines": {"indonesia", "malaysia", "china", "vietnam"},
    "australia": {"new", "zealand", "indonesia", "england", "britain"},
    "canada": {"united", "states", "america", "britain", "france"},
    "mexico": {"united", "states", "guatemala", "belize", "america"},
    "cuba": {"haiti", "jamaica", "mexico", "united", "states"},
    "colombia": {"venezuela", "ecuador", "peru", "panama", "brazil"},
    "venezuela": {"colombia", "brazil", "guyana", "ecuador"},
    "ecuador": {"colombia", "peru"},
    "peru": {"ecuador", "colombia", "brazil", "bolivia", "chile"},
    "bolivia": {"peru", "brazil", "paraguay", "argentina", "chile"},
    "paraguay": {"brazil", "argentina", "bolivia", "uruguay"},
    "uruguay": {"argentina", "brazil", "paraguay"},
    "brazil": {"argentina", "uruguay", "paraguay", "bolivia", "peru", "colombia", "venezuela"},
    "kenya": {"tanzania", "uganda", "somalia", "ethiopia", "sudan"},
    "uganda": {"kenya", "tanzania", "rwanda", "sudan", "congo"},
    "tanzania": {"kenya", "uganda", "mozambique", "zambia", "malawi", "rwanda", "burundi"},
    "somalia": {"ethiopia", "kenya", "djibouti"},
    "ethiopia": {"somalia", "kenya", "sudan", "eritrea", "djibouti"},
    "eritrea": {"ethiopia", "sudan", "djibouti"},
    "djibouti": {"somalia", "ethiopia", "eritrea"},
    "sudan": {"egypt", "ethiopia", "kenya", "uganda", "chad", "libya"},
    "libya": {"egypt", "tunisia", "algeria", "chad", "niger", "sudan"},
    "tunisia": {"libya", "algeria"},
    "algeria": {"tunisia", "libya", "morocco", "niger", "mali", "mauritania"},
    "morocco": {"algeria", "spain", "mauritania"},
    "ghana": {"nigeria", "ivory", "togo", "burkina", "faso"},
    "nigeria": {"ghana", "niger", "chad", "cameroon", "benin"},
    "cameroon": {"nigeria", "chad", "central", "african", "republic", "congo"},
    "zimbabwe": {"zambia", "mozambique", "botswana", "south", "africa"},
    "zambia": {"zimbabwe", "mozambique", "malawi", "tanzania", "congo"},
    "mozambique": {"tanzania", "malawi", "zambia", "zimbabwe", "south", "africa"},
    "malawi": {"tanzania", "mozambique", "zambia"},
    "rwanda": {"uganda", "tanzania", "burundi", "congo"},
    "burundi": {"rwanda", "tanzania", "congo"},
    "chad": {"niger", "libya", "sudan", "cameroon", "nigeria"},
    "niger": {"chad", "libya", "algeria", "mali", "nigeria", "benin"},
    "mali": {"niger", "algeria", "senegal", "guinea", "burkina", "faso"},
    "senegal": {"mali", "mauritania", "guinea", "gambia"},
    "qatar": {"saudi", "arabia", "united", "emirates", "bahrain", "kuwait", "oman"},
    "oman": {"qatar", "united", "emirates", "saudi", "arabia", "yemen"},
    "kuwait": {"iraq", "saudi", "arabia", "qatar"},
    "bahrain": {"qatar", "saudi", "arabia", "kuwait"},
    "yemen": {"saudi", "arabia", "oman"},
    "kazakhstan": {"russia", "china", "uzbekistan", "turkmenistan", "kyrgyzstan"},
    "uzbekistan": {"kazakhstan", "turkmenistan", "tajikistan", "kyrgyzstan", "afghanistan"},
    "turkmenistan": {"uzbekistan", "kazakhstan", "iran", "afghanistan"},
    "kyrgyzstan": {"kazakhstan", "uzbekistan", "tajikistan", "china"},
    "tajikistan": {"uzbekistan", "kyrgyzstan", "afghanistan", "china"},
    "mongolia": {"china", "russia"},
    "albania": {"greece", "macedonia", "montenegro", "serbia"},
    "macedonia": {"albania", "greece", "bulgaria", "serbia"},
    "bosnia": {"croatia", "serbia", "montenegro"},
    "montenegro": {"croatia", "bosnia", "serbia", "albania"},
    "moldova": {"romania", "ukraine"},
    "luxembourg": {"belgium", "france", "germany"},
    "liechtenstein": {"austria", "switzerland"},
    "monaco": {"france", "italy"},
    "singapore": {"malaysia", "indonesia"},
    "fiji": {"australia", "new", "zealand", "tonga", "samoa"},
    "tonga": {"fiji", "samoa", "new", "zealand"},
    "samoa": {"tonga", "fiji", "new", "zealand"},
    "madagascar": {"mozambique", "tanzania", "mauritius"},
    "botswana": {"south", "africa", "zimbabwe", "namibia"},
    "namibia": {"south", "africa", "botswana", "angola", "zambia"},
    "angola": {"namibia", "zambia", "congo"},
    "liberia": {"sierra", "leone", "guinea", "ivory", "coast"},
    "panama": {"colombia", "costa", "rica"},
    "honduras": {"guatemala", "nicaragua", "el", "salvador"},
    "guatemala": {"honduras", "mexico", "belize", "el", "salvador"},
    "nicaragua": {"honduras", "costa", "rica"},
    "jamaica": {"cuba", "haiti", "dominican", "republic"},
    "haiti": {"cuba", "jamaica", "dominican", "republic"},
    "bhutan": {"india", "china", "nepal", "bangladesh"},
}


class MissingWordError(KeyError):
    """Raised when an input word is absent from the embedding vocabulary."""

    def __init__(self, words):
        self.words = list(words)
        super().__init__(", ".join(self.words))

    def __str__(self) -> str:
        return (
            "not found in the embedding vocabulary: " + ", ".join(self.words)
            + " (GloVe is case-insensitive; check spelling or use a word "
            "that appears in the 400k-word vocabulary)"
        )


def _has_word2vec_header(path: Path) -> bool:
    """Return True if the first line looks like a word2vec 'count dim' header."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    parts = first_line.split()
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def load_embeddings(model_path: str | None = None):
    """Load word vectors and return them with a small metadata dictionary."""
    from gensim.models import KeyedVectors

    start = time.time()
    print("[model] Loading word embeddings ...", flush=True)

    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        source = f"local file: {path}"
        print(f"[model] Source: {source}", flush=True)
        if path.suffix == ".kv":
            vectors = KeyedVectors.load(str(path))
        else:
            binary = path.suffix == ".bin"
            no_header = not binary and not _has_word2vec_header(path)
            vectors = KeyedVectors.load_word2vec_format(
                str(path), binary=binary, no_header=no_header
            )
    else:
        import gensim.downloader as api

        source = "gensim-data cache (~/gensim-data, 128 MB download on first use)"
        print(f"[model] Source: {source}", flush=True)
        vectors = api.load(MODEL_NAME)

    elapsed = time.time() - start
    info = {
        "name": MODEL_NAME if not model_path else Path(model_path).name,
        "source": source,
        "vocabulary_size": len(vectors.key_to_index),
        "dimension": vectors.vector_size,
        "load_seconds": round(elapsed, 2),
    }
    print(
        f"[model] {info['name']}: {info['vocabulary_size']:,} words, "
        f"{info['dimension']} dimensions, loaded in {elapsed:.1f}s",
        flush=True,
    )
    return vectors, info


def load_records(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_candidate_countries(path: Path | None = None) -> list[str]:
    path = Path(path) if path else DATASET_DIR / "candidate_countries.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate list not found: {path}. Run BuildDatasets.py first."
        )
    return [row["country"].strip().lower() for row in load_records(path)]


class CountryAnalogyPredictor:
    """Rank country candidates for capital-country analogies."""

    def __init__(self, vectors, candidate_countries):
        self.vectors = vectors
        self.normalized = np.asarray(
            vectors.get_normed_vectors(), dtype=np.float32
        )
        self.vocabulary = list(vectors.index_to_key)
        self.word_index = vectors.key_to_index

        self.countries = [
            country
            for country in candidate_countries
            if country in self.word_index
        ]
        self.country_index = {country: i for i, country in enumerate(self.countries)}
        country_ids = [self.word_index[country] for country in self.countries]
        self.country_matrix = self.normalized[country_ids]
        self.alpha_mask = np.fromiter(
            (word.isalpha() for word in self.vocabulary),
            dtype=bool,
            count=len(self.vocabulary),
        )

    # ------------------------------------------------------------------ maths
    def vector(self, word: str) -> np.ndarray:
        return self.normalized[self.word_index[word]]

    def analogy_vector(self, capital1: str, country1: str, capital2: str) -> np.ndarray:
        return (
            self.vector(country1)
            - self.vector(capital1)
            + self.vector(capital2)
        )

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero-length vector.")
        return vector / norm

    # ------------------------------------------------------------- validation
    @staticmethod
    def _clean_record(record: dict) -> tuple[str, str, str, str | None]:
        capital1 = str(record["capital1"]).strip().lower()
        country1 = str(record["country1"]).strip().lower()
        capital2 = str(record["capital2"]).strip().lower()
        actual = record.get("country2")
        actual = str(actual).strip().lower() if actual else None
        return capital1, country1, capital2, actual

    def _check_words(self, words) -> None:
        missing = [word for word in words if word not in self.word_index]
        if missing:
            raise MissingWordError(sorted(set(missing)))

    # --------------------------------------------------------------- ranking
    def _rank_countries(self, scores, excluded, topn):
        order = np.argsort(-scores, kind="stable")
        ranked = []
        for index in order:
            word = self.countries[index]
            if word in excluded:
                continue
            ranked.append((word, float(scores[index])))
            if len(ranked) >= topn:
                break
        return ranked

    def _rank_vocabulary(self, scores, excluded, topn, pool_size):
        pool = min(max(pool_size, topn + 10), scores.size)
        candidates = np.argpartition(-scores, pool - 1)[:pool]
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        ranked = []
        for index in candidates:
            word = self.vocabulary[index]
            if not word.isalpha() or word in excluded:
                continue
            ranked.append((word, float(scores[index])))
            if len(ranked) >= topn:
                break
        return ranked

    def run(
        self,
        records: list[dict],
        topn: int = DEFAULT_TOP_N,
        country_filter: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        with_actual: bool = False,
    ) -> list[dict]:
        """Rank candidates for every record, processing records in batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        outputs: list[dict] = []
        total = len(records)
        show_progress = total >= 500

        for start in range(0, total, batch_size):
            batch = records[start : start + batch_size]
            queries = []
            exclusions = []

            for record in batch:
                capital1, country1, capital2, _ = self._clean_record(record)
                self._check_words((capital1, country1, capital2))
                queries.append(
                    self.normalize(
                        self.analogy_vector(capital1, country1, capital2)
                    )
                )
                exclusions.append({capital1, country1, capital2})

            query_matrix = np.asarray(queries, dtype=np.float32)
            if country_filter:
                scores_matrix = query_matrix @ self.country_matrix.T
            else:
                scores_matrix = query_matrix @ self.normalized.T

            for row, record in enumerate(batch):
                scores = scores_matrix[row]
                excluded = exclusions[row]
                _, _, _, actual = self._clean_record(record)

                if country_filter:
                    ranked = self._rank_countries(scores, excluded, topn)
                else:
                    ranked = self._rank_vocabulary(
                        scores, excluded, topn, VOCAB_POOL_SIZE
                    )

                actual_score = None
                actual_rank = None
                if with_actual and actual:
                    if country_filter:
                        index_map = self.country_index
                        pool_mask = np.ones(len(scores), dtype=bool)
                    else:
                        index_map = self.word_index
                        pool_mask = self.alpha_mask.copy()

                    index = index_map.get(actual)
                    if index is not None:
                        actual_score = float(scores[index])
                        excluded_ids = [
                            index_map[word]
                            for word in excluded
                            if word in index_map
                        ]
                        pool_mask[excluded_ids] = False
                        actual_rank = int(
                            np.count_nonzero(scores[pool_mask] > actual_score)
                        ) + 1

                outputs.append(
                    {
                        "record": record,
                        "ranked": ranked,
                        "actual_score": actual_score,
                        "actual_rank": actual_rank,
                    }
                )

            completed = start + len(batch)
            if show_progress and (completed == total or completed % 500 == 0):
                print(
                    f"\r[ranking] {completed}/{total} analogies",
                    end="",
                    flush=True,
                )

        if show_progress:
            print(flush=True)
        return outputs

    def rank(
        self,
        capital1: str,
        country1: str,
        capital2: str,
        topn: int = DEFAULT_TOP_N,
        country_filter: bool = True,
    ):
        record = {
            "capital1": capital1,
            "country1": country1,
            "capital2": capital2,
        }
        return self.run(
            [record],
            topn=topn,
            country_filter=country_filter,
            batch_size=1,
        )[0]["ranked"]

    def predict(
        self,
        capital1: str,
        country1: str,
        capital2: str,
        country_filter: bool = True,
    ) -> str:
        return self.rank(
            capital1, country1, capital2, topn=1, country_filter=country_filter
        )[0][0]


# ------------------------------------------------------------------ reporting
def format_ranked(ranked, topn: int = DEFAULT_TOP_N) -> str:
    return ", ".join(f"{word} ({score:.3f})" for word, score in ranked[:topn])


def show_prediction(
    predictor: CountryAnalogyPredictor,
    capital1: str,
    country1: str,
    capital2: str,
    actual: str | None = None,
) -> None:
    filtered = predictor.rank(capital1, country1, capital2, country_filter=True)
    full = predictor.rank(capital1, country1, capital2, country_filter=False)

    print(f"  Input: {capital1.title()} : {country1.title()} :: {capital2.title()} : ?")
    print(f"  country-filtered: {format_ranked(filtered)}")
    print(f"  full vocabulary : {format_ranked(full)}")

    prediction = filtered[0][0]
    if actual is None:
        status = ""
    elif prediction == actual:
        status = "  -> CORRECT"
    else:
        status = f"  -> WRONG (expected {actual})"
    print(f"  Predicted Country 2 = {prediction.upper()}{status}")


def evaluate_predictions(
    predictor: CountryAnalogyPredictor,
    records: list[dict],
    country_filter: bool = True,
    topk: int = DEFAULT_TOP_N,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict], dict]:
    start = time.time()
    raw = predictor.run(
        records,
        topn=topk,
        country_filter=country_filter,
        batch_size=batch_size,
        with_actual=True,
    )

    rows = []
    for item in raw:
        record = item["record"]
        ranked = item["ranked"]
        predicted, predicted_score = ranked[0]
        actual = item["record"].get("country2")
        actual = actual.strip().lower() if actual else None
        correct = predicted == actual
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
        rows.append(
            {
                "capital1": record["capital1"],
                "country1": record["country1"],
                "capital2": record["capital2"],
                "actual_country2": actual,
                "predicted_country": predicted,
                "correct": correct,
                "predicted_score": predicted_score,
                "actual_score": item["actual_score"],
                "actual_rank": item["actual_rank"],
                "margin": margin,
                "top_results": ranked,
            }
        )

    total = len(rows)
    top1_correct = sum(row["correct"] for row in rows)
    top5_correct = sum(
        row["actual_rank"] is not None and row["actual_rank"] <= topk
        for row in rows
    )
    margins = [row["margin"] for row in rows if row["margin"] is not None]
    predicted_scores = [row["predicted_score"] for row in rows]
    actual_scores = [
        row["actual_score"] for row in rows if row["actual_score"] is not None
    ]

    metrics = {
        "total": total,
        "correct_top1": int(top1_correct),
        "incorrect_top1": int(total - top1_correct),
        "top1_accuracy_pct": round(100.0 * top1_correct / total, 2),
        f"correct_top{topk}": int(top5_correct),
        f"top{topk}_accuracy_pct": round(100.0 * top5_correct / total, 2),
        "mean_predicted_score": round(float(np.mean(predicted_scores)), 4),
        "mean_actual_score": (
            round(float(np.mean(actual_scores)), 4) if actual_scores else None
        ),
        "mean_top1_margin": (
            round(float(np.mean(margins)), 4) if margins else None
        ),
        "median_top1_margin": (
            round(float(np.median(margins)), 4) if margins else None
        ),
        "runtime_seconds": round(time.time() - start, 2),
    }
    return rows, metrics


def write_predictions_csv(
    path: Path, rows: list[dict], topk: int = DEFAULT_TOP_N
) -> None:
    header = [
        "capital1",
        "country1",
        "capital2",
        "actual_country2",
        "predicted_country",
        "correct",
        "predicted_score",
        "actual_score",
        "actual_rank",
        "margin",
    ]
    for rank in range(1, topk + 1):
        header += [f"rank{rank}_word", f"rank{rank}_score"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            values = [
                row["capital1"],
                row["country1"],
                row["capital2"],
                row["actual_country2"],
                row["predicted_country"],
                "Correct" if row["correct"] else "Incorrect",
                f"{row['predicted_score']:.4f}",
                f"{row['actual_score']:.4f}" if row["actual_score"] is not None else "",
                row["actual_rank"] if row["actual_rank"] is not None else "",
                f"{row['margin']:.4f}" if row["margin"] is not None else "",
            ]
            ranked = row["top_results"]
            for rank in range(topk):
                if rank < len(ranked):
                    word, score = ranked[rank]
                    values += [word, f"{score:.4f}"]
                else:
                    values += ["", ""]
            writer.writerow(values)


def explain_error(row: dict) -> tuple[str, str]:
    """Return a short category and a human-readable explanation for an error."""
    actual = row["actual_country2"]
    predicted = row["predicted_country"]
    rank = row["actual_rank"]
    margin = row["margin"]

    if actual in AMBIGUOUS_WORDS:
        return "ambiguous word", AMBIGUOUS_WORDS[actual]

    if predicted in NEIGHBOURING_COUNTRIES.get(actual, set()):
        return (
            "neighbouring / culturally close country",
            f"{actual.title()} and {predicted.title()} share regional and "
            "cultural contexts in the training corpus, so their vectors are "
            "close.",
        )

    if rank is not None and rank <= 5:
        return (
            "close competition",
            f"The correct country {actual.title()} was ranked #{rank} with a "
            f"score of {row['actual_score']:.3f}, while {predicted.title()} "
            f"scored {row['predicted_score']:.3f}. The margin was only "
            f"{margin:.3f}, so small vector differences decided the result."
            if margin is not None
            else f"The correct country {actual.title()} was ranked #{rank}, "
            "so the analogy was almost resolved correctly.",
        )

    return (
        "weak analogy offset",
        f"The vector Country1 - Capital1 + Capital2 was closer to "
        f"{predicted.title()} than to {actual.title()}. The capital-country "
        "offset is not consistent for this pair in a static embedding space.",
    )


def write_error_csv(path: Path, rows: list[dict]) -> list[dict]:
    errors = []
    for row in rows:
        if row["correct"]:
            continue
        category, explanation = explain_error(row)
        errors.append(
            {
                "capital1": row["capital1"],
                "country1": row["country1"],
                "capital2": row["capital2"],
                "actual_country2": row["actual_country2"],
                "predicted_country": row["predicted_country"],
                "predicted_score": round(row["predicted_score"], 4),
                "actual_score": (
                    round(row["actual_score"], 4)
                    if row["actual_score"] is not None
                    else None
                ),
                "actual_rank": row["actual_rank"],
                "error_category": category,
                "explanation": explanation,
            }
        )

    header = [
        "capital1",
        "country1",
        "capital2",
        "actual_country2",
        "predicted_country",
        "predicted_score",
        "actual_score",
        "actual_rank",
        "error_category",
        "explanation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(errors)
    return errors


def error_breakdown(rows: list[dict], limit: int = 10) -> list[dict]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["correct"]:
            continue
        actual = row["actual_country2"]
        counts[actual] = counts.get(actual, 0) + 1
    return [
        {"country": country, "errors": count}
        for country, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def run_evaluation(predictor, results_dir: Path = RESULTS_DIR) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)

    datasets = [
        ("test15", DATASET_DIR / "test_analogies.csv"),
        ("large1000", DATASET_DIR / "eval_analogies_large.csv"),
    ]

    summary = {
        "experiment": "Country Prediction using Word Embeddings",
        "model": {
            "name": MODEL_NAME,
            "source": predictor.model_info["source"],
            "vocabulary_size": predictor.model_info["vocabulary_size"],
            "dimension": predictor.model_info["dimension"],
        },
        "candidate_countries": len(predictor.countries),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": {},
        "results": {},
        "comparison": {},
    }

    for dataset_name, path in datasets:
        records = load_records(path)
        summary["datasets"][dataset_name] = {
            "file": str(path.relative_to(BASE_DIR)),
            "size": len(records),
        }
        summary["results"][dataset_name] = {}

        for country_filter, mode in (
            (True, "country_filtered"),
            (False, "full_vocabulary"),
        ):
            label = f"{dataset_name}/{mode}"
            print(f"\n[evaluate] {label} ({len(records)} analogies)", flush=True)
            rows, metrics = evaluate_predictions(
                predictor, records, country_filter=country_filter
            )
            summary["results"][dataset_name][mode] = metrics

            predictions_path = (
                results_dir / f"predictions_{dataset_name}_{mode}.csv"
            )
            write_predictions_csv(predictions_path, rows)

            errors = write_error_csv(
                results_dir / f"error_analysis_{dataset_name}_{mode}.csv", rows
            )
            print(
                f"  top-1 {metrics['correct_top1']}/{metrics['total']} "
                f"({metrics['top1_accuracy_pct']:.2f}%), "
                f"top-5 {metrics['top5_accuracy_pct']:.2f}%, "
                f"errors {len(errors)}, "
                f"{metrics['runtime_seconds']:.1f}s",
                flush=True,
            )

            if dataset_name == "large1000" and country_filter:
                summary["error_breakdown"] = error_breakdown(rows)

    for dataset_name in summary["results"]:
        filtered = summary["results"][dataset_name]["country_filtered"]
        full = summary["results"][dataset_name]["full_vocabulary"]
        summary["comparison"][dataset_name] = {
            "top1_gain_country_filtered_pct_points": round(
                filtered["top1_accuracy_pct"] - full["top1_accuracy_pct"], 2
            ),
            "top5_gain_country_filtered_pct_points": round(
                filtered["top5_accuracy_pct"] - full["top5_accuracy_pct"], 2
            ),
        }

    summary_path = results_dir / "accuracy_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 78)
    print("ACCURACY SUMMARY")
    print("=" * 78)
    print(f"{'dataset/mode':<32}{'n':>6}{'top-1':>10}{'top-1 %':>10}{'top-5 %':>10}")
    print("-" * 78)
    for dataset_name, modes in summary["results"].items():
        for mode, metrics in modes.items():
            print(
                f"{dataset_name + '/' + mode:<32}"
                f"{metrics['total']:>6}"
                f"{metrics['correct_top1']:>10}"
                f"{metrics['top1_accuracy_pct']:>10.2f}"
                f"{metrics['top5_accuracy_pct']:>10.2f}"
            )
    print("-" * 78)
    for dataset_name, comparison in summary["comparison"].items():
        print(
            f"{dataset_name}: country filtering changes top-1 accuracy by "
            f"{comparison['top1_gain_country_filtered_pct_points']:+.2f} "
            "percentage points."
        )
    print(f"\nResults written to: {results_dir.resolve()}")
    return summary


def run_demo(predictor: CountryAnalogyPredictor) -> None:
    test_records = load_records(DATASET_DIR / "test_analogies.csv")

    print("=" * 78)
    print("EXAMPLE FROM THE ASSIGNMENT SHEET")
    print("=" * 78)
    show_prediction(predictor, "athens", "greece", "baghdad", "iraq")

    print("\n" + "=" * 78)
    print(f"TEST SET ({len(test_records)} analogies) - country-filtered ranking")
    print("=" * 78)
    for number, record in enumerate(test_records, 1):
        capital1 = record["capital1"]
        country1 = record["country1"]
        capital2 = record["capital2"]
        actual = record["country2"]
        ranked = predictor.rank(capital1, country1, capital2)
        status = "CORRECT" if ranked[0][0] == actual else f"WRONG (expected {actual})"
        print(
            f"{number:>2}. {capital1.title()} : {country1.title()} :: "
            f"{capital2.title()} : ?"
        )
        print(f"    -> {ranked[0][0].upper()} ({ranked[0][1]:.3f})  {status}")
        print(f"    top-5: {format_ranked(ranked)}")

    correct = sum(
        predictor.predict(
            record["capital1"], record["country1"], record["capital2"]
        )
        == record["country2"]
        for record in test_records
    )
    print(
        f"\nDemo accuracy: {correct}/{len(test_records)} "
        f"({100.0 * correct / len(test_records):.2f}%)"
    )


def interactive_loop(predictor: CountryAnalogyPredictor) -> None:
    print(
        "Interactive mode - enter three words "
        "(Capital1 Country1 Capital2), or 'q' to quit."
    )
    while True:
        try:
            raw = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw in {"q", "quit", "exit"}:
            return
        words = raw.split()
        if len(words) != 3:
            print("Please enter exactly three words, e.g.: athens greece baghdad")
            continue
        try:
            show_prediction(predictor, *words)
        except (MissingWordError, ValueError) as error:
            print(f"Error: {error}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Country 2 from Capital 1, Country 1 and Capital 2 "
        "using GloVe word embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python3 country_prediction.py --words athens greece baghdad\n"
            "  python3 country_prediction.py --demo\n"
            "  python3 country_prediction.py --evaluate\n"
        ),
    )
    parser.add_argument(
        "--words",
        nargs=3,
        metavar=("CAPITAL1", "COUNTRY1", "CAPITAL2"),
        help="Predict Country 2 for a single analogy.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the required example and the 15-example test set.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on the 15-example and 1,000-example datasets.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local GloVe file (.gz/.txt/.kv). Default: gensim-data cache.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of ranked candidates to show (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        print("Error: --batch-size must be at least 1.", file=sys.stderr)
        return 2

    vectors, model_info = load_embeddings(args.model_path)
    candidates = load_candidate_countries()
    predictor = CountryAnalogyPredictor(vectors, candidates)
    predictor.model_info = model_info
    print(
        f"[data] {len(predictor.countries)} candidate countries loaded from "
        f"{DATASET_DIR.name}/{ 'candidate_countries.csv' }\n",
        flush=True,
    )

    try:
        if args.words:
            show_prediction(predictor, *args.words)
            return 0

        if args.demo or args.evaluate:
            run_demo(predictor)

        if args.evaluate:
            run_evaluation(predictor)
            return 0

        if args.demo:
            return 0

        interactive_loop(predictor)
        return 0
    except MissingWordError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
