#!/usr/bin/env python3
"""
NLP Laboratory Assignment 3
Survey and Implementation of Part-of-Speech (POS) Tagging
=========================================================

A reproducible implementation of a statistical Hidden Markov Model (HMM)
POS tagger with Viterbi decoding for English, trained on the NLTK Penn
Treebank corpus and evaluated with the NLTK Universal tagset.

Main components
---------------
1. NLTK resource checks and corpus loading.
2. Removal of Penn Treebank trace tokens (-NONE-).
3. Penn Treebank -> Universal tagset mapping.
4. Reproducible 80:20 sentence-level train/test split (seed = 42).
5. Supervised HMM training: transition counts, emission counts,
   add-k smoothed transitions (k = 0.001), log probabilities.
6. Viterbi decoding with backpointers.
7. OOV strategy: digit-shape priors and suffix statistics from rare words.
8. Most-frequent-tag baseline.
9. Evaluation: accuracy, macro/weighted P/R/F1, per-tag metrics, OOV accuracy,
   confusion matrix, error analysis with context.
10. Custom sentence tagging, CSV/JSON results, five charts, and optional
    Excel/Word report generation through companion modules.

Run:
    python3 pos_tagging_hmm.py

The script writes every generated artefact inside this project directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Use a non-interactive backend so the script also works over SSH/CI.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Project paths and configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
VIS_DIR = PROJECT_DIR / "visualizations"
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"
TESTS_DIR = PROJECT_DIR / "tests"

SEED = 42
TEST_SIZE = 0.20
TRANSITION_K = 0.001
RARE_WORD_THRESHOLD = 2
MAX_SUFFIX_LEN = 4
OOV_ALPHA = 0.1
N_CUSTOM_SENTENCES = 25

UNIVERSAL_TAG_DESCRIPTIONS = {
    "ADJ": "Adjective (e.g., quick, old, national)",
    "ADP": "Adposition / preposition (e.g., in, of, over)",
    "ADV": "Adverb (e.g., quickly, very, yesterday)",
    "CONJ": "Conjunction (e.g., and, or, but)",
    "DET": "Determiner / article (e.g., the, a, this)",
    "NOUN": "Noun, including proper nouns (e.g., dog, Shravani, Pune)",
    "NUM": "Numeral (e.g., 61, 2025, three)",
    "PRON": "Pronoun (e.g., he, she, it, their)",
    "PRT": "Particle (e.g., to, up, off)",
    "VERB": "Verb (e.g., run, is, reported, studying)",
    ".": "Punctuation / sentence delimiter (., ,, ?, !, quotes, brackets)",
    "X": "Other: foreign word, symbol, or unclassifiable item",
}

# Manually written, previously unseen sentences used for qualitative testing.
DEFAULT_CUSTOM_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Shravani is studying Natural Language Processing at her university.",
    "The committee approved the new budget after a long discussion.",
    "She bought three apples and two oranges at the market.",
    "The weather in Pune is pleasant during October.",
    "Can you explain how the Viterbi algorithm works?",
    "The company reported a 15% increase in revenue for 2025.",
    "He quickly finished his assignment and submitted it on time.",
    "The children were playing football in the park.",
    "Although it was raining, we decided to continue the journey.",
    "The professor asked the students to submit their reports by Friday.",
    "This smartphone has a battery life of more than two days.",
    "The train to Mumbai was delayed by 30 minutes.",
    "My brother wants to become a software engineer.",
    "The government announced a new policy for renewable energy.",
    "Please close the door and turn off the lights.",
    "The museum displays ancient artifacts from South India.",
    "Artificial intelligence is transforming the healthcare industry.",
    "They have been working on this project since January.",
    "The little girl smiled at the photographer.",
    "Researchers at IIT Bombay developed an efficient language model.",
    "Neither the manager nor the employees were satisfied with the result.",
    "The old man walked slowly along the riverbank.",
    "Wow, that was an amazing performance!",
    "\"Stop!\" shouted the officer near the crowded station.",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    """Create all output directories used by the pipeline."""
    for directory in (OUTPUTS_DIR, VIS_DIR, SCREENSHOTS_DIR, TESTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_random_seeds(seed: int = SEED) -> None:
    """Set Python and NumPy seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_native(value):
    """Recursively convert NumPy/Pandas types into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _shorten(text: str, width: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


class Tee:
    """Duplicate stdout/stderr to the console and a log file."""

    def __init__(self, path: Path):
        self.path = path
        self.file = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __exit__(self, exc_type, exc, tb):
        self.flush()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()
        return False


# ---------------------------------------------------------------------------
# NLTK resources and corpus loading
# ---------------------------------------------------------------------------

NLTK_RESOURCES = {
    "treebank": "corpora/treebank",
    "universal_tagset": "taggers/universal_tagset",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_resources(allow_download: bool = True, verbose: bool = True) -> Dict[str, str]:
    """
    Verify that the NLTK resources required by this project are available.

    If a resource is missing, try the NLTK downloader once.  If downloading is
    disabled or fails, raise a RuntimeError with the exact remedial command.
    """
    import nltk

    resolved: Dict[str, str] = {}
    for package, resource_path in NLTK_RESOURCES.items():
        try:
            found = nltk.data.find(resource_path)
            resolved[package] = str(found)
            continue
        except LookupError:
            pass

        if allow_download:
            if verbose:
                print(f"[resources] Missing '{package}'. Attempting one-time download ...")
            try:
                ok = nltk.download(package, quiet=True)
            except Exception as exc:  # network / SSL / permission problems
                ok = False
                detail = f"{type(exc).__name__}: {exc}"
            else:
                detail = "downloader returned False"
            if ok:
                try:
                    resolved[package] = str(nltk.data.find(resource_path))
                    continue
                except LookupError:
                    pass
        else:
            detail = "automatic download disabled"

        raise RuntimeError(
            f"NLTK resource '{package}' is unavailable ({detail}).\n"
            f"Remedy: run  python3 -m nltk.downloader {package}\n"
            "If the network is blocked, download the package on another machine and "
            "place it in ~/nltk_data, then rerun the script."
        )

    if verbose:
        print("[resources] All required NLTK resources are available.")
    return resolved


def load_treebank_sentences(remove_traces: bool = True) -> Tuple[List[List[Tuple[str, str]]], Dict[str, int]]:
    """
    Load Penn Treebank tagged sentences from NLTK.

    Trace / empty-category tokens carry the PTB tag '-NONE-' (for example
    '*', '0', '*-1', '*T*-1').  They are not real surface words, so they are
    removed by default.  Brackets '-LRB-' and '-RRB-' are retained and later
    mapped to punctuation.
    """
    from nltk.corpus import treebank

    raw = treebank.tagged_sents()
    raw_sentence_count = len(raw)
    raw_token_count = sum(len(s) for s in raw)

    sentences: List[List[Tuple[str, str]]] = []
    removed_traces = 0
    for sent in raw:
        cleaned = []
        for word, tag in sent:
            if remove_traces and tag == "-NONE-":
                removed_traces += 1
                continue
            if not str(word).strip():
                continue
            cleaned.append((str(word), str(tag)))
        if cleaned:  # never keep an empty sentence
            sentences.append(cleaned)

    stats = {
        "raw_sentences": raw_sentence_count,
        "raw_tokens": raw_token_count,
        "trace_tokens_removed": removed_traces,
        "sentences_after_trace_removal": len(sentences),
        "tokens_after_trace_removal": sum(len(s) for s in sentences),
    }
    return sentences, stats


def map_ptb_to_universal(
    sentences: Sequence[Sequence[Tuple[str, str]]],
) -> Tuple[List[List[Tuple[str, str]]], Dict[str, str]]:
    """Map every Penn Treebank tag in the corpus to the NLTK Universal tagset."""
    from nltk.tag.mapping import map_tag

    ptb_tags = sorted({tag for sent in sentences for _, tag in sent})
    mapping = {}
    for tag in ptb_tags:
        try:
            mapping[tag] = map_tag("en-ptb", "universal", tag)
        except Exception:
            # Defensive fallback; the standard mapping covers every PTB tag.
            mapping[tag] = "X"
    mapped = [[(word, mapping[tag]) for word, tag in sent] for sent in sentences]
    return mapped, mapping


def deduplicate_sentences(
    sentences: Sequence[Sequence[Tuple[str, str]]],
) -> Tuple[List[List[Tuple[str, str]]], int]:
    """Remove exact duplicate tagged sentences while preserving order."""
    seen = set()
    unique = []
    removed = 0
    for sent in sentences:
        key = tuple(sent)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(list(sent))
    return unique, removed


def prepare_dataset(
    sentences: Sequence[Sequence[Tuple[str, str]]],
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]], Dict[str, int]]:
    """
    Create a reproducible sentence-level 80:20 train/test split.

    The split is made over shuffled integer indices, so each sentence appears
    in exactly one split.  Duplicate sentences were removed before this step.
    """
    if not sentences:
        raise ValueError("Cannot split an empty corpus.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie strictly between 0 and 1.")

    indices = list(range(len(sentences)))
    random.Random(seed).shuffle(indices)

    n_test = max(1, int(math.ceil(len(sentences) * test_size)))
    n_train = len(sentences) - n_test
    if n_train <= 0:
        raise ValueError("Training split is empty; provide more sentences.")

    train_idx = sorted(indices[:n_train])
    test_idx = sorted(indices[n_train:])

    train = [list(sentences[i]) for i in train_idx]
    test = [list(sentences[i]) for i in test_idx]

    # Hard guarantee against sentence-level leakage.
    train_keys = {tuple(s) for s in train}
    test_keys = {tuple(s) for s in test}
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} sentences in both splits.")

    meta = {
        "train_sentences": len(train),
        "test_sentences": len(test),
        "train_tokens": sum(len(s) for s in train),
        "test_tokens": sum(len(s) for s in test),
        "split_seed": seed,
        "test_size": test_size,
        "sentence_overlap": 0,
    }
    return train, test, meta


def write_sentence_file(path: Path, sentences: Sequence[Sequence[Tuple[str, str]]]) -> None:
    """Write one whitespace-tokenised sentence per line (no tags)."""
    with path.open("w", encoding="utf-8") as handle:
        for sent in sentences:
            handle.write(" ".join(word for word, _ in sent) + "\n")


def load_custom_sentences(path: Path) -> List[str]:
    """
    Load custom test sentences.  If the file does not exist, write the
    manually prepared defaults.  Existing user content is never overwritten.
    """
    if path.exists():
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if lines:
            return lines
    path.write_text("\n".join(DEFAULT_CUSTOM_SENTENCES) + "\n", encoding="utf-8")
    return list(DEFAULT_CUSTOM_SENTENCES)


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenise one sentence safely; empty/whitespace input yields []."""
    if sentence is None or not str(sentence).strip():
        return []
    from nltk.tokenize import word_tokenize

    return [tok for tok in word_tokenize(str(sentence)) if tok.strip()]


# ---------------------------------------------------------------------------
# HMM training and Viterbi decoding
# ---------------------------------------------------------------------------


class HMMPosTagger:
    """
    Supervised first-order HMM POS tagger.

    Probability model
    -----------------
    P(t_1..t_n, w_1..w_n) =
        P(t_1 | START) * prod_i P(w_i | t_i) * prod_{i>1} P(t_i | t_{i-1})
        * P(END | t_n)

    Transition probabilities use add-k smoothing (k = 0.001):
        P(t_i | t_{i-1}) = (C(t_{i-1}, t_i) + k) / (C(t_{i-1}) + k * (|T| + 1))

    Emission probabilities for observed words use maximum likelihood:
        P(w | t) = C(w, t) / C(t)

    OOV words are handled by a separate, data-driven estimate described in
    `_build_oov_models` and `oov_emission_probs`.
    """

    START = "<START>"
    END = "<END>"

    def __init__(
        self,
        transition_k: float = TRANSITION_K,
        rare_word_threshold: int = RARE_WORD_THRESHOLD,
        max_suffix_len: int = MAX_SUFFIX_LEN,
        oov_alpha: float = OOV_ALPHA,
    ):
        self.transition_k = float(transition_k)
        self.rare_word_threshold = int(rare_word_threshold)
        self.max_suffix_len = int(max_suffix_len)
        self.oov_alpha = float(oov_alpha)

        self.tags: List[str] = []
        self.tag_counts: Counter = Counter()
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.emission_counts: Dict[str, Counter] = defaultdict(Counter)
        self.word_counts: Counter = Counter()
        self.word_tag_counts: Dict[str, Counter] = defaultdict(Counter)

        self.log_transition: Dict[str, Dict[str, float]] = {}
        self.log_tag_prior: Dict[str, float] = {}

        self.vocab: set = set()
        self._oov_cache: Dict[str, Dict[str, float]] = {}
        self.is_trained = False

        # OOV model data
        self.suffix_counts: Dict[int, Dict[str, Counter]] = {}
        self.suffix_tag_totals: Dict[int, Counter] = {}
        self.suffix_vocab: Dict[int, set] = {}
        self.numeric_tag_counts: Counter = Counter()
        self.digit_tag_counts: Counter = Counter()
        self.capitalized_tag_counts: Counter = Counter()
        self.rare_token_occurrences = 0
        self.numeric_rare_occurrences = 0
        self.digit_rare_occurrences = 0
        self.capitalized_rare_occurrences = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, tagged_sentences: Sequence[Sequence[Tuple[str, str]]]) -> "HMMPosTagger":
        if not tagged_sentences:
            raise ValueError("Training requires at least one tagged sentence.")

        self.tag_counts.clear()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.word_counts.clear()
        self.word_tag_counts = defaultdict(Counter)

        for sent in tagged_sentences:
            if not sent:
                continue
            prev_tag = self.START
            for word, tag in sent:
                word = str(word)
                tag = str(tag)
                self.tag_counts[tag] += 1
                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][word] += 1
                self.word_counts[word] += 1
                self.word_tag_counts[word][tag] += 1
                prev_tag = tag
            self.transition_counts[prev_tag][self.END] += 1

        self.tags = sorted(self.tag_counts)
        self.vocab = set(self.word_counts)
        total_tokens = sum(self.tag_counts.values())
        self.log_tag_prior = {
            tag: math.log(self.tag_counts[tag] / total_tokens) for tag in self.tags
        }

        self._build_transition_logprobs()
        self._build_oov_models()
        self._oov_cache.clear()
        self.is_trained = True
        return self

    def _build_transition_logprobs(self) -> None:
        """Pre-compute add-k smoothed log transition probabilities."""
        n_next = len(self.tags) + 1  # all tags + END
        states = [self.START] + self.tags
        self.log_transition = {}
        for prev in states:
            row = self.transition_counts.get(prev, Counter())
            denominator = sum(row.values()) + self.transition_k * n_next
            if denominator <= 0:
                # A state with no outgoing counts is impossible in a trained
                # model, but keep a safe uniform fallback.
                denominator = self.transition_k * n_next
            self.log_transition[prev] = {}
            for curr in self.tags + [self.END]:
                count = row.get(curr, 0)
                prob = (count + self.transition_k) / denominator
                self.log_transition[prev][curr] = math.log(max(prob, 1e-300))

    def _build_oov_models(self) -> None:
        """
        Build suffix and shape statistics for OOV words.

        Rare training words (frequency <= rare_word_threshold) supply suffix
        counts for lengths 1..4.  Rare numeric tokens, digit-containing tokens,
        and capitalised tokens supply separate shape priors.  All estimates use
        add-alpha smoothing and are therefore strictly positive.
        """
        self.suffix_counts = {length: defaultdict(Counter) for length in range(1, self.max_suffix_len + 1)}
        self.suffix_tag_totals = {length: Counter() for length in range(1, self.max_suffix_len + 1)}
        self.suffix_vocab = {length: set() for length in range(1, self.max_suffix_len + 1)}
        self.numeric_tag_counts = Counter()
        self.digit_tag_counts = Counter()
        self.capitalized_tag_counts = Counter()
        self.rare_token_occurrences = 0
        self.numeric_rare_occurrences = 0
        self.digit_rare_occurrences = 0
        self.capitalized_rare_occurrences = 0

        for word, total_count in self.word_counts.items():
            if total_count > self.rare_word_threshold:
                continue
            tag_counts = self.word_tag_counts[word]
            is_numeric = bool(re.fullmatch(r"[+\-]?\d[\d,.:/%\-]*", word))
            has_digit = any(ch.isdigit() for ch in word)
            is_capitalized = word[:1].isupper()

            for tag, count in tag_counts.items():
                self.rare_token_occurrences += count
                if is_numeric:
                    self.numeric_tag_counts[tag] += count
                    self.numeric_rare_occurrences += count
                if has_digit:
                    self.digit_tag_counts[tag] += count
                    self.digit_rare_occurrences += count
                if is_capitalized:
                    self.capitalized_tag_counts[tag] += count
                    self.capitalized_rare_occurrences += count

                lowered = word.lower()
                for length in range(1, self.max_suffix_len + 1):
                    if len(lowered) >= length:
                        suffix = lowered[-length:]
                        self.suffix_counts[length][tag][suffix] += count
                        self.suffix_tag_totals[length][tag] += count
                        self.suffix_vocab[length].add(suffix)

    # ------------------------------------------------------------------
    # Probability helpers
    # ------------------------------------------------------------------
    def _smoothed_suffix_prob(self, suffix: str, length: int, tag: str) -> float:
        count = self.suffix_counts.get(length, {}).get(tag, Counter()).get(suffix, 0)
        total = self.suffix_tag_totals.get(length, Counter()).get(tag, 0)
        vocabulary_size = len(self.suffix_vocab.get(length, set()))
        denominator = total + self.oov_alpha * (vocabulary_size + 1)
        return (count + self.oov_alpha) / denominator

    def _shape_prior(self, counts: Counter, total: int) -> Dict[str, float]:
        """Add-alpha smoothed shape distribution over the tagset."""
        if total <= 0:
            return {tag: 1.0 / len(self.tags) for tag in self.tags}
        denominator = total + self.oov_alpha * len(self.tags)
        return {
            tag: (counts.get(tag, 0) + self.oov_alpha) / denominator
            for tag in self.tags
        }

    def oov_emission_probs(self, word: str) -> Dict[str, float]:
        """
        Estimate P(word | tag) for an out-of-vocabulary word.

        Strategy:
        * Numeric tokens use a numeric-shape prior learned from rare training
          numbers (strong NUM signal).
        * Other words use the longest matching suffix seen in rare training
          words, combined with the tag prior.
        * If the token contains a digit or starts with a capital letter, the
          corresponding shape prior is mixed in with a fixed weight.
        * The resulting scores are normalised over tags (softmax).  The
          normalising constant depends only on the word, so it does not alter
          Viterbi path ranking.
        """
        cache_key = word
        if cache_key in self._oov_cache:
            return self._oov_cache[cache_key]

        if not self.tags:
            raise RuntimeError("The HMM must be trained before OOV scoring.")

        lowered = word.lower()
        is_numeric = bool(re.fullmatch(r"[+\-]?\d[\d,.:/%\-]*", word))
        has_digit = any(ch.isdigit() for ch in word)
        is_capitalized = word[:1].isupper()

        if is_numeric and self.numeric_rare_occurrences > 0:
            probs = self._shape_prior(self.numeric_tag_counts, self.numeric_rare_occurrences)
            self._oov_cache[cache_key] = probs
            return probs

        max_len = min(self.max_suffix_len, len(lowered))
        chosen_length = max_len
        if max_len > 0:
            for length in range(max_len, 0, -1):
                suffix = lowered[-length:]
                if any(self.suffix_counts[length][tag].get(suffix, 0) > 0 for tag in self.tags):
                    chosen_length = length
                    break

        log_scores = {}
        for tag in self.tags:
            score = self.log_tag_prior[tag]
            if chosen_length > 0:
                suffix = lowered[-chosen_length:]
                score += math.log(max(self._smoothed_suffix_prob(suffix, chosen_length, tag), 1e-300))
            if has_digit and self.digit_rare_occurrences > 0:
                digit_prior = self._shape_prior(self.digit_tag_counts, self.digit_rare_occurrences)
                score = 0.5 * score + 0.5 * math.log(max(digit_prior[tag], 1e-300))
            elif is_capitalized and self.capitalized_rare_occurrences > 0:
                cap_prior = self._shape_prior(self.capitalized_tag_counts, self.capitalized_rare_occurrences)
                score = 0.8 * score + 0.2 * math.log(max(cap_prior[tag], 1e-300))
            log_scores[tag] = score

        max_score = max(log_scores.values())
        exp_values = {tag: math.exp(value - max_score) for tag, value in log_scores.items()}
        normaliser = sum(exp_values.values())
        probs = {tag: value / normaliser for tag, value in exp_values.items()}
        self._oov_cache[cache_key] = probs
        return probs

    def emission_logprob(self, word: str, tag: str) -> float:
        """Return log P(word | tag); observed words use MLE, OOV words the OOV model."""
        if word in self.vocab:
            count = self.emission_counts.get(tag, Counter()).get(word, 0)
            if count <= 0:
                return float("-inf")
            return math.log(count / self.tag_counts[tag])
        return math.log(max(self.oov_emission_probs(word).get(tag, 0.0), 1e-300))

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------
    def viterbi_decode(self, tokens: Sequence[str], return_details: bool = False):
        """
        Decode the most probable tag sequence with the Viterbi algorithm.

        Returns
        -------
        (tags, log_probability) normally, or
        (tags, log_probability, details) when return_details=True.

        Complexity: O(n * |T|^2) time and O(n * |T|) space.
        Backpointers store the best predecessor for every position and tag.
        """
        if not self.is_trained:
            raise RuntimeError("Train the HMM before decoding.")
        tokens = [str(tok) for tok in tokens]
        if not tokens:
            if return_details:
                return [], 0.0, {"score_matrix": [], "backpointers": [], "emissions": []}
            return [], 0.0

        tags = self.tags
        n = len(tokens)
        m = len(tags)

        emission_scores = [
            [self.emission_logprob(token, tag) for tag in tags] for token in tokens
        ]

        score = [[float("-inf")] * m for _ in range(n)]
        backpointer = [[-1] * m for _ in range(n)]

        for j, tag in enumerate(tags):
            emission = emission_scores[0][j]
            if emission == float("-inf"):
                continue
            score[0][j] = self.log_transition[self.START][tag] + emission

        for i in range(1, n):
            for j, tag in enumerate(tags):
                emission = emission_scores[i][j]
                if emission == float("-inf"):
                    continue
                best_prev_score = float("-inf")
                best_prev_index = -1
                for k, prev_tag in enumerate(tags):
                    previous = score[i - 1][k]
                    if previous == float("-inf"):
                        continue
                    candidate = previous + self.log_transition[prev_tag][tag]
                    if candidate > best_prev_score:
                        best_prev_score = candidate
                        best_prev_index = k
                if best_prev_index >= 0:
                    score[i][j] = best_prev_score + emission
                    backpointer[i][j] = best_prev_index

        final_scores = []
        for j, tag in enumerate(tags):
            if score[n - 1][j] == float("-inf"):
                final_scores.append(float("-inf"))
            else:
                final_scores.append(score[n - 1][j] + self.log_transition[tag][self.END])

        best_final_index = int(np.argmax(final_scores))
        best_final_score = final_scores[best_final_index]

        if best_final_score == float("-inf"):
            # Extremely defensive fallback; with smoothed transitions and the
            # OOV model this should not occur.
            fallback = [self.tags[0]] * n
            if return_details:
                return fallback, float("-inf"), {
                    "score_matrix": score,
                    "backpointers": backpointer,
                    "emissions": emission_scores,
                }
            return fallback, float("-inf")

        best_path = [None] * n
        best_path[n - 1] = tags[best_final_index]
        for i in range(n - 1, 0, -1):
            best_final_index = backpointer[i][best_final_index]
            best_path[i - 1] = tags[best_final_index]

        if return_details:
            details = {
                "tags": tags,
                "tokens": tokens,
                "score_matrix": score,
                "backpointers": backpointer,
                "emissions": emission_scores,
            }
            return best_path, best_final_score, details
        return best_path, best_final_score

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def tag(self, tokens: Sequence[str]) -> List[str]:
        return self.viterbi_decode(tokens)[0]


class MostFrequentTagBaseline:
    """Most-frequent-tag baseline fitted only on the training split."""

    def __init__(self):
        self.word_tag: Dict[str, str] = {}
        self.default_tag: str = "NOUN"
        self.tag_counts: Counter = Counter()

    def train(self, tagged_sentences: Sequence[Sequence[Tuple[str, str]]]) -> "MostFrequentTagBaseline":
        word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.tag_counts = Counter()
        for sent in tagged_sentences:
            for word, tag in sent:
                word_counts[str(word)][str(tag)] += 1
                self.tag_counts[str(tag)] += 1
        if not self.tag_counts:
            raise ValueError("Baseline training data is empty.")

        for word, counts in word_counts.items():
            # Deterministic tie-break: higher count first, then alphabetical tag.
            self.word_tag[word] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        self.default_tag = sorted(self.tag_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return self

    def tag(self, tokens: Sequence[str]) -> List[str]:
        return [self.word_tag.get(str(token), self.default_tag) for token in tokens]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def flatten_tagged(sentences: Sequence[Sequence[Tuple[str, str]]]) -> Tuple[List[str], List[str]]:
    words = [word for sent in sentences for word, _ in sent]
    tags = [tag for sent in sentences for _, tag in sent]
    return words, tags


def flatten_predictions(predictions: Sequence[Sequence[str]]) -> List[str]:
    return [tag for sent in predictions for tag in sent]


def evaluate_tags(
    gold_tags: Sequence[str],
    predicted_tags: Sequence[str],
    labels: Sequence[str],
    words: Sequence[str],
    training_vocab: set,
) -> Tuple[Dict, pd.DataFrame, np.ndarray]:
    """Compute overall, per-tag, OOV and confusion-matrix statistics."""
    if len(gold_tags) != len(predicted_tags):
        raise ValueError("Gold and predicted tag sequences have different lengths.")
    if len(gold_tags) != len(words):
        raise ValueError("Token list and tag list have different lengths.")

    label_list = list(labels)
    accuracy = float(accuracy_score(gold_tags, predicted_tags))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold_tags, predicted_tags, labels=label_list, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        gold_tags, predicted_tags, labels=label_list, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        gold_tags, predicted_tags, labels=label_list, average=None, zero_division=0
    )

    per_tag_rows = []
    for idx, tag in enumerate(label_list):
        per_tag_rows.append(
            {
                "tag": tag,
                "precision": float(per_p[idx]),
                "recall": float(per_r[idx]),
                "f1_score": float(per_f1[idx]),
                "support": int(per_support[idx]),
            }
        )
    per_tag_df = pd.DataFrame(per_tag_rows)

    oov_mask = np.array([word not in training_vocab for word in words], dtype=bool)
    gold_array = np.array(gold_tags)
    pred_array = np.array(predicted_tags)
    oov_count = int(oov_mask.sum())
    known_count = int((~oov_mask).sum())
    oov_accuracy = float((gold_array[oov_mask] == pred_array[oov_mask]).mean()) if oov_count else 0.0
    known_accuracy = float((gold_array[~oov_mask] == pred_array[~oov_mask]).mean()) if known_count else 0.0

    cm = confusion_matrix(gold_tags, predicted_tags, labels=label_list)
    report_text = classification_report(
        gold_tags, predicted_tags, labels=label_list, zero_division=0, digits=4
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "support": int(len(gold_tags)),
        "oov_count": oov_count,
        "oov_rate": float(oov_count / len(words)) if words else 0.0,
        "oov_accuracy": oov_accuracy,
        "known_count": known_count,
        "known_accuracy": known_accuracy,
        "per_tag": {row["tag"]: {k: v for k, v in row.items() if k != "tag"} for row in per_tag_rows},
        "classification_report": report_text,
    }
    return metrics, per_tag_df, cm


def build_error_rows(
    test_sentences: Sequence[Sequence[Tuple[str, str]]],
    predicted_tags: Sequence[Sequence[str]],
    baseline_tags: Sequence[Sequence[str]],
    training_vocab: set,
) -> Tuple[List[Dict], List[Dict]]:
    """Create per-error rows (with context) and a ranked confusion summary."""
    rows: List[Dict] = []
    pair_counter: Counter = Counter()
    pair_example: Dict[Tuple[str, str], str] = {}

    for sent_id, (sent, preds, baseline_preds) in enumerate(
        zip(test_sentences, predicted_tags, baseline_tags), start=1
    ):
        words = [word for word, _ in sent]
        golds = [tag for _, tag in sent]
        if len(preds) != len(sent) or len(baseline_preds) != len(sent):
            raise ValueError("Prediction length does not match sentence length.")
        for idx, (word, gold, pred) in enumerate(zip(words, golds, preds)):
            if gold == pred:
                continue
            pair = (gold, pred)
            pair_counter[pair] += 1
            left = " ".join(words[max(0, idx - 3):idx])
            right = " ".join(words[idx + 1: idx + 4])
            marked = f"{left + ' ' if left else ''}[{word}/{gold}->{pred}]{' ' + right if right else ''}"
            pair_example.setdefault(pair, marked)
            rows.append(
                {
                    "sentence_id": sent_id,
                    "token_index": idx,
                    "token": word,
                    "gold_tag": gold,
                    "predicted_tag": pred,
                    "baseline_tag": baseline_preds[idx],
                    "is_oov": int(word not in training_vocab),
                    "error_pair": f"{gold} -> {pred}",
                    "pair_count": 0,  # filled below
                    "context": marked,
                }
            )

    total_errors = len(rows)
    for row in rows:
        row["pair_count"] = int(pair_counter[(row["gold_tag"], row["predicted_tag"])])

    ranked = []
    for (gold, pred), count in sorted(pair_counter.items(), key=lambda item: (-item[1], item[0])):
        ranked.append(
            {
                "gold_tag": gold,
                "predicted_tag": pred,
                "count": int(count),
                "percentage_of_errors": float(count / total_errors * 100.0) if total_errors else 0.0,
                "example_context": pair_example.get((gold, pred), ""),
            }
        )
    return rows, ranked


def build_sample_rows(
    test_sentences: Sequence[Sequence[Tuple[str, str]]],
    predicted_tags: Sequence[Sequence[str]],
    baseline_tags: Sequence[Sequence[str]],
    training_vocab: set,
    max_sentences: int = 15,
) -> List[Dict]:
    """Flatten the first few test sentences for the sample output CSV."""
    rows = []
    for sent_id, (sent, preds, baseline_preds) in enumerate(
        zip(test_sentences, predicted_tags, baseline_tags), start=1
    ):
        if sent_id > max_sentences:
            break
        for idx, ((word, gold), pred, base) in enumerate(zip(sent, preds, baseline_preds)):
            rows.append(
                {
                    "sentence_id": sent_id,
                    "token_index": idx,
                    "token": word,
                    "gold_tag": gold,
                    "hmm_predicted_tag": pred,
                    "baseline_predicted_tag": base,
                    "is_oov": int(word not in training_vocab),
                }
            )
    return rows


def run_custom_tests(
    sentences: Sequence[str],
    hmm: HMMPosTagger,
    baseline: MostFrequentTagBaseline,
) -> Tuple[List[Dict], List[Dict]]:
    """Tag every custom sentence and build display/CSV records."""
    records: List[Dict] = []
    rows: List[Dict] = []
    for sent_id, sentence in enumerate(sentences, start=1):
        tokens = tokenize_sentence(sentence)
        hmm_tags = hmm.tag(tokens) if tokens else []
        base_tags = baseline.tag(tokens) if tokens else []
        if len(hmm_tags) != len(tokens) or len(base_tags) != len(tokens):
            raise RuntimeError("Custom sentence tagging produced an invalid output length.")
        records.append(
            {
                "sentence_id": sent_id,
                "sentence": sentence,
                "tokens": tokens,
                "hmm_tags": hmm_tags,
                "baseline_tags": base_tags,
                "hmm_tagged": " ".join(f"{w}/{t}" for w, t in zip(tokens, hmm_tags)),
                "baseline_tagged": " ".join(f"{w}/{t}" for w, t in zip(tokens, base_tags)),
            }
        )
        for idx, (token, hmm_tag, base_tag) in enumerate(zip(tokens, hmm_tags, base_tags)):
            rows.append(
                {
                    "sentence_id": sent_id,
                    "sentence": sentence,
                    "token_index": idx,
                    "token": token,
                    "hmm_predicted_tag": hmm_tag,
                    "baseline_predicted_tag": base_tag,
                }
            )
    return records, rows


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_visualizations(result: Dict, gold_tags: Sequence[str], hmm_tags: Sequence[str]) -> List[str]:
    """Generate the five required charts from actual computed values."""
    labels = result["confusion_matrix"]["labels"]
    cm = np.array(result["confusion_matrix"]["matrix"], dtype=float)
    created = []

    # 1. Row-normalised confusion matrix.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("HMM POS Tagging: Normalised Confusion Matrix (Test Set)", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Universal Tag")
    ax.set_ylabel("Gold Universal Tag")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_norm[i, j]
            if value >= 0.005:
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.55 else "black", fontsize=7.5,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion of gold class")
    _save_figure(fig, VIS_DIR / "confusion_matrix.png")
    created.append("visualizations/confusion_matrix.png")

    # 2. Per-tag precision/recall/F1 grouped bars.
    per_tag = result["hmm"]["per_tag"]
    tags = [tag for tag in labels if per_tag.get(tag, {}).get("support", 0) > 0]
    precision = [per_tag[t]["precision"] for t in tags]
    recall = [per_tag[t]["recall"] for t in tags]
    f1 = [per_tag[t]["f1_score"] for t in tags]
    x = np.arange(len(tags))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x - width, precision, width, label="Precision", color="#2E75B6")
    ax.bar(x, recall, width, label="Recall", color="#70AD47")
    ax.bar(x + width, f1, width, label="F1-score", color="#ED7D31")
    ax.set_title("Per-Tag Precision, Recall and F1-score (HMM, Test Set)", fontsize=12)
    ax.set_xlabel("Universal POS Tag")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_figure(fig, VIS_DIR / "per_tag_scores.png")
    created.append("visualizations/per_tag_scores.png")

    # 3. Baseline vs HMM.
    metrics = [
        ("Accuracy", result["baseline"]["accuracy"], result["hmm"]["accuracy"]),
        ("Macro F1", result["baseline"]["macro_f1"], result["hmm"]["macro_f1"]),
        ("Weighted F1", result["baseline"]["weighted_f1"], result["hmm"]["weighted_f1"]),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    base_vals = [m[1] for m in metrics]
    hmm_vals = [m[2] for m in metrics]
    bars1 = ax.bar(x - width / 2, base_vals, width, label="Most-frequent-tag baseline", color="#A5A5A5")
    bars2 = ax.bar(x + width / 2, hmm_vals, width, label="HMM + Viterbi", color="#2E75B6")
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Baseline versus HMM Performance", fontsize=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_figure(fig, VIS_DIR / "model_comparison.png")
    created.append("visualizations/model_comparison.png")

    # 4. OOV versus known-token accuracy.
    groups = ["OOV tokens", "Known tokens"]
    hmm_values = [result["hmm"]["oov_accuracy"], result["hmm"]["known_accuracy"]]
    base_values = [result["baseline"]["oov_accuracy"], result["baseline"]["known_accuracy"]]
    counts = [result["hmm"]["oov_count"], result["hmm"]["known_count"]]
    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bars1 = ax.bar(x - width / 2, base_values, width, label="Baseline", color="#A5A5A5")
    bars2 = ax.bar(x + width / 2, hmm_values, width, label="HMM", color="#4472C4")
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_title("OOV versus Known-Token Accuracy", fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={c:,})" for g, c in zip(groups, counts)])
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_figure(fig, VIS_DIR / "oov_analysis.png")
    created.append("visualizations/oov_analysis.png")

    # 5. Tag distribution in train and test splits.
    train_counts = result["corpus"]["train_tag_counts"]
    test_counts = result["corpus"]["test_tag_counts"]
    tags = [tag for tag in labels if train_counts.get(tag, 0) + test_counts.get(tag, 0) > 0]
    train_vals = [train_counts.get(tag, 0) for tag in tags]
    test_vals = [test_counts.get(tag, 0) for tag in tags]
    x = np.arange(len(tags))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x - width / 2, train_vals, width, label="Training split (80%)", color="#4472C4")
    ax.bar(x + width / 2, test_vals, width, label="Test split (20%)", color="#ED7D31")
    ax.set_title("Universal POS-Tag Distribution by Split", fontsize=12)
    ax.set_xlabel("Universal POS Tag")
    ax.set_ylabel("Token count")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_figure(fig, VIS_DIR / "tag_distribution.png")
    created.append("visualizations/tag_distribution.png")

    return created


# ---------------------------------------------------------------------------
# Result assembly and file output
# ---------------------------------------------------------------------------


def _environment_versions() -> Dict[str, str]:
    import nltk
    import sklearn

    versions = {
        "python": sys.version.split()[0],
        "nltk": nltk.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import openpyxl

        versions["openpyxl"] = openpyxl.__version__
    except Exception:
        versions["openpyxl"] = "not available"
    try:
        import docx

        versions["python-docx"] = getattr(docx, "__version__", "installed")
    except Exception:
        versions["python-docx"] = "not available"
    return versions


def build_summary_rows(result: Dict) -> List[Dict]:
    """Key-value rows used by the console, JSON and workbook summary."""
    corpus = result["corpus"]
    hmm = result["hmm"]
    baseline = result["baseline"]
    return [
        {"metric": "Corpus", "value": corpus["name"]},
        {"metric": "Universal tagset size", "value": corpus["universal_tag_count"]},
        {"metric": "Prepared sentences", "value": corpus["prepared_sentences"]},
        {"metric": "Training sentences", "value": corpus["train_sentences"]},
        {"metric": "Test sentences", "value": corpus["test_sentences"]},
        {"metric": "Training tokens", "value": corpus["train_tokens"]},
        {"metric": "Test tokens", "value": corpus["test_tokens"]},
        {"metric": "Test OOV tokens", "value": hmm["oov_count"]},
        {"metric": "Baseline accuracy", "value": round(baseline["accuracy"], 6)},
        {"metric": "HMM accuracy", "value": round(hmm["accuracy"], 6)},
        {"metric": "HMM macro F1", "value": round(hmm["macro_f1"], 6)},
        {"metric": "HMM weighted F1", "value": round(hmm["weighted_f1"], 6)},
        {"metric": "HMM OOV accuracy", "value": round(hmm["oov_accuracy"], 6)},
        {"metric": "HMM known-token accuracy", "value": round(hmm["known_accuracy"], 6)},
    ]


def save_core_outputs(bundle: Dict) -> List[str]:
    """Write JSON, CSV, sentence files and charts from a pipeline bundle."""
    ensure_directories()
    result = bundle["result"]
    created: List[str] = []

    write_sentence_file(PROJECT_DIR / "train_sentences.txt", bundle["train_sentences"])
    write_sentence_file(PROJECT_DIR / "test_sentences.txt", bundle["test_sentences"])
    created.extend(["train_sentences.txt", "test_sentences.txt"])

    # Custom sentence file is created by load_custom_sentences if missing.
    if not (PROJECT_DIR / "custom_test_sentences.txt").exists():
        (PROJECT_DIR / "custom_test_sentences.txt").write_text(
            "\n".join(DEFAULT_CUSTOM_SENTENCES) + "\n", encoding="utf-8"
        )
    created.append("custom_test_sentences.txt")

    # Per-tag and overall evaluation table for both models.
    evaluation_rows = []
    for model_name, metrics in (("Baseline", result["baseline"]), ("HMM", result["hmm"])):
        evaluation_rows.append(
            {
                "model": model_name,
                "scope": "Overall",
                "tag": "ALL",
                "precision": metrics["macro_precision"],
                "recall": metrics["macro_recall"],
                "f1_score": metrics["macro_f1"],
                "support": metrics["support"],
                "accuracy": metrics["accuracy"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        for tag, values in metrics["per_tag"].items():
            evaluation_rows.append(
                {
                    "model": model_name,
                    "scope": "Per-tag",
                    "tag": tag,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1_score"],
                    "support": values["support"],
                    "accuracy": None,
                    "weighted_precision": None,
                    "weighted_recall": None,
                    "weighted_f1": None,
                }
            )
    pd.DataFrame(evaluation_rows).to_csv(PROJECT_DIR / "evaluation_results.csv", index=False)
    created.append("evaluation_results.csv")

    pd.DataFrame(bundle["sample_rows"]).to_csv(
        PROJECT_DIR / "sample_tagged_output.csv", index=False
    )
    created.append("sample_tagged_output.csv")

    pd.DataFrame(bundle["error_rows"]).to_csv(
        PROJECT_DIR / "error_analysis.csv", index=False
    )
    created.append("error_analysis.csv")

    pd.DataFrame(bundle["custom_rows"]).to_csv(
        PROJECT_DIR / "custom_predictions.csv", index=False
    )
    created.append("custom_predictions.csv")

    with (PROJECT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(to_native(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
    created.append("results.json")

    charts = generate_visualizations(
        result,
        bundle["y_true"],
        bundle["y_pred"],
    )
    created.extend(charts)
    result["artifacts"] = {
        "files": created,
        "visualizations": charts,
    }
    # Rewrite JSON so it records the artefacts too.
    with (PROJECT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(to_native(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
    return created


def run_pipeline(write_files: bool = True, verbose: bool = True) -> Dict:
    """
    Execute the complete experiment and return a bundle containing the result
    dictionary, trained models, splits, predictions and analysis rows.
    """
    set_random_seeds(SEED)
    ensure_directories()
    started = time.time()

    if verbose:
        print("=" * 78)
        print("NLP LABORATORY ASSIGNMENT 3 - HMM PART-OF-SPEECH TAGGING")
        print("=" * 78)

    ensure_nltk_resources(verbose=verbose)

    raw_sentences, load_stats = load_treebank_sentences(remove_traces=True)
    if verbose:
        print(
            f"[corpus] Raw PTB sentences: {load_stats['raw_sentences']:,}; "
            f"raw tokens: {load_stats['raw_tokens']:,}"
        )
        print(f"[corpus] Trace (-NONE-) tokens removed: {load_stats['trace_tokens_removed']:,}")

    universal_sentences, tag_mapping = map_ptb_to_universal(raw_sentences)
    universal_sentences, duplicates_removed = deduplicate_sentences(universal_sentences)
    if verbose:
        print(f"[corpus] Exact duplicate sentences removed: {duplicates_removed}")
        print(f"[corpus] Sentences after cleaning: {len(universal_sentences):,}")

    train_sentences, test_sentences, split_meta = prepare_dataset(
        universal_sentences, test_size=TEST_SIZE, seed=SEED
    )
    train_words, train_tags = flatten_tagged(train_sentences)
    test_words, test_tags = flatten_tagged(test_sentences)
    train_vocab = set(train_words)
    test_oov_mask = [word not in train_vocab for word in test_words]
    test_oov_count = int(sum(test_oov_mask))

    if verbose:
        print(
            f"[split] Seed={SEED}, test_size={TEST_SIZE:.2f} -> "
            f"{len(train_sentences):,} train / {len(test_sentences):,} test sentences"
        )
        print(
            f"[split] {len(train_words):,} train tokens / {len(test_words):,} test tokens; "
            f"train vocabulary={len(train_vocab):,}; test OOV={test_oov_count:,} "
            f"({test_oov_count / max(1, len(test_words)) * 100:.2f}%)"
        )
        print(f"[split] Sentence overlap between train and test: {split_meta['sentence_overlap']}")

    if not (PROJECT_DIR / "custom_test_sentences.txt").exists():
        (PROJECT_DIR / "custom_test_sentences.txt").write_text(
            "\n".join(DEFAULT_CUSTOM_SENTENCES) + "\n", encoding="utf-8"
        )
    custom_sentences = load_custom_sentences(PROJECT_DIR / "custom_test_sentences.txt")
    if verbose:
        print(f"[custom] Loaded {len(custom_sentences)} manually written test sentences.")

    if verbose:
        print("[train] Training the HMM ...")
    hmm = HMMPosTagger(
        transition_k=TRANSITION_K,
        rare_word_threshold=RARE_WORD_THRESHOLD,
        max_suffix_len=MAX_SUFFIX_LEN,
        oov_alpha=OOV_ALPHA,
    ).train(train_sentences)

    baseline = MostFrequentTagBaseline().train(train_sentences)
    if verbose:
        print(
            f"[train] HMM tags={len(hmm.tags)}, vocabulary={len(hmm.vocab):,}, "
            f"emission pairs={sum(len(c) for c in hmm.emission_counts.values()):,}"
        )
        print(
            f"[train] OOV suffix model built from {hmm.rare_token_occurrences:,} rare-word "
            f"occurrences; numeric rare tokens={hmm.numeric_rare_occurrences:,}"
        )
        print("[predict] Decoding the held-out test set with Viterbi ...")

    hmm_predictions = [hmm.tag([word for word, _ in sent]) for sent in test_sentences]
    baseline_predictions = [
        baseline.tag([word for word, _ in sent]) for sent in test_sentences
    ]

    for sent, preds in zip(test_sentences, hmm_predictions):
        if len(preds) != len(sent):
            raise RuntimeError("HMM prediction length mismatch on the test set.")
    for sent, preds in zip(test_sentences, baseline_predictions):
        if len(preds) != len(sent):
            raise RuntimeError("Baseline prediction length mismatch on the test set.")

    y_true = flatten_predictions([[tag for _, tag in sent] for sent in test_sentences])
    y_pred = flatten_predictions(hmm_predictions)
    y_base = flatten_predictions(baseline_predictions)

    hmm_metrics, hmm_per_tag_df, cm = evaluate_tags(
        y_true, y_pred, hmm.tags, test_words, train_vocab
    )
    baseline_metrics, baseline_per_tag_df, _ = evaluate_tags(
        y_true, y_base, hmm.tags, test_words, train_vocab
    )

    error_rows, top_errors = build_error_rows(
        test_sentences, hmm_predictions, baseline_predictions, train_vocab
    )
    sample_rows = build_sample_rows(
        test_sentences, hmm_predictions, baseline_predictions, train_vocab,
        max_sentences=15,
    )
    custom_records, custom_rows = run_custom_tests(custom_sentences, hmm, baseline)

    tag_counts_all = Counter(train_tags)
    tag_counts_all.update(test_tags)
    universal_tag_counts = {
        tag: int(tag_counts_all.get(tag, 0)) for tag in sorted(tag_counts_all)
    }
    train_tag_counts = {tag: int(Counter(train_tags).get(tag, 0)) for tag in hmm.tags}
    test_tag_counts = {tag: int(Counter(test_tags).get(tag, 0)) for tag in hmm.tags}

    corpus = {
        "name": "NLTK Penn Treebank (treebank.tagged_sents())",
        "raw_sentences": load_stats["raw_sentences"],
        "raw_tokens": load_stats["raw_tokens"],
        "trace_tokens_removed": load_stats["trace_tokens_removed"],
        "sentences_after_trace_removal": load_stats["sentences_after_trace_removal"],
        "tokens_after_trace_removal": load_stats["tokens_after_trace_removal"],
        "duplicate_sentences_removed": duplicates_removed,
        "prepared_sentences": len(universal_sentences),
        "prepared_tokens": len(train_words) + len(test_words),
        "train_sentences": split_meta["train_sentences"],
        "test_sentences": split_meta["test_sentences"],
        "train_tokens": split_meta["train_tokens"],
        "test_tokens": split_meta["test_tokens"],
        "train_vocabulary": len(train_vocab),
        "test_oov_tokens": test_oov_count,
        "test_oov_rate": test_oov_count / max(1, len(test_words)),
        "ptb_tag_count": len(tag_mapping),
        "universal_tag_count": len(hmm.tags),
        "universal_tags": hmm.tags,
        "universal_tag_counts": universal_tag_counts,
        "train_tag_counts": train_tag_counts,
        "test_tag_counts": test_tag_counts,
    }

    result = {
        "assignment": "NLP Laboratory Assignment 3",
        "title": "Survey and Implementation of POS Tagging using a Hidden Markov Model",
        "student_name": "Shravani Kurkute",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_directory": str(PROJECT_DIR),
        "environment": _environment_versions(),
        "config": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "transition_smoothing": "add-k",
            "transition_k": TRANSITION_K,
            "rare_word_threshold": RARE_WORD_THRESHOLD,
            "max_suffix_length": MAX_SUFFIX_LEN,
            "oov_alpha": OOV_ALPHA,
            "tagset": "NLTK Universal tagset (12 tags; '.' denotes punctuation)",
            "decoder": "Viterbi with log probabilities and backpointers",
            "baseline": "Most frequent tag per word; global most frequent tag for OOV",
        },
        "corpus": corpus,
        "tagset": UNIVERSAL_TAG_DESCRIPTIONS,
        "tag_mapping": tag_mapping,
        "model": {
            "num_tags": len(hmm.tags),
            "vocabulary_size": len(hmm.vocab),
            "emission_pairs": int(sum(len(c) for c in hmm.emission_counts.values())),
            "transition_rows": len(hmm.log_transition),
            "rare_token_occurrences": int(hmm.rare_token_occurrences),
            "numeric_rare_occurrences": int(hmm.numeric_rare_occurrences),
            "suffix_lengths": list(range(1, MAX_SUFFIX_LEN + 1)),
        },
        "baseline": baseline_metrics,
        "hmm": hmm_metrics,
        "baseline_per_tag": baseline_metrics["per_tag"],
        "comparison": {
            "accuracy_gain": hmm_metrics["accuracy"] - baseline_metrics["accuracy"],
            "macro_f1_gain": hmm_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "weighted_f1_gain": hmm_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
        },
        "oov": {
            "count": hmm_metrics["oov_count"],
            "rate": hmm_metrics["oov_rate"],
            "hmm_accuracy": hmm_metrics["oov_accuracy"],
            "baseline_accuracy": baseline_metrics["oov_accuracy"],
            "known_count": hmm_metrics["known_count"],
            "hmm_known_accuracy": hmm_metrics["known_accuracy"],
            "baseline_known_accuracy": baseline_metrics["known_accuracy"],
        },
        "confusion_matrix": {
            "labels": hmm.tags,
            "matrix": cm.tolist(),
        },
        "top_errors": top_errors[:20],
        "top_error_count": len(top_errors),
        "total_error_tokens": len(error_rows),
        "custom_sentences": custom_records,
        "sample_tagged_sentences": sample_rows,
        "summary": build_summary_rows(
            {
                "corpus": corpus,
                "hmm": hmm_metrics,
                "baseline": baseline_metrics,
            }
        ),
    }

    bundle = {
        "result": result,
        "train_sentences": train_sentences,
        "test_sentences": test_sentences,
        "hmm": hmm,
        "baseline": baseline,
        "hmm_predictions": hmm_predictions,
        "baseline_predictions": baseline_predictions,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_base": y_base,
        "error_rows": error_rows,
        "sample_rows": sample_rows,
        "custom_records": custom_records,
        "custom_rows": custom_rows,
        "hmm_per_tag_df": hmm_per_tag_df,
        "baseline_per_tag_df": baseline_per_tag_df,
    }

    if write_files:
        save_core_outputs(bundle)

    elapsed = time.time() - started

    if verbose:
        print("-" * 78)
        print(f"Baseline accuracy          : {baseline_metrics['accuracy']:.6f}")
        print(
            f"Baseline macro / weighted F1: {baseline_metrics['macro_f1']:.6f} / "
            f"{baseline_metrics['weighted_f1']:.6f}"
        )
        print(f"HMM accuracy               : {hmm_metrics['accuracy']:.6f}")
        print(
            f"HMM macro / weighted F1     : {hmm_metrics['macro_f1']:.6f} / "
            f"{hmm_metrics['weighted_f1']:.6f}"
        )
        print(
            f"HMM OOV accuracy           : {hmm_metrics['oov_accuracy']:.6f} "
            f"on {hmm_metrics['oov_count']:,} OOV tokens"
        )
        print(
            f"HMM known-token accuracy   : {hmm_metrics['known_accuracy']:.6f} "
            f"on {hmm_metrics['known_count']:,} known tokens"
        )
        print(f"Total HMM token errors     : {len(error_rows):,}")
        print(f"Pipeline time              : {elapsed:.2f} seconds")
        print("-" * 78)

    result["runtime_seconds"] = round(elapsed, 3)
    if write_files:
        # Persist the runtime too.
        with (PROJECT_DIR / "results.json").open("w", encoding="utf-8") as handle:
            json.dump(to_native(result), handle, indent=2, ensure_ascii=False, allow_nan=False)

    return bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate an HMM POS tagger.")
    parser.add_argument("--no-report", action="store_true", help="Skip DOCX report generation")
    parser.add_argument("--no-workbook", action="store_true", help="Skip XLSX workbook generation")
    args = parser.parse_args()

    ensure_directories()
    with Tee(OUTPUTS_DIR / "terminal_output.txt"):
        bundle = run_pipeline(write_files=True, verbose=True)
        result = bundle["result"]

        if not args.no_workbook:
            try:
                from generate_workbook import build_workbook

                path = build_workbook(result)
                print(f"[workbook] Created: {path}")
            except Exception as exc:
                print(f"[workbook] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

        if not args.no_report:
            try:
                from generate_report import build_report

                path = build_report(result)
                print(f"[report] Created: {path}")
            except Exception as exc:
                print(f"[report] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

        print("=" * 78)
        print("All core artefacts have been generated in:")
        print(PROJECT_DIR)
        print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
