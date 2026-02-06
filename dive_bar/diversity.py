#!/usr/bin/env python3
"""Diversity scoring for response quality checking."""

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from dive_bar.models import Message

STOP_WORDS = frozenset({
    "i", "me", "my", "we", "you", "your", "he", "she",
    "it", "they", "them", "the", "a", "an", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "am", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will",
    "just", "not", "no", "so", "if", "that", "this",
    "what", "when", "how", "all", "up", "out", "about",
    "like", "got", "get", "go", "can", "would", "could",
    "should", "there", "here", "from", "its", "than",
    "into", "over", "some", "then", "too", "very",
    "dont", "im", "ive", "thats", "yeah", "oh",
})

# Scoring weights
W_NGRAM = 0.50
W_OPENER = 0.25
W_STRUCT = 0.25

# Thresholds
NGRAM_OVERLAP_THRESHOLD = 0.30
MAX_OPENER_REPEATS = 2


@dataclass
class DiversityResult:
    """Result from diversity scoring."""

    score: float  # 0.0 (bad) to 1.0 (good)
    passed: bool  # score >= threshold
    problems: list[str] = field(default_factory=list)
    repeated_ngrams: list[str] = field(default_factory=list)
    formulaic_opener: str | None = None
    structural_score: float = 1.0


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stops."""
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    words = text.split()
    return [w for w in words if w not in STOP_WORDS]


def extract_ngrams(
    words: list[str], n: int
) -> list[tuple[str, ...]]:
    """Return contiguous n-grams from word list."""
    if len(words) < n:
        return []
    return [
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]


def _build_history_ngrams(
    history: list[Message],
    ngram_min: int = 3,
    ngram_max: int = 6,
) -> set[tuple[str, ...]]:
    """Extract all n-grams from history messages."""
    ngrams: set[tuple[str, ...]] = set()
    for msg in history:
        words = tokenize(msg.content)
        for n in range(ngram_min, ngram_max + 1):
            ngrams.update(extract_ngrams(words, n))
    return ngrams


def _check_ngram_overlap(
    response: str,
    history_ngrams: set[tuple[str, ...]],
    ngram_min: int = 3,
    ngram_max: int = 6,
) -> tuple[float, list[str]]:
    """Check response n-grams against history.

    Returns (overlap_ratio, list of repeated phrases).
    """
    words = tokenize(response)
    if len(words) < ngram_min:
        return 0.0, []
    response_ngrams: list[tuple[str, ...]] = []
    for n in range(ngram_min, ngram_max + 1):
        response_ngrams.extend(extract_ngrams(words, n))
    if not response_ngrams:
        return 0.0, []
    overlapping = [
        ng for ng in response_ngrams
        if ng in history_ngrams
    ]
    ratio = len(overlapping) / len(response_ngrams)
    phrases = list({" ".join(ng) for ng in overlapping})
    return ratio, phrases[:5]


def _get_opener(text: str) -> str:
    """Extract first 6 words as opener signature."""
    words = text.lower().split()[:6]
    opener = " ".join(words)
    return re.sub(r"[^\w\s]", "", opener).strip()


def _build_opener_counts(
    history: list[Message],
    agent_name: str,
) -> Counter:
    """Count openers used by this agent in history."""
    counts: Counter = Counter()
    for msg in history:
        if msg.agent_name == agent_name:
            opener = _get_opener(msg.content)
            if opener:
                counts[opener] += 1
    return counts


def _check_formulaic_opener(
    response: str,
    opener_counts: Counter,
) -> str | None:
    """Check if response opener was used too often.

    Returns the opener if it exceeds threshold, else None.
    """
    opener = _get_opener(response)
    if not opener:
        return None
    if opener_counts.get(opener, 0) >= MAX_OPENER_REPEATS:
        return opener
    return None


def _compute_structural_features(text: str) -> dict:
    """Extract structural features from text."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return {
        "sentence_count": len(sentences),
        "question_marks": text.count("?"),
        "commas": text.count(","),
        "exclamations": text.count("!"),
        "word_count": len(text.split()),
    }


def _check_structural_similarity(
    response: str,
    history: list[Message],
    agent_name: str,
) -> float:
    """Compare structure against last 3 agent responses.

    Returns similarity score 0-1 (1 = very similar = bad).
    """
    agent_msgs = [
        m for m in history
        if m.agent_name == agent_name
    ][-3:]
    if not agent_msgs:
        return 0.0
    resp_features = _compute_structural_features(response)
    similarities = []
    for msg in agent_msgs:
        msg_features = _compute_structural_features(
            msg.content
        )
        sim = _feature_similarity(resp_features, msg_features)
        similarities.append(sim)
    return sum(similarities) / len(similarities)


def _feature_similarity(f1: dict, f2: dict) -> float:
    """Compute similarity between two feature dicts."""
    matches = 0
    total = len(f1)
    for key in f1:
        if f1[key] == f2.get(key):
            matches += 1
    return matches / total if total > 0 else 0.0


def compute_diversity_score(
    response: str,
    history: list[Message],
    agent_name: str,
    window: int = 10,
    threshold: float = 0.6,
    ngram_min: int = 3,
    ngram_max: int = 6,
) -> DiversityResult:
    """Check response against recent history.

    Returns DiversityResult with score (0-1) and problems.
    Score of 1.0 means fully diverse, 0.0 means fully
    repetitive.
    """
    recent = history[-window:] if history else []
    problems: list[str] = []
    repeated_ngrams: list[str] = []
    formulaic_opener: str | None = None
    # 1. N-gram overlap detection
    history_ngrams = _build_history_ngrams(
        recent, ngram_min, ngram_max
    )
    overlap_ratio, phrases = _check_ngram_overlap(
        response, history_ngrams, ngram_min, ngram_max
    )
    if overlap_ratio > NGRAM_OVERLAP_THRESHOLD:
        problems.append(
            f"Repeated phrases: {', '.join(phrases[:3])}"
        )
        repeated_ngrams = phrases
    ngram_score = 1.0 - min(1.0, overlap_ratio / 0.5)
    # 2. Formulaic opener detection
    opener_counts = _build_opener_counts(recent, agent_name)
    formulaic_opener = _check_formulaic_opener(
        response, opener_counts
    )
    if formulaic_opener:
        problems.append(
            f"Repeated opener: \"{formulaic_opener[:40]}\""
        )
    opener_score = 0.0 if formulaic_opener else 1.0
    # 3. Structural similarity detection
    struct_sim = _check_structural_similarity(
        response, recent, agent_name
    )
    if struct_sim > 0.7:
        problems.append("Similar structure to recent msgs")
    struct_score = 1.0 - struct_sim
    # 4. Aggregate weighted score
    final_score = (
        W_NGRAM * ngram_score +
        W_OPENER * opener_score +
        W_STRUCT * struct_score
    )
    return DiversityResult(
        score=round(final_score, 3),
        passed=final_score >= threshold,
        problems=problems,
        repeated_ngrams=repeated_ngrams,
        formulaic_opener=formulaic_opener,
        structural_score=struct_score,
    )
