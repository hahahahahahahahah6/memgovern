"""Decay scoring and lightweight text similarity. No dependencies."""

import math
import re

_STOPWORDS = frozenset(
    "a an the and or but if then of to in on at for with is are was were be been "
    "it its this that these those i you he she we they my your his her our their "
    "as by from not no do does did will would can could should have has had".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def decayed_score(importance: float, age_seconds: float, half_life: float) -> float:
    """Exponential decay: score halves every `half_life` seconds.

    importance in [0, 1] is the starting weight; time does the rest.
    """
    if half_life <= 0:
        return importance if age_seconds <= 0 else 0.0
    return importance * 0.5 ** (age_seconds / half_life)


def tokenize(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def jaccard(a: str, b: str) -> float:
    """Token-overlap similarity in [0, 1]. Cheap, deterministic, honest about limits."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def combined_score(decay: float, relevance: float) -> float:
    return 0.4 * decay + 0.6 * relevance
