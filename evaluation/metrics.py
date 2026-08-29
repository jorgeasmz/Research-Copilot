"""Ranking metrics, defined once so the benchmark and the corpus report agree."""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """Position of the first relevant result, inverted. Zero when none is found."""
    for rank, identifier in enumerate(ranked, start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """
    Normalised discounted cumulative gain with binary or graded judgements.

    The ideal ranking places every judged document in descending grade order, so
    a query whose relevant documents are all outside the corpus scores zero
    rather than being silently excluded from the average.
    """
    gains = [relevance.get(identifier, 0) for identifier in ranked[:k]]
    actual = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_grades, start=1))

    return actual / ideal if ideal else 0.0
