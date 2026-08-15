from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    *,
    k: int,
) -> float:
    """Calculate query-level Recall@K for set-valued relevance labels."""
    _validate_inputs(rankings, relevant, k)
    if not rankings:
        return 0.0
    scores = [
        bool(set(ranking[:k]) & set(expected)) for ranking, expected in zip(rankings, relevant)
    ]
    return sum(scores) / len(scores)


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    *,
    k: int | None = None,
) -> float:
    """Calculate MRR, optionally truncating each ranking at ``k``."""
    if not rankings:
        return 0.0
    if len(rankings) != len(relevant):
        raise ValueError("rankings and relevant must have the same length")
    if k is not None and k < 1:
        raise ValueError("k must be positive")
    reciprocal_ranks = []
    for ranking, expected in zip(rankings, relevant):
        relevant_ids = set(expected)
        reciprocal = 0.0
        for index, item in enumerate(ranking[:k] if k else ranking, start=1):
            if item in relevant_ids:
                reciprocal = 1.0 / index
                break
        reciprocal_ranks.append(reciprocal)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    *,
    k: int,
) -> float:
    """Calculate binary-label NDCG@K."""
    _validate_inputs(rankings, relevant, k)
    if not rankings:
        return 0.0
    values = []
    for ranking, expected in zip(rankings, relevant):
        relevant_ids = set(expected)
        dcg = sum(
            1.0 / math.log2(index + 2)
            for index, item in enumerate(ranking[:k])
            if item in relevant_ids
        )
        ideal_hits = min(k, len(relevant_ids))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
        values.append(dcg / ideal if ideal else 0.0)
    return sum(values) / len(values)


def _validate_inputs(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    k: int,
) -> None:
    if len(rankings) != len(relevant):
        raise ValueError("rankings and relevant must have the same length")
    if k < 1:
        raise ValueError("k must be positive")
