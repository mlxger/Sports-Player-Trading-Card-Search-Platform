import math

import pytest

from eval import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_retrieval_metrics() -> None:
    rankings = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [{"b"}, {"z"}]
    assert recall_at_k(rankings, relevant, k=2) == 0.5
    assert mean_reciprocal_rank(rankings, relevant, k=3) == (0.5 + 1 / 3) / 2
    assert ndcg_at_k(rankings, relevant, k=2) == pytest.approx(1 / (2 * math.log2(3)))
