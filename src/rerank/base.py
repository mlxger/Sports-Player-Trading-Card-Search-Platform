from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from models.contracts import SearchHit


class IdentityReranker:
    """Explicit fallback until a trained ranking feature contract is supplied."""

    def rerank(self, image: Image.Image, hits: Sequence[SearchHit]) -> list[SearchHit]:
        return list(hits)


class LightGBMReranker:
    """Optional LightGBM reranker for the migrated three-feature contract."""

    feature_names = ("initial_score", "hog_similarity", "border_similarity")

    def __init__(self, model_path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("install the 'ranking' extra to use LightGBM reranking") from exc
        if not model_path.is_file():
            raise FileNotFoundError(f"ranking model not found at {model_path}")
        self._model: Any = joblib.load(model_path)

    def rerank(self, image: Image.Image, hits: Sequence[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        rows = [self._features(hit) for hit in hits]
        scores = np.asarray(self._model.predict(np.asarray(rows, dtype=np.float32))).reshape(-1)
        ranked = sorted(
            zip(hits, scores, strict=True), key=lambda item: float(item[1]), reverse=True
        )
        return [
            replace(
                hit,
                score=float(score),
                rank=rank,
                metadata={**hit.metadata, "vector_score": hit.score},
            )
            for rank, (hit, score) in enumerate(ranked, start=1)
        ]

    @classmethod
    def _features(cls, hit: SearchHit) -> list[float]:
        metadata = hit.metadata
        return [
            float(hit.score),
            _number(metadata.get("hog_similarity")),
            _number(metadata.get("border_similarity")),
        ]


def _number(value: object) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
