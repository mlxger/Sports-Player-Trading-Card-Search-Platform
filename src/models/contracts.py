from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class SearchRequest:
    top_k: int = 10
    status: int | None = None
    player_ids: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()
    real_photo: bool = True


@dataclass(frozen=True, slots=True)
class SearchHit:
    image_id: str
    tool_id: str
    score: float
    rank: int = 0
    player_id: str = ""
    status: int = 0
    primary_key: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CardRecord:
    image_id: str
    tool_id: str
    embedding: Sequence[float]
    player_id: str = ""
    status: int = 0


class ImagePreprocessor(Protocol):
    def process(self, image: Image.Image) -> Image.Image: ...


class ImageEmbedder(Protocol):
    dimension: int

    def encode(self, image: Image.Image, *, real_photo: bool = True) -> np.ndarray: ...


class VectorStore(Protocol):
    def search(self, vector: Sequence[float], request: SearchRequest) -> list[SearchHit]: ...

    def insert(self, records: Sequence[CardRecord]) -> int: ...

    def close(self) -> None: ...


class CandidateReranker(Protocol):
    def rerank(self, image: Image.Image, hits: Sequence[SearchHit]) -> list[SearchHit]: ...
