from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from models.contracts import CardRecord, SearchHit, SearchRequest
from preprocessing import PassthroughPreprocessor
from service import ImageRetrievalService


class FakeEmbedder:
    dimension = 4

    def __init__(self) -> None:
        self.real_photo: bool | None = None

    def encode(self, image: Image.Image, *, real_photo: bool = True) -> np.ndarray:
        self.real_photo = real_photo
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.request: SearchRequest | None = None
        self.vector: list[float] | None = None
        self.closed = False

    def search(self, vector, request: SearchRequest) -> list[SearchHit]:
        self.vector = list(vector)
        self.request = request
        return [SearchHit(image_id="image-1", tool_id="tool-1", score=0.97)]

    def insert(self, records: list[CardRecord]) -> int:
        return len(records)

    def close(self) -> None:
        self.closed = True


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 24), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_service_orchestrates_preprocessing_embedding_and_search() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    service = ImageRetrievalService(
        preprocessor=PassthroughPreprocessor(),
        embedder=embedder,
        vector_store=store,
        max_upload_bytes=10_000,
        max_top_k=20,
    )
    request = SearchRequest(top_k=5, status=4, player_ids=("p1",), real_photo=False)

    hits = service.search(make_png(), request)

    assert hits[0].rank == 1
    assert hits[0].score == pytest.approx(0.97)
    assert embedder.real_photo is False
    assert store.request == request
    assert store.vector == [1.0, 0.0, 0.0, 0.0]


def test_service_rejects_out_of_range_top_k() -> None:
    service = ImageRetrievalService(
        preprocessor=PassthroughPreprocessor(),
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        max_upload_bytes=10_000,
        max_top_k=20,
    )

    with pytest.raises(ValueError, match="top_k"):
        service.search(make_png(), SearchRequest(top_k=21))


def test_service_rejects_rerank_without_configured_model() -> None:
    service = ImageRetrievalService(
        preprocessor=PassthroughPreprocessor(),
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        max_upload_bytes=10_000,
    )

    with pytest.raises(ValueError, match="no ranking model"):
        service.search(make_png(), SearchRequest(top_k=5), rerank=True)
