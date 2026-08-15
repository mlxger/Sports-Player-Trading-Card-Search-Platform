from __future__ import annotations

from dataclasses import replace

from models.contracts import (
    CandidateReranker,
    CardRecord,
    ImageEmbedder,
    ImagePreprocessor,
    SearchHit,
    SearchRequest,
    VectorStore,
)
from preprocessing import load_image


class ImageRetrievalService:
    def __init__(
        self,
        *,
        preprocessor: ImagePreprocessor,
        embedder: ImageEmbedder,
        vector_store: VectorStore,
        max_upload_bytes: int,
        max_top_k: int = 100,
        reranker: CandidateReranker | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._embedder = embedder
        self._vector_store = vector_store
        self._max_upload_bytes = max_upload_bytes
        self._max_top_k = max_top_k
        self._reranker = reranker

    def search(
        self,
        image_data: bytes,
        request: SearchRequest,
        *,
        rerank: bool = False,
    ) -> list[SearchHit]:
        if not 1 <= request.top_k <= self._max_top_k:
            raise ValueError(f"top_k must be between 1 and {self._max_top_k}")
        image = load_image(image_data, max_bytes=self._max_upload_bytes)
        normalized = self._preprocessor.process(image)
        vector = self._embedder.encode(normalized, real_photo=request.real_photo)
        if len(vector) != self._embedder.dimension:
            raise RuntimeError("embedder returned an inconsistent vector dimension")
        hits = self._vector_store.search(vector, request)
        if rerank:
            if self._reranker is None:
                raise ValueError("reranking was requested but no ranking model is configured")
            hits = self._reranker.rerank(normalized, hits)
        return [replace(hit, rank=rank) for rank, hit in enumerate(hits, start=1)]

    def close(self) -> None:
        self._vector_store.close()


class ImageIndexingService:
    def __init__(
        self,
        *,
        preprocessor: ImagePreprocessor,
        embedder: ImageEmbedder,
        vector_store: VectorStore,
        max_upload_bytes: int,
    ) -> None:
        self._preprocessor = preprocessor
        self._embedder = embedder
        self._vector_store = vector_store
        self._max_upload_bytes = max_upload_bytes

    def index(
        self,
        image_data: bytes,
        *,
        image_id: str,
        tool_id: str,
        player_id: str = "",
        status: int = 0,
        real_photo: bool = False,
    ) -> int:
        record = self.prepare_record(
            image_data,
            image_id=image_id,
            tool_id=tool_id,
            player_id=player_id,
            status=status,
            real_photo=real_photo,
        )
        return self._vector_store.insert([record])

    def prepare_record(
        self,
        image_data: bytes,
        *,
        image_id: str,
        tool_id: str,
        player_id: str = "",
        status: int = 0,
        real_photo: bool = False,
    ) -> CardRecord:
        image = load_image(image_data, max_bytes=self._max_upload_bytes)
        normalized = self._preprocessor.process(image)
        vector = self._embedder.encode(normalized, real_photo=real_photo)
        return CardRecord(
            image_id=image_id,
            tool_id=tool_id,
            player_id=player_id,
            status=status,
            embedding=vector.tolist(),
        )

    def insert_records(self, records: list[CardRecord]) -> int:
        return self._vector_store.insert(records)

    def close(self) -> None:
        self._vector_store.close()
