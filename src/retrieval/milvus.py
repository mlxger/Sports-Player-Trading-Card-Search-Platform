from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from models.contracts import CardRecord, SearchHit, SearchRequest
from settings import Settings

LOGGER = logging.getLogger(__name__)


def build_filter_expression(request: SearchRequest) -> str | None:
    clauses: list[str] = []
    if request.status is not None:
        clauses.append(f"status == {int(request.status)}")
    if request.player_ids:
        clauses.append(_in_expression("player_id", request.player_ids))
    if request.tool_ids:
        clauses.append(_in_expression("tool_id", request.tool_ids))
    return " && ".join(clauses) or None


def _in_expression(field: str, values: Sequence[str]) -> str:
    normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    encoded = ", ".join(json.dumps(value, ensure_ascii=False) for value in normalized)
    return f"{field} in [{encoded}]"


class MilvusVectorStore:
    """A small Milvus adapter with no import-time connection or collection mutation."""

    def __init__(self, settings: Settings) -> None:
        try:
            from pymilvus import Collection, connections
        except ImportError as exc:
            raise RuntimeError("install the 'retrieval' extra to use Milvus") from exc

        self._settings = settings
        self._connections = connections
        connect_args: dict[str, str] = {
            "alias": settings.milvus_alias,
            "uri": settings.milvus_uri,
        }
        if settings.milvus_token:
            connect_args["token"] = settings.milvus_token
        connections.connect(**connect_args)
        self._collection = Collection(
            settings.milvus_collection,
            using=settings.milvus_alias,
        )
        self._validate_dimension()
        self._collection.load()

    @classmethod
    def create_collection(cls, settings: Settings) -> None:
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise RuntimeError("install the 'retrieval' extra to use Milvus") from exc

        connect_args: dict[str, str] = {
            "alias": settings.milvus_alias,
            "uri": settings.milvus_uri,
        }
        if settings.milvus_token:
            connect_args["token"] = settings.milvus_token
        connections.connect(**connect_args)
        try:
            if utility.has_collection(settings.milvus_collection, using=settings.milvus_alias):
                LOGGER.info("Milvus collection %s already exists", settings.milvus_collection)
                return

            fields = [
                FieldSchema("pk_id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("image_id", DataType.VARCHAR, max_length=128),
                FieldSchema("tool_id", DataType.VARCHAR, max_length=512),
                FieldSchema("player_id", DataType.VARCHAR, max_length=512),
                FieldSchema("status", DataType.INT64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dimension),
            ]
            collection = Collection(
                settings.milvus_collection,
                schema=CollectionSchema(fields, description="Trading card multimodal embeddings"),
                using=settings.milvus_alias,
            )
            collection.create_index(
                "embedding",
                {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256},
                },
            )
            for field in ("image_id", "tool_id", "player_id", "status"):
                collection.create_index(field)
            LOGGER.info("Created Milvus collection %s", settings.milvus_collection)
        finally:
            connections.disconnect(settings.milvus_alias)

    def search(self, vector: Sequence[float], request: SearchRequest) -> list[SearchHit]:
        self._require_dimension(vector)
        results = self._collection.search(
            data=[list(vector)],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": {"ef": max(self._settings.milvus_search_ef, request.top_k)},
            },
            limit=request.top_k,
            expr=build_filter_expression(request),
            output_fields=["image_id", "tool_id", "player_id", "status"],
        )
        hits: list[SearchHit] = []
        for rank, hit in enumerate(results[0], start=1):
            entity = hit.entity
            hits.append(
                SearchHit(
                    primary_key=int(hit.id),
                    image_id=entity.get("image_id") or "",
                    tool_id=entity.get("tool_id") or "",
                    player_id=entity.get("player_id") or "",
                    status=int(entity.get("status") or 0),
                    score=float(hit.distance),
                    rank=rank,
                )
            )
        return hits

    def insert(self, records: Sequence[CardRecord]) -> int:
        if not records:
            return 0
        entities = []
        for record in records:
            self._require_dimension(record.embedding)
            entities.append(
                {
                    "image_id": record.image_id,
                    "tool_id": record.tool_id,
                    "player_id": record.player_id,
                    "status": record.status,
                    "embedding": list(record.embedding),
                }
            )
        self._collection.insert(entities)
        self._collection.flush()
        return len(entities)

    def count(self) -> int:
        """Return the number of entities currently stored in the collection."""
        return int(self._collection.num_entities)

    def get_by_ids(self, primary_keys: Sequence[int]) -> list[dict[str, object]]:
        """Read inserted entities by their Milvus auto-generated primary keys."""
        ids = [int(value) for value in primary_keys]
        if not ids:
            return []
        encoded = ", ".join(str(value) for value in ids)
        rows = self._collection.query(
            expr=f"pk_id in [{encoded}]",
            output_fields=["pk_id", "image_id", "tool_id", "player_id", "status"],
        )
        return [dict(row) for row in rows]

    def delete_by_ids(self, primary_keys: Sequence[int]) -> int:
        """Delete entities by primary key and flush the mutation."""
        ids = [int(value) for value in primary_keys]
        if not ids:
            return 0
        encoded = ", ".join(str(value) for value in ids)
        result = self._collection.delete(expr=f"pk_id in [{encoded}]")
        self._collection.flush()
        return int(getattr(result, "delete_count", len(ids)))

    def close(self) -> None:
        self._connections.disconnect(self._settings.milvus_alias)

    def _validate_dimension(self) -> None:
        field = next(
            (field for field in self._collection.schema.fields if field.name == "embedding"),
            None,
        )
        if field is None:
            raise RuntimeError("Milvus collection has no 'embedding' field")
        dimension = int(field.params["dim"])
        if dimension != self._settings.embedding_dimension:
            raise RuntimeError(
                f"Milvus embedding dimension is {dimension}; expected "
                f"{self._settings.embedding_dimension}"
            )

    def _require_dimension(self, vector: Sequence[float]) -> None:
        if len(vector) != self._settings.embedding_dimension:
            raise ValueError(
                f"embedding dimension is {len(vector)}; expected "
                f"{self._settings.embedding_dimension}"
            )
