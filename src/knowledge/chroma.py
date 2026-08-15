from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from models.rag_model import BilingualEmbeddingFunction

CARD_FIELDS: tuple[str, ...] = (
    "ids",
    "图片名字",
    "运动类型",
    "运动类型英文",
    "发行商中文",
    "发行商英文",
    "大系列中文简称",
    "大系列英文简称",
    "赛季",
    "版权",
    "小系列中文",
    "小系列英文",
    "俱乐部中文",
    "俱乐部英文",
    "俱乐部别名",
    "球员中文",
    "球员英文",
)
_EMPTY_VALUES = {"", "nan", "NaN", "None", "none", "null"}


class ChromaCardKnowledgeBase:
    """Thread-safe ChromaDB repository for bilingual trading-card metadata."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
        device: str = "auto",
        allow_model_downloads: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("install the 'parsing' extra to use ChromaDB") from exc
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._embedding = BilingualEmbeddingFunction(
            embedding_model, device=device, allow_downloads=allow_model_downloads
        )
        self._collection_name = collection_name
        self._write_lock = threading.Lock()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def fields(self) -> tuple[str, ...]:
        return CARD_FIELDS

    @staticmethod
    def list_collection_names(persist_directory: Path) -> list[str]:
        """Return collection names without loading an embedding model."""
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("install the 'parsing' extra to use ChromaDB") from exc
        if not persist_directory.exists():
            return []
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        collections = client.list_collections()
        return sorted(item if isinstance(item, str) else item.name for item in collections)

    def list_cards(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return a bounded metadata page for administrative viewers."""
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset cannot be negative")
        raw = self._collection.get(limit=min(limit, 1000), offset=offset, include=["metadatas"])
        return _format_get(raw)

    def add(self, card: dict[str, Any], *, overwrite: bool = False) -> str:
        card_id = _card_id(card)
        with self._write_lock:
            if not overwrite and self.get(card_id) is not None:
                raise ValueError(f"card id '{card_id}' already exists")
            self._collection.upsert(
                ids=[card_id], documents=[_build_document(card)], metadatas=[_metadata(card)]
            )
        return card_id

    def add_batch(
        self,
        cards: Sequence[dict[str, Any]],
        *,
        overwrite: bool = False,
        batch_size: int = 256,
    ) -> list[str]:
        if not cards:
            return []
        ids = [_card_id(card) for card in cards]
        duplicates = sorted({card_id for card_id in ids if ids.count(card_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate ids in batch: {duplicates}")
        if not overwrite:
            existing = self._collection.get(ids=ids, include=[]).get("ids") or []
            if existing:
                raise ValueError(f"card ids already exist: {existing}")
        documents = [_build_document(card) for card in cards]
        embeddings = self._embedding(documents)
        with self._write_lock:
            for start in range(0, len(cards), batch_size):
                end = min(start + batch_size, len(cards))
                self._collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=[_metadata(card) for card in cards[start:end]],
                )
        return ids

    def update(self, card_id: str, card: dict[str, Any]) -> str:
        if self.get(card_id) is None:
            raise ValueError(f"card id '{card_id}' does not exist")
        updated = dict(card)
        updated["ids"] = card_id
        with self._write_lock:
            self._collection.update(
                ids=[card_id],
                documents=[_build_document(updated)],
                metadatas=[_metadata(updated)],
            )
        return card_id

    def get(self, card_id: str) -> dict[str, Any] | None:
        records = _format_get(self._collection.get(ids=[card_id], include=["metadatas"]))
        return records[0] if records else None

    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("semantic query cannot be empty")
        total = self.count()
        if total == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, total),
            "include": ["metadatas", "distances"],
        }
        where = _where(filters or {})
        if where:
            kwargs["where"] = where
        return _format_query(self._collection.query(**kwargs))

    def search_field(self, field: str, value: str, *, limit: int = 20):
        _validate_field(field)
        raw = self._collection.get(
            where={field: {"$eq": _clean(value)}}, limit=limit, include=["metadatas"]
        )
        return _format_get(raw)

    def search_multi(self, filters: dict[str, str], *, limit: int = 20):
        if not filters:
            raise ValueError("filters cannot be empty")
        raw = self._collection.get(where=_where(filters), limit=limit, include=["metadatas"])
        return _format_get(raw)

    def remove(self, card_id: str) -> None:
        with self._write_lock:
            self._collection.delete(ids=[card_id])

    def remove_batch(self, card_ids: Sequence[str]) -> None:
        with self._write_lock:
            self._collection.delete(ids=list(card_ids))

    def count(self) -> int:
        return int(self._collection.count())

    def import_excel(
        self,
        path: Path,
        *,
        sheet_index: int = 0,
        batch_size: int = 256,
        overwrite: bool = True,
    ) -> int:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and openpyxl are required for Excel import") from exc
        frame = pd.read_excel(path, sheet_name=sheet_index, header=0, dtype=str)
        if "ids" not in frame.columns:
            raise ValueError("Excel file must contain an 'ids' column")
        columns = [field for field in CARD_FIELDS if field in frame.columns]
        cards = frame[columns].where(frame.notna(), other=None).to_dict(orient="records")
        self.add_batch(cards, overwrite=overwrite, batch_size=batch_size)
        return len(cards)

    def drop(self) -> None:
        self._client.delete_collection(name=self._collection_name)


def build_ocr_query(extracted: dict[str, Any]) -> str:
    parts: list[str] = []
    if extracted.get("name"):
        parts.append(f"球员 player: {extracted['name']}")
    if extracted.get("brand"):
        parts.append(f"发行商 brand: {extracted['brand']}")
    if extracted.get("series"):
        parts.append(f"大系列 series: {extracted['series']}")
    return " ".join(parts)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in _EMPTY_VALUES else text


def _card_id(card: dict[str, Any]) -> str:
    card_id = _clean(card.get("ids"))
    if not card_id:
        raise ValueError("card metadata must contain a non-empty 'ids' field")
    try:
        numeric = float(card_id)
        if numeric == int(numeric):
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return card_id


def _build_document(card: dict[str, Any]) -> str:
    specs = (
        ("球员 player", ("球员中文", "球员英文")),
        ("运动类型 sport", ("运动类型", "运动类型英文")),
        ("发行商 brand", ("发行商中文", "发行商英文")),
        ("大系列 series", ("大系列中文简称", "大系列英文简称")),
        ("赛季 season", ("赛季",)),
        ("版权 license", ("版权",)),
        ("小系列 sub-series", ("小系列中文", "小系列英文")),
        ("俱乐部 club", ("俱乐部中文", "俱乐部英文", "俱乐部别名")),
        ("图片名字 picture-name", ("图片名字",)),
    )
    lines = []
    for label, fields in specs:
        values = " ".join(filter(None, (_clean(card.get(field)) for field in fields)))
        if values:
            lines.append(f"{label}: {values}")
    return "\n".join(lines) or "unknown card"


def _metadata(card: dict[str, Any]) -> dict[str, str]:
    return {field: _clean(card.get(field)) for field in CARD_FIELDS}


def _restore(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value if value != "" else None for key, value in metadata.items()}


def _format_get(raw: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for card_id, metadata in zip(raw.get("ids") or [], raw.get("metadatas") or []):
        record = _restore(metadata or {})
        record["_id"] = card_id
        output.append(record)
    return output


def _format_query(raw: dict[str, Any]) -> list[dict[str, Any]]:
    ids = (raw.get("ids") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    output = []
    for card_id, metadata, distance in zip(ids, metadatas, distances):
        record = _restore(metadata or {})
        record["_id"] = card_id
        record["_score"] = round(1.0 - float(distance), 4)
        output.append(record)
    return output


def _validate_field(field: str) -> None:
    if field not in CARD_FIELDS:
        raise ValueError(f"unknown card field: {field}")


def _where(filters: dict[str, Any]) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses = []
    for field, value in filters.items():
        _validate_field(field)
        clauses.append({field: value if isinstance(value, dict) else {"$eq": _clean(value)}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
