from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from knowledge import CARD_FIELDS
from service.knowledge import KnowledgeService

router = APIRouter(prefix="/rag", tags=["RAG Knowledge Base"])


class BatchCards(BaseModel):
    cards: list[dict[str, Any]]
    overwrite: bool = False
    batch_size: int = Field(default=256, ge=1, le=1000)


class SemanticSearch(BaseModel):
    query: str
    n_results: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Any] | None = None


class FieldSearch(BaseModel):
    field: str
    value: str
    limit: int = Field(default=20, ge=1, le=1000)


class MultiFieldSearch(BaseModel):
    filters: dict[str, str]
    limit: int = Field(default=20, ge=1, le=1000)


@router.get("/fields")
async def list_fields() -> dict[str, Any]:
    return {"fields": list(CARD_FIELDS)}


@router.get("/count")
async def count(request: Request, collection: str | None = None) -> dict[str, Any]:
    knowledge = _registry(request).get(collection)
    return {
        "collection": collection or request.app.state.settings.chroma_collection,
        "count": await run_in_threadpool(knowledge.count),
    }


@router.post("/cards")
async def add_card(
    request: Request,
    card: dict[str, Any],
    overwrite: bool = Query(False),
    collection: str | None = None,
) -> dict[str, Any]:
    try:
        card_id = await run_in_threadpool(
            _registry(request).get(collection).add, card, overwrite=overwrite
        )
        return {"id": card_id}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/cards/batch")
async def add_cards_batch(
    request: Request,
    payload: BatchCards,
    collection: str | None = None,
) -> dict[str, Any]:
    try:
        ids = await run_in_threadpool(
            _registry(request).get(collection).add_batch,
            payload.cards,
            overwrite=payload.overwrite,
            batch_size=payload.batch_size,
        )
        return {"ids": ids, "count": len(ids)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/cards/{card_id}")
async def get_card(card_id: str, request: Request, collection: str | None = None) -> dict[str, Any]:
    card = await run_in_threadpool(_registry(request).get(collection).get, card_id)
    if card is None:
        raise HTTPException(status_code=404, detail="card not found")
    return card


@router.put("/cards/{card_id}")
async def update_card(
    card_id: str,
    card: dict[str, Any],
    request: Request,
    collection: str | None = None,
) -> dict[str, Any]:
    try:
        updated = await run_in_threadpool(_registry(request).get(collection).update, card_id, card)
        return {"id": updated}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/cards/{card_id}")
async def delete_card(
    card_id: str, request: Request, collection: str | None = None
) -> dict[str, Any]:
    await run_in_threadpool(_registry(request).get(collection).remove, card_id)
    return {"deleted": card_id}


@router.post("/search")
async def semantic_search(
    request: Request,
    payload: SemanticSearch,
    collection: str | None = None,
) -> dict[str, Any]:
    try:
        results = await run_in_threadpool(
            _registry(request).get(collection).search,
            payload.query,
            n_results=payload.n_results,
            filters=payload.filters,
        )
        return {"results": results}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search/field")
async def field_search(
    request: Request,
    payload: FieldSearch,
    collection: str | None = None,
) -> dict[str, Any]:
    try:
        results = await run_in_threadpool(
            _registry(request).get(collection).search_field,
            payload.field,
            payload.value,
            limit=payload.limit,
        )
        return {"results": results}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search/multi")
async def multi_search(
    request: Request,
    payload: MultiFieldSearch,
    collection: str | None = None,
) -> dict[str, Any]:
    try:
        results = await run_in_threadpool(
            _registry(request).get(collection).search_multi,
            payload.filters,
            limit=payload.limit,
        )
        return {"results": results}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/import/excel")
async def import_excel(
    request: Request,
    file: UploadFile = File(description="Excel knowledge-base file"),
    collection: str | None = None,
    overwrite: bool = Query(True),
    sheet_index: int = Query(0, ge=0),
    batch_size: int = Query(256, ge=1, le=1000),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="only .xlsx and .xls are supported")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await file.read())
        imported = await run_in_threadpool(
            _registry(request).get(collection).import_excel,
            temp_path,
            sheet_index=sheet_index,
            batch_size=batch_size,
            overwrite=overwrite,
        )
        return {"imported": imported}
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _registry(request: Request) -> KnowledgeService:
    service = getattr(request.app.state, "knowledge_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is disabled")
    return service
