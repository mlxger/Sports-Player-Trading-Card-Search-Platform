from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from models.contracts import CardRecord
from retrieval import MilvusVectorStore
from settings import Settings

router = APIRouter(prefix="/milvus", tags=["Milvus vector management"])
StoreFactory = Callable[[Settings], MilvusVectorStore]


class MilvusRecord(BaseModel):
    image_id: str = Field(min_length=1, max_length=128)
    tool_id: str = Field(default="", max_length=512)
    player_id: str = Field(default="", max_length=512)
    status: int = 0
    embedding: list[float] = Field(min_length=1)


class MilvusBatch(BaseModel):
    records: list[MilvusRecord] = Field(min_length=1, max_length=1000)


class PrimaryKeyBatch(BaseModel):
    primary_keys: list[int] = Field(min_length=1, max_length=1000)


class MilvusMutationResponse(BaseModel):
    code: int = 0
    message: str = "success"
    count: int


def build_store(settings: Settings) -> MilvusVectorStore:
    return MilvusVectorStore(settings)


def _store(request: Request, factory: StoreFactory) -> MilvusVectorStore:
    settings: Settings = request.app.state.settings
    try:
        return factory(settings)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {exc}") from exc


@router.post("/collections/create", response_model=MilvusMutationResponse)
async def create_collection(request: Request) -> MilvusMutationResponse:
    settings: Settings = request.app.state.settings
    try:
        MilvusVectorStore.create_collection(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Milvus collection creation failed: {exc}"
        ) from exc
    return MilvusMutationResponse(count=0, message="collection ready")


@router.get("/count", response_model=MilvusMutationResponse)
async def count_records(request: Request) -> MilvusMutationResponse:
    store = _store(request, build_store)
    try:
        return MilvusMutationResponse(count=store.count())
    finally:
        store.close()


@router.post("/records", response_model=MilvusMutationResponse)
async def insert_record(record: MilvusRecord, request: Request) -> MilvusMutationResponse:
    store = _store(request, build_store)
    try:
        count = store.insert([_to_card_record(record)])
        return MilvusMutationResponse(count=count, message="record inserted")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        store.close()


@router.post("/records/batch", response_model=MilvusMutationResponse)
async def insert_records(batch: MilvusBatch, request: Request) -> MilvusMutationResponse:
    store = _store(request, build_store)
    try:
        count = store.insert([_to_card_record(item) for item in batch.records])
        return MilvusMutationResponse(count=count, message="records inserted")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        store.close()


@router.get("/records/{primary_key}")
async def get_record(primary_key: int, request: Request) -> dict[str, object]:
    store = _store(request, build_store)
    try:
        rows = store.get_by_ids([primary_key])
    finally:
        store.close()
    if not rows:
        raise HTTPException(status_code=404, detail="Milvus record not found")
    return rows[0]


@router.delete("/records", response_model=MilvusMutationResponse)
async def delete_records(batch: PrimaryKeyBatch, request: Request) -> MilvusMutationResponse:
    store = _store(request, build_store)
    try:
        count = store.delete_by_ids(batch.primary_keys)
        return MilvusMutationResponse(count=count, message="records deleted")
    finally:
        store.close()


def _to_card_record(record: MilvusRecord) -> CardRecord:
    return CardRecord(
        image_id=record.image_id,
        tool_id=record.tool_id,
        player_id=record.player_id,
        status=record.status,
        embedding=record.embedding,
    )
