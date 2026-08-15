from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.contracts import SearchRequest
from preprocessing import InvalidImageError
from service.bootstrap import build_retrieval_service
from service.knowledge import KnowledgeService, build_knowledge_service
from service.ocr import CardOcrService, build_ocr_service
from service.retrieval import ImageRetrievalService
from settings import Settings, get_settings

from .ocr import router as ocr_router
from .rag import router as rag_router
from .system import router as system_router

LOGGER = logging.getLogger(__name__)
ServiceFactory = Callable[[Settings], ImageRetrievalService]
KnowledgeFactory = Callable[[Settings], KnowledgeService]
OcrFactory = Callable[[Settings, KnowledgeService | None], CardOcrService]


class SearchResult(BaseModel):
    rank: int
    image_id: str
    tool_id: str
    player_id: str
    status: int
    similarity: float
    primary_key: int | None = None


class SearchData(BaseModel):
    results: list[SearchResult]


class SearchResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: SearchData


def create_app(
    settings: Settings | None = None,
    service_factory: ServiceFactory = build_retrieval_service,
    knowledge_factory: KnowledgeFactory = build_knowledge_service,
    ocr_factory: OcrFactory = build_ocr_service,
) -> FastAPI:
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=runtime_settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        app.state.settings = runtime_settings
        service = None
        knowledge = None
        ocr_service = None
        if runtime_settings.retrieval_enabled:
            service = await run_in_threadpool(service_factory, runtime_settings)
            app.state.retrieval_service = service
        if runtime_settings.rag_enabled:
            knowledge = await run_in_threadpool(knowledge_factory, runtime_settings)
            app.state.knowledge_service = knowledge
        if runtime_settings.ocr_enabled:
            ocr_service = await run_in_threadpool(ocr_factory, runtime_settings, knowledge)
            app.state.ocr_service = ocr_service
        app.state.search_semaphore = asyncio.Semaphore(runtime_settings.milvus_search_concurrency)
        app.state.ocr_semaphore = asyncio.Semaphore(runtime_settings.ocr_concurrency)
        try:
            yield
        finally:
            if service is not None:
                await run_in_threadpool(service.close)

    app = FastAPI(title=runtime_settings.app_name, version="1.0.0", lifespan=lifespan)
    if runtime_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=runtime_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "retrieval": "enabled" if runtime_settings.retrieval_enabled else "disabled",
            "ocr": "enabled" if runtime_settings.ocr_enabled else "disabled",
            "rag": "enabled" if runtime_settings.rag_enabled else "disabled",
            "ranker": "enabled" if runtime_settings.ranker_enabled else "disabled",
        }

    @app.post(
        f"{runtime_settings.api_prefix}/retrieval/search",
        response_model=SearchResponse,
    )
    async def search_cards(
        image: UploadFile = File(description="Trading card image"),
        top_k: int = Form(
            default=runtime_settings.default_top_k,
            ge=1,
            le=runtime_settings.max_top_k,
        ),
        status: int | None = Form(default=None),
        player_ids: str | None = Form(default=None),
        tool_ids: str | None = Form(default=None),
        real_photo: bool = Form(default=True),
        rerank: bool = Form(default=False),
    ) -> SearchResponse:
        try:
            content = await image.read(runtime_settings.max_upload_bytes + 1)
            request = SearchRequest(
                top_k=top_k,
                status=status,
                player_ids=_split_values(player_ids),
                tool_ids=_split_values(tool_ids),
                real_photo=real_photo,
            )
            service: ImageRetrievalService | None = getattr(app.state, "retrieval_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail="retrieval service is disabled")
            async with app.state.search_semaphore:
                hits = await run_in_threadpool(service.search, content, request, rerank=rerank)
        except (InvalidImageError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("card retrieval failed")
            raise HTTPException(status_code=503, detail="retrieval service unavailable") from exc
        finally:
            await image.close()

        return SearchResponse(
            data=SearchData(
                results=[
                    SearchResult(
                        rank=hit.rank,
                        image_id=hit.image_id,
                        tool_id=hit.tool_id,
                        player_id=hit.player_id,
                        status=hit.status,
                        similarity=hit.score,
                        primary_key=hit.primary_key,
                    )
                    for hit in hits
                ]
            )
        )

    app.include_router(ocr_router, prefix=runtime_settings.api_prefix)
    app.include_router(rag_router, prefix=runtime_settings.api_prefix)
    app.include_router(system_router, prefix=runtime_settings.api_prefix)
    return app


def _split_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "router.api:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )
