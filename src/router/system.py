from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from models.ocr_model import check_ollama

router = APIRouter(prefix="/system", tags=["System diagnostics"])


def run_doctor() -> None:
    """CLI diagnostic for embedding and Ollama readiness."""
    from settings import get_settings

    settings = get_settings()
    ollama = check_ollama(
        base_url=settings.ollama_url,
        model_name=settings.ocr_model_name,
        timeout=settings.dependency_check_timeout,
    )
    report = _dependency_report(settings, ollama.as_dict())
    print(json.dumps(report, ensure_ascii=False, indent=2))


@router.get("/dependencies")
async def dependency_status(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    ollama = await run_in_threadpool(
        check_ollama,
        base_url=settings.ollama_url,
        model_name=settings.ocr_model_name,
        timeout=settings.dependency_check_timeout,
    )
    return _dependency_report(settings, ollama.as_dict())


def _dependency_report(settings, ollama: dict[str, object]) -> dict[str, object]:
    embedding_path = Path(settings.rag_embedding_model)
    yolo_path = settings.yolo_model_path
    ranker_path = settings.ranker_model_path
    return {
        "embedding": {
            "configured_model": settings.rag_embedding_model,
            "local_path": str(embedding_path),
            "available_locally": embedding_path.exists(),
            "downloads_allowed": settings.allow_model_downloads,
        },
        "retrieval_models": {
            "cache_directory": str(settings.model_cache_dir),
            "cache_available": settings.model_cache_dir.exists(),
            "downloads_allowed": settings.allow_model_downloads,
        },
        "yolo": {
            "enabled": settings.preprocessing_mode == "yolo",
            "model_path": str(yolo_path) if yolo_path else None,
            "model_available": bool(yolo_path and yolo_path.is_file()),
        },
        "ranker": {
            "enabled": settings.ranker_enabled,
            "model_path": str(ranker_path),
            "model_available": ranker_path.is_file(),
        },
        "ollama": ollama,
    }
