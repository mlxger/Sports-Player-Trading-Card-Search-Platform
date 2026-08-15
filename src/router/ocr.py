from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from preprocessing import InvalidImageError
from service.ocr import CardOcrService

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.get("/fields")
async def list_ocr_fields() -> dict[str, Any]:
    from ocr_parsing import ALL_FIELDS

    return {"fields": list(ALL_FIELDS)}


@router.post("/recognize/single")
async def recognize_single(
    request: Request,
    image: UploadFile = File(description="Trading-card image"),
    fields: str | None = Form(default=None, description="JSON string array"),
    rag: str | None = Form(default=None, description="JSON string array"),
    graded: bool = Form(default=False),
) -> dict[str, Any]:
    service = _service(request)
    try:
        settings = request.app.state.settings
        content = await image.read(settings.max_upload_bytes + 1)
        async with request.app.state.ocr_semaphore:
            return await run_in_threadpool(
                service.recognize_single,
                content,
                fields=_parse_list("fields", fields),
                rag_fields=_parse_list("rag", rag),
                graded=graded,
            )
    except (InvalidImageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("single-card OCR failed")
        raise HTTPException(status_code=502, detail="OCR model request failed") from exc
    finally:
        await image.close()


@router.post("/recognize/double")
async def recognize_double(
    request: Request,
    front: UploadFile = File(description="Front image"),
    back: UploadFile = File(description="Back image"),
    fields: str | None = Form(default=None, description="JSON string array"),
    rag: str | None = Form(default=None, description="JSON string array"),
    graded: bool = Form(default=False),
) -> dict[str, Any]:
    service = _service(request)
    try:
        settings = request.app.state.settings
        front_data = await front.read(settings.max_upload_bytes + 1)
        back_data = await back.read(settings.max_upload_bytes + 1)
        async with request.app.state.ocr_semaphore:
            return await run_in_threadpool(
                service.recognize_double,
                front_data,
                back_data,
                fields=_parse_list("fields", fields),
                rag_fields=_parse_list("rag", rag),
                graded=graded,
            )
    except (InvalidImageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("double-card OCR failed")
        raise HTTPException(status_code=502, detail="OCR model request failed") from exc
    finally:
        await front.close()
        await back.close()


def _service(request: Request) -> CardOcrService:
    service = getattr(request.app.state, "ocr_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="OCR service is disabled")
    return service


def _parse_list(name: str, raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip() or raw.strip().lower() == "null":
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be a JSON array") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HTTPException(status_code=422, detail=f"{name} must be a string array")
    return value
