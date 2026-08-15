from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

from knowledge import build_ocr_query
from models.ocr_model import OllamaVisionClient, VisionModelClient
from ocr_parsing import ALL_FIELDS, ModelConfig, extract_card_metadata
from preprocessing import load_image
from service.knowledge import KnowledgeService
from settings import Settings

_RAG_FIELDS = ("brand", "series")


class CardOcrService:
    def __init__(
        self,
        *,
        client: VisionModelClient,
        model_config: ModelConfig,
        max_upload_bytes: int,
        knowledge_service: KnowledgeService | None = None,
        rag_candidate_limit: int = 5,
    ) -> None:
        self._client = client
        self._model_config = model_config
        self._max_upload_bytes = max_upload_bytes
        self._knowledge_service = knowledge_service
        self._rag_candidate_limit = rag_candidate_limit

    def recognize_single(
        self,
        image_data: bytes,
        *,
        fields: Sequence[str] | None = None,
        rag_fields: Sequence[str] | None = None,
        graded: bool = False,
    ) -> dict[str, Any]:
        payload = _encode_image(image_data, self._max_upload_bytes)
        return self._extract(
            [payload],
            "Single trading-card image.",
            fields=fields,
            rag_fields=rag_fields,
            graded=graded,
            double_sided=False,
        )

    def recognize_double(
        self,
        front_data: bytes,
        back_data: bytes,
        *,
        fields: Sequence[str] | None = None,
        rag_fields: Sequence[str] | None = None,
        graded: bool = False,
    ) -> dict[str, Any]:
        payloads = [
            _encode_image(front_data, self._max_upload_bytes),
            _encode_image(back_data, self._max_upload_bytes),
        ]
        return self._extract(
            payloads,
            "Front and back images of the same trading card.",
            fields=fields,
            rag_fields=rag_fields,
            graded=graded,
            double_sided=True,
        )

    def _extract(
        self,
        payloads: list[str],
        context: str,
        *,
        fields: Sequence[str] | None,
        rag_fields: Sequence[str] | None,
        graded: bool,
        double_sided: bool,
    ) -> dict[str, Any]:
        requested = _validate_fields(fields)
        requested_rag = _validate_fields(rag_fields)
        use_rag = bool(requested_rag and self._knowledge_service is not None)
        active_fields = _active_fields(requested, requested_rag) if use_rag else requested
        result = extract_card_metadata(
            image_payloads=payloads,
            image_context=context,
            client=self._client,
            cfg=self._model_config,
            requested_fields=active_fields,
            is_double_sided=double_sided,
            is_graded_card=graded,
            pair_fields=True,
            use_rag=use_rag,
            rag_candidate_provider=self._rag_candidates if use_rag else None,
        )
        return _merge_requested_rag_fields(result, requested, requested_rag)

    def _rag_candidates(self, extracted: dict[str, Any]) -> list[dict[str, Any]]:
        query = build_ocr_query(extracted)
        if not query or self._knowledge_service is None:
            return []
        return self._knowledge_service.default.search(query, n_results=self._rag_candidate_limit)


def build_ocr_service(
    settings: Settings,
    knowledge_service: KnowledgeService | None = None,
) -> CardOcrService:
    return CardOcrService(
        client=OllamaVisionClient(
            host=settings.ollama_url,
            timeout=settings.ocr_model_timeout,
        ),
        model_config=ModelConfig(
            model_name=settings.ocr_model_name,
            temperature=settings.ocr_temperature,
            max_output_tokens=settings.ocr_max_output_tokens,
        ),
        max_upload_bytes=settings.max_upload_bytes,
        knowledge_service=knowledge_service,
        rag_candidate_limit=settings.rag_candidate_limit,
    )


def _encode_image(data: bytes, max_upload_bytes: int) -> str:
    image = load_image(data, max_bytes=max_upload_bytes)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _validate_fields(fields: Sequence[str] | None) -> list[str] | None:
    if fields is None:
        return None
    values = list(dict.fromkeys(fields))
    invalid = sorted(set(values) - set(ALL_FIELDS))
    if invalid:
        raise ValueError(f"unsupported OCR fields: {invalid}")
    return values


def _active_fields(
    fields: list[str] | None,
    rag_fields: list[str] | None,
) -> list[str]:
    active = set(fields or ALL_FIELDS)
    active.update(_RAG_FIELDS)
    active.update(rag_fields or ())
    return [field for field in ALL_FIELDS if field in active]


def _merge_requested_rag_fields(
    result: dict[str, Any],
    fields: list[str] | None,
    rag_fields: list[str] | None,
) -> dict[str, Any]:
    if not rag_fields:
        return result
    requested = set(fields or ALL_FIELDS)
    rag_match = result.get("rag_match")
    if not isinstance(rag_match, dict):
        rag_match = {}
    for field in rag_fields:
        if field not in _RAG_FIELDS and field in requested and field in result:
            rag_match[field] = result[field]
    result["rag_match"] = rag_match or None
    return result
