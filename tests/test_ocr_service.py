from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image

from ocr_parsing import ModelConfig, build_field_groups
from service.ocr import CardOcrService


class FakeVisionClient:
    def chat(self, **kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": '{"name": "姚明"}'}}


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.query = ""
        self.limit = 0

    def search(self, query: str, *, n_results: int) -> list[dict[str, Any]]:
        self.query = query
        self.limit = n_results
        return [{"球员中文": "姚明", "发行商英文": "Topps"}]


class FakeKnowledgeService:
    def __init__(self) -> None:
        self.default = FakeKnowledgeBase()


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 18), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def make_service(knowledge_service: Any = None) -> CardOcrService:
    return CardOcrService(
        client=FakeVisionClient(),
        model_config=ModelConfig(model_name="fake"),
        max_upload_bytes=1024 * 1024,
        knowledge_service=knowledge_service,
        rag_candidate_limit=3,
    )


def test_recognize_single_injects_rag_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    knowledge = FakeKnowledgeService()

    def fake_extract(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        candidates = kwargs["rag_candidate_provider"](
            {"name": "姚明", "brand": "Topps", "series": "Chrome"}
        )
        assert candidates[0]["球员中文"] == "姚明"
        return {"name": "姚明", "brand": "Topps", "series": "Chrome", "rag_match": {}}

    monkeypatch.setattr("service.ocr.extract_card_metadata", fake_extract)
    result = make_service(knowledge).recognize_single(
        make_png(), fields=["name"], rag_fields=["brand"]
    )

    assert captured["requested_fields"] == ["name", "brand", "series"]
    assert captured["use_rag"] is True
    assert knowledge.default.query == (
        "球员 player: 姚明 发行商 brand: Topps 大系列 series: Chrome"
    )
    assert knowledge.default.limit == 3
    assert result["name"] == "姚明"


def test_recognize_single_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unsupported OCR fields"):
        make_service().recognize_single(make_png(), fields=["unknown"])


def test_recognize_single_runs_with_fake_vision_client() -> None:
    result = make_service().recognize_single(make_png(), fields=["name"])
    assert result["name"] == "姚明"
    assert result["status"] == "0"
    assert result["rag_match"] is None


def test_build_field_groups_keeps_standalone_and_limited_order() -> None:
    groups = build_field_groups(["name", "brand", "limited_edition", "information", "card_number"])
    assert groups == [
        ["name", "brand"],
        ["limited_edition"],
        ["information"],
        ["card_number"],
    ]
