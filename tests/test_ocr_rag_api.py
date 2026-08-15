from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from router.api import create_app
from settings import Settings


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 18), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeOcrService:
    def recognize_single(self, image_data: bytes, **kwargs: Any) -> dict[str, Any]:
        assert image_data
        assert kwargs == {"fields": ["name"], "rag_fields": ["brand"], "graded": False}
        return {"name": "姚明", "status": "0", "rag_match": {"brand": "Topps"}}


class FakeKnowledgeBase:
    def count(self) -> int:
        return 1

    def search(
        self, query: str, *, n_results: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        assert query == "姚明 Topps"
        assert n_results == 2
        assert filters == {"运动类型英文": "basketball"}
        return [{"_id": "1", "球员中文": "姚明", "_score": 0.98}]


class FakeKnowledgeService:
    def __init__(self) -> None:
        self.default = FakeKnowledgeBase()

    def get(self, name: str | None = None) -> FakeKnowledgeBase:
        assert name is None
        return self.default


def test_ocr_single_endpoint_with_injected_service() -> None:
    settings = Settings(retrieval_enabled=False, ocr_enabled=True, rag_enabled=False)
    app = create_app(settings, ocr_factory=lambda _settings, _knowledge: FakeOcrService())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ocr/recognize/single",
            files={"image": ("card.png", make_png(), "image/png")},
            data={"fields": '["name"]', "rag": '["brand"]'},
        )

    assert response.status_code == 200
    assert response.json()["rag_match"] == {"brand": "Topps"}


def test_rag_semantic_search_endpoint_with_injected_registry() -> None:
    settings = Settings(retrieval_enabled=False, ocr_enabled=False, rag_enabled=True)
    app = create_app(settings, knowledge_factory=lambda _settings: FakeKnowledgeService())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "姚明 Topps",
                "n_results": 2,
                "filters": {"运动类型英文": "basketball"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {"results": [{"_id": "1", "球员中文": "姚明", "_score": 0.98}]}


def test_dependency_diagnostics_endpoint(monkeypatch) -> None:
    from models.ocr_model.health import OllamaStatus

    monkeypatch.setattr(
        "router.system.check_ollama",
        lambda **kwargs: OllamaStatus(
            reachable=True,
            version="0.6.1",
            configured_model=kwargs["model_name"],
            model_available=True,
            models=(kwargs["model_name"],),
        ),
    )
    settings = Settings(retrieval_enabled=False, ocr_enabled=False, rag_enabled=False)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/v1/system/dependencies")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ollama"]["version"] == "0.6.1"
    assert payload["ollama"]["model_available"] is True
    assert payload["embedding"]["downloads_allowed"] is False
