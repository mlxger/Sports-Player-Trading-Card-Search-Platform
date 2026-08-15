from __future__ import annotations

import json
from typing import Any

from models.ocr_model.health import check_ollama


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_check_ollama_reports_version_and_configured_model(monkeypatch) -> None:
    responses = iter(
        [FakeResponse({"version": "0.6.1"}), FakeResponse({"models": [{"name": "qwen3-vl:8b"}]})]
    )
    monkeypatch.setattr("models.ocr_model.health.urlopen", lambda *args, **kwargs: next(responses))

    status = check_ollama(
        base_url="http://127.0.0.1:11434/",
        model_name="qwen3-vl:8b",
    )

    assert status.reachable is True
    assert status.version == "0.6.1"
    assert status.model_available is True


def test_check_ollama_is_safe_when_server_is_unavailable(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object):
        raise OSError("connection refused")

    monkeypatch.setattr("models.ocr_model.health.urlopen", fail)
    status = check_ollama(base_url="http://127.0.0.1:11434", model_name="qwen3-vl:8b")

    assert status.reachable is False
    assert status.model_available is False
    assert "connection refused" in (status.error or "")
