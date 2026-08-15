from __future__ import annotations

from typing import Any, Protocol


class VisionModelClient(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...


class OllamaVisionClient:
    """Lazy Ollama adapter that keeps the OCR core independent of the SDK."""

    def __init__(self, *, host: str, timeout: float) -> None:
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError("install the 'parsing' extra to use Ollama OCR") from exc
        self._client = Client(host=host, timeout=timeout)

    def chat(self, **kwargs: Any) -> Any:
        return self._client.chat(**kwargs)
