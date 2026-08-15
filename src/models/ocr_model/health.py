"""Read-only Ollama connectivity and model availability diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaStatus:
    reachable: bool
    version: str | None
    configured_model: str
    model_available: bool
    models: tuple[str, ...]
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def check_ollama(*, base_url: str, model_name: str, timeout: float = 3.0) -> OllamaStatus:
    """Check Ollama's version endpoint and whether the configured model is installed."""
    root = base_url.rstrip("/")
    try:
        version = _get_json(f"{root}/api/version", timeout).get("version")
        tags = _get_json(f"{root}/api/tags", timeout)
        names = tuple(
            str(item.get("name"))
            for item in tags.get("models", [])
            if isinstance(item, dict) and item.get("name")
        )
        return OllamaStatus(
            reachable=True,
            version=str(version) if version else None,
            configured_model=model_name,
            model_available=model_name in names,
            models=names,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return OllamaStatus(
            reachable=False,
            version=None,
            configured_model=model_name,
            model_available=False,
            models=(),
            error=str(exc),
        )


def _get_json(url: str, timeout: float) -> dict[str, object]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local URL
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected Ollama response from {url}")
    return payload
