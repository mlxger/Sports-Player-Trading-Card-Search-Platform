from __future__ import annotations

from pathlib import Path
from typing import Any


class BilingualEmbeddingFunction:
    """Chroma-compatible SentenceTransformer embedding function."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        allow_downloads: bool = False,
        batch_size: int | None = None,
    ) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the 'parsing' extra to use ChromaDB RAG") from exc
        model_path = Path(model_name)
        if not allow_downloads and not model_path.exists():
            raise FileNotFoundError(
                f"RAG embedding model not found at {model_path}; "
                "enable downloads or populate the model directory"
            )
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self._batch_size = batch_size or _default_batch_size(torch, resolved_device)

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(input),
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    @staticmethod
    def name() -> str:
        return "bilingual-sentence-transformer"

    def get_config(self) -> dict[str, Any]:
        return {"batch_size": self._batch_size}


def _default_batch_size(torch_module: Any, device: str) -> int:
    if not device.startswith("cuda"):
        return 32
    memory_gb = torch_module.cuda.get_device_properties(0).total_memory / 1e9
    if memory_gb >= 16:
        return 256
    if memory_gb >= 8:
        return 128
    return 64
