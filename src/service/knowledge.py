from __future__ import annotations

import threading

from knowledge import ChromaCardKnowledgeBase
from settings import Settings


class KnowledgeService:
    """Collection registry sharing one configuration boundary."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collections: dict[str, ChromaCardKnowledgeBase] = {}
        self._lock = threading.Lock()

    @property
    def default(self) -> ChromaCardKnowledgeBase:
        return self.get(self._settings.chroma_collection)

    def get(self, name: str | None = None) -> ChromaCardKnowledgeBase:
        collection_name = name or self._settings.chroma_collection
        with self._lock:
            if collection_name not in self._collections:
                self._collections[collection_name] = ChromaCardKnowledgeBase(
                    persist_directory=self._settings.chroma_persist_dir,
                    collection_name=collection_name,
                    embedding_model=self._settings.rag_embedding_model,
                    device=self._settings.rag_device,
                    allow_model_downloads=self._settings.allow_model_downloads,
                )
            return self._collections[collection_name]

    def drop(self, name: str | None = None) -> None:
        collection_name = name or self._settings.chroma_collection
        knowledge = self.get(collection_name)
        knowledge.drop()
        with self._lock:
            self._collections.pop(collection_name, None)


def build_knowledge_service(settings: Settings) -> KnowledgeService:
    return KnowledgeService(settings)
