"""ChromaDB-backed knowledge retrieval and candidate correction."""

from .chroma import CARD_FIELDS, ChromaCardKnowledgeBase, build_ocr_query

__all__ = ["CARD_FIELDS", "ChromaCardKnowledgeBase", "build_ocr_query"]
