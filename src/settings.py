from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from ``CARD_PIPELINE_*`` variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CARD_PIPELINE_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Trading Card Intelligence API"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=list)
    max_upload_bytes: int = 15 * 1024 * 1024
    retrieval_enabled: bool = True
    ocr_enabled: bool = False
    rag_enabled: bool = False

    device: str = "auto"
    model_cache_dir: Path = Path("models/embedding")
    allow_model_downloads: bool = False
    enable_real_photo_enhancement: bool = True
    insightface_model: str = "buffalo_s"
    dinov2_model: str = "facebook/dinov2-with-registers-base"
    openclip_model: str = "ViT-H-14"
    openclip_model_repo: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # The legacy ``texture_*`` names point to the supplied SLIP ConvNeXt checkpoint.
    texture_model: str = "convnext_base_w"
    texture_model_repo: str = "laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
    face_weight: float = 0.05
    structure_weight: float = 0.15
    fusion_weight: float = 0.80
    texture_weight: float = 0.60
    color_weight: float = 0.40
    embedding_dimension: int = 2304

    preprocessing_mode: Literal["none", "yolo"] = "none"
    yolo_model_path: Path | None = None

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str | None = None
    milvus_collection: str = "trading_card_embeddings_v1"
    milvus_alias: str = "retrieval"
    milvus_search_ef: int = 256
    milvus_search_concurrency: int = 16
    create_collection_on_startup: bool = False

    default_top_k: int = 10
    max_top_k: int = 100
    ranker_enabled: bool = False
    ranker_model_path: Path = Path("models/ranking/ranking_model.joblib")

    ollama_url: str = "http://127.0.0.1:11434"
    ocr_model_name: str = "qwen3-vl:8b-instruct-q8_0"
    ocr_model_timeout: float = 180.0
    ocr_temperature: float = 0.0
    ocr_max_output_tokens: int = 4096
    ocr_concurrency: int = 2
    dependency_check_timeout: float = 3.0

    chroma_persist_dir: Path = Path("data/chroma_db")
    chroma_collection: str = "cards"
    rag_embedding_model: str = "models/rag/BAAI/bge-m3"
    rag_device: str = "auto"
    rag_candidate_limit: int = 5

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_pipeline(self) -> Settings:
        weights = (self.face_weight, self.structure_weight, self.fusion_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("embedding weights must be non-negative and not all zero")
        if self.preprocessing_mode == "yolo" and self.yolo_model_path is None:
            raise ValueError("yolo_model_path is required when preprocessing_mode='yolo'")
        if self.embedding_dimension != 2304:
            raise ValueError("the migrated multimodal encoder produces exactly 2304 dimensions")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
