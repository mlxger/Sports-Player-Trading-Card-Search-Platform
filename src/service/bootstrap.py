from __future__ import annotations

from models import EmbeddingWeights, MultimodalCardEmbedder
from preprocessing import PassthroughPreprocessor, YoloCardPreprocessor
from rerank import LightGBMReranker
from retrieval import MilvusVectorStore
from service.retrieval import ImageIndexingService, ImageRetrievalService
from settings import Settings


def build_retrieval_service(settings: Settings) -> ImageRetrievalService:
    preprocessor, embedder, vector_store = _build_components(settings)
    reranker = LightGBMReranker(settings.ranker_model_path) if settings.ranker_enabled else None
    return ImageRetrievalService(
        preprocessor=preprocessor,
        embedder=embedder,
        vector_store=vector_store,
        max_upload_bytes=settings.max_upload_bytes,
        max_top_k=settings.max_top_k,
        reranker=reranker,
    )


def build_indexing_service(settings: Settings) -> ImageIndexingService:
    preprocessor, embedder, vector_store = _build_components(settings)
    return ImageIndexingService(
        preprocessor=preprocessor,
        embedder=embedder,
        vector_store=vector_store,
        max_upload_bytes=settings.max_upload_bytes,
    )


def _build_components(settings: Settings):
    if settings.create_collection_on_startup:
        MilvusVectorStore.create_collection(settings)
    if settings.preprocessing_mode == "yolo":
        preprocessor = YoloCardPreprocessor(str(settings.yolo_model_path))
    else:
        preprocessor = PassthroughPreprocessor()
    embedder = MultimodalCardEmbedder(
        cache_dir=settings.model_cache_dir,
        device=settings.device,
        weights=EmbeddingWeights(
            face=settings.face_weight,
            structure=settings.structure_weight,
            fusion=settings.fusion_weight,
            texture=settings.texture_weight,
            color=settings.color_weight,
        ),
        enable_real_photo_enhancement=settings.enable_real_photo_enhancement,
        allow_downloads=settings.allow_model_downloads,
        insightface_model=settings.insightface_model,
        dinov2_model=settings.dinov2_model,
        openclip_model=settings.openclip_model,
        openclip_repo=settings.openclip_model_repo,
        texture_model=settings.texture_model,
        texture_repo=settings.texture_model_repo,
    )
    return preprocessor, embedder, MilvusVectorStore(settings)
