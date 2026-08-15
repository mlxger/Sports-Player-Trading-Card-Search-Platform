from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingWeights:
    face: float = 0.05
    structure: float = 0.15
    fusion: float = 0.80
    texture: float = 0.60
    color: float = 0.40


class MultimodalCardEmbedder:
    """InsightFace + DINOv2 + SLIP/CLIP encoder.

    The output layout is stable and compatible with the latest supplied pipeline:
    face (512) + DINOv2 (768) + fused SLIP texture/CLIP color (1024) = 2304 dimensions.

    ``texture_model`` is intentionally backed by the supplied SLIP ConvNeXt checkpoint;
    the legacy parameter name is retained for configuration compatibility.
    """

    dimension = 2304

    def __init__(
        self,
        *,
        cache_dir: Path,
        device: str = "auto",
        weights: EmbeddingWeights | None = None,
        enable_real_photo_enhancement: bool = True,
        allow_downloads: bool = False,
        insightface_model: str = "buffalo_s",
        dinov2_model: str = "facebook/dinov2-with-registers-base",
        openclip_model: str = "ViT-H-14",
        openclip_repo: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        texture_model: str = "convnext_base_w",
        texture_repo: str = "laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
    ) -> None:
        try:
            import torch
            from torchvision import transforms
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("install the 'retrieval' extra to load embedding models") from exc

        self._torch = torch
        self._device = _resolve_device(device, torch)
        self._weights = weights or EmbeddingWeights()
        self._enhance_real_photos = enable_real_photo_enhancement
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._face_app = self._load_face_model(insightface_model, allow_downloads)
        dino_path = self._resolve_model_path("dinov2", dinov2_model, allow_downloads)
        self._dino = (
            AutoModel.from_pretrained(
                dino_path,
                local_files_only=not allow_downloads,
            )
            .to(self._device)
            .eval()
        )
        self._dino_transform = transforms.Compose(
            [
                transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._color_model, self._color_transform = self._load_openclip(
            directory="openclip",
            model_name=openclip_model,
            repo=openclip_repo,
            allow_downloads=allow_downloads,
        )
        self._texture_model, self._texture_transform = self._load_openclip(
            directory="texture",
            model_name=texture_model,
            repo=texture_repo,
            allow_downloads=allow_downloads,
        )

    def encode(self, image: Image.Image, *, real_photo: bool = True) -> np.ndarray:
        raw = image.convert("RGB")
        enhanced = self._enhance(raw) if real_photo and self._enhance_real_photos else raw
        face = self._extract_face(raw)
        structure = self._extract_dino(raw)
        texture = _fit_dimension(self._extract_openclip(enhanced, "texture"), 1024)
        color = _fit_dimension(self._extract_openclip(enhanced, "color"), 1024)

        fusion = _normalize(texture * self._weights.texture + color * self._weights.color)
        combined = np.concatenate(
            [
                face * self._weights.face,
                structure * self._weights.structure,
                fusion * self._weights.fusion,
            ]
        ).astype(np.float32, copy=False)
        if combined.size != self.dimension:
            raise RuntimeError(f"unexpected embedding dimension: {combined.size}")
        return _normalize(combined)

    def _resolve_model_path(self, directory: str, repo: str, allow_downloads: bool) -> str:
        local_path = self._cache_dir / directory
        if local_path.exists():
            return str(local_path)
        if not allow_downloads:
            raise FileNotFoundError(
                f"model cache not found at {local_path}; enable model downloads or populate it"
            )
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo, local_dir=local_path)
        return str(local_path)

    def _load_face_model(self, model_name: str, allow_downloads: bool):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("insightface is required for face embeddings") from exc
        insightface_root = self._cache_dir / "insightface"
        expected_model = insightface_root / "models" / model_name
        if not allow_downloads and not expected_model.exists():
            raise FileNotFoundError(
                f"InsightFace model cache not found at {expected_model}; "
                "enable model downloads or populate it"
            )
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if self._device.startswith("cuda"):
            providers.insert(0, "CUDAExecutionProvider")
            ctx_id = int(self._device.partition(":")[2] or 0)
        app = FaceAnalysis(
            name=model_name,
            root=str(insightface_root),
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return app

    def _load_openclip(
        self,
        *,
        directory: str,
        model_name: str,
        repo: str,
        allow_downloads: bool,
    ):
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("open-clip-torch is required for multimodal embeddings") from exc

        local_path = self._cache_dir / directory
        checkpoint = _find_checkpoint(local_path)
        if checkpoint is None and allow_downloads:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, local_dir=local_path)
            checkpoint = _find_checkpoint(local_path)
        if checkpoint is None:
            raise FileNotFoundError(f"no model checkpoint found in {local_path} for {repo}")

        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=str(checkpoint)
        )
        return model.to(self._device).eval(), transform

    def _extract_face(self, image: Image.Image) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        faces = self._face_app.get(bgr)
        if not faces:
            return np.zeros(512, dtype=np.float32)
        largest = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
        )
        return _normalize(np.asarray(largest.embedding, dtype=np.float32))

    def _extract_dino(self, image: Image.Image) -> np.ndarray:
        tensor = self._dino_transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            output = self._dino(tensor).last_hidden_state[:, :5].mean(dim=1)
        return _normalize(output.cpu().numpy().ravel().astype(np.float32))

    def _extract_openclip(self, image: Image.Image, kind: str) -> np.ndarray:
        model = self._texture_model if kind == "texture" else self._color_model
        transform = self._texture_transform if kind == "texture" else self._color_transform
        tensor = transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            vector = model.encode_image(tensor)
        return _normalize(vector.cpu().numpy().ravel().astype(np.float32))

    @staticmethod
    def _enhance(image: Image.Image) -> Image.Image:
        import cv2

        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(ycrcb[:, :, 0])
        balanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(balanced, -1, kernel)
        mixed = cv2.addWeighted(balanced, 0.6, sharpened, 0.4, 0)
        enhanced = Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
        enhanced = ImageEnhance.Color(enhanced).enhance(1.5)
        return ImageEnhance.Contrast(enhanced).enhance(1.2)


def _resolve_device(value: str, torch_module) -> str:
    if value == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    return value


def _find_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    preferred = (
        "open_clip_pytorch_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
        "model.pt",
    )
    for name in preferred:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        match = next(directory.glob(pattern), None)
        if match is not None:
            return match
    return None


def _fit_dimension(vector: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    if vector.size == dimension:
        return vector
    if vector.size > dimension:
        return vector[:dimension]
    return np.pad(vector, (0, dimension - vector.size))


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else vector
