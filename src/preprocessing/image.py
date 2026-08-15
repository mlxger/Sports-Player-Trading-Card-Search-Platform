from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


def load_image(data: bytes, *, max_bytes: int) -> Image.Image:
    if not data:
        raise InvalidImageError("uploaded image is empty")
    if len(data) > max_bytes:
        raise InvalidImageError(f"uploaded image exceeds the {max_bytes}-byte limit")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("uploaded file is not a valid image") from exc


@dataclass(slots=True)
class PassthroughPreprocessor:
    def process(self, image: Image.Image) -> Image.Image:
        return image.copy()


class YoloCardPreprocessor:
    """Detect a card segmentation mask and rectify it into a front-facing crop."""

    def __init__(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("install the 'preprocessing' extra to enable YOLO") from exc
        self._model = YOLO(model_path)

    def process(self, image: Image.Image) -> Image.Image:
        import cv2

        rgb = np.asarray(image.convert("RGB"))
        result = self._model.predict(source=rgb, verbose=False)
        if not result or result[0].masks is None or not result[0].masks.xy:
            return image.copy()

        contour = np.asarray(result[0].masks.xy[0], dtype=np.float32)
        corners = _quadrilateral_corners(contour)
        if corners is None:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect)

        ordered = _order_points(corners)
        width = max(
            int(np.linalg.norm(ordered[2] - ordered[3])),
            int(np.linalg.norm(ordered[1] - ordered[0])),
        )
        height = max(
            int(np.linalg.norm(ordered[1] - ordered[2])),
            int(np.linalg.norm(ordered[0] - ordered[3])),
        )
        if width < 2 or height < 2:
            return image.copy()

        target = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, target)
        warped = cv2.warpPerspective(rgb, matrix, (width, height))
        return Image.fromarray(warped, mode="RGB")


def _quadrilateral_corners(points: np.ndarray) -> np.ndarray | None:
    import cv2

    hull = cv2.convexHull(points.reshape((-1, 1, 2)))
    perimeter = cv2.arcLength(hull, True)
    for factor in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
        polygon = cv2.approxPolyDP(hull, factor * perimeter, True)
        if len(polygon) == 4:
            return polygon.reshape(4, 2).astype(np.float32)
    return None


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered
