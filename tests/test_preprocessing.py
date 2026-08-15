from __future__ import annotations

import io

import pytest
from PIL import Image

from preprocessing import InvalidImageError, load_image


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 30), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_returns_detached_rgb_image() -> None:
    image = load_image(image_bytes(), max_bytes=10_000)

    assert image.mode == "RGB"
    assert image.size == (20, 30)
    assert image.getpixel((0, 0)) == (10, 20, 30)


@pytest.mark.parametrize("payload", [b"", b"not-an-image"])
def test_load_image_rejects_invalid_payload(payload: bytes) -> None:
    with pytest.raises(InvalidImageError):
        load_image(payload, max_bytes=10_000)


def test_load_image_enforces_size_limit() -> None:
    with pytest.raises(InvalidImageError, match="exceeds"):
        load_image(image_bytes(), max_bytes=10)
