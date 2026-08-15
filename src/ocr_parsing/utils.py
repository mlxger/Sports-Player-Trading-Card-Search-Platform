"""编码、归一化、JSON 解析、超时等通用工具函数。"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import signal
from collections.abc import Generator
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import cv2

MAX_INFERENCE_LONG_SIDE: int = 1280

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── 超时机制（仅支持 Unix/SIGALRM/Windows）


class TimeoutException(Exception):
    """stage 执行超时时抛出。"""


if hasattr(signal, "SIGALRM"):
    # Unix：使用 SIGALRM，精确且无需额外线程
    def _sigalrm_handler(signum: int, frame: Any) -> None:
        raise TimeoutException("Stage execution exceeded time limit")

    @contextlib.contextmanager
    def time_limit(seconds: int) -> Generator[None, None, None]:
        """Unix 专用：通过 SIGALRM 限制代码块最大执行时间。"""
        signal.signal(signal.SIGALRM, _sigalrm_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

else:
    # Windows / 无 SIGALRM 平台：time_limit 降级为无操作。
    # 实际超时由 stage_runner.py 中的 ThreadPoolExecutor 处理。
    @contextlib.contextmanager  # type: ignore[misc]
    def time_limit(seconds: int) -> Generator[None, None, None]:
        """Windows 兼容占位：不做任何超时操作，仅保留接口。"""
        yield


# ── 噪声抑制


@contextlib.contextmanager
def suppress_cli_noise() -> Generator[None, None, None]:
    """临时屏蔽第三方库写入 stdout/stderr 的内容。"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ── 推理图像预缩放 ─────────────────────────────────────────────────────────────


def resize_for_inference(
    image: Any,
    max_long_side: int = MAX_INFERENCE_LONG_SIDE,
) -> tuple[Any, bool]:
    """
    等比缩放图像至长边不超过 max_long_side。
    算法选择
    --------
    - 下采样使用 cv2.INTER_AREA（面积平均插值）：
      抗锯齿效果最佳，对文字、细线保留清晰度优于双线性插值。
    - 若图像已在限制内则原样返回，零拷贝零计算。
    """
    h, w = image.shape[:2]
    long_side = max(h, w)

    if long_side <= max_long_side:
        return image, False

    scale = max_long_side / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    logger.debug(
        "Pre-scaling image for inference: %dx%d → %dx%d  (scale=%.3f, pixel reduction=%.1f%%)",
        w,
        h,
        new_w,
        new_h,
        scale,
        (1 - scale**2) * 100,
    )

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, True


# ── 图像编码
"""
def encode_image_to_b64(image_path: Path) -> str:
    #将图像文件读取为 base64 字符串。
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_array_to_b64(image_array: Any, ext: str = ".jpg") -> str:
    #将 numpy 内存图像编码为 base64 字符串。
    success, buffer = cv2.imencode(ext, image_array)
    if not success:
        raise ValueError("Failed to encode image array to base64.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
"""


def encode_image_to_b64(
    image_path: Path,
    max_long_side: int = MAX_INFERENCE_LONG_SIDE,
    jpeg_quality: int = 95,
) -> str:
    """
    将图像文件编码为 base64 字符串。

    流程
    ----
    1. cv2.imread 读取原图
    2. resize_for_inference 等比缩放（如需要）
    3. cv2.imencode 重新编码为 JPEG
    4. base64 序列化

    Parameters
    ----------
    image_path   : 图像文件路径
    max_long_side: 长边像素上限（0 = 跳过缩放，直接读取原始字节）
    jpeg_quality : JPEG 重编码质量（1–100），95 在质量和体积间取得良好平衡
    """
    # max_long_side=0：向后兼容模式，原始字节直接编码，不经过 cv2
    if max_long_side == 0:
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cv2 无法读取图像: {image_path}")

    img, was_resized = resize_for_inference(img, max_long_side)
    if not was_resized:
        # 尺寸已在限制内：仍走 cv2 编码路径以统一输出格式
        logger.debug("Image %s is within size limit, no resize needed.", image_path.name)

    return encode_array_to_b64(img, jpeg_quality=jpeg_quality)


def encode_array_to_b64(
    image_array: Any,
    ext: str = ".jpg",
    max_long_side: int = MAX_INFERENCE_LONG_SIDE,
    jpeg_quality: int = 95,
) -> str:
    """
    将 numpy 内存图像编码为 base64 字符串。

    Parameters
    ----------
    image_array  : numpy 图像数组（BGR）
    ext          : 编码格式（'.jpg' 或 '.png'）
    max_long_side: 长边像素上限（0 = 跳过缩放）
    jpeg_quality : JPEG 质量（仅 ext='.jpg' 时生效）
    """
    if max_long_side > 0:
        image_array, _ = resize_for_inference(image_array, max_long_side)

    encode_params: list = []
    if ext.lower() in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    success, buffer = cv2.imencode(ext, image_array, encode_params)
    if not success:
        raise ValueError("Failed to encode image array to base64.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ── 字段归一化


def normalize_text_field(value: Any) -> str | None:
    """通用文本归一化：去空白、将 'null'/空值转为 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return None if (not s or s.lower() == "null") else s
    if isinstance(value, (list, tuple)):
        joined = "&".join(str(i).strip() for i in value if str(i).strip())
        return joined or None
    s = str(value).strip()
    return s or None


def normalize_sport_type(value: Any) -> str | None:
    """将 sport_type 映射为标准值。"""
    # 延迟导入避免循环依赖
    from .config import SPORT_TYPE_CANONICAL

    text = normalize_text_field(value)
    if text is None:
        return None
    return SPORT_TYPE_CANONICAL.get(text.lower(), text)


# ── JSON 解析


def extract_last_json_object(text: str) -> dict[str, Any]:
    """从模型回复文本中提取最后一个 JSON 对象。"""
    cleaned = text.strip()
    decoder = json.JSONDecoder()
    idx, last_obj = 0, None
    while idx < len(cleaned):
        try:
            obj, end = decoder.raw_decode(cleaned, idx)
            last_obj, idx = obj, end
        except JSONDecodeError:
            idx += 1
    if last_obj is None:
        raise ValueError(f"No JSON object found in response: {text!r}")
    if not isinstance(last_obj, dict):
        raise ValueError(f"Last JSON value is not an object: {last_obj!r}")
    return last_obj
