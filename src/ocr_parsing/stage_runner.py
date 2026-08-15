"""
通用 Stage 执行器。

所有 stage（无论字段数量和类型）统一通过 run_stage() 调用，
内部完成：提示词构建 → 模型调用 → JSON 解析 → 超时保护。
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from models.ocr_model import VisionModelClient

from .config import FIELD_TIMEOUTS, ModelConfig
from .prompts import build_rag_correction_messages, build_stage_messages
from .utils import extract_last_json_object

if TYPE_CHECKING:
    from .timing import StageTimer

# RAG 选择阶段默认超时（秒）
_RAG_SELECTION_TIMEOUT: int = 10

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class StageResult:
    """单次 stage 执行结果。"""

    fields: list[str]
    values: dict[str, Any]
    success: bool
    timed_out: bool = False
    error: str | None = None


# ── 内部：纯模型调用（在子线程中执行


def _call_model(
    client: VisionModelClient,
    cfg: ModelConfig,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any]:
    """
    执行模型调用并解析 JSON 响应。
    此函数在独立线程中运行，由 executor 管理其生命周期。
    """
    response = client.chat(
        model=cfg.model_name,
        messages=messages,
        options=options,
        stream=False,
        think=False,
    )
    content = response.get("message", {}).get("content", "").strip()
    if not content:
        raise ValueError("Model returned empty content.")
    return extract_last_json_object(content)


def _compute_timeout(fields: list[str]) -> int:
    """取字段列表中最大的超时值作为本次 stage 超时上限。"""
    return max((FIELD_TIMEOUTS.get(f, 20) for f in fields), default=20)


def run_stage(
    *,
    client: VisionModelClient,
    cfg: ModelConfig,
    image_payloads: list[str],
    fields: list[str],
    image_context: str,
    is_double_sided: bool = False,
    dependencies: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
    timer: StageTimer | None = None,
) -> StageResult:
    """
    通用 stage 执行函数：适配任意字段组合与场景。

    Parameters
    ----------
    client          : Ollama 客户端
    cfg             : 模型配置
    image_payloads  : base64 图像列表
    fields          : 本次 stage 需提取的字段名列表
    image_context   : 图像描述字符串
    is_double_sided : 是否为双面卡（正面 + 背面）
    dependencies    : 上游 stage 已确认的字段值（用作上下文）
    timeout_seconds : 超时秒数（None 则取字段最大值）
    timer : 若传入，本次 Stage 的耗时与字段分组将被自动记录。
    Returns
    -------
    StageResult
    """
    if not fields:
        return StageResult(fields=[], values={}, success=True)

    timeout = timeout_seconds if timeout_seconds is not None else _compute_timeout(fields)
    messages = build_stage_messages(
        image_b64_list=image_payloads,
        fields=fields,
        image_context=image_context,
        is_double_sided=is_double_sided,
        dependencies=dependencies,
    )
    options = {
        "temperature": cfg.temperature,
        "num_predict": cfg.max_output_tokens,
    }

    t_start = time.perf_counter()

    # ── 使用 ThreadPoolExecutor 实现跨平台超时 ────────────────────────────────
    # 注意：future.cancel() 在线程已开始时不会真正终止线程，
    # 但 future.result(timeout=...) 会立即返回并让主流程继续，
    # 后台线程会在模型响应后自然结束，不会泄漏资源。
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call_model, client, cfg, messages, options)
    try:
        try:
            raw = future.result(timeout=timeout)
            logger.debug("Stage %s → %s", fields, raw)
            result = StageResult(fields=fields, values=raw, success=True)

        except concurrent.futures.TimeoutError:
            logger.warning("Stage %s timed out after %ds.", fields, timeout)
            future.cancel()
            return StageResult(
                fields=fields,
                values={f: None for f in fields},
                success=False,
                timed_out=True,
                error=f"Timed out after {timeout}s",
            )

        except Exception as exc:
            logger.error("Stage %s failed: %s", fields, exc)
            return StageResult(
                fields=fields,
                values={f: None for f in fields},
                success=False,
                error=str(exc),
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    elapsed = time.perf_counter() - t_start

    if timer is not None:
        timer.record_stage(
            fields=fields,
            elapsed=elapsed,
            success=result.success,
            timed_out=result.timed_out,
            error=result.error,
        )

    return result


# ── RAG 候选选择 Stage


def run_rag_correction(
    *,
    client: VisionModelClient,
    cfg: ModelConfig,
    image_payloads: list[str],  # 新增：传入图像
    extracted: dict[str, Any],
    candidates: list[dict[str, Any]],
    timeout_seconds: int = 30,
    timer: StageTimer | None = None,
) -> dict[str, str] | None:
    """
    调用模型基于视觉信息和 RAG 候选记录纠正 brand/series/sub_series。

    Parameters
    ----------
    client          : Ollama 客户端
    cfg             : 模型配置
    image_payloads  : base64 图像列表（与普通 stage 一致）
    extracted       : 已提取的完整卡片元数据字典
    candidates      : RAG 检索返回的候选记录列表
    timeout_seconds : 超时秒数（默认 30s）
    timer           : 可选 StageTimer

    Returns
    -------
    Optional[Dict[str, str]]
        包含纠正后的三个字段：{"brand": ..., "series": ..., "sub_series": ...}
        如果 candidates 为空、超时或解析失败，返回 None。
    """
    if not candidates:
        logger.debug("run_rag_correction: candidates list is empty, skipping.")
        return None

    messages = build_rag_correction_messages(image_payloads, extracted, candidates)
    options = {
        "temperature": cfg.temperature,
        "num_predict": cfg.max_output_tokens,
    }

    t_start = time.perf_counter()
    success = False
    timed_out = False
    error_msg: str | None = None
    corrected: dict[str, str] | None = None

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call_model, client, cfg, messages, options)
    try:
        try:
            raw = future.result(timeout=timeout_seconds)
            logger.debug("RAG correction raw response: %s", raw)

            # 验证返回格式
            if not isinstance(raw, dict):
                error_msg = f"Invalid response format: {type(raw)}"
                logger.warning("RAG correction: %s", error_msg)
            else:
                corrected = {
                    "brand": raw.get("brand"),
                    "series": raw.get("series"),
                    # "sub_series": raw.get("sub_series"),
                }
                logger.info(
                    # "RAG correction completed: brand=%s, series=%s, sub_series=%s",
                    # corrected["brand"], corrected["series"], corrected["sub_series"]
                    "RAG correction completed: brand=%s, series=%s",
                    corrected["brand"],
                    corrected["series"],
                )
                success = True

        except concurrent.futures.TimeoutError:
            timed_out = True
            error_msg = f"Timed out after {timeout_seconds}s"
            logger.warning("RAG correction timed out after %ds.", timeout_seconds)
            future.cancel()

        except Exception as exc:
            error_msg = str(exc)
            logger.error("RAG correction failed: %s", exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.perf_counter() - t_start

    if timer is not None:
        timer.record_stage(
            fields=["rag_correction"],
            elapsed=elapsed,
            success=success,
            timed_out=timed_out,
            error=error_msg,
        )

    return corrected
