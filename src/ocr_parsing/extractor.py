"""
字段分组与多阶段编排。

核心职责
--------
1. build_field_groups() : 将字段列表按规则分组为 stage 执行单元
2. extract_card_metadata(): 按序调用 run_stage()，汇总结果
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from models.ocr_model import VisionModelClient

from .config import ALL_FIELDS, GRADING_FIELDS, STANDALONE_FIELDS, ModelConfig
from .stage_runner import StageResult, run_rag_correction, run_stage
from .timing import StageTimer
from .utils import normalize_sport_type, normalize_text_field

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── 字段分组 ───────────────────────────────────────────────────────────────────


def build_field_groups(
    fields: list[str],
    pair_fields: bool = True,
) -> list[list[str]]:
    """
    将字段列表组织为 stage 执行分组。

    规则
    ----
    1. STANDALONE_FIELDS (limited_edition, information) 始终单独一组。
    2. LOCKED_PAIRS (series+sub_series) 中的字段若同时出现，作为整体令牌参与
       配对流程，保证永远输出为同一组，不受前置字段奇偶数量影响。
    3. 若 pair_fields=True，其余普通字段两两配对；否则每字段独占一组。
    4. limited_edition 强制先于 card_number 执行。
    5. 分组顺序与输入 fields 的顺序保持一致（locked pair 以 series 出现位置为准）。

    Parameters
    ----------
    fields      : 待提取字段列表（顺序即执行顺序）
    pair_fields : True → 非 standalone 字段自动两两配对

    Returns
    -------
    List[List[str]] — 每个子列表为一次 stage 的字段集合
    """
    # 定义锁定对：这些字段组合若同时出现，永远绑定为同一 stage
    # 每个元素为有序的 (先出现字段, 后出现字段)
    _LOCKED_PAIRS: list[tuple[str, str]] = [
        # ("series", "sub_series"),
    ]

    # ① 强制顺序调整
    ordered = list(fields)

    # 锁定对：将后者移到前者的紧邻位置（保证令牌化时被识别为相邻对）
    for first, second in _LOCKED_PAIRS:
        if first in ordered and second in ordered:
            f_pos = ordered.index(first)
            s_pos = ordered.index(second)
            if s_pos != f_pos + 1:
                ordered.remove(second)
                ordered.insert(ordered.index(first) + 1, second)

    # limited_edition 先于 card_number
    if "card_number" in ordered and "limited_edition" in ordered:
        le_pos = ordered.index("limited_edition")
        cn_pos = ordered.index("card_number")
        if cn_pos < le_pos:
            ordered.remove("card_number")
            ordered.insert(le_pos + 1, "card_number")

    if not pair_fields:
        return [[f] for f in ordered]

    # ② 令牌化：将 ordered 转换为三类令牌序列
    #   ("standalone", field)        → 独占一组，不参与配对
    #   ("locked", [first, second])  → 锁定对，直接作为一组输出
    #   ("single", field)            → 参与两两顺序配对的普通字段
    locked_pair_map: dict[str, str] = {first: second for first, second in _LOCKED_PAIRS}

    tokens: list[tuple[str, Any]] = []
    i = 0
    while i < len(ordered):
        f = ordered[i]
        if f in STANDALONE_FIELDS:
            tokens.append(("standalone", f))
            i += 1
        elif f in locked_pair_map and i + 1 < len(ordered) and ordered[i + 1] == locked_pair_map[f]:
            tokens.append(("locked", [f, ordered[i + 1]]))
            i += 2
        else:
            tokens.append(("single", f))
            i += 1

    # ③ 将所有 single 令牌两两顺序配对
    singles = [t for t in tokens if t[0] == "single"]
    single_pairs: list[list[str]] = []
    j = 0
    while j < len(singles):
        if j + 1 < len(singles):
            single_pairs.append([singles[j][1], singles[j + 1][1]])
            j += 2
        else:
            single_pairs.append([singles[j][1]])
            j += 1

    field_to_pair_idx: dict[str, int] = {
        f: pi for pi, pair in enumerate(single_pairs) for f in pair
    }

    # ④ 按令牌顺序输出最终分组
    result: list[list[str]] = []
    emitted: set[int] = set()
    for token_type, token_val in tokens:
        if token_type == "standalone":
            result.append([token_val])
        elif token_type == "locked":
            result.append(token_val)
        else:  # single
            pi = field_to_pair_idx[token_val]
            if pi not in emitted:
                result.append(single_pairs[pi])
                emitted.add(pi)

    return result


def _validate_limited_edition(value: Any) -> str | None:
    """校验 limited_edition 必须为 X/Y 且 Y >= X，否则返回 None。"""
    text = normalize_text_field(value)
    if text is None:
        return None
    if text == "无编":
        return text
    parts = text.split("/")
    if len(parts) != 2:
        logger.warning("Rejecting limited_edition '%s' — not X/Y format.", text)
        return None
    try:
        x, y = int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        logger.warning("Rejecting limited_edition '%s' — non-integer parts.", text)
        return None
    if x > y:
        logger.warning("Rejecting limited_edition '%s' — X > Y.", text)
        return None
    return f"{x}/{y}"


# ── 字段值归一化 ───────────────────────────────────────────────────────────────


def _normalize(field: str, value: Any) -> Any:
    """对单个字段的原始模型输出进行类型安全的归一化。"""
    if field == "sport_type":
        return normalize_sport_type(value)
    if field == "limited_edition":
        return _validate_limited_edition(value)
    if field == "information":
        if isinstance(value, list):
            return [str(i).strip() for i in value if str(i).strip()]
        logger.warning("'information' is not a list: %s", value)
        return []
    if field == "card_number":
        text = normalize_text_field(value)
        if text and ("/" in text or "of" in text.lower()):
            logger.warning("Rejecting card_number '%s' — contains '/' or 'of'.", text)
            return None
        return text
    return normalize_text_field(value)


# ── 依赖上下文构建


def _build_dependencies(
    group: list[str],
    accumulated: dict[str, Any],
) -> dict[str, Any] | None:
    """
    根据当前 stage 的字段组，从已提取结果中提取所需依赖上下文。

    依赖规则
    --------
    - season / country_or_club  → 需要 name, sport_type
    - series / sub_series       → 需要 brand
    - card_number               → 需要 limited_edition
    """
    deps: dict[str, Any] = {}

    if any(f in group for f in ("season", "country_or_club")):
        for k in ("name", "sport_type"):
            if accumulated.get(k) is not None:
                deps[k] = accumulated[k]

    # if any(f in group for f in ("series", "sub_series")):
    if any(f in group for f in ("series")):
        if accumulated.get("brand") is not None:
            deps["brand"] = accumulated["brand"]

    if "card_number" in group and accumulated.get("limited_edition") is not None:
        deps["limited_edition"] = accumulated["limited_edition"]

    return deps or None


# ── 主提取编排函数


def extract_card_metadata(
    image_payloads: list[str],
    image_context: str,
    *,
    client: VisionModelClient,
    cfg: ModelConfig,
    requested_fields: list[str] | None = None,
    is_double_sided: bool = False,
    is_graded_card: bool = False,
    pair_fields: bool = True,
    use_rag: bool = True,
    rag_candidate_provider: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    return_timer: bool = False,
) -> dict[str, Any]:
    """
    多阶段提取编排入口。

    Parameters
    ----------
    image_payloads    : base64 图像列表
    image_context     : 图像描述字符串
    client            : Ollama 客户端
    cfg               : 模型配置
    requested_fields  : 需提取字段（None = 全部 ALL_FIELDS）
    is_double_sided   : 是否为双面卡
    is_graded_card    : 是否为评级卡（启用 rating_agencies/score）
    pair_fields       : 非 standalone 字段是否自动两两配对
    use_rag           : True（默认）时执行 RAG 候选选择；False 时跳过，
                        final["rag_match"] 置为 None

    Returns
    -------
    Dict  包含所有 ALL_FIELDS 键及 'status' 二进制字符串
    """
    if requested_fields is None:
        excluded = {"information", "rating_agencies", "score"}
        requested_fields = [field for field in ALL_FIELDS if field not in excluded]

    # 非评级卡过滤掉评级字段
    active_fields: list[str] = [
        f for f in requested_fields if not (f in GRADING_FIELDS and not is_graded_card)
    ]

    groups = build_field_groups(active_fields, pair_fields=pair_fields)
    # logger.debug("Execution groups: %s", groups)

    # 初始化累计结果
    accumulated: dict[str, Any] = {f: ([] if f == "information" else None) for f in active_fields}
    stage_flags: list[int] = []
    timer = StageTimer().start()
    for group in groups:
        # 跳过条件：仅当该组不含 name/sport_type 时，season/country_or_club 才依赖前置结果
        if any(f in group for f in ("season", "country_or_club")) and not any(
            f in group for f in ("name", "sport_type")
        ):
            name_val = accumulated.get("name")
            sport_val = accumulated.get("sport_type")
            if sport_val == "movie and television" or (name_val is None and sport_val is None):
                logger.info("Skipping %s — stage 1 yielded no useful result.", group)
                stage_flags.append(1)
                continue

        deps = _build_dependencies(group, accumulated)

        result: StageResult = run_stage(
            client=client,
            cfg=cfg,
            image_payloads=image_payloads,
            fields=group,
            image_context=image_context,
            is_double_sided=is_double_sided,
            dependencies=deps,
            timer=timer,
        )
        stage_flags.append(0 if result.success else 1)

        if result.success:
            for f in group:
                accumulated[f] = _normalize(f, result.values.get(f))

    # 后处理：movie and television → name 强制为 null
    if accumulated.get("sport_type") == "movie and television":
        accumulated["name"] = None

    # 组装最终结果（仅包含实际请求的字段）
    final: dict[str, Any] = {f: accumulated[f] for f in active_fields}

    final["status"] = "".join(str(b) for b in stage_flags)

    # ── RAG 候选选择 Stage ────────────────────────────────────────────────────
    if use_rag and rag_candidate_provider is not None:
        candidates = rag_candidate_provider(final)
        if candidates:
            # logger.info("RAG: %d candidates retrieved, running selection.", len(candidates))
            rag_result = run_rag_correction(
                client=client,
                cfg=cfg,
                image_payloads=image_payloads,
                extracted=final,
                candidates=candidates,
                timer=timer,  # RAG 阶段也纳入计时
            )
        else:
            logger.info("RAG: no candidates returned, skipping selection.")
            rag_result = None
        final["rag_match"] = rag_result
    else:
        # logger.debug("RAG disabled (use_rag=False).")
        final["rag_match"] = None

    # ── 计时汇总（包含 RAG stage）────────────────────────────────────────────
    timer.stop().print_summary()
    if return_timer:  # ← 新增
        return final, timer  # ← 新增

    return final
