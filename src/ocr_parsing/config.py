# config.py
"""全局常量、ModelConfig 及字段分组规则。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "qwen3-vl:8b-instruct-q8_0"
    temperature: float = 0.0
    max_output_tokens: int = 4096


# ── YOLO 类优先级 ──────────────────────────────────────────────────────────────
CROP_CLASS_PRIORITY: dict[int, int] = {
    0: 0,  # front-card
    1: 1,  # front-patch-card
    3: 2,  # front-rate-card
    2: 3,  # front-strange-card
    4: 4,  # front-book-card
}

SHAPE_LABELS: dict[str, str] = {
    "circle": "circular",
    "rectangle": "rectangular",
    "polygon": "polygonal",
}

SPORT_TYPE_CANONICAL: dict[str, str] = {
    "basketball": "basketball",
    "football": "football",
    "baseball": "baseball",
    "soccer": "soccer",
    "mma": "MMA",
    "m.m.a": "MMA",
    "mixed martial arts": "MMA",
    "movie and television": "movie and television",
}

# ── 可提取字段（顺序决定默认自动配对结果）─────────────────────────────────────
# 排列原则：相邻两两配对语义合理；standalone 字段单独处理。
ALL_FIELDS: list[str] = [
    "name",
    "sport_type",  # 配对 1
    "season",
    "country_or_club",  # 配对 2
    "brand",
    "license",  # 配对 3
    "series",
    "sub_series",  # 配对 4
    "rating_agencies",
    "score",  # 配对 5（评级卡专用）
    "limited_edition",  # 独立 stage
    "card_number",  # 独立 stage（依赖 limited_edition）
    "information",  # 独立 stage
]

# 必须单独处理、不参与配对的字段
STANDALONE_FIELDS: set[str] = {"limited_edition", "card_number", "information"}

# 仅限评级卡提取的字段
GRADING_FIELDS: set[str] = {"rating_agencies", "score"}

# 各字段默认超时（秒）
FIELD_TIMEOUTS: dict[str, int] = {
    "name": 10,
    "sport_type": 10,
    "season": 10,
    "country_or_club": 10,
    "limited_edition": 10,
    "card_number": 10,
    "brand": 10,
    "license": 10,
    "series": 10,
    "sub_series": 10,
    "rating_agencies": 10,
    "score": 10,
    "information": 15,
}
