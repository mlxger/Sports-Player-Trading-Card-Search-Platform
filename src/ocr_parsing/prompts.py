"""
各字段的独立 Prompt 片段，以及将多字段组合为单次 stage Prompt 的构建器。

设计原则
--------
- FIELD_TASK_PROMPTS : 每字段一条说明，可独立使用，也可拼接
- FIELD_OUTPUT_SCHEMAS: 对应字段的 JSON Schema 片段
- build_stage_prompt() : 将任意字段列表组装为完整的提示词
- build_stage_messages(): 生成 Ollama chat 消息列表
- build_rag_selection_prompt()  : 生成 RAG 候选选择提示词（文本模式）
- build_rag_selection_messages(): 生成对应的 Ollama chat 消息列表
"""

from __future__ import annotations

import json
from typing import Any

# ── 字段任务描述（可单独使用，也可多字段拼接）

FIELD_TASK_PROMPTS: dict[str, str] = {
    # "name": (
    #     "NAME — Player/Subject Name:\n"
    #     "• Identify the player name(s) printed on the card.\n"
    #     "• Join multiple names with \"&\" (e.g., \"Name1&Name2\").\n"
    #     "• If sport_type is \"movie and television\", name MUST be null.\n"
    #     "• If the name is not printed and you cannot recognise the person, return null."
    # ),
    # 球员搜索优化
    "name": (
        "NAME — Player/Subject Name:\n"
        "• Identify the player name(s) printed on the card.\n"
        "• MULTIPLE DIFFERENT PLAYERS: ONLY use '&' to join names if there are two or more DIFFERENT people on the card(e.g., \"Name1&Name2\").\n"
        "• BILINGUAL/PINYIN NAMES (CRITICAL): If the card displays the SAME person's name in both Chinese and English/Pinyin, extract ONLY the primary Chinese name (e.g., for '潘晓婷 PAN XIAOTING', return '潘晓婷').\n"
        '• If sport_type is "movie and television", name MUST be null.\n'
        "• If the name is not printed and you cannot recognise the person, return null."
    ),
    "sport_type": (
        "SPORT_TYPE — Card Category:\n"
        "• Identify from this EXACT list ONLY:\n"
        '  "basketball", "football", "baseball", "soccer",\n'
        '  "MMA", "movie and television"\n'
        '• "football" = American NFL; "soccer" = association football;\n'
        '  "MMA" = mixed martial arts.\n'
        "• If not clearly identifiable from the list, return null."
    ),
    # "season": (
    #     "SEASON — Season/Year:\n"
    #     "• Only extract if explicitly printed (e.g., \"2023\", \"2023-24\").\n"
    #     "• Must be a full year or year range.\n"
    #     "• Do NOT accept '#' or '/' separators (e.g., '09/10' is NOT valid).\n"
    #     "• If not clearly visible, return null."
    # ),
    # 优化发行年份的提取 明确视觉锚点在图片背面底部
    "season": (
        "SEASON — Card Issue Season/Year:\n"
        "• CRITICAL LOCATION: You MUST ONLY extract the issue year from the PUBLISHER COPYRIGHT TEXT on the BACK of the card. This is typically located at the very bottom edge and often accompanied by a '©' or 'TM' symbol.\n"
        "• PROHIBITED LOCATIONS: NEVER extract years from the player's biography, descriptive paragraphs, or statistical tables.\n"
        '• FORMAT: Must be a full year or year range explicitly printed in the copyright line (e.g., "2023", "2023-24").\n'
        "• Do NOT accept '#' or '/' separators.\n"
        "• If the copyright year is not clearly visible at the bottom of the back, return null."
    ),
    "country_or_club": (
        "COUNTRY_OR_CLUB — National Team or Club:\n"
        "• Look for country names, flags, club/team names, or logos.\n"
        "• Return the MOST SPECIFIC identifier:\n"
        "  - National team → country name\n"
        "  - Club team → club/team name (takes precedence over country)\n"
        "• Multiple subjects → join with '&' in the same order as name.\n"
        "• If any person lacks a clear identifier, set the entire field to null.\n"
        "• If only a league/city/event name appears, return null.\n"
        "• Do NOT infer from player knowledge or history."
    ),
    # "limited_edition": (
    #     "LIMITED_EDITION — Limited Edition Marking:\n"
    #     "• Scan the ENTIRE card for a limited edition mark.\n"
    #     "• Accepted patterns: 'X/Y' (e.g. '17/99', '1/1'), 'X of Y', 'One of One'.\n"
    #     "• X must be ≤ Y; if X > Y, return null.\n"
    #     "• Normalise to 'X/Y' format ('X of Y' or 'One of One' → '1/1').\n"
    #     "• Only return '1/1' if the card explicitly and clearly displays '1/1' (or equivalent like 'One of One' or 'X of Y' with X=1 and Y=1).\n"
    #     "• If '1/1' is not visible, not legible, ambiguous, or visually obscured (e.g., looks like an 'H' or part of a logo), return null\n"
    #     "• Do not guess. Do not interpret design elements (like logos, stylized text, or symbols) as edition numbers. If there is any doubt, return null.\n"
    #     "• If unclear or blurry, return null. Never guess."
    # ),
    # 优化无编限量级的识别
    "limited_edition": (
        "LIMITED_EDITION — Serial Number / Unnumbered:\n"
        "• Check BOTH the front and back of the SAME card.\n"
        "• Your task is ONLY to find ONE continuous printed serial-number string.\n"
        "• Accepted exact patterns ONLY:\n"
        "  - digits '/' digits (e.g. '17/99', '1/1', '#23/50')\n"
        "  - digits ' of ' digits (e.g. '3 of 25')\n"
        "  - 'One of One'\n"
        "• The numbering pattern must appear as ONE continuous printed string on the card.\n"
        "• Do NOT combine text from different areas.\n"
        "• If a valid numbering string is clearly visible, return it normalised to 'X/Y' format:\n"
        "  - '3 of 25' → '3/25'\n"
        "  - 'One of One' or '1 of 1' → '1/1'\n"
        "• If both visible sides are clear and there is clearly NO such numbering string anywhere on the card, return '无编'.\n"
        "• Return null ONLY if the image is blurry, blocked, cropped, incomplete, or the possible numbering area is unreadable.\n"
        "• NEVER use jersey numbers, card numbers, years, logos, stamps, foil text, decorative text, or design patterns as serial numbering.\n"
        "• NEVER infer serial numbering from rarity, colour, refractor pattern, or parallel type.\n"
        "• NEVER guess.\n"
        "• Decision order:\n"
        "  1. Exact visible numbering pattern → return normalised 'X/Y'\n"
        "  2. Clearly no exact numbering pattern anywhere → return '无编'\n"
        "  3. Unclear or insufficient evidence → return null"
    ),
    # "card_number": (
    #    "CARD_NUMBER — Official Card Number:\n"
    #    "• NEVER extract limited edition numbers (anything with '/' or 'of').\n"
    #    "• NEVER extract numbers with 5+ digits or a '.' inside.\n"
    #    "• NEVER extract jersey numbers or statistics.\n"
    #    "• Look for SHORT identifiers (1-6 chars) near top corners/edges:\n"
    #    "  ✓ '#' prefix:  \"#45\", \"#OCRAM\", \"#AC-15\"\n"
    #    "  ✓ 'No.' prefix: \"No.SSP-KDU\", \"No.128\"\n"
    #    "  ✓ Short alphanumeric: \"128\", \"EE-15\" (max one hyphen)\n"
    #    "• Priority: '#' prefix > 'No.' prefix > short alphanumeric.\n"
    #    "• If a number appears in 'X of Y' or 'X/Y' format, treat it as limited edition and DO NOT extract as card_number.\n"
    #    "• When in doubt, return null."
    # ),
    "card_number": (
        "CARD_NUMBER — Official Card Number:\n"
        "• NEVER extract numbers from limited edition phrases (e.g., 8 of 11, 5/10, 1/250), even if they meet length/format criteria.\n"
        "• NEVER extract numbers with 5+ digits or a '.' inside (e.g., 12.5).\n"
        "• NEVER extract jersey numbers or statistics.\n"
        "• The number must appear as a standalone identifier (not part of a limited edition, year, or stat).\n"
        "• Look for SHORT identifiers (1-6 chars) near top corners/edges:\n"
        '  ✓ \'#\' prefix:  "#45", "#OCRAM", "#AC-15"\n'
        '  ✓ \'No.\' prefix: "No.SSP-KDU", "No.128"\n'
        '  ✓ Short alphanumeric: "128", "EE-15" (max one hyphen)\n'
        "• Priority: '#' prefix > 'No.' prefix > short alphanumeric.\n"
        "• If a number appears in 'X of Y' or 'X/Y' format, treat it as limited edition and DO NOT extract as card_number."
        "  Critical Examples: 9 of 10 → Invalid (limited edition; return null for both 9 and 10)\n"
        "• If the number appears in a phrase containing of or / (e.g., 9 of 11, 5/10), return null immediately."
        "• When in doubt, return null."
    ),
    "brand": (
        "BRAND — Card Publisher Brand:\n"
        "• Identify from this list ONLY:\n"
        '  "Panini", "Donruss", "Topps", "Upper Deck", "Leaf", "Fleer", "SkyBox", "QICA China Sports", \n'
        '  "Fleer", "Bowman", "Leaf", "Sage"\n'
        "• In combinations like 'Topps Chrome': 'Topps' = brand, 'Chrome' = series.\n"
        "• If not in the list but clearly visible with copyright, report the exact name.\n"
        "• Do NOT guess — only report if explicitly shown."
    ),
    "license": (
        "LICENSE — Authorization/Copyright:\n"
        "• Shows which event, league, club, or IP authorised this card series.\n"
        "• Three types:\n"
        '  Type 1 EVENT:  e.g. "World Cup", "UEFA Champions League", "NBA Finals"\n'
        "  Type 2 CLUB:   specific club branding (not just a player wearing a uniform)\n"
        '  Type 3 MEDIA:  movie/TV name, e.g. "Spider-Man: Far From Home"\n'
        "• If not clearly visible, return null."
    ),
    # "series": (
    #     "SERIES — Product Line Name:\n"
    #     "• FOR PANINI: check BACK near copyright.\n"
    #     "  Pattern: 'YYYY-YY PANINI-[SERIES] [SPORT]'\n"
    #     "  Steps: 1) Remove year prefix  2) Remove 'PANINI-'  3) Remove sport suffix.\n"
    #     "  Example: '2023-24 PANINI-DONRUSS SOCCER' → 'DONRUSS'\n"
    #     "• FOR NON-PANINI: check FRONT for series logo; remove brand name.\n"
    #     "  Example: 'Topps Chrome' → 'Chrome'\n"
    #     "• If uncertain, return null."
    # ),
    # 优化系列名称在提取时中文识别不准及发行商和系列混淆问题
    "series": (
        "SERIES — Product Line or Set Name:\n"
        "• DEFINITION: Extract the specific product line, set, or sub-brand (e.g., 'Prizm', 'Donruss', 'Bowman', '凌云', '光芒').\n"
        "• CORPORATE PUBLISHER EXCLUSION (CRITICAL): NEVER return the primary corporate parent company as the series. \n"
        "  - STRICTLY EXCLUDE: 'Panini', 'Topps', 'Upper Deck', 'DAKA', '奇卡', 'ChiKa', '中国体育', 'China Sports', '卡游', 'Kayou'.\n"
        "• ACQUIRED/LEGACY BRANDS (VALID AS SERIES): \n"
        "  - Names like 'Donruss', 'Bowman', 'Fleer', 'Score', and 'Hoops' ARE VALID series names when printed alongside a corporate publisher (e.g., for 'Topps Bowman', extract 'Bowman'; for 'Panini Donruss', extract 'Donruss').\n"
        "• FOR WESTERN CARDS:\n"
        "  - Look at the BACK copyright line. If it follows 'YYYY-YY [PUBLISHER]-[SERIES]' or 'YYYY-YY [PUBLISHER] [SERIES]', extract ONLY the [SERIES] part (e.g., 'PANINI-DONRUSS' → 'DONRUSS').\n"
        "  - Look for prominent logos on the FRONT. If multiple logos exist, extract the specific set name, ignoring 'Panini' or 'Topps'.\n"
        "• FOR CHINESE CARDS:\n"
        "  - Identify the most prominent ARTISTIC, CALLIGRAPHY, or STYLIZED text on the FRONT (e.g., '凌云', '青龙'). This is usually the series name.\n"
        "  - On the BACK, look for text ending in '系列' (Series) or '集合' (Collection).\n"
        "• CLEANING: Remove any sport name suffixes (e.g., 'Basketball', 'Soccer', '明星卡') and year prefixes.\n"
        "• If uncertain, return null."
    ),
    "sub_series": (
        "SUB_SERIES — Themed Subset Name:\n"
        "• A subdivision or themed subset WITHIN a series.\n"
        "• Look primarily at FRONT in decorative/artistic/stylised text.\n"
        "• Do NOT confuse event names or copyright text as sub-series.\n"
        "• If unclear or ambiguous, return null."
    ),
    "rating_agencies": (
        "RATING_AGENCIES — Grading Company:\n"
        "• Identify from: PSA, BGS (Beckett), CGC, CSG, SGC, HGA, ISA, GMA.\n"
        "• Look for company logo/name on the grading label (usually at top).\n"
        "• Return exact abbreviation (e.g., 'PSA', 'BGS').\n"
        "• If not in the list but clearly visible, report as-is.\n"
        "• If no grading company is visible, return null."
    ),
    "score": (
        "SCORE — Overall Card Grade:\n"
        "• Extract the OVERALL/FINAL card grade (e.g., '10', '9.5', '8.5').\n"
        "• Usually the LARGEST number on the grading label.\n"
        "• DO NOT extract autograph grades ('AUTOGRAPH 10', 'AUTO 10').\n"
        "• DO NOT extract sub-grades (Centering, Corners, Edges, Surface).\n"
        "• Only extract if CLEARLY visible on the grading label."
    ),
    "information": (
        "INFORMATION — All Visible Clearly Text:\n"
        "• Extract ALL clear, readable text from the card image(s).\n"
        "• Skip blurry, tiny, or unclear text and statistical/numeric data.\n"
        "• Return as a JSON array of strings — fast extraction, do not overthink. do not extract unclear text."
    ),
}


# ── 字段 JSON Schema 片段

FIELD_OUTPUT_SCHEMAS: dict[str, str] = {
    "name": '"name": <string or null>',
    "sport_type": '"sport_type": <string or null>',
    "season": '"season": <string or null>',
    "country_or_club": '"country_or_club": <string or null>',
    # "limited_edition": '"limited_edition": <"X/Y" or null>',
    "limited_edition": '"limited_edition": <"X/Y" or "无编" or null>',
    "card_number": '"card_number": <string or null>',
    "brand": '"brand": <string or null>',
    "license": '"license": <string or null>',
    "series": '"series": <string or null>',
    "sub_series": '"sub_series": <string or null>',
    "rating_agencies": '"rating_agencies": <string or null>',
    "score": '"score": <string or null>',
    "information": '"information": ["text1", "text2", ...]',
}


# ── 系统消息

_FIELD_ROLES: dict[str, str] = {
    "name": "extracting player names",
    "sport_type": "identifying sport categories",
    "season": "extracting season/year information",
    "country_or_club": "identifying country or club affiliations",
    "limited_edition": "reading limited-edition markings",
    "card_number": "identifying official card numbers",
    "brand": "identifying brand information",
    "license": "identifying license and copyright information",
    "series": "identifying product series names",
    "sub_series": "identifying themed subsets",
    "rating_agencies": "reading grading company labels",
    "score": "extracting card grade scores",
    "information": "reading all visible clearly text",
}


# def build_system_message(fields: List[str]) -> str:
#     roles = ", ".join(_FIELD_ROLES.get(f, f) for f in fields)
#     return (
#         f"You are Qwen3-VL, an expert assistant specialised in {roles} "
#         "from trading cards. Always respond with valid JSON only."
#         "If you are unsure whether you have extracted the correct information, do not guess; you can return null instead."
#     )


# 强化全局系统提示
def build_system_message(fields: list[str]) -> str:
    roles = ", ".join(_FIELD_ROLES.get(f, f) for f in fields)

    extra_rules = []
    if "limited_edition" in fields:
        extra_rules.append(
            "For limited_edition, '无编' is a valid required output when both visible sides have been checked and there is clearly no serial-number pattern printed anywhere; do not default to null in that case."
        )
    if "name" in fields:
        extra_rules.append(
            "For name, bilingual Chinese + English/Pinyin versions of the SAME person count as one person, not multiple people."
        )
    if "season" in fields:
        extra_rules.append(
            "For season, only extract from the publisher copyright text on the BACK side, not from biography or stats."
        )

    extra = (" " + " ".join(extra_rules)) if extra_rules else ""

    return (
        f"You are Qwen3-VL, an expert assistant specialised in {roles} "
        "from trading cards. Always respond with valid JSON only. "
        "Only extract information that is clearly supported by the image. "
        "Do not guess. "
        "If a field has a special explicit rule, follow that field rule exactly instead of defaulting to null."
        f"{extra}"
    )


# 双面说明

_DS_HINTS: dict[str, str] = {
    "brand": "Brand logo and copyright usually appear on the BACK side.",
    "license": "License/copyright text usually appears on the BACK side.",
    "series": "Series name typically on FRONT; copyright/code on BACK.",
    "sub_series": "Sub-series name typically on FRONT in stylised text.",
    # "limited_edition":  "Limited edition marking usually appears on the FRONT side, but may appear on BACK.",
    "limited_edition": (
        "For limited_edition, you MUST check BOTH front and back completely. "
        "If neither side shows any exact serial-number pattern, return '无编' rather than null."
    ),
    "card_number": "Card number typically on BACK, but may appear on FRONT.",
    "rating_agencies": "For graded cards, grading info is on the slab label.",
    "score": "For graded cards, grade score is on the slab label.",
}


# def _double_sided_note(fields: List[str]) -> str:
#     base = (
#         "DOUBLE-SIDED CARD: You are viewing BOTH the front and back of the SAME card.\n"
#         "• Both sides are complementary — extract information from either.\n"
#         "• Do not treat front/back as multiple subjects."
#     )
#     hints = [_DS_HINTS[f] for f in fields if f in _DS_HINTS]
#     if hints:
#         base += "\n" + "\n".join(f"• {h}" for h in hints)
#     return base


def _double_sided_note(fields: list[str]) -> str:
    base = (
        "DOUBLE-SIDED CARD: You are viewing BOTH the front and back of the SAME card.\n"
        "• Both sides are complementary — extract information from either side.\n"
        "• Do not treat front/back as multiple subjects.\n"
        "• You must use BOTH images together before deciding that a field is missing.\n"
        "• If one side lacks the information, check the other side before returning null."
    )
    hints = [_DS_HINTS[f] for f in fields if f in _DS_HINTS]
    if hints:
        base += "\n" + "\n".join(f"• {h}" for h in hints)
    return base


#  主 Prompt 构建器
def build_stage_prompt(
    fields: list[str],
    image_context: str,
    is_double_sided: bool = False,
    dependencies: dict[str, Any] | None = None,
) -> str:
    """
    将任意字段列表组装为一次 stage 所需的完整提示词。

    Parameters
    ----------
    fields        : 本次 stage 需要提取的字段列表
    image_context : 图像描述字符串
    is_double_sided: 是否为正/反面双图
    dependencies  : 已确认的字段值（用于提供上下文）
    """
    if not image_context.strip():
        image_context = (
            "• Images: front and back of the same trading card."
            if is_double_sided
            else "• Image: uncropped scan/photo of the trading card."
        )

    lines: list[str] = [
        "Extract structured data from the trading card image(s) below.",
        f"Image context: {image_context}",
    ]

    if is_double_sided:
        lines += ["", _double_sided_note(fields)]

    # 已确认字段上下文
    if dependencies:
        dep_str = json.dumps(
            {k: v for k, v in dependencies.items() if v is not None},
            ensure_ascii=False,
        )
        lines += ["", f"Previously confirmed fields (do not modify): {dep_str}"]

    # card_number 需要明确排除 limited_edition 值
    if "card_number" in fields and dependencies and dependencies.get("limited_edition"):
        le = dependencies["limited_edition"]
        lines += [
            "",
            f'   IMPORTANT: The limited edition number is "{le}".',
            "    Do NOT return this as card_number — find a DIFFERENT identifier.",
        ]

    # 任务列表
    lines.append("")
    if len(fields) == 1:
        lines.append("TASK:")
        lines.append(FIELD_TASK_PROMPTS.get(fields[0], f"Extract: {fields[0]}"))
    else:
        lines.append("TASKS:")
        for i, f in enumerate(fields, 1):
            lines += [f"\nTask {i}:", FIELD_TASK_PROMPTS.get(f, f"Extract: {f}")]

    # 输出格式
    schema_parts = [FIELD_OUTPUT_SCHEMAS.get(f, f'"{f}": null') for f in fields]
    # lines += [
    #     "",
    #     "OUTPUT (no markdown, plain JSON only):",
    #     "{" + ", ".join(schema_parts) + "}",
    #     "",
    #     "When uncertain about any field, return null rather than guessing.",
    # ]

    lines += [
        "",
        "OUTPUT (no markdown, plain JSON only):",
        "{" + ", ".join(schema_parts) + "}",
        "",
        "When uncertain about any field, return null rather than guessing.",
        "If a field has an explicit special rule, follow that field rule exactly.",
    ]
    if "limited_edition" in fields:
        lines += [
            "For limited_edition specifically: if both visible sides are clear and no exact serial-number pattern appears anywhere on the card, return '无编' instead of null.",
            "Examples:",
            '• visible "23/99" -> {"limited_edition": "23/99"}',
            '• no serial-number string on either side -> {"limited_edition": "无编"}',
            '• blurry or incomplete numbering area -> {"limited_edition": null}',
        ]

    # print("\n".join(lines))
    return "\n".join(lines)


def build_stage_messages(
    image_b64_list: list[str],
    fields: list[str],
    image_context: str,
    is_double_sided: bool = False,
    dependencies: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """生成 Ollama chat 接口所需的消息列表。"""
    messages = [
        {
            "role": "system",
            "content": build_system_message(fields),
        },
        {
            "role": "user",
            "content": build_stage_prompt(fields, image_context, is_double_sided, dependencies),
            "images": image_b64_list,
        },
    ]
    # print(messages[0])
    return messages


# ── RAG 候选选择 Prompt ────────────────────────────────────────────────────────

# 不参与匹配的附加键（仅元数据字段参与比对）
_RAG_SKIP_KEYS = frozenset({"status", "rag_match", "information"})


def build_rag_correction_prompt(
    extracted: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    """
    生成让模型基于视觉信息和 RAG 候选记录纠正 brand/series/sub_series 的提示词。

    Parameters
    ----------
    extracted  : 模型已提取的卡片元数据
    candidates : RAG 检索返回的候选记录列表

    Returns
    -------
    str  完整提示词字符串
    """
    # 提取当前已识别的三个字段
    current_brand = extracted.get("brand")
    current_series = extracted.get("series")
    # current_sub_series = extracted.get("sub_series")

    extracted_core = {
        "name": extracted.get("name"),
        "sport_type": extracted.get("sport_type"),
        "brand": current_brand,
        "series": current_series,
        # "sub_series": current_sub_series,
    }
    extracted_str = json.dumps(extracted_core, ensure_ascii=False, indent=2)

    # 候选列表：只展示关键字段
    candidate_lines: list[str] = []
    for i, cand in enumerate(candidates):
        cand_core = {
            "球员英文": cand.get("球员英文"),
            "发行商英文": cand.get("发行商英文"),
            "大系列英文简称": cand.get("大系列英文简称"),
            # "小系列英文": cand.get("小系列英文"),
        }
        candidate_lines.append(f"  [{i}] {json.dumps(cand_core, ensure_ascii=False)}")
    candidates_str = "\n".join(candidate_lines)
    # print(candidates_str)
    return "\n".join(
        [
            "You are viewing a trading card image and have extracted initial metadata.",
            "A database search has returned semantically similar candidate records.",
            "",
            "YOUR TASK:",
            "Use the VISUAL INFORMATION from the card image AND the candidate records",
            # "to CORRECT the three fields: brand, series, sub_series.",
            "to CORRECT the three fields: brand, series.",
            "",
            "CURRENTLY EXTRACTED:",
            extracted_str,
            "",
            f"RAG CANDIDATES FROM DATABASE ({len(candidates)} records):",
            candidates_str,
            "",
            "CORRECTION RULES:",
            # "• RE-EXAMINE the card image carefully for brand logo, series name, and sub-series text.",
            "• RE-EXAMINE the card image carefully for brand logo, series name text.",
            "• Compare what you see visually with the candidate suggestions.",
            "• Field mapping:",
            "    brand       ← 发行商英文 (e.g., Panini, Topps, Upper Deck)",
            "    series      ← 大系列英文简称 (e.g., Prizm, Chrome, Select)",
            # "    sub_series  ← 小系列英文 (e.g., Silver, Base, Rookie)",
            "",
            "• PRIORITY ORDER when deciding corrections:",
            "    1. VISUAL EVIDENCE from the image (text, logo, design) — HIGHEST priority",
            "    2. Candidate consistency (multiple candidates with same value)",
            "    3. Player name match (球员英文 matches extracted name)",
            "",
            "• If the current value is CORRECT based on visual evidence, KEEP it (do not change).",
            "• If the current value is WRONG or missing, and candidates provide a plausible",
            "  correction that matches visual evidence, UPDATE it.",
            "",
            "CRITICAL:",
            "• Do NOT blindly copy from candidates — always verify against the IMAGE.",
            "• If the card shows 'Topps Chrome' but candidates suggest 'Panini Prizm',",
            "  trust the IMAGE and keep 'Topps' / 'Chrome'.",
            "• Only correct if you can SEE supporting evidence in the image.",
            "",
            "OUTPUT (plain JSON only, no markdown, no explanation):",
            "{",
            '  "brand": <corrected string or null>,',
            '  "series": <corrected string or null>,',
            #'  "sub_series": <corrected string or null>',
            "}",
        ]
    )


def build_rag_correction_messages(
    image_b64_list: list[str],
    extracted: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    生成 RAG 纠正任务的 Ollama chat 消息列表。

    与普通 stage 不同：这是"纠正模式"，模型需结合视觉和候选记录。

    Parameters
    ----------
    image_b64_list : 卡片图像的 base64 列表
    extracted      : 模型已提取的卡片元数据
    candidates     : RAG 检索候选列表

    Returns
    -------
    List[Dict[str, Any]]  Ollama chat messages
    """
    return [
        {
            "role": "system",
            "content": (
                # "You are Qwen3-VL, an expert in trading card brand, series, and "
                # "sub-series identification. You combine visual recognition with "
                "You are Qwen3-VL, an expert in trading card brand, and series"
                "identification. You combine visual recognition with "
                "database knowledge to correct extraction errors. "
                "Always respond with valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": build_rag_correction_prompt(extracted, candidates),
            "images": image_b64_list,
        },
    ]
