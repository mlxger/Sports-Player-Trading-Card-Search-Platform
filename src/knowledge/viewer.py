"""Full Streamlit administration UI for the configured ChromaDB knowledge base."""

from __future__ import annotations

import io
import math
import uuid
from collections import Counter
from typing import Any

from knowledge.chroma import CARD_FIELDS, ChromaCardKnowledgeBase
from settings import Settings


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("install the 'parsing' extra to run the Chroma viewer") from exc

    settings = Settings()
    st.set_page_config(page_title="ChromaDB 查看器", page_icon="🔍", layout="wide")
    st.title("🔍 ChromaDB 数据查看器")

    try:
        names = ChromaCardKnowledgeBase.list_collection_names(settings.chroma_persist_dir)
    except Exception as exc:
        st.error(f"❌ 无法连接到 ChromaDB：{exc}")
        st.info(f"请检查数据库路径：{settings.chroma_persist_dir}")
        return
    if not names:
        st.warning("当前数据库中没有集合")
        st.info("请先通过 RAG API 或 Excel 导入接口写入卡片数据。")
        return

    st.sidebar.header("集合列表")
    selected = st.sidebar.selectbox("选择集合", names)
    try:
        knowledge = _get_knowledge(settings, selected)
    except Exception as exc:
        st.error(f"❌ 集合加载失败：{exc}")
        return

    total = knowledge.count()
    st.sidebar.metric("数据总数", total)
    st.sidebar.success(f"✅ 嵌入模型：{settings.rag_embedding_model}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 系统信息")
    st.sidebar.text(f"ChromaDB 路径:\n{settings.chroma_persist_dir}")
    st.sidebar.text(f"模型路径:\n{settings.rag_embedding_model}")

    browse_tab, search_tab, stats_tab, manage_tab = st.tabs(
        ["📋 浏览数据", "🔎 搜索", "📊 统计", "✏️ 数据管理"]
    )
    with browse_tab:
        _render_browse(st, pd, knowledge, selected, total)
    with search_tab:
        _render_search(st, knowledge, total)
    with stats_tab:
        _render_statistics(st, pd, knowledge, total, selected)
    with manage_tab:
        _render_management(st, knowledge)


def _render_browse(st: Any, pd: Any, knowledge: ChromaCardKnowledgeBase, name: str, total: int):
    st.header(f"集合：{name}")
    col_size, col_page, col_refresh = st.columns([1, 2, 1])
    with col_size:
        page_size = st.selectbox("每页显示", [10, 20, 50, 100], index=2)
    pages = max(1, math.ceil(total / page_size))
    with col_page:
        page = st.number_input(f"页码（共 {pages} 页）", min_value=1, max_value=pages, value=1)
    with col_refresh:
        st.write("")
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    try:
        cards = knowledge.list_cards(limit=page_size, offset=(int(page) - 1) * page_size)
    except Exception as exc:
        st.error(f"❌ 获取数据失败：{exc}")
        return
    if not cards:
        st.warning("当前页没有数据")
        return

    frame = pd.DataFrame(cards)
    st.dataframe(frame, use_container_width=True, height=420)
    st.subheader("详细信息")
    index = st.selectbox(
        "选择要查看的条目",
        range(len(cards)),
        format_func=lambda item: str(cards[item].get("_id", "unknown")),
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**卡片字段**")
        st.json({key: value for key, value in cards[index].items() if key != "_id"})
    with right:
        st.markdown("**记录信息**")
        st.metric("记录 ID", cards[index].get("_id", "unknown"))
        st.caption(f"第 {page} 页 / 共 {pages} 页")

    st.download_button(
        "📥 导出当前页为 CSV",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{name}_page{page}.csv",
        mime="text/csv",
    )


def _render_search(st: Any, knowledge: ChromaCardKnowledgeBase, total: int) -> None:
    st.header("搜索")
    st.success("✅ 搜索功能已就绪")
    mode = st.radio(
        "搜索模式",
        ["🔍 语义搜索（自然语言）", "🏷️ 字段精确过滤", "🆔 按 ID 直接查找"],
        horizontal=True,
    )
    st.markdown("---")

    if mode.startswith("🔍"):
        st.info("输入球员、俱乐部、系列、赛季等简短关键词，可获得更准确的语义结果。")
        query = st.text_input(
            "输入搜索关键词", placeholder="例如：Stephen Curry / 切尔西 Chrome / Topps 2024"
        )
        limit = st.slider("返回结果数量", 1, min(20, max(total, 1)), min(10, max(total, 1)))
        if query and len(query) > 80:
            st.warning("查询文本较长，建议保留球员名、系列名和赛季等核心关键词。")
        if st.button("🔍 搜索", type="primary"):
            if not query.strip():
                st.warning("请输入搜索关键词")
            elif total == 0:
                st.warning("当前集合为空")
            else:
                with st.spinner("搜索中..."):
                    try:
                        _render_results(st, knowledge.search(query, n_results=limit), scored=True)
                    except Exception as exc:
                        _render_error(st, exc)

    elif mode.startswith("🏷️"):
        st.info("按具体字段精确筛选，可组合球员、系列、俱乐部和赛季等条件。")
        filters = _filter_editor(st, "search_filters")
        limit = st.slider("最多返回条数", 5, 200, 50)
        if st.button("🏷️ 精确查找", type="primary"):
            if not filters:
                st.warning("请至少输入一个过滤条件")
            else:
                with st.spinner("查找中..."):
                    try:
                        _render_results(st, knowledge.search_multi(filters, limit=limit))
                    except Exception as exc:
                        _render_error(st, exc)

    else:
        st.info("输入精确 ID 直接定位卡片记录。")
        card_id = st.text_input("输入 ID", placeholder="例如：1001")
        find_similar = st.checkbox("同时查找相似记录")
        if st.button("🆔 查找", type="primary"):
            if not card_id.strip():
                st.warning("请输入 ID")
                return
            try:
                card = knowledge.get(card_id.strip())
                if card is None:
                    st.warning(f"未找到 ID：{card_id.strip()}")
                    return
                st.success(f"✅ 找到 ID：{card_id.strip()}")
                st.json(card)
                if find_similar:
                    query = " ".join(
                        str(card.get(field) or "")
                        for field in ("球员中文", "球员英文", "大系列中文简称", "赛季")
                    )
                    results = [
                        item
                        for item in knowledge.search(query, n_results=min(6, max(total, 1)))
                        if item.get("_id") != card_id.strip()
                    ]
                    st.subheader("📎 相似记录")
                    _render_results(st, results[:5], scored=True)
            except Exception as exc:
                _render_error(st, exc)


def _render_statistics(
    st: Any,
    pd: Any,
    knowledge: ChromaCardKnowledgeBase,
    total: int,
    collection_name: str,
) -> None:
    st.header("数据统计")
    if total == 0:
        st.info("没有元数据可供统计")
        return
    try:
        cards = _all_cards(knowledge, total)
    except Exception as exc:
        _render_error(st, exc)
        return

    available = [field for field in CARD_FIELDS if any(card.get(field) for card in cards)]
    summary = [
        {
            "字段": field,
            "有效值": sum(bool(card.get(field)) for card in cards),
            "唯一值": len({card.get(field) for card in cards if card.get(field)}),
        }
        for field in available
    ]
    st.subheader("元数据字段统计")
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    if available:
        st.subheader("字段值分布")
        field = st.selectbox("选择要分析的字段", available)
        counts = Counter(str(card[field]) for card in cards if card.get(field))
        chart = pd.DataFrame(counts.most_common(20), columns=[field, "数量"]).set_index(field)
        st.bar_chart(chart)
        st.dataframe(chart, use_container_width=True)

    output = io.BytesIO()
    pd.DataFrame(cards).to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        "📥 导出所有数据为 Excel",
        data=output.getvalue(),
        file_name=f"{collection_name}_all.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _render_management(st: Any, knowledge: ChromaCardKnowledgeBase) -> None:
    st.header("数据管理")
    mode = st.radio("操作类型", ["➕ 插入新数据", "✏️ 编辑数据", "🗑️ 删除数据"], horizontal=True)
    st.markdown("---")

    if mode.startswith("➕"):
        st.subheader("插入新记录")
        default_id = st.session_state.get("new_card_id", "")
        card_id = st.text_input("记录 ID", value=default_id, placeholder="留空可自动生成")
        if st.button("🎲 自动生成 ID"):
            st.session_state.new_card_id = str(uuid.uuid4())[:8]
            st.rerun()
        values = _card_form(st, "insert")
        if st.button("💾 确认插入", type="primary"):
            final_id = card_id.strip() or str(uuid.uuid4())[:8]
            try:
                knowledge.add({"ids": final_id, **values})
                st.success(f"✅ 成功插入记录，ID：{final_id}")
                st.balloons()
            except Exception as exc:
                _render_error(st, exc)

    elif mode.startswith("✏️"):
        st.subheader("编辑现有记录")
        card_id = st.text_input("输入要编辑的记录 ID")
        if st.button("🔍 加载记录"):
            card = knowledge.get(card_id.strip()) if card_id.strip() else None
            if card is None:
                st.warning("没有找到该记录")
            else:
                st.session_state.edit_card = card
                st.rerun()
        card = st.session_state.get("edit_card")
        if card:
            st.success(f"✅ 已加载记录：{card.get('_id')}")
            values = _card_form(st, "edit", card)
            if st.button("💾 保存修改", type="primary"):
                try:
                    knowledge.update(str(card["_id"]), values)
                    st.success("✅ 记录已更新")
                    st.session_state.edit_card = {"_id": card["_id"], **values}
                except Exception as exc:
                    _render_error(st, exc)

    else:
        st.subheader("删除记录")
        st.error("删除操作不可撤销，请先确认记录 ID。")
        ids_text = st.text_area("输入要删除的 ID（每行一个）")
        ids = [item.strip() for item in ids_text.splitlines() if item.strip()]
        if ids:
            existing = [item for item in ids if knowledge.get(item) is not None]
            st.write(f"找到 {len(existing)} 条可删除记录")
            confirmation = st.text_input(f"请输入 DELETE {len(existing)} 以确认")
            if st.button(
                f"🗑️ 删除 {len(existing)} 条记录",
                type="primary",
                disabled=confirmation != f"DELETE {len(existing)}" or not existing,
            ):
                try:
                    knowledge.remove_batch(existing)
                    st.success(f"✅ 已删除 {len(existing)} 条记录")
                except Exception as exc:
                    _render_error(st, exc)


def _filter_editor(st: Any, key: str) -> dict[str, str]:
    if key not in st.session_state:
        st.session_state[key] = [{"field": CARD_FIELDS[1], "value": ""}]
    add_col, clear_col = st.columns([1, 5])
    with add_col:
        if st.button("➕ 添加条件", key=f"{key}_add"):
            st.session_state[key].append({"field": CARD_FIELDS[1], "value": ""})
            st.rerun()
    with clear_col:
        if st.button("🗑️ 清空条件", key=f"{key}_clear"):
            st.session_state[key] = [{"field": CARD_FIELDS[1], "value": ""}]
            st.rerun()

    filters = {}
    remove = []
    for index, row in enumerate(st.session_state[key]):
        field_col, value_col, remove_col = st.columns([3, 4, 1])
        with field_col:
            field = st.selectbox(
                f"字段 {index + 1}",
                CARD_FIELDS,
                index=CARD_FIELDS.index(row["field"]),
                key=f"{key}_field_{index}",
            )
        with value_col:
            value = st.text_input(f"值 {index + 1}", value=row["value"], key=f"{key}_value_{index}")
        with remove_col:
            st.write("")
            if st.button("❌", key=f"{key}_remove_{index}") and len(st.session_state[key]) > 1:
                remove.append(index)
        st.session_state[key][index] = {"field": field, "value": value}
        if value.strip():
            filters[field] = value.strip()
    if remove:
        st.session_state[key] = [
            row for index, row in enumerate(st.session_state[key]) if index not in remove
        ]
        st.rerun()
    return filters


def _card_form(st: Any, key: str, values: dict[str, Any] | None = None) -> dict[str, str]:
    values = values or {}
    output = {}
    with st.expander("📋 卡片字段", expanded=True):
        columns = st.columns(3)
        for index, field in enumerate(CARD_FIELDS):
            if field == "ids":
                continue
            with columns[index % 3]:
                output[field] = st.text_input(
                    field, value=str(values.get(field) or ""), key=f"{key}_{field}"
                ).strip()
    return output


def _render_results(st: Any, results: list[dict[str, Any]], *, scored: bool = False) -> None:
    if not results:
        st.warning("没有找到匹配结果")
        return
    st.success(f"✅ 找到 {len(results)} 条结果")
    for index, result in enumerate(results, 1):
        score = float(result.get("_score") or 0)
        badge = "🟢" if score >= 0.85 else "🟡" if score >= 0.70 else "🔴"
        suffix = f" · 相似度 {score:.4f}" if scored else ""
        with st.expander(
            f"{badge if scored else '📄'} 结果 {index}{suffix} · ID {result.get('_id')}"
        ):
            st.json(result)


def _all_cards(knowledge: ChromaCardKnowledgeBase, total: int) -> list[dict[str, Any]]:
    cards = []
    for offset in range(0, total, 1000):
        cards.extend(knowledge.list_cards(limit=min(1000, total - offset), offset=offset))
    return cards


def _render_error(st: Any, exc: Exception) -> None:
    st.error(f"❌ 操作失败：{exc}")
    with st.expander("查看错误详情"):
        st.code(str(exc))


def _get_knowledge(settings: Settings, collection_name: str) -> ChromaCardKnowledgeBase:
    return ChromaCardKnowledgeBase(
        persist_directory=settings.chroma_persist_dir,
        collection_name=collection_name,
        embedding_model=settings.rag_embedding_model,
        device=settings.rag_device,
        allow_model_downloads=settings.allow_model_downloads,
    )


if __name__ == "__main__":
    main()
