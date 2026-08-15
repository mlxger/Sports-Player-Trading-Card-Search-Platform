"""Optional Streamlit viewer for the configured ChromaDB knowledge base.

Run with ``streamlit run src/knowledge/viewer.py`` after installing ``.[parsing]``.
The viewer reuses the same repository and embedding policy as the API; it never carries
its own hard-coded paths or silently enables model downloads.
"""

from __future__ import annotations

import json

from knowledge.chroma import ChromaCardKnowledgeBase
from settings import Settings


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("install the 'parsing' extra to run the Chroma viewer") from exc

    settings = Settings()
    st.set_page_config(page_title="Trading Card Chroma Viewer", layout="wide")
    st.title("Trading Card Knowledge Base")
    st.caption("Read-only inspection and semantic search using the API's Chroma configuration.")

    st.sidebar.header("Configuration")
    st.sidebar.code(str(settings.chroma_persist_dir))
    st.sidebar.caption(f"Embedding: {settings.rag_embedding_model}")
    st.sidebar.caption(f"Downloads allowed: {settings.allow_model_downloads}")

    try:
        names = ChromaCardKnowledgeBase.list_collection_names(settings.chroma_persist_dir)
    except Exception as exc:
        st.error(f"Unable to open ChromaDB: {exc}")
        return
    if not names:
        st.info("No Chroma collections found.")
        return

    selected = st.sidebar.selectbox("Collection", names)
    try:
        knowledge = _get_knowledge(settings, selected)
    except Exception as exc:
        st.error(f"Unable to load collection: {exc}")
        return

    count = knowledge.count()
    st.metric("Cards", count)
    search_tab, browse_tab = st.tabs(["Semantic search", "Browse metadata"])
    with search_tab:
        query = st.text_input("Query", placeholder="player, brand, series")
        result_limit = st.slider("Results", 1, min(100, max(count, 1)), min(10, max(count, 1)))
        if query and st.button("Search"):
            try:
                results = knowledge.search(query, n_results=result_limit)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.write(f"Found {len(results)} result(s)")
                for result in results:
                    score = result.get("_score") or 0.0
                    with st.expander(f"{result.get('_id', 'unknown')} · score {score:.4f}"):
                        st.json(result)
    with browse_tab:
        page_size = st.number_input("Page size", min_value=1, max_value=1000, value=50)
        offset = st.number_input("Offset", min_value=0, value=0, step=1)
        if st.button("Load metadata"):
            try:
                cards = knowledge.list_cards(limit=int(page_size), offset=int(offset))
            except Exception as exc:
                st.error(str(exc))
            else:
                st.dataframe(cards, use_container_width=True)
                st.download_button(
                    "Download JSON",
                    data=json.dumps(cards, ensure_ascii=False, indent=2),
                    file_name=f"{selected}-cards.json",
                    mime="application/json",
                )


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
