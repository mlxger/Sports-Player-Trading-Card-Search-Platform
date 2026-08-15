from knowledge import build_ocr_query


def test_build_ocr_query_uses_available_bilingual_search_fields() -> None:
    assert (
        build_ocr_query({"name": "姚明", "brand": "Topps", "series": "Chrome", "score": "9.5"})
        == "球员 player: 姚明 发行商 brand: Topps 大系列 series: Chrome"
    )
    assert build_ocr_query({"name": None, "brand": "", "series": None}) == ""
