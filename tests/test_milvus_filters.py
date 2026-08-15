from models.contracts import SearchRequest
from retrieval import build_filter_expression


def test_build_filter_expression_combines_and_escapes_filters() -> None:
    request = SearchRequest(
        status=4,
        player_ids=('player"1', "player-2"),
        tool_ids=("tool-1", "tool-1"),
    )

    expression = build_filter_expression(request)

    assert expression == (
        'status == 4 && player_id in ["player\\"1", "player-2"] && tool_id in ["tool-1"]'
    )


def test_build_filter_expression_returns_none_without_filters() -> None:
    assert build_filter_expression(SearchRequest()) is None
