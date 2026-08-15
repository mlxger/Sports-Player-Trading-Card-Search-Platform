from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from models.contracts import SearchHit
from router.api import create_app
from settings import Settings


class FakeService:
    def search(self, image_data, request, *, rerank=False):
        assert image_data
        assert request.top_k == 3
        assert request.player_ids == ("p1", "p2")
        assert rerank is True
        return [
            SearchHit(
                rank=1,
                image_id="image-1",
                tool_id="tool-1",
                player_id="p1",
                status=4,
                score=0.991,
                primary_key=12,
            )
        ]

    def close(self) -> None:
        pass


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 18), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_search_endpoint_contract() -> None:
    settings = Settings(default_top_k=10, max_top_k=20)
    app = create_app(settings, service_factory=lambda _: FakeService())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/retrieval/search",
            files={"image": ("card.png", make_png(), "image/png")},
            data={"top_k": "3", "player_ids": "p1,p2", "rerank": "true"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "code": 0,
        "message": "success",
        "data": {
            "results": [
                {
                    "rank": 1,
                    "image_id": "image-1",
                    "tool_id": "tool-1",
                    "player_id": "p1",
                    "status": 4,
                    "similarity": 0.991,
                    "primary_key": 12,
                }
            ]
        },
    }
