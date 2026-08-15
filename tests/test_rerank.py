from __future__ import annotations

import numpy as np
from PIL import Image

from models.contracts import SearchHit
from rerank import LightGBMReranker
from scripts.train_ranker import enable_ranker


class FakeRankModel:
    def predict(self, rows: np.ndarray) -> np.ndarray:
        assert rows.shape == (2, 3)
        return rows[:, 0] * 0.2 + rows[:, 1] * 0.3 + rows[:, 2] * 0.5


def test_lightgbm_reranker_uses_contract_features() -> None:
    reranker = LightGBMReranker.__new__(LightGBMReranker)
    reranker._model = FakeRankModel()
    hits = [
        SearchHit("a", "tool", 0.2, metadata={"hog_similarity": 0.9, "border_similarity": 0.9}),
        SearchHit("b", "tool", 0.9, metadata={"hog_similarity": 0.1, "border_similarity": 0.1}),
    ]

    ranked = reranker.rerank(Image.new("RGB", (4, 4)), hits)

    assert [hit.image_id for hit in ranked] == ["a", "b"]
    assert ranked[0].metadata["vector_score"] == 0.2


def test_enable_ranker_updates_dotenv(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("CARD_PIPELINE_RANKER_ENABLED=false\nKEEP_ME=yes\n")
    model_path = tmp_path / "ranking_model.joblib"

    enable_ranker(env_path, model_path)

    text = env_path.read_text()
    assert "CARD_PIPELINE_RANKER_ENABLED=true" in text
    assert f"CARD_PIPELINE_RANKER_MODEL_PATH={model_path.as_posix()}" in text
    assert "KEEP_ME=yes" in text
