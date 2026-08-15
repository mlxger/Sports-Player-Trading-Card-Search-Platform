from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ocr_trainer.evaluate import evaluate_predictions
from ocr_trainer.prepare_dataset import prepare_dataset
from ocr_trainer.train import build_config, write_yaml


def test_prepare_dataset_and_evaluate(tmp_path: Path) -> None:
    image = tmp_path / "card.png"
    Image.new("RGB", (8, 8), "white").save(image)
    source = tmp_path / "cards.jsonl"
    records = [
        {"id": "a", "image": "card.png", "fields": {"name": "姚明", "brand": "Topps"}},
        {"id": "b", "image": "card.png", "fields": {"name": "Jordan", "brand": "Upper Deck"}},
    ]
    source.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    output = tmp_path / "prepared"
    manifest = prepare_dataset(source, output, validation_ratio=0.5, seed=7)

    assert manifest == {"total": 2, "groups": 2, "train": 1, "validation": 1}
    assert (output / "dataset_info.json").is_file()
    prepared = json.loads((output / "train.json").read_text(encoding="utf-8"))
    assert prepared[0]["messages"][1]["role"] == "assistant"
    assert prepared[0]["images"] == [str(image.resolve())]

    report = evaluate_predictions(
        {"a": {"name": "姚明", "brand": "Topps"}},
        {"a": {"name": "姚明", "brand": "topps"}},
    )
    assert report["record_exact_match"] == 1.0
    assert report["micro_field_accuracy"] == 1.0


def test_train_config_is_inspectable(tmp_path: Path) -> None:
    (tmp_path / "dataset_info.json").write_text("{}", encoding="utf-8")
    config = build_config(tmp_path, "Qwen/Qwen3-VL-8B-Instruct", tmp_path / "out")
    path = tmp_path / "train.yaml"
    write_yaml(config, path)
    text = path.read_text(encoding="utf-8")
    assert 'finetuning_type: "lora"' in text
    assert "dataset_dir:" in text
