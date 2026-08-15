from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .schema import CardTrainingExample, load_examples


def prepare_dataset(
    source: Path,
    output_dir: Path,
    *,
    image_root: Path | None = None,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    examples = load_examples(source, image_root=image_root)
    grouped: dict[str, list[CardTrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.group_id].append(example)
    group_ids = list(grouped)
    if len(group_ids) < 2:
        raise ValueError("at least two group_id values are required for leakage-safe splitting")
    random.Random(seed).shuffle(group_ids)
    validation_groups = min(len(group_ids) - 1, max(1, round(len(group_ids) * validation_ratio)))
    validation_ids = set(group_ids[:validation_groups])
    validation = [example for example in examples if example.group_id in validation_ids]
    train = [example for example in examples if example.group_id not in validation_ids]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train.json", train)
    _write_json(output_dir / "validation.json", validation)
    dataset_info = {
        "card_ocr_train": _dataset_entry("train.json"),
        "card_ocr_validation": _dataset_entry("validation.json"),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "total": len(examples),
        "groups": len(group_ids),
        "train": len(train),
        "validation": len(validation),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _write_json(path: Path, examples: list[CardTrainingExample]) -> None:
    payload = [example.to_llamafactory() for example in examples]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dataset_entry(file_name: str) -> dict[str, object]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen3-VL OCR data for LLaMA Factory")
    parser.add_argument("source", type=Path, help="canonical source JSONL")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.source,
        args.output_dir,
        image_root=args.image_root,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
