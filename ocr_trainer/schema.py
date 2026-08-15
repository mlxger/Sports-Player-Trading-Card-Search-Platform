from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALL_FIELDS = (
    "name",
    "sport_type",
    "season",
    "country_or_club",
    "brand",
    "license",
    "series",
    "sub_series",
    "rating_agencies",
    "score",
    "limited_edition",
    "card_number",
    "information",
)

DEFAULT_INSTRUCTION = (
    "Extract the trading-card metadata visible in the supplied image(s). "
    "Return one valid JSON object only. Do not guess unsupported values."
)


@dataclass(frozen=True, slots=True)
class CardTrainingExample:
    example_id: str
    group_id: str
    images: tuple[Path, ...]
    fields: dict[str, Any]
    instruction: str = DEFAULT_INSTRUCTION

    def to_llamafactory(self) -> dict[str, Any]:
        image_tokens = "".join("<image>" for _ in self.images)
        return {
            "id": self.example_id,
            "messages": [
                {"role": "user", "content": f"{image_tokens}\n{self.instruction}"},
                {
                    "role": "assistant",
                    "content": json.dumps(self.fields, ensure_ascii=False, sort_keys=True),
                },
            ],
            "images": [str(path) for path in self.images],
        }


def load_examples(path: Path, *, image_root: Path | None = None) -> list[CardTrainingExample]:
    root = (image_root or path.parent).resolve()
    examples: list[CardTrainingExample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        example = _parse_example(raw, root, line_number)
        if example.example_id in seen:
            raise ValueError(f"duplicate example id: {example.example_id}")
        seen.add(example.example_id)
        examples.append(example)
    if not examples:
        raise ValueError("dataset is empty")
    return examples


def _parse_example(raw: object, image_root: Path, line_number: int) -> CardTrainingExample:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number} must be a JSON object")
    example_id = str(raw.get("id") or "").strip()
    if not example_id:
        raise ValueError(f"line {line_number} has no id")
    image_values = raw.get("images")
    if image_values is None and raw.get("image"):
        image_values = [raw["image"]]
    if not isinstance(image_values, list) or not image_values:
        raise ValueError(f"line {line_number} must contain image or images")
    images = tuple(_resolve_image(value, image_root, line_number) for value in image_values)
    fields = raw.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"line {line_number} fields must be an object")
    invalid = sorted(set(fields) - set(ALL_FIELDS))
    if invalid:
        raise ValueError(f"line {line_number} contains unsupported fields: {invalid}")
    if not fields:
        raise ValueError(f"line {line_number} fields cannot be empty")
    normalized = {field: fields.get(field) for field in ALL_FIELDS if field in fields}
    instruction = str(raw.get("instruction") or DEFAULT_INSTRUCTION).strip()
    group_id = str(raw.get("group_id") or example_id).strip()
    return CardTrainingExample(example_id, group_id, images, normalized, instruction)


def _resolve_image(value: object, root: Path, line_number: int) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"line {line_number} image does not exist: {resolved}")
    return resolved
