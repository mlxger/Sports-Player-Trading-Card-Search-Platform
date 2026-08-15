from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import ALL_FIELDS, load_examples


def evaluate_predictions(
    truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing = sorted(set(truth) - set(predictions))
    extras = sorted(set(predictions) - set(truth))
    counters: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "exact": 0,
            "gold_present": 0,
            "predicted_present": 0,
            "true_present": 0,
        }
    )
    record_exact = 0
    for example_id, expected in truth.items():
        predicted = predictions.get(example_id, {})
        all_exact = True
        for field in ALL_FIELDS:
            if field not in expected:
                continue
            gold = _normalize(field, expected.get(field))
            pred = _normalize(field, predicted.get(field))
            stats = counters[field]
            stats["total"] += 1
            stats["gold_present"] += int(gold is not None)
            stats["predicted_present"] += int(pred is not None)
            stats["true_present"] += int(gold is not None and pred == gold)
            exact = gold == pred
            stats["exact"] += int(exact)
            all_exact = all_exact and exact
        record_exact += int(all_exact)

    field_metrics: dict[str, dict[str, float | int]] = {}
    exact_sum = total_sum = 0
    for field, stats in counters.items():
        total = stats["total"]
        precision = _safe_div(stats["true_present"], stats["predicted_present"])
        recall = _safe_div(stats["true_present"], stats["gold_present"])
        field_metrics[field] = {
            **stats,
            "accuracy": _safe_div(stats["exact"], total),
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_f1": _safe_div(2 * precision * recall, precision + recall),
        }
        exact_sum += stats["exact"]
        total_sum += total
    return {
        "records": len(truth),
        "predicted_records": len(predictions),
        "missing_prediction_ids": missing,
        "unexpected_prediction_ids": extras,
        "record_exact_match": _safe_div(record_exact, len(truth)),
        "micro_field_accuracy": _safe_div(exact_sum, total_sum),
        "macro_field_accuracy": _safe_div(
            sum(float(metrics["accuracy"]) for metrics in field_metrics.values()),
            len(field_metrics),
        ),
        "fields": field_metrics,
    }


def load_truth(path: Path, *, image_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {
        example.example_id: example.fields for example in load_examples(path, image_root=image_root)
    }


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        example_id = str(raw.get("id") or "").strip()
        value = raw.get("prediction", raw.get("fields"))
        if not example_id or not isinstance(value, dict):
            raise ValueError(f"invalid prediction on line {line_number}")
        if example_id in predictions:
            raise ValueError(f"duplicate prediction id: {example_id}")
        predictions[example_id] = value
    return predictions


def _normalize(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(_normalize(field, item) for item in value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return None
    if field == "limited_edition":
        text = re.sub(r"\s+of\s+", "/", text, flags=re.IGNORECASE)
        text = text.replace("One of One", "1/1").replace("one of one", "1/1")
    return text.casefold()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured OCR prediction JSONL")
    parser.add_argument("truth", type=Path, help="canonical source JSONL")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_predictions(
        load_truth(args.truth, image_root=args.image_root),
        load_predictions(args.predictions),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
