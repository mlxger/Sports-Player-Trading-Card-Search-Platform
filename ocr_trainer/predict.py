from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import CardTrainingExample, load_examples


def load_model(base_model: str, adapter_path: Path | None = None, *, device_map: str = "auto"):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("install the 'ocr-training' extra to run adapter evaluation") from exc
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model.eval(), processor


def predict_example(model, processor, example: CardTrainingExample, *, max_new_tokens: int) -> dict:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": str(image)} for image in example.images
    ]
    content.append({"type": "text", "text": example.instruction})
    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = inputs["input_ids"].shape[1]
    text = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
    return _extract_json(text)


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last = value
    if last is None:
        raise ValueError(f"model response contains no JSON object: {text!r}")
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL/LoRA predictions for OCR evaluation")
    parser.add_argument("dataset", type=Path, help="canonical source JSONL")
    parser.add_argument("base_model")
    parser.add_argument("output", type=Path, help="prediction JSONL")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    examples = load_examples(args.dataset, image_root=args.image_root)
    model, processor = load_model(args.base_model, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for example in examples:
            prediction = predict_example(
                model, processor, example, max_new_tokens=args.max_new_tokens
            )
            output.write(
                json.dumps(
                    {"id": example.example_id, "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
