# Qwen3-VL LoRA OCR Trainer

This directory contains the project-specific data, training, inference and evaluation workflow.
It does not bundle model weights or fabricate a labeled dataset.

## Canonical source format

One JSON object per line:

```json
{"id":"card-001","group_id":"product-001","images":["images/card-001-front.jpg","images/card-001-back.jpg"],"fields":{"name":"姚明","brand":"Topps","series":"Chrome","season":"2023-24"}}
```

Use the same `group_id` for augmented, rotated, front/back, or near-duplicate samples. Dataset
splitting is group-aware to prevent train/validation leakage.

## Prepare LLaMA Factory data

```bash
python -m ocr_trainer.prepare_dataset data/cards.jsonl data/llamafactory --validation-ratio 0.1
```

This creates `train.json`, `validation.json`, `dataset_info.json`, and `manifest.json`.

## Download the base model or an existing adapter

```bash
python -m ocr_trainer.download_model Qwen/Qwen3-VL-8B-Instruct models/qwen3-vl-8b
```

Use the exact repository id supported by your training environment.

## Train

Install LLaMA Factory according to its official instructions, then run:

```bash
python -m ocr_trainer.train data/llamafactory models/qwen3-vl-8b models/qwen3-vl-card-lora
```

Use `--dry-run` to generate and inspect `train_config.yaml` without starting GPU training.

## Predict and evaluate

```bash
python -m ocr_trainer.predict data/validation.jsonl models/qwen3-vl-8b predictions.jsonl --adapter models/qwen3-vl-card-lora
python -m ocr_trainer.evaluate data/validation.jsonl predictions.jsonl --output metrics.json
```

Metrics include record exact match, micro/macro field accuracy, and per-field presence
precision/recall/F1.
