from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def build_config(
    dataset_dir: Path,
    base_model: str,
    output_dir: Path,
    *,
    epochs: float = 3.0,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
) -> dict[str, Any]:
    if not (dataset_dir / "dataset_info.json").is_file():
        raise FileNotFoundError(f"dataset_info.json not found in {dataset_dir}")
    return {
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": lora_rank,
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.05,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset": "card_ocr_train",
        "eval_dataset": "card_ocr_validation",
        "template": "qwen3_vl",
        "cutoff_len": 4096,
        "output_dir": str(output_dir.resolve()),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_steps": 250,
        "eval_steps": 250,
        "eval_strategy": "steps",
        "save_total_limit": 2,
        "plot_loss": True,
    }


def write_yaml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {_yaml_value(value)}" for key, value in config.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_training(config_path: Path, *, executable: str = "llamafactory-cli") -> None:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(
            f"{executable} was not found; install LLaMA Factory in the training environment"
        )
    subprocess.run([resolved, "train", str(config_path)], check=True)


def _yaml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Qwen3-VL LoRA training")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("base_model", help="local path or Hugging Face model id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = build_config(
        args.dataset_dir,
        args.base_model,
        args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
    )
    config_path = args.config_output or args.output_dir / "train_config.yaml"
    write_yaml(config, config_path)
    if not args.dry_run:
        launch_training(config_path)
    print(config_path)


if __name__ == "__main__":
    main()
