from __future__ import annotations

import argparse
from pathlib import Path


def download_model(repo_id: str, output_dir: Path, *, revision: str | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install the 'ocr-training' extra to download model weights") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=output_dir,
    )
    return Path(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-VL base or LoRA adapter weights")
    parser.add_argument("repo_id", help="Hugging Face repository id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    print(download_model(args.repo_id, args.output_dir, revision=args.revision))


if __name__ == "__main__":
    main()
