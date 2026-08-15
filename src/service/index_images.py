from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from models.contracts import CardRecord
from service.bootstrap import build_indexing_service
from settings import get_settings

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index card images and adjacent JSON metadata")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--real-photo", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    settings = get_settings()
    service = build_indexing_service(settings)
    records: list[CardRecord] = []
    indexed = 0
    try:
        for image_path in _image_paths(args.input_dir):
            metadata_path = image_path.with_suffix(".json")
            if not metadata_path.exists():
                LOGGER.warning("Skipping %s: adjacent JSON metadata is missing", image_path)
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records.append(
                service.prepare_record(
                    image_path.read_bytes(),
                    image_id=str(metadata["image_id"]),
                    tool_id=str(metadata.get("tool_id") or ""),
                    player_id=str(metadata.get("player_id") or ""),
                    status=int(metadata.get("status") or 0),
                    real_photo=args.real_photo,
                )
            )
            if len(records) >= args.batch_size:
                indexed += service.insert_records(records)
                records.clear()
        if records:
            indexed += service.insert_records(records)
    finally:
        service.close()
    LOGGER.info("Indexed %d images", indexed)


def _image_paths(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            yield path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
