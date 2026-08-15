"""Download and validate a project-trained LightGBM ranker artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.train_ranker import download_model, enable_ranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="HTTPS URL for ranking_model.joblib")
    parser.add_argument("output", type=Path, help="destination .joblib path")
    parser.add_argument("--sha256", required=True, help="expected SHA-256 checksum")
    parser.add_argument("--enable-env", type=Path, help="dotenv file to update")
    args = parser.parse_args()
    download_model(args.url, args.output, sha256=args.sha256)
    if args.enable_env:
        enable_ranker(args.enable_env, args.output)


if __name__ == "__main__":
    main()
