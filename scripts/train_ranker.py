"""Train the LightGBM ranking model used by ``LightGBMReranker``.

Input CSV columns must include ``query_id``, ``label`` and the three feature columns
``initial_score``, ``hog_similarity`` and ``border_similarity``. Each query_id forms
one ranking group.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

FEATURE_COLUMNS = ["initial_score", "hog_similarity", "border_similarity"]


def train(input_path: Path, output_path: Path, *, estimators: int = 300) -> None:
    try:
        import joblib
        import pandas as pd
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise RuntimeError("install the 'ranking' extra to train the ranker") from exc

    frame = pd.read_csv(input_path)
    required = {"query_id", "label", *FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"training CSV is missing columns: {missing}")
    if frame.empty:
        raise ValueError("training CSV is empty")
    frame = frame.sort_values("query_id", kind="stable")
    group_sizes = frame.groupby("query_id", sort=False).size().tolist()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[10],
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(frame[FEATURE_COLUMNS], frame["label"], group=group_sizes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def enable_ranker(env_path: Path, model_path: Path) -> None:
    """Create/update only the ranker settings in a dotenv file."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updates = {
        "CARD_PIPELINE_RANKER_ENABLED": "true",
        "CARD_PIPELINE_RANKER_MODEL_PATH": str(model_path).replace("\\", "/"),
    }
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    if output and output[-1] != "":
        output.append("")
    output.extend(f"{key}={value}" for key, value in updates.items() if key not in seen)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def download_model(url: str, output_path: Path, *, sha256: str | None = None) -> None:
    """Download a user-provided, project-trained joblib model and validate its contract."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("model URL must use http or https")
    digest = hashlib.sha256()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, output_path.open("wb") as target:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    actual = digest.hexdigest()
    if sha256 and actual.lower() != sha256.lower():
        output_path.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch: expected {sha256}, got {actual}")
    try:
        import joblib

        model = joblib.load(output_path)
        if not hasattr(model, "predict"):
            raise TypeError("downloaded artifact has no predict() method")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing ranking samples")
    parser.add_argument("output", type=Path, help="output .joblib path")
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--enable-env", type=Path, help="dotenv file to update after training")
    args = parser.parse_args()
    train(args.input, args.output, estimators=args.estimators)
    if args.enable_env:
        enable_ranker(args.enable_env, args.output)


if __name__ == "__main__":
    main()
