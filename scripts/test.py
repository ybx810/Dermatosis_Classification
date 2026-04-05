from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.test import run_test_from_checkpoint
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained whole-image classification checkpoint on the test split.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_yaml(resolve_path(args.config))
    checkpoint_path = resolve_path(args.checkpoint)
    run_dir = resolve_path(args.run_dir) if args.run_dir is not None else checkpoint_path.parent

    metrics = run_test_from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
    )
    logging.info(
        "Finished test evaluation | sample_level=%s acc=%.4f macro_f1=%.4f loss=%.4f",
        metrics.get("sample_level", "image"),
        metrics.get("accuracy", 0.0),
        metrics.get("macro_f1", 0.0),
        metrics.get("loss", 0.0),
    )


if __name__ == "__main__":
    main()