from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.features import extract_all_splits
from src.ml.io import PROJECT_ROOT as SRC_PROJECT_ROOT
from src.ml.io import build_feature_run_dir, resolve_project_path, save_yaml_snapshot
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract whole-image CNN embeddings for train/val/test splits.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
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
    return resolve_project_path(path_value, SRC_PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_yaml(resolve_path(args.config))
    ml_enabled = bool(config.get("ml_experiment", {}).get("enabled", False))
    if not ml_enabled:
        logging.warning("ml_experiment.enabled is false. Running feature extraction because this script was called explicitly.")

    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        configured_output = resolve_path(config.get("ml_experiment", {}).get("feature_dir"))
        if configured_output is not None:
            output_dir = configured_output
        else:
            output_dir = build_feature_run_dir(config, run_name=args.run_name, project_root=PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_snapshot(config, output_dir / "ml_config.yaml")

    logging.info("Feature output directory: %s", output_dir)
    info = extract_all_splits(config=config, output_dir=output_dir, run_tag=args.run_name)

    logging.info(
        "Feature extraction completed | source=%s backbone=%s train=%s val=%s test=%s",
        info.get("feature_source"),
        info.get("backbone"),
        info.get("splits", {}).get("train", {}).get("num_samples", "N/A"),
        info.get("splits", {}).get("val", {}).get("num_samples", "N/A"),
        info.get("splits", {}).get("test", {}).get("num_samples", "N/A"),
    )
    logging.info("feature_info.json: %s", output_dir / "feature_info.json")


if __name__ == "__main__":
    main()
