from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = "configs/default.yaml"
DEFAULT_SUMMARY_DIR = "outputs/comparison"
METRIC_FIELDS = ["accuracy", "precision", "recall", "macro_f1", "auc_ovr"]
CSV_COLUMNS = [
    "experiment",
    "metrics_path",
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "auc_ovr",
    "status",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics under outputs/.")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--summary-dir", type=str, default=None)
    return parser.parse_args()



def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path



def load_yaml_config(config_path: Path) -> dict[str, Any]:
    # Load YAML config as a top-level mapping and surface explicit read/parse errors.
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {config_path}. {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read config file: {config_path}. {exc}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML config format (expected mapping): {config_path}")
    return payload



def resolve_summary_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    # Priority: CLI --summary-dir > YAML summary.summarize_dir > default.
    if args.summary_dir:
        return resolve_path(args.summary_dir)

    summary_config = config.get("summary", {})
    if isinstance(summary_config, dict):
        yaml_summary_dir = summary_config.get("summarize_dir")
        if yaml_summary_dir:
            return resolve_path(str(yaml_summary_dir))

    return resolve_path(DEFAULT_SUMMARY_DIR)



def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)



def infer_experiment_dir(metrics_path: Path, outputs_dir: Path) -> Path:
    if metrics_path.parent.name == "test":
        return metrics_path.parent.parent
    return metrics_path.parent



def collect_metric_rows(outputs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metrics_files = sorted(path for path in outputs_dir.rglob("metrics.json") if "summary" not in path.parts)

    for metrics_path in metrics_files:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment_dir = infer_experiment_dir(metrics_path, outputs_dir)
        try:
            experiment_name = experiment_dir.relative_to(outputs_dir).as_posix()
        except ValueError:
            experiment_name = experiment_dir.as_posix()

        row: dict[str, str] = {
            "experiment": experiment_name,
            "metrics_path": metrics_path.as_posix(),
        }
        missing_metrics: list[str] = []
        for field in METRIC_FIELDS:
            if field not in payload or payload.get(field) is None:
                row[field] = "N/A"
                missing_metrics.append(field)
            else:
                row[field] = format_metric(payload[field])

        row["status"] = "missing: " + ", ".join(missing_metrics) if missing_metrics else "ok"
        rows.append(row)

    return rows



def write_summary_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return output_path



def build_markdown_table(rows: list[dict[str, str]]) -> str:
    header = "| Experiment | Accuracy | Precision | Recall | Macro F1 | AUC OVR | Status |"
    separator = "|---|---:|---:|---:|---:|---:|---|"
    if not rows:
        return "# Experiment Summary\n\nNo `metrics.json` files were found under `outputs/`."

    lines = ["# Experiment Summary", "", header, separator]
    for row in rows:
        lines.append(
            "| {experiment} | {accuracy} | {precision} | {recall} | {macro_f1} | {auc_ovr} | {status} |".format(
                **row
            )
        )
    return "\n".join(lines)



def write_summary_markdown(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_table(rows), encoding="utf-8")
    return output_path



def summarize_results(outputs_dir: Path, summary_dir: Path) -> tuple[Path, Path, int]:
    rows = collect_metric_rows(outputs_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_summary_csv(rows, summary_dir / "summary.csv")
    markdown_path = write_summary_markdown(rows, summary_dir / "summary.md")
    return csv_path, markdown_path, len(rows)



def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    try:
        config = load_yaml_config(config_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    outputs_dir = resolve_path(args.outputs_dir)
    summary_dir = resolve_summary_dir(args, config)
    csv_path, markdown_path, count = summarize_results(outputs_dir, summary_dir)
    print(f"Collected {count} experiment result(s).")
    print(f"CSV summary saved to: {csv_path}")
    print(f"Markdown summary saved to: {markdown_path}")


if __name__ == "__main__":
    main()
