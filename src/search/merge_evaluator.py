from __future__ import annotations

import copy
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from src.main import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CV_METRIC_KEYS = ("accuracy", "precision", "recall", "macro_f1", "auc_ovr")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_fold_csvs(folds_dir: str | Path, n_splits: int) -> list[dict[str, Any]]:
    resolved_folds_dir = resolve_path(folds_dir)
    if not resolved_folds_dir.exists():
        raise FileNotFoundError(
            f"Fold directory does not exist: {resolved_folds_dir}. "
            "Run scripts/build_image_splits.py with build_image_splits.mode=kfold first."
        )

    fold_items: list[dict[str, Any]] = []
    for fold_idx in range(int(n_splits)):
        fold_name = f"fold_{fold_idx}"
        train_csv = resolved_folds_dir / f"{fold_name}_train_images.csv"
        val_csv = resolved_folds_dir / f"{fold_name}_val_images.csv"
        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError(
                f"Missing fold CSV files for {fold_name}. Expected:\n"
                f"- {train_csv}\n"
                f"- {val_csv}\n"
                "Please regenerate folds via scripts/build_image_splits.py."
            )

        fold_items.append(
            {
                "fold_idx": int(fold_idx),
                "fold_name": fold_name,
                "train_csv": train_csv,
                "val_csv": val_csv,
            }
        )
    return fold_items


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def compute_metric_statistics(per_fold_rows: list[dict[str, Any]], prefix: str = "val") -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric_name in CV_METRIC_KEYS:
        field_name = f"{prefix}_{metric_name}"
        values: list[float] = []
        for row in per_fold_rows:
            numeric = _to_float(row.get(field_name))
            if numeric is not None:
                values.append(numeric)

        if values:
            mean_value = float(statistics.mean(values))
            std_value = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        else:
            mean_value = None
            std_value = None

        summary[metric_name] = {
            "mean": mean_value,
            "std": std_value,
            "num_folds_with_value": len(values),
        }
    return summary


def write_csv(rows: list[dict[str, Any]], fieldnames: list[str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_mean(metric_summary: dict[str, Any], metric_name: str) -> float | None:
    metric_key = metric_name
    if metric_key.startswith("val_"):
        metric_key = metric_key[4:]
    return _to_float(metric_summary.get(metric_key, {}).get("mean"))


def _metric_std(metric_summary: dict[str, Any], metric_name: str) -> float | None:
    metric_key = metric_name
    if metric_key.startswith("val_"):
        metric_key = metric_key[4:]
    return _to_float(metric_summary.get(metric_key, {}).get("std"))


def _extract_repeat_score(repeat_summary: dict[str, Any], score_metric: str) -> float | None:
    val_summary = repeat_summary.get("val_metrics_summary", {})
    return _metric_mean(val_summary, score_metric)


def _build_fold_result_row(
    fold_item: dict[str, Any],
    run_result: dict[str, Any],
) -> dict[str, Any]:
    val_metrics = run_result.get("best_metrics", {}) or {}
    row: dict[str, Any] = {
        "fold": str(fold_item["fold_name"]),
        "fold_index": int(fold_item["fold_idx"]),
        "train_csv": str(fold_item["train_csv"]),
        "val_csv": str(fold_item["val_csv"]),
        "run_dir": run_result.get("run_dir"),
        "best_model_path": run_result.get("best_model_path"),
        "best_epoch": run_result.get("best_epoch"),
        "primary_metric": run_result.get("primary_metric"),
        "best_metric_value": run_result.get("best_metric_value"),
        "history_path": run_result.get("final_history_path"),
    }
    for metric_name in CV_METRIC_KEYS:
        row[f"val_{metric_name}"] = val_metrics.get(metric_name)
    return row


def _repeat_summary_from_rows(
    repeat_index: int,
    repeat_seed: int,
    per_fold_rows: list[dict[str, Any]],
    repeat_dir: Path,
    score_metric: str,
) -> dict[str, Any]:
    per_fold_rows = sorted(per_fold_rows, key=lambda item: int(item.get("fold_index", 0)))
    per_fold_fields = [
        "fold",
        "fold_index",
        "train_csv",
        "val_csv",
        "run_dir",
        "best_model_path",
        "best_epoch",
        "primary_metric",
        "best_metric_value",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_macro_f1",
        "val_auc_ovr",
        "history_path",
    ]
    per_fold_csv_path = write_csv(per_fold_rows, per_fold_fields, repeat_dir / "per_fold_results.csv")

    val_metrics_summary = compute_metric_statistics(per_fold_rows, prefix="val")
    score_value = _metric_mean(val_metrics_summary, score_metric)

    payload = {
        "repeat_index": int(repeat_index),
        "seed": int(repeat_seed),
        "num_folds": int(len(per_fold_rows)),
        "score_metric": str(score_metric),
        "score_value": score_value,
        "val_metrics_summary": val_metrics_summary,
        "per_fold_results": per_fold_rows,
        "per_fold_results_csv": str(per_fold_csv_path),
        "completed": True,
    }
    summary_path = repeat_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload


def _load_or_run_repeat(
    base_config: dict[str, Any],
    fold_items: list[dict[str, Any]],
    repeat_index: int,
    repeat_seed: int,
    score_metric: str,
    cv_epochs: int | None,
    partition: tuple[tuple[str, ...], ...],
    old_to_new: dict[str, int],
    merged_label_names: list[str],
    scheme_id: str,
    repeat_dir: Path,
) -> dict[str, Any]:
    repeat_dir.mkdir(parents=True, exist_ok=True)
    summary_path = repeat_dir / "summary.json"
    cached_summary = _read_json(summary_path)
    if cached_summary is not None and bool(cached_summary.get("completed", False)):
        cached_summary["summary_path"] = str(summary_path)
        return cached_summary

    per_fold_rows: list[dict[str, Any]] = []
    for fold_item in fold_items:
        fold_name = str(fold_item["fold_name"])
        fold_cache_path = repeat_dir / f"{fold_name}_result.json"
        fold_cached = _read_json(fold_cache_path)
        if fold_cached is not None and bool(fold_cached.get("completed", False)):
            per_fold_rows.append(fold_cached["row"])
            continue

        fold_config = copy.deepcopy(base_config)
        fold_config.setdefault("whole_image", {})
        fold_config["whole_image"]["train_csv"] = str(fold_item["train_csv"])
        fold_config["whole_image"]["val_csv"] = str(fold_item["val_csv"])

        fold_config.setdefault("train", {})
        fold_config["train"]["seed"] = int(repeat_seed)
        if cv_epochs is not None:
            fold_config["train"]["epochs"] = int(cv_epochs)

        fold_config.setdefault("data", {})
        fold_config["data"]["num_classes"] = int(len(merged_label_names))
        fold_config["label_merge"] = {
            "enabled": True,
            "scheme_id": str(scheme_id),
            "partition": [list(group) for group in partition],
            "old_to_new": {str(label): int(index) for label, index in old_to_new.items()},
            "merged_label_names": [str(name) for name in merged_label_names],
        }

        run_result = run_training(
            fold_config,
            run_dir=repeat_dir,
            run_name=fold_name,
        )
        row = _build_fold_result_row(fold_item=fold_item, run_result=run_result)
        fold_cache_path.write_text(
            json.dumps({"completed": True, "row": row}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        per_fold_rows.append(row)

    return _repeat_summary_from_rows(
        repeat_index=repeat_index,
        repeat_seed=repeat_seed,
        per_fold_rows=per_fold_rows,
        repeat_dir=repeat_dir,
        score_metric=score_metric,
    )


def _build_aggregate_repeat_reference(repeat_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric_name in CV_METRIC_KEYS:
        values: list[float] = []
        for repeat_summary in repeat_summaries:
            val_summary = repeat_summary.get("val_metrics_summary", {})
            mean_value = _to_float(val_summary.get(metric_name, {}).get("mean"))
            if mean_value is not None:
                values.append(mean_value)

        if values:
            aggregate[metric_name] = {
                "mean_of_repeat_means": float(statistics.mean(values)),
                "std_of_repeat_means": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                "num_repeats_with_value": len(values),
            }
        else:
            aggregate[metric_name] = {
                "mean_of_repeat_means": None,
                "std_of_repeat_means": None,
                "num_repeats_with_value": 0,
            }
    return aggregate


def evaluate_scheme_with_repeats(
    base_config: dict[str, Any],
    fold_items: list[dict[str, Any]],
    scheme_dir: str | Path,
    scheme_id: str,
    partition: tuple[tuple[str, ...], ...],
    old_to_new: dict[str, int],
    merged_label_names: list[str],
    repeat_seeds: list[int],
    score_metric: str = "val_macro_f1",
    cv_epochs: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_scheme_dir = resolve_path(scheme_dir)
    resolved_scheme_dir.mkdir(parents=True, exist_ok=True)

    partition_payload = {
        "scheme_id": str(scheme_id),
        "num_classes": int(len(partition)),
        "partition": [list(group) for group in partition],
        "old_to_new": {str(label): int(index) for label, index in old_to_new.items()},
        "merged_label_names": [str(name) for name in merged_label_names],
    }
    if extra_metadata:
        partition_payload.update(extra_metadata)
    (resolved_scheme_dir / "partition.json").write_text(
        json.dumps(partition_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = resolved_scheme_dir / "scheme_summary.json"
    cached_summary = _read_json(summary_path)
    if cached_summary is not None and bool(cached_summary.get("completed", False)):
        if int(cached_summary.get("num_repeats", 0)) == int(len(repeat_seeds)):
            return cached_summary

    repeat_summaries: list[dict[str, Any]] = []
    for repeat_index, repeat_seed in enumerate(repeat_seeds):
        repeat_dir = resolved_scheme_dir / f"repeat_{repeat_index}"
        repeat_summary = _load_or_run_repeat(
            base_config=base_config,
            fold_items=fold_items,
            repeat_index=repeat_index,
            repeat_seed=int(repeat_seed),
            score_metric=score_metric,
            cv_epochs=cv_epochs,
            partition=partition,
            old_to_new=old_to_new,
            merged_label_names=merged_label_names,
            scheme_id=scheme_id,
            repeat_dir=repeat_dir,
        )
        repeat_summaries.append(repeat_summary)

    best_repeat_index: int | None = None
    best_repeat_score: float | None = None
    for index, repeat_summary in enumerate(repeat_summaries):
        score = _extract_repeat_score(repeat_summary, score_metric)
        if score is None:
            continue
        if best_repeat_score is None or score > best_repeat_score:
            best_repeat_score = score
            best_repeat_index = index

    best_repeat_summary = repeat_summaries[best_repeat_index] if best_repeat_index is not None else None
    best_val_summary = {} if best_repeat_summary is None else (best_repeat_summary.get("val_metrics_summary", {}) or {})
    aggregate_all_repeats = _build_aggregate_repeat_reference(repeat_summaries)

    summary_payload: dict[str, Any] = {
        "scheme_id": str(scheme_id),
        "num_classes": int(len(partition)),
        "partition": [list(group) for group in partition],
        "old_to_new": {str(label): int(index) for label, index in old_to_new.items()},
        "merged_label_names": [str(name) for name in merged_label_names],
        "score_metric": str(score_metric),
        "score_source": f"best_of_{len(repeat_seeds)}_repeats",
        "score_macro_f1": _metric_mean(best_val_summary, "val_macro_f1"),
        "num_repeats": int(len(repeat_seeds)),
        "repeat_seeds": [int(seed) for seed in repeat_seeds],
        "best_repeat_index": best_repeat_index,
        "best_repeat_mean_macro_f1": _metric_mean(best_val_summary, "val_macro_f1"),
        "best_repeat_std_macro_f1": _metric_std(best_val_summary, "val_macro_f1"),
        "mean_acc_of_best_repeat": _metric_mean(best_val_summary, "val_accuracy"),
        "mean_precision_of_best_repeat": _metric_mean(best_val_summary, "val_precision"),
        "mean_recall_of_best_repeat": _metric_mean(best_val_summary, "val_recall"),
        "mean_auc_ovr_of_best_repeat": _metric_mean(best_val_summary, "val_auc_ovr"),
        "repeat_summaries": repeat_summaries,
        "aggregate_all_repeats": aggregate_all_repeats,
        "completed": True,
    }

    for repeat_index, repeat_summary in enumerate(repeat_summaries):
        val_summary = repeat_summary.get("val_metrics_summary", {})
        summary_payload[f"repeat_{repeat_index}_mean_macro_f1"] = _metric_mean(val_summary, "val_macro_f1")
        summary_payload[f"repeat_{repeat_index}_std_macro_f1"] = _metric_std(val_summary, "val_macro_f1")

    if extra_metadata:
        summary_payload.update(extra_metadata)

    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload
