from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.confusion_prior import load_historical_confusion_prior
from src.search.merge_evaluator import collect_fold_csvs, evaluate_scheme_with_repeats
from src.utils.io import load_yaml
from src.utils.label_merge import build_merged_label_names, build_old_to_new_mapping_from_partition, load_label_mapping_from_json

DEFAULT_SEARCH_MERGE: dict[str, Any] = {
    "enabled": True,
    "target_num_classes": [6, 5, 4, 3],
    "beam_width": 8,
    "top_n_final_per_target": 3,
    "scoring": {
        "metric": "val_macro_f1",
        "choose_best_repeat": True,
    },
    "cv": {
        "enabled": True,
        "n_splits": 3,
        "epochs": 15,
        "reuse_existing_folds": True,
        "use_fixed_test_for_search": False,
    },
    "repeats": {
        "num_repeats": 3,
        "seed_mode": "offset",
        "base_seed": 42,
        "seed_offsets": [0, 1, 2],
    },
    "cache": {
        "enabled": True,
        "dir": "outputs/merge_search_cache",
    },
    "historical_confusion": {
        "enabled": False,
        "paths": [],
        "input_type": "auto",
        "normalize": True,
        "average_multiple": True,
    },
    "medical_constraints": {
        "must_stay_single": [],
        "forbidden_pairs": [],
        "preferred_pairs": [],
    },
    "output": {
        "dir": "outputs/merge_search",
        "save_all_candidates": True,
        "save_repeat_details": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search best label-merge schemes with beam search on validation-only CV.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("search_label_merge")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(output_dir / "search.log", encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_target_num_classes(values: list[Any], max_classes: int) -> list[int]:
    normalized = sorted({int(value) for value in values}, reverse=True)
    if not normalized:
        raise ValueError("search_merge.target_num_classes cannot be empty.")
    if any(value < 2 for value in normalized):
        raise ValueError(f"search_merge.target_num_classes must be >=2. Got {normalized}")
    if any(value >= max_classes for value in normalized):
        raise ValueError(
            "target_num_classes must be less than the number of original classes. "
            f"max_classes={max_classes}, target_num_classes={normalized}"
        )
    return normalized


def _parse_pair_item(item: Any, field_name: str, allowed_labels: set[str], label_order: dict[str, int]) -> tuple[str, str]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        first, second = str(item[0]), str(item[1])
    elif isinstance(item, dict):
        first = str(item.get("a", item.get("left", "")))
        second = str(item.get("b", item.get("right", "")))
    else:
        raise ValueError(f"{field_name} items must be [label_a, label_b] pairs. Invalid item: {item}")

    if first not in allowed_labels or second not in allowed_labels:
        raise ValueError(f"{field_name} contains unknown labels: {item}")
    if first == second:
        raise ValueError(f"{field_name} cannot contain identical labels in one pair: {item}")

    if label_order[first] <= label_order[second]:
        return first, second
    return second, first


def canonicalize_partition(partition: list[list[str]] | tuple[tuple[str, ...], ...], label_order: dict[str, int]) -> tuple[tuple[str, ...], ...]:
    normalized_groups: list[tuple[str, ...]] = []
    for group in partition:
        unique_labels = sorted({str(label) for label in group}, key=lambda label: (label_order[label], label))
        if not unique_labels:
            raise ValueError("Partition contains an empty group.")
        normalized_groups.append(tuple(unique_labels))

    normalized_groups = sorted(
        normalized_groups,
        key=lambda group: tuple(label_order[label] for label in group),
    )
    return tuple(normalized_groups)


def partition_signature(partition: tuple[tuple[str, ...], ...]) -> str:
    return "||".join([",".join(group) for group in partition])


def scheme_id_from_partition(partition: tuple[tuple[str, ...], ...]) -> str:
    signature = partition_signature(partition)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    return digest[:12]


def is_partition_valid(
    partition: tuple[tuple[str, ...], ...],
    must_stay_single: set[str],
    forbidden_pairs: set[tuple[str, str]],
) -> bool:
    for group in partition:
        group_set = set(group)
        if len(group_set) > 1 and any(label in must_stay_single for label in group_set):
            return False

        for first, second in forbidden_pairs:
            if first in group_set and second in group_set:
                return False
    return True


def _group_pair_metadata(
    group_a: tuple[str, ...],
    group_b: tuple[str, ...],
    preferred_pairs: set[tuple[str, str]],
    label_to_index: dict[str, int],
    confusion_scores: Any,
    use_confusion: bool,
) -> dict[str, Any]:
    preferred_hit = False
    for label_a in group_a:
        for label_b in group_b:
            pair = (label_a, label_b) if label_to_index[label_a] <= label_to_index[label_b] else (label_b, label_a)
            if pair in preferred_pairs:
                preferred_hit = True
                break
        if preferred_hit:
            break

    confusion_score = 0.0
    if use_confusion and confusion_scores is not None:
        for label_a in group_a:
            for label_b in group_b:
                i = label_to_index[label_a]
                j = label_to_index[label_b]
                confusion_score = max(confusion_score, float(confusion_scores[i, j]))

    return {
        "preferred_hit": preferred_hit,
        "confusion_score": float(confusion_score),
        "used_confusion_prior": bool(use_confusion and confusion_score > 0.0),
        "used_medical_preference": bool(preferred_hit),
    }


def expand_beam_layer(
    beam_states: list[dict[str, Any]],
    label_order: dict[str, int],
    must_stay_single: set[str],
    forbidden_pairs: set[tuple[str, str]],
    preferred_pairs: set[tuple[str, str]],
    label_to_index: dict[str, int],
    confusion_scores: Any,
    use_confusion: bool,
) -> list[dict[str, Any]]:
    candidate_by_partition: dict[str, dict[str, Any]] = {}

    for parent in beam_states:
        partition = parent["partition"]
        parent_score = parent.get("score_macro_f1")
        parent_score_value = float(parent_score) if parent_score is not None else -1.0

        groups = list(partition)
        group_count = len(groups)
        for i in range(group_count):
            for j in range(i + 1, group_count):
                merged_group = tuple(sorted({*groups[i], *groups[j]}, key=lambda label: (label_order[label], label)))
                next_groups = [groups[index] for index in range(group_count) if index not in {i, j}]
                next_groups.append(merged_group)
                next_partition = canonicalize_partition(next_groups, label_order)

                if not is_partition_valid(next_partition, must_stay_single, forbidden_pairs):
                    continue

                metadata = _group_pair_metadata(
                    group_a=groups[i],
                    group_b=groups[j],
                    preferred_pairs=preferred_pairs,
                    label_to_index=label_to_index,
                    confusion_scores=confusion_scores,
                    use_confusion=use_confusion,
                )

                signature = partition_signature(next_partition)
                generation_priority = (
                    1 if metadata["preferred_hit"] else 0,
                    float(metadata["confusion_score"]),
                    parent_score_value,
                )

                existing = candidate_by_partition.get(signature)
                candidate_payload = {
                    "partition": next_partition,
                    "partition_signature": signature,
                    "parent_scheme_id": parent.get("scheme_id"),
                    "beam_depth": int(parent.get("beam_depth", 0)) + 1,
                    "used_confusion_prior": bool(metadata["used_confusion_prior"]),
                    "used_medical_preference": bool(metadata["used_medical_preference"]),
                    "generation_priority": generation_priority,
                }

                if existing is None or candidate_payload["generation_priority"] > existing["generation_priority"]:
                    candidate_by_partition[signature] = candidate_payload

    return list(candidate_by_partition.values())


def write_csv(rows: list[dict[str, Any]], fieldnames: list[str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _build_repeat_seeds(repeat_cfg: dict[str, Any]) -> list[int]:
    num_repeats = int(repeat_cfg.get("num_repeats", 3))
    if num_repeats <= 0:
        raise ValueError(f"search_merge.repeats.num_repeats must be positive, got {num_repeats}")

    seed_mode = str(repeat_cfg.get("seed_mode", "offset")).lower()
    if seed_mode == "offset":
        base_seed = int(repeat_cfg.get("base_seed", 42))
        offsets = repeat_cfg.get("seed_offsets", list(range(num_repeats)))
        offsets = [int(offset) for offset in offsets]
        if len(offsets) < num_repeats:
            raise ValueError(
                "search_merge.repeats.seed_offsets length must be >= num_repeats in offset mode. "
                f"Got offsets={offsets}, num_repeats={num_repeats}"
            )
        return [base_seed + offsets[index] for index in range(num_repeats)]

    if seed_mode == "explicit":
        values = [int(item) for item in repeat_cfg.get("seeds", [])]
        if len(values) != num_repeats:
            raise ValueError(
                "search_merge.repeats.seeds length must match num_repeats in explicit mode. "
                f"Got seeds={values}, num_repeats={num_repeats}"
            )
        return values

    raise ValueError(f"Unsupported search_merge.repeats.seed_mode: {seed_mode}")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def build_candidate_row(
    candidate: dict[str, Any],
    scheme_summary: dict[str, Any],
    repeat_count: int,
) -> dict[str, Any]:
    partition = candidate["partition"]
    old_to_new = build_old_to_new_mapping_from_partition([list(group) for group in partition])

    row: dict[str, Any] = {
        "scheme_id": candidate["scheme_id"],
        "target_num_classes": int(len(partition)),
        "partition_canonical": json.dumps([list(group) for group in partition], ensure_ascii=False),
        "label_map_json": json.dumps(old_to_new, ensure_ascii=False, sort_keys=True),
        "beam_depth": int(candidate.get("beam_depth", 0)),
        "parent_scheme_id": candidate.get("parent_scheme_id"),
        "score_macro_f1": scheme_summary.get("score_macro_f1"),
        "score_source": scheme_summary.get("score_source"),
        "best_repeat_index": scheme_summary.get("best_repeat_index"),
        "best_repeat_mean_macro_f1": scheme_summary.get("best_repeat_mean_macro_f1"),
        "best_repeat_std_macro_f1": scheme_summary.get("best_repeat_std_macro_f1"),
        "mean_acc_of_best_repeat": scheme_summary.get("mean_acc_of_best_repeat"),
        "mean_precision_of_best_repeat": scheme_summary.get("mean_precision_of_best_repeat"),
        "mean_recall_of_best_repeat": scheme_summary.get("mean_recall_of_best_repeat"),
        "mean_auc_ovr_of_best_repeat": scheme_summary.get("mean_auc_ovr_of_best_repeat"),
        "used_confusion_prior": bool(candidate.get("used_confusion_prior", False)),
        "used_medical_preference": bool(candidate.get("used_medical_preference", False)),
        "scheme_dir": candidate.get("scheme_dir"),
    }

    for repeat_index in range(max(3, repeat_count)):
        row[f"repeat_{repeat_index}_mean_macro_f1"] = scheme_summary.get(f"repeat_{repeat_index}_mean_macro_f1")
        row[f"repeat_{repeat_index}_std_macro_f1"] = scheme_summary.get(f"repeat_{repeat_index}_std_macro_f1")

    return row


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
    score = _float_or_none(candidate.get("score_macro_f1"))
    if score is None:
        score = -1.0
    best_std = _float_or_none(candidate.get("best_repeat_std_macro_f1"))
    if best_std is None:
        best_std = 1.0
    return (score, -best_std, str(candidate.get("scheme_id", "")))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml(config_path)
    search_merge_cfg = deep_merge_dict(DEFAULT_SEARCH_MERGE, config.get("search_merge", {}))
    if not bool(search_merge_cfg.get("enabled", True)):
        search_merge_cfg["enabled"] = True

    if args.beam_width is not None:
        search_merge_cfg["beam_width"] = int(args.beam_width)
    if args.epochs is not None:
        search_merge_cfg.setdefault("cv", {})["epochs"] = int(args.epochs)
    if args.output_dir is not None:
        search_merge_cfg.setdefault("output", {})["dir"] = str(args.output_dir)
    if args.cache_dir is not None:
        search_merge_cfg.setdefault("cache", {})["dir"] = str(args.cache_dir)

    output_dir = resolve_path(search_merge_cfg.get("output", {}).get("dir", "outputs/merge_search"))
    cache_dir = resolve_path(search_merge_cfg.get("cache", {}).get("dir", "outputs/merge_search_cache"))
    logger = setup_logger(output_dir)

    split_dir_value = config.get("data", {}).get("split_dir", "data/splits")
    default_label_mapping = Path(split_dir_value) / "label_mapping.json"
    label_mapping_path = resolve_path(
        config.get("build_image_splits", {}).get("label_mapping_path")
        or default_label_mapping
    )
    label_mapping = load_label_mapping_from_json(label_mapping_path)
    if label_mapping is None:
        raise FileNotFoundError(
            f"Could not load label mapping: {label_mapping_path}. "
            "Please run scripts/build_image_splits.py first."
        )

    base_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    label_order = {label: index for index, label in enumerate(base_labels)}
    label_set = set(base_labels)

    target_num_classes = _normalize_target_num_classes(
        values=search_merge_cfg.get("target_num_classes", [6, 5, 4, 3]),
        max_classes=len(base_labels),
    )
    minimum_target_classes = int(min(target_num_classes))

    beam_width = int(search_merge_cfg.get("beam_width", 8))
    if beam_width <= 0:
        raise ValueError(f"search_merge.beam_width must be positive, got: {beam_width}")

    top_n_final_per_target = int(search_merge_cfg.get("top_n_final_per_target", 3))
    if top_n_final_per_target <= 0:
        raise ValueError(
            "search_merge.top_n_final_per_target must be positive. "
            f"Got: {top_n_final_per_target}"
        )

    scoring_cfg = search_merge_cfg.get("scoring", {})
    score_metric = str(scoring_cfg.get("metric", "val_macro_f1")).lower()
    if score_metric != "val_macro_f1":
        raise ValueError(
            "Only search_merge.scoring.metric=val_macro_f1 is supported for now. "
            f"Got: {score_metric}"
        )

    cv_cfg = search_merge_cfg.get("cv", {})
    if not bool(cv_cfg.get("enabled", True)):
        raise ValueError("search_merge.cv.enabled must be true.")
    if not bool(cv_cfg.get("reuse_existing_folds", True)):
        raise ValueError("search_merge.cv.reuse_existing_folds=false is not supported; please reuse existing folds.")
    if bool(cv_cfg.get("use_fixed_test_for_search", False)):
        raise ValueError("search_merge.cv.use_fixed_test_for_search must be false during search.")

    n_splits = int(cv_cfg.get("n_splits", config.get("build_image_splits", {}).get("n_splits", 3)))
    cv_epochs = int(cv_cfg.get("epochs", config.get("train", {}).get("epochs", 15)))

    folds_dir = resolve_path(config.get("build_image_splits", {}).get("folds_dir", "data/splits/cv3"))
    fold_items = collect_fold_csvs(folds_dir=folds_dir, n_splits=n_splits)

    medical_cfg = search_merge_cfg.get("medical_constraints", {})
    must_stay_single = {str(label) for label in medical_cfg.get("must_stay_single", [])}
    unknown_must_labels = sorted(must_stay_single.difference(label_set))
    if unknown_must_labels:
        raise ValueError(f"medical_constraints.must_stay_single contains unknown labels: {unknown_must_labels}")

    forbidden_pairs = {
        _parse_pair_item(item, "medical_constraints.forbidden_pairs", label_set, label_order)
        for item in medical_cfg.get("forbidden_pairs", [])
    }
    preferred_pairs = {
        _parse_pair_item(item, "medical_constraints.preferred_pairs", label_set, label_order)
        for item in medical_cfg.get("preferred_pairs", [])
    }

    historical_cfg = search_merge_cfg.get("historical_confusion", {})
    raw_historical_paths = historical_cfg.get("paths", [])
    historical_paths: list[Any] = []
    for entry in raw_historical_paths:
        if isinstance(entry, dict):
            normalized_entry = dict(entry)
            if "path" in normalized_entry:
                normalized_entry["path"] = str(resolve_path(normalized_entry["path"]))
            if "y_true_path" in normalized_entry:
                normalized_entry["y_true_path"] = str(resolve_path(normalized_entry["y_true_path"]))
            if "y_pred_path" in normalized_entry:
                normalized_entry["y_pred_path"] = str(resolve_path(normalized_entry["y_pred_path"]))
            historical_paths.append(normalized_entry)
        else:
            historical_paths.append(str(resolve_path(entry)))

    use_confusion_prior = bool(historical_cfg.get("enabled", False) and historical_paths)
    confusion_scores = None
    confusion_info: dict[str, Any] = {
        "enabled": bool(historical_cfg.get("enabled", False)),
        "used": False,
        "sources": [],
        "source_count": 0,
    }
    if use_confusion_prior:
        prior_payload = load_historical_confusion_prior(
            paths=historical_paths,
            base_labels=base_labels,
            input_type=str(historical_cfg.get("input_type", "auto")),
            normalize=bool(historical_cfg.get("normalize", True)),
            average_multiple=bool(historical_cfg.get("average_multiple", True)),
        )
        confusion_scores = prior_payload.get("conf_scores")
        confusion_info = {
            "enabled": bool(prior_payload.get("enabled", False)),
            "used": bool(prior_payload.get("used", False)),
            "sources": prior_payload.get("sources", []),
            "source_count": int(prior_payload.get("source_count", 0)),
        }

    repeat_seeds = _build_repeat_seeds(search_merge_cfg.get("repeats", {}))

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Search config loaded from %s", config_path)
    logger.info("Original classes: %d", len(base_labels))
    logger.info("Target num_classes: %s", target_num_classes)
    logger.info("Beam width: %d", beam_width)
    logger.info("Repeat seeds: %s", repeat_seeds)
    logger.info("Folds dir: %s | n_splits=%d", folds_dir, n_splits)
    logger.info("Search scoring metric: %s (validation-only)", score_metric)

    initial_partition = canonicalize_partition([[label] for label in base_labels], label_order)
    initial_state = {
        "scheme_id": "root",
        "partition": initial_partition,
        "beam_depth": 0,
        "score_macro_f1": None,
    }
    beam_states: list[dict[str, Any]] = [initial_state]

    all_candidates_by_id: dict[str, dict[str, Any]] = {}
    target_top_candidates: dict[int, list[dict[str, Any]]] = {target: [] for target in target_num_classes}
    beam_history: dict[str, Any] = {
        str(len(base_labels)): {
            "beam_scheme_ids": ["root"],
            "num_candidates": 1,
        }
    }

    for next_num_classes in range(len(base_labels) - 1, minimum_target_classes - 1, -1):
        logger.info("Expanding to %d classes...", next_num_classes)
        generated = expand_beam_layer(
            beam_states=beam_states,
            label_order=label_order,
            must_stay_single=must_stay_single,
            forbidden_pairs=forbidden_pairs,
            preferred_pairs=preferred_pairs,
            label_to_index=label_order,
            confusion_scores=confusion_scores,
            use_confusion=bool(confusion_info.get("used", False)),
        )

        logger.info("Generated %d unique candidates at %d classes.", len(generated), next_num_classes)

        evaluated_candidates: list[dict[str, Any]] = []
        for candidate in generated:
            partition = candidate["partition"]
            candidate["scheme_id"] = scheme_id_from_partition(partition)
            old_to_new = build_old_to_new_mapping_from_partition([list(group) for group in partition])
            merged_label_names = build_merged_label_names([list(group) for group in partition])

            scheme_dir = cache_dir / f"scheme_{candidate['scheme_id']}"
            candidate["scheme_dir"] = str(scheme_dir)

            scheme_summary = evaluate_scheme_with_repeats(
                base_config=config,
                fold_items=fold_items,
                scheme_dir=scheme_dir,
                scheme_id=candidate["scheme_id"],
                partition=partition,
                old_to_new=old_to_new,
                merged_label_names=merged_label_names,
                repeat_seeds=repeat_seeds,
                score_metric=score_metric,
                cv_epochs=cv_epochs,
                extra_metadata={
                    "beam_depth": int(candidate.get("beam_depth", 0)),
                    "parent_scheme_id": candidate.get("parent_scheme_id"),
                    "used_confusion_prior": bool(candidate.get("used_confusion_prior", False)),
                    "used_medical_preference": bool(candidate.get("used_medical_preference", False)),
                },
            )

            candidate["score_macro_f1"] = scheme_summary.get("score_macro_f1")
            candidate["best_repeat_std_macro_f1"] = scheme_summary.get("best_repeat_std_macro_f1")

            row = build_candidate_row(candidate=candidate, scheme_summary=scheme_summary, repeat_count=len(repeat_seeds))
            all_candidates_by_id[candidate["scheme_id"]] = row
            evaluated_candidates.append(row)

        evaluated_candidates = sorted(evaluated_candidates, key=_candidate_sort_key, reverse=True)

        beam_states = []
        for item in evaluated_candidates[:beam_width]:
            beam_states.append(
                {
                    "scheme_id": item["scheme_id"],
                    "partition": canonicalize_partition(json.loads(item["partition_canonical"]), label_order),
                    "beam_depth": int(item.get("beam_depth", 0)),
                    "score_macro_f1": item.get("score_macro_f1"),
                }
            )

        if next_num_classes in target_top_candidates:
            target_top_candidates[next_num_classes] = evaluated_candidates[:top_n_final_per_target]

        layer_payload = {
            "num_classes": int(next_num_classes),
            "generated_candidate_count": int(len(generated)),
            "evaluated_candidate_count": int(len(evaluated_candidates)),
            "beam_width": int(beam_width),
            "beam_scheme_ids": [state["scheme_id"] for state in beam_states],
            "top_candidates": evaluated_candidates[: min(top_n_final_per_target, len(evaluated_candidates))],
        }
        beam_history[str(next_num_classes)] = layer_payload
        (layer_dir / f"layer_{next_num_classes}.json").write_text(
            json.dumps(layer_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        best_layer_score = None if not evaluated_candidates else _float_or_none(evaluated_candidates[0].get("score_macro_f1"))
        logger.info(
            "Layer %d complete | kept %d beam states | best score=%s",
            next_num_classes,
            len(beam_states),
            "N/A" if best_layer_score is None else f"{best_layer_score:.4f}",
        )

    all_candidates = sorted(
        all_candidates_by_id.values(),
        key=lambda item: (
            int(item.get("target_num_classes", 0)),
            _candidate_sort_key(item)[0],
            str(item.get("scheme_id", "")),
        ),
        reverse=True,
    )

    all_candidate_fields = [
        "scheme_id",
        "target_num_classes",
        "partition_canonical",
        "label_map_json",
        "beam_depth",
        "parent_scheme_id",
        "score_macro_f1",
        "score_source",
        "best_repeat_index",
        "best_repeat_mean_macro_f1",
        "best_repeat_std_macro_f1",
        "repeat_0_mean_macro_f1",
        "repeat_1_mean_macro_f1",
        "repeat_2_mean_macro_f1",
        "repeat_0_std_macro_f1",
        "repeat_1_std_macro_f1",
        "repeat_2_std_macro_f1",
        "mean_acc_of_best_repeat",
        "mean_precision_of_best_repeat",
        "mean_recall_of_best_repeat",
        "mean_auc_ovr_of_best_repeat",
        "used_confusion_prior",
        "used_medical_preference",
        "scheme_dir",
    ]
    all_candidates_csv = write_csv(all_candidates, all_candidate_fields, output_dir / "all_candidates.csv")

    best_by_target_rows: list[dict[str, Any]] = []
    for target in target_num_classes:
        candidates = target_top_candidates.get(target, [])
        if not candidates:
            continue
        best_by_target_rows.append(candidates[0])

    best_by_target_csv = write_csv(best_by_target_rows, all_candidate_fields, output_dir / "best_by_target.csv")

    recommended_scheme = None
    if best_by_target_rows:
        recommended_scheme = sorted(best_by_target_rows, key=_candidate_sort_key, reverse=True)[0]

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "search_config": search_merge_cfg,
        "base_label_mapping_path": str(label_mapping_path),
        "base_labels": base_labels,
        "medical_constraints": {
            "must_stay_single": sorted(must_stay_single, key=lambda label: (label_order[label], label)),
            "forbidden_pairs": sorted([list(pair) for pair in forbidden_pairs]),
            "preferred_pairs": sorted([list(pair) for pair in preferred_pairs]),
        },
        "historical_confusion": confusion_info,
        "beam_history": beam_history,
        "top_candidates_by_target": {
            str(target): target_top_candidates.get(target, [])
            for target in target_num_classes
        },
        "best_by_target": {
            str(row["target_num_classes"]): row
            for row in best_by_target_rows
        },
        "recommended_scheme": recommended_scheme,
        "all_candidates_csv": str(all_candidates_csv),
        "best_by_target_csv": str(best_by_target_csv),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "repeat_seeds": repeat_seeds,
        "n_splits": int(n_splits),
        "cv_epochs": int(cv_epochs),
        "search_score_source": "best_of_repeats_validation_only",
    }
    summary_path = output_dir / "search_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    resolved_config_path = output_dir / "search_config_resolved.yaml"
    resolved_config_path.write_text(yaml.safe_dump(search_merge_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    logger.info("Saved all candidates to %s", all_candidates_csv)
    logger.info("Saved best-by-target to %s", best_by_target_csv)
    logger.info("Saved search summary to %s", summary_path)


if __name__ == "__main__":
    main()
