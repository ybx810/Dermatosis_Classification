from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_label_mapping_from_json(label_mapping_path: Path | None) -> dict[str, int] | None:
    if label_mapping_path is None or not label_mapping_path.exists():
        return None

    payload = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    mapping_payload = payload.get("label_to_index", payload)
    if not isinstance(mapping_payload, dict):
        raise ValueError(f"label mapping file must contain an object: {label_mapping_path}")

    return {str(label): int(index) for label, index in mapping_payload.items()}


def load_label_names_from_json(
    label_mapping_path: Path | None,
    num_classes: int | None = None,
) -> list[str] | None:
    mapping = load_label_mapping_from_json(label_mapping_path)
    if mapping is None:
        return None

    index_to_label = {int(index): str(label) for label, index in mapping.items()}
    inferred_num_classes = int(num_classes) if num_classes is not None else (max(index_to_label.keys()) + 1 if index_to_label else 0)
    return [index_to_label.get(index, str(index)) for index in range(inferred_num_classes)]


def build_old_to_new_mapping_from_partition(partition: list[list[str]] | tuple[tuple[str, ...], ...]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for merged_idx, group in enumerate(partition):
        for label in group:
            normalized = str(label)
            if normalized in mapping:
                raise ValueError(f"Label '{normalized}' appears multiple times in label_merge.partition.")
            mapping[normalized] = int(merged_idx)
    return mapping


def build_partition_from_old_to_new(
    old_to_new: dict[str, int],
    label_order: list[str] | None = None,
) -> list[list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for label, merged_idx in old_to_new.items():
        groups[int(merged_idx)].append(str(label))

    sorted_group_indices = sorted(groups.keys())
    if sorted_group_indices != list(range(len(sorted_group_indices))):
        raise ValueError(
            "label_merge.old_to_new values must be contiguous integers starting from 0. "
            f"Got indices: {sorted_group_indices}"
        )

    order_lookup = {str(label): index for index, label in enumerate(label_order or [])}

    def _sort_labels(labels: list[str]) -> list[str]:
        if not order_lookup:
            return sorted(labels)
        return sorted(labels, key=lambda item: (order_lookup.get(item, len(order_lookup)), item))

    return [_sort_labels(groups[group_idx]) for group_idx in sorted_group_indices]


def build_merged_label_names(
    partition: list[list[str]],
    explicit_names: list[str] | None = None,
) -> list[str]:
    if explicit_names is not None:
        normalized = [str(item) for item in explicit_names]
        if len(normalized) != len(partition):
            raise ValueError(
                "label_merge.merged_label_names length must match merged class count. "
                f"Got {len(normalized)} names for {len(partition)} classes."
            )
        return normalized

    names: list[str] = []
    for group in partition:
        if len(group) == 1:
            names.append(str(group[0]))
            continue
        names.append(" + ".join(str(label) for label in group))
    return names


def resolve_label_merge_runtime(
    config: dict[str, Any],
    label_mapping_path: Path | None,
    fallback_num_classes: int | None = None,
) -> dict[str, Any]:
    merge_cfg = config.get("label_merge", {})
    merge_enabled = bool(isinstance(merge_cfg, dict) and merge_cfg.get("enabled", False))
    base_mapping = load_label_mapping_from_json(label_mapping_path)
    base_label_order = None if base_mapping is None else [label for label, _ in sorted(base_mapping.items(), key=lambda item: item[1])]

    if not merge_enabled:
        label_names = load_label_names_from_json(label_mapping_path, num_classes=fallback_num_classes)
        resolved_num_classes = (
            len(label_names)
            if label_names is not None
            else (int(fallback_num_classes) if fallback_num_classes is not None else None)
        )
        return {
            "active": False,
            "dataset_label_mapping": None,
            "label_names": label_names,
            "num_classes": resolved_num_classes,
            "old_to_new": None,
            "partition": None,
        }

    raw_old_to_new = merge_cfg.get("old_to_new")
    raw_partition = merge_cfg.get("partition")
    if raw_old_to_new is None:
        if raw_partition is None:
            raise ValueError("label_merge.enabled=true requires label_merge.old_to_new or label_merge.partition.")
        old_to_new = build_old_to_new_mapping_from_partition(raw_partition)
    else:
        if not isinstance(raw_old_to_new, dict):
            raise ValueError("label_merge.old_to_new must be a mapping from original label name to merged class index.")
        old_to_new = {str(label): int(merged_idx) for label, merged_idx in raw_old_to_new.items()}

    if base_mapping is not None:
        base_labels = set(base_mapping.keys())
        merge_labels = set(old_to_new.keys())
        missing_labels = sorted(base_labels.difference(merge_labels))
        unknown_labels = sorted(merge_labels.difference(base_labels))
        if missing_labels:
            raise ValueError(
                "label_merge.old_to_new is missing labels from label mapping: "
                f"{missing_labels}"
            )
        if unknown_labels:
            raise ValueError(
                "label_merge.old_to_new includes labels not present in label mapping: "
                f"{unknown_labels}"
            )

    partition = build_partition_from_old_to_new(old_to_new, label_order=base_label_order)
    merged_label_names = build_merged_label_names(
        partition=partition,
        explicit_names=merge_cfg.get("merged_label_names"),
    )

    resolved_num_classes = len(partition)
    if fallback_num_classes is not None and int(fallback_num_classes) != resolved_num_classes:
        raise ValueError(
            "data.num_classes does not match merged class count. "
            f"data.num_classes={fallback_num_classes}, merged={resolved_num_classes}."
        )

    return {
        "active": True,
        "dataset_label_mapping": old_to_new,
        "label_names": merged_label_names,
        "num_classes": resolved_num_classes,
        "old_to_new": old_to_new,
        "partition": partition,
    }
