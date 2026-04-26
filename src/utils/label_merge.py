from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


def _get_label_merge_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("label_merge", {}) or {})


def is_label_merge_enabled(config: dict[str, Any]) -> bool:
    return bool(_get_label_merge_config(config).get("enabled", False))


def _ensure_contiguous_indices(mapping: dict[str, Any]) -> None:
    indices = sorted(int(index) for index in mapping["merged_name_to_index"].values())
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            "Merged label indices must be contiguous integers from 0 to num_classes-1. "
            f"Got indices={indices}, expected={expected}."
        )


def build_label_merge_mapping(config: dict[str, Any]) -> dict[str, Any]:
    """Build original-label to merged-label mappings from config['label_merge']['groups']."""

    label_merge_config = _get_label_merge_config(config)
    groups = label_merge_config.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise ValueError("label_merge.enabled=true requires a non-empty label_merge.groups mapping.")

    original_to_merged_name: dict[str, str] = {}
    merged_name_to_index: dict[str, int] = {}
    index_to_merged_name: dict[int, str] = {}

    for merged_index, (merged_name, original_labels) in enumerate(groups.items()):
        merged_name = str(merged_name)
        if merged_name in merged_name_to_index:
            raise ValueError(f"Duplicate merged label group name in label_merge.groups: {merged_name}")
        if not isinstance(original_labels, list) or not original_labels:
            raise ValueError(f"label_merge.groups.{merged_name} must be a non-empty list of original labels.")

        merged_name_to_index[merged_name] = int(merged_index)
        index_to_merged_name[int(merged_index)] = merged_name

        for original_label in original_labels:
            original_label = str(original_label)
            previous = original_to_merged_name.get(original_label)
            if previous is not None and previous != merged_name:
                raise ValueError(
                    f"Original label '{original_label}' is assigned to multiple merged groups: "
                    f"'{previous}' and '{merged_name}'."
                )
            original_to_merged_name[original_label] = merged_name

    original_to_merged_index = {
        original_label: int(merged_name_to_index[merged_name])
        for original_label, merged_name in original_to_merged_name.items()
    }

    mapping = {
        "original_to_merged_name": original_to_merged_name,
        "merged_name_to_index": merged_name_to_index,
        "index_to_merged_name": index_to_merged_name,
        "original_to_merged_index": original_to_merged_index,
        "num_classes": len(merged_name_to_index),
    }
    _ensure_contiguous_indices(mapping)
    return mapping


def _extend_mapping_for_unmapped_labels(mapping: dict[str, Any], labels: list[str]) -> None:
    for original_label in sorted(labels):
        if original_label in mapping["original_to_merged_name"]:
            continue

        merged_name = original_label
        if merged_name not in mapping["merged_name_to_index"]:
            merged_index = len(mapping["merged_name_to_index"])
            mapping["merged_name_to_index"][merged_name] = int(merged_index)
            mapping["index_to_merged_name"][int(merged_index)] = merged_name
        merged_index = int(mapping["merged_name_to_index"][merged_name])
        mapping["original_to_merged_name"][original_label] = merged_name
        mapping["original_to_merged_index"][original_label] = merged_index

    mapping["num_classes"] = len(mapping["merged_name_to_index"])
    _ensure_contiguous_indices(mapping)


def validate_label_merge_coverage(
    df_list: list[pd.DataFrame],
    mapping: dict[str, Any],
    strict: bool = True,
) -> None:
    """Validate that every original label in split dataframes is covered by the merge mapping."""

    observed_labels: set[str] = set()
    for dataframe in df_list:
        if "label" not in dataframe.columns:
            raise ValueError("Every split dataframe must contain a 'label' column for label_merge.")
        observed_labels.update(dataframe["label"].dropna().astype(str).tolist())

    missing_labels = sorted(observed_labels.difference(mapping["original_to_merged_name"]))
    if not missing_labels:
        return

    if strict:
        raise ValueError(
            "label_merge.strict=true but these labels are present in split CSVs and missing from "
            f"label_merge.groups: {missing_labels}"
        )

    logging.warning(
        "label_merge.strict=false: keeping unmapped labels as their own classes: %s",
        missing_labels,
    )
    _extend_mapping_for_unmapped_labels(mapping, missing_labels)


def apply_label_merge_to_dataframe(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    label_col: str = "label",
    strict: bool = True,
) -> pd.DataFrame:
    """Return a copy of df with merged_label and merged_label_idx columns.

    The old label_idx column, if present, is intentionally ignored because it belongs
    to the original label space.
    """

    if label_col not in df.columns:
        raise ValueError(f"Cannot apply label_merge: dataframe is missing label column '{label_col}'.")

    result = df.copy()
    labels = result[label_col].astype(str)
    missing_labels = sorted(set(labels.tolist()).difference(mapping["original_to_merged_name"]))
    if missing_labels:
        if strict:
            raise ValueError(
                f"Cannot apply label_merge: labels missing from mapping: {missing_labels}. "
                "Add them to label_merge.groups or set label_merge.strict=false."
            )
        logging.warning(
            "label_merge.strict=false: keeping unmapped labels as their own classes: %s",
            missing_labels,
        )
        _extend_mapping_for_unmapped_labels(mapping, missing_labels)

    result["merged_label"] = labels.map(mapping["original_to_merged_name"])
    result["merged_label_idx"] = labels.map(mapping["original_to_merged_index"]).astype(int)
    return result


def get_label_names_from_mapping(mapping: dict[str, Any]) -> list[str]:
    index_to_name = {int(index): str(name) for index, name in mapping["index_to_merged_name"].items()}
    return [index_to_name[index] for index in range(int(mapping["num_classes"]))]


def update_config_num_classes_from_mapping(config: dict[str, Any], mapping: dict[str, Any]) -> None:
    label_merge_config = _get_label_merge_config(config)
    auto_set = bool(label_merge_config.get("auto_set_num_classes", True))
    merged_num_classes = int(mapping["num_classes"])

    data_config = config.setdefault("data", {})
    configured_num_classes = data_config.get("num_classes")
    if configured_num_classes is None or auto_set:
        data_config["num_classes"] = merged_num_classes
        return

    if int(configured_num_classes) != merged_num_classes:
        raise ValueError(
            "data.num_classes does not match label_merge num_classes: "
            f"{configured_num_classes} vs {merged_num_classes}. "
            "Set label_merge.auto_set_num_classes=true to override automatically."
        )


def serialize_label_merge_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_to_merged_name": dict(sorted(mapping["original_to_merged_name"].items())),
        "original_to_merged_index": {
            original: int(index)
            for original, index in sorted(mapping["original_to_merged_index"].items())
        },
        "merged_name_to_index": {
            name: int(index)
            for name, index in sorted(mapping["merged_name_to_index"].items(), key=lambda item: item[1])
        },
        "index_to_merged_name": {
            str(index): str(name)
            for index, name in sorted(
                ((int(index), name) for index, name in mapping["index_to_merged_name"].items()),
                key=lambda item: item[0],
            )
        },
        "num_classes": int(mapping["num_classes"]),
    }


def save_label_merge_mapping(mapping: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialize_label_merge_mapping(mapping), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
