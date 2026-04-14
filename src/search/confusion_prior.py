from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _normalize_label_value(value: Any, label_to_index: dict[str, int]) -> int:
    if isinstance(value, (np.integer, int)):
        numeric = int(value)
        if 0 <= numeric < len(label_to_index):
            return numeric
        raise ValueError(f"Class index out of range: {numeric}")

    text = str(value)
    if text in label_to_index:
        return int(label_to_index[text])

    try:
        numeric = int(text)
    except ValueError as exc:
        raise ValueError(f"Unknown label value: {value}") from exc

    if 0 <= numeric < len(label_to_index):
        return numeric
    raise ValueError(f"Class index out of range: {numeric}")


def _load_vector_values(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=True)
        return np.asarray(values).tolist()

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("values", "y_true", "y_pred", "labels"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise ValueError(f"Cannot parse vector values from JSON: {path}")

    if suffix == ".csv":
        frame = pd.read_csv(path)
        if frame.shape[1] == 1:
            return frame.iloc[:, 0].tolist()
        raise ValueError(f"CSV vector file must have exactly one column: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _build_confusion_from_ytrue_ypred(
    y_true_values: list[Any],
    y_pred_values: list[Any],
    label_to_index: dict[str, int],
) -> np.ndarray:
    if len(y_true_values) != len(y_pred_values):
        raise ValueError(
            "y_true and y_pred lengths do not match: "
            f"{len(y_true_values)} vs {len(y_pred_values)}"
        )

    y_true_indices = [_normalize_label_value(value, label_to_index) for value in y_true_values]
    y_pred_indices = [_normalize_label_value(value, label_to_index) for value in y_pred_values]
    labels = list(range(len(label_to_index)))
    matrix = confusion_matrix(y_true_indices, y_pred_indices, labels=labels)
    return matrix.astype(np.float64)


def _align_matrix_to_base_labels(
    matrix: np.ndarray,
    source_labels: list[Any] | None,
    base_labels: list[str],
    label_to_index: dict[str, int],
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix must be square. Got shape={matrix.shape}")

    n_classes = len(base_labels)
    if source_labels is None:
        if matrix.shape[0] != n_classes:
            raise ValueError(
                "Confusion matrix size does not match class count without explicit labels. "
                f"matrix={matrix.shape[0]}, expected={n_classes}"
            )
        return matrix

    if len(source_labels) != matrix.shape[0]:
        raise ValueError(
            "source labels length must match confusion matrix dimension. "
            f"labels={len(source_labels)} matrix={matrix.shape[0]}"
        )

    aligned = np.zeros((n_classes, n_classes), dtype=np.float64)
    mapped_indices = [_normalize_label_value(value, label_to_index) for value in source_labels]
    for src_i, base_i in enumerate(mapped_indices):
        for src_j, base_j in enumerate(mapped_indices):
            aligned[base_i, base_j] = float(matrix[src_i, src_j])
    return aligned


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums != 0)


def _load_matrix_from_file(
    path: Path,
    input_type: str,
    base_labels: list[str],
    label_to_index: dict[str, int],
) -> tuple[np.ndarray, str]:
    resolved_type = input_type.lower()
    suffix = path.suffix.lower()

    if resolved_type in {"auto", "matrix"} and suffix == ".npy":
        matrix = np.load(path, allow_pickle=True)
        aligned = _align_matrix_to_base_labels(matrix, source_labels=None, base_labels=base_labels, label_to_index=label_to_index)
        return aligned, "matrix"

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if resolved_type in {"auto", "y_true_pred", "ytrue_ypred"} and isinstance(payload, dict):
            if "y_true" in payload and "y_pred" in payload:
                matrix = _build_confusion_from_ytrue_ypred(payload["y_true"], payload["y_pred"], label_to_index)
                return matrix, "y_true_pred"

        if isinstance(payload, dict):
            matrix_payload = payload.get("confusion_matrix", payload.get("matrix"))
            if matrix_payload is None:
                raise ValueError(f"JSON confusion payload must provide confusion_matrix/matrix or y_true/y_pred: {path}")
            source_labels = payload.get("labels")
            matrix = np.asarray(matrix_payload, dtype=np.float64)
            aligned = _align_matrix_to_base_labels(matrix, source_labels, base_labels, label_to_index)
            return aligned, "matrix"

        if isinstance(payload, list):
            matrix = np.asarray(payload, dtype=np.float64)
            aligned = _align_matrix_to_base_labels(matrix, None, base_labels, label_to_index)
            return aligned, "matrix"

        raise ValueError(f"Unsupported JSON confusion payload: {path}")

    if suffix == ".csv":
        frame = pd.read_csv(path)
        lower_columns = {str(column).lower(): str(column) for column in frame.columns}
        if resolved_type in {"auto", "y_true_pred", "ytrue_ypred"} and "y_true" in lower_columns and "y_pred" in lower_columns:
            y_true = frame[lower_columns["y_true"]].tolist()
            y_pred = frame[lower_columns["y_pred"]].tolist()
            matrix = _build_confusion_from_ytrue_ypred(y_true, y_pred, label_to_index)
            return matrix, "y_true_pred"

        source_labels: list[Any] | None = None
        numeric_frame = frame.copy()
        if frame.shape[1] > 0:
            first_column_name = str(frame.columns[0])
            first_column_values = frame.iloc[:, 0]
            if first_column_name.lower() in {"label", "labels", "class", "class_name"}:
                source_labels = first_column_values.tolist()
                numeric_frame = frame.iloc[:, 1:]
            elif not np.issubdtype(first_column_values.dtype, np.number):
                source_labels = first_column_values.tolist()
                numeric_frame = frame.iloc[:, 1:]

        if source_labels is None and all(str(column) in label_to_index for column in frame.columns):
            source_labels = [str(column) for column in frame.columns]

        matrix = numeric_frame.to_numpy(dtype=np.float64)
        aligned = _align_matrix_to_base_labels(matrix, source_labels, base_labels, label_to_index)
        return aligned, "matrix"

    raise ValueError(f"Unsupported confusion source format: {path}")


def load_historical_confusion_prior(
    paths: list[Any],
    base_labels: list[str],
    input_type: str = "auto",
    normalize: bool = True,
    average_multiple: bool = True,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(base_labels)}

    matrices: list[np.ndarray] = []
    source_descriptions: list[dict[str, Any]] = []

    for entry in paths:
        if isinstance(entry, dict) and entry.get("y_true_path") and entry.get("y_pred_path"):
            y_true_path = Path(entry["y_true_path"])
            y_pred_path = Path(entry["y_pred_path"])
            y_true = _load_vector_values(y_true_path)
            y_pred = _load_vector_values(y_pred_path)
            matrix = _build_confusion_from_ytrue_ypred(y_true, y_pred, label_to_index)
            source_type = "y_true_pred"
            source_ref = {
                "y_true_path": str(y_true_path),
                "y_pred_path": str(y_pred_path),
            }
        else:
            if isinstance(entry, dict):
                path_value = entry.get("path")
                if path_value is None:
                    raise ValueError(f"Invalid historical_confusion entry: {entry}")
                source_input_type = str(entry.get("input_type", input_type))
            else:
                path_value = entry
                source_input_type = input_type

            path = Path(path_value)
            matrix, source_type = _load_matrix_from_file(
                path=path,
                input_type=source_input_type,
                base_labels=base_labels,
                label_to_index=label_to_index,
            )
            source_ref = {"path": str(path)}

        if normalize:
            matrix = _row_normalize(matrix)

        matrices.append(matrix)
        source_descriptions.append(
            {
                **source_ref,
                "type": source_type,
                "normalize": bool(normalize),
                "shape": list(matrix.shape),
            }
        )

    if not matrices:
        return {
            "enabled": False,
            "used": False,
            "source_count": 0,
            "sources": [],
            "matrix": None,
        }

    if len(matrices) == 1 or not average_multiple:
        merged = matrices[0]
    else:
        merged = np.mean(np.stack(matrices, axis=0), axis=0)

    conf_scores = merged + merged.T
    return {
        "enabled": True,
        "used": True,
        "source_count": len(matrices),
        "sources": source_descriptions,
        "matrix": merged,
        "conf_scores": conf_scores,
    }
