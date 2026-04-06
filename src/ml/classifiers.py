from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ml.evaluate import compute_metrics_for_split, get_prediction_scores

SUPPORTED_CLASSIFIERS = {
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "knn",
}
SUPPORTED_PRIMARY_METRICS = {"accuracy", "precision", "recall", "macro_f1", "auc_ovr"}


@dataclass
class PreprocessorBundle:
    scaler: StandardScaler | None
    pca: PCA | None


@dataclass
class ModelSelectionResult:
    best_classifier: str
    best_hyperparams: dict[str, Any]
    best_model: Any
    preprocessor: PreprocessorBundle
    val_records: list[dict[str, Any]]
    best_val_metrics: dict[str, Any]


def _as_list(value: Any, name: str) -> list[Any]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Parameter list for {name} cannot be empty.")
        return value
    return [value]


def _build_param_grid(parameter_space: dict[str, Any]) -> list[dict[str, Any]]:
    if not parameter_space:
        return [{}]

    keys = list(parameter_space.keys())
    values = [_as_list(parameter_space[key], key) for key in keys]
    combinations = []
    for items in product(*values):
        combinations.append(dict(zip(keys, items)))
    return combinations


def _resolve_selected_classifiers(config: dict[str, Any]) -> list[str]:
    classifiers_config = config.get("ml_experiment", {}).get("classifiers", {})
    selected = classifiers_config.get("selected") or ["logistic_regression"]
    selected = [str(name).lower() for name in selected]

    unsupported = sorted(set(selected).difference(SUPPORTED_CLASSIFIERS))
    if unsupported:
        raise ValueError(
            f"Unsupported classifiers in ml_experiment.classifiers.selected: {unsupported}. "
            f"Supported: {sorted(SUPPORTED_CLASSIFIERS)}"
        )
    return selected


def _build_classifier(
    classifier_name: str,
    hyperparams: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> Any:
    classifiers_config = config.get("ml_experiment", {}).get("classifiers", {})
    classifier_config = classifiers_config.get(classifier_name, {})

    if classifier_name == "logistic_regression":
        max_iter = int(classifier_config.get("max_iter", 3000))
        class_weight = classifier_config.get("class_weight", "balanced")
        return LogisticRegression(
            C=float(hyperparams.get("C", 1.0)),
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=seed,
            multi_class="auto",
        )

    if classifier_name == "linear_svm":
        class_weight = classifier_config.get("class_weight", "balanced")
        probability = bool(classifier_config.get("probability", False))
        return SVC(
            kernel="linear",
            C=float(hyperparams.get("C", 1.0)),
            class_weight=class_weight,
            probability=probability,
            random_state=seed,
        )

    if classifier_name == "rbf_svm":
        class_weight = classifier_config.get("class_weight", "balanced")
        probability = bool(classifier_config.get("probability", True))
        return SVC(
            kernel="rbf",
            C=float(hyperparams.get("C", 1.0)),
            gamma=hyperparams.get("gamma", "scale"),
            class_weight=class_weight,
            probability=probability,
            random_state=seed,
        )

    if classifier_name == "random_forest":
        class_weight = classifier_config.get("class_weight", "balanced")
        return RandomForestClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 200)),
            max_depth=hyperparams.get("max_depth", None),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )

    if classifier_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(hyperparams.get("n_neighbors", 5)),
            weights=str(hyperparams.get("weights", "uniform")),
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported classifier: {classifier_name}")


def fit_preprocessor(train_features: np.ndarray, config: dict[str, Any], seed: int) -> tuple[np.ndarray, PreprocessorBundle]:
    preprocessing_config = config.get("ml_experiment", {}).get("preprocessing", {})
    standardize = bool(preprocessing_config.get("standardize", True))
    pca_config = preprocessing_config.get("pca", {}) or {}
    pca_enabled = bool(pca_config.get("enabled", False))

    transformed = np.asarray(train_features, dtype=np.float32)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(transformed)

    pca_model = None
    if pca_enabled:
        n_components = pca_config.get("n_components", 0.95)
        pca_model = PCA(n_components=n_components, random_state=seed)
        transformed = pca_model.fit_transform(transformed)

    return transformed.astype(np.float32), PreprocessorBundle(scaler=scaler, pca=pca_model)


def transform_features(features: np.ndarray, preprocessor: PreprocessorBundle) -> np.ndarray:
    transformed = np.asarray(features, dtype=np.float32)
    if preprocessor.scaler is not None:
        transformed = preprocessor.scaler.transform(transformed)
    if preprocessor.pca is not None:
        transformed = preprocessor.pca.transform(transformed)
    return transformed.astype(np.float32)


def describe_preprocessor(preprocessor: PreprocessorBundle) -> dict[str, Any]:
    pca_components = None
    explained_variance = None
    if preprocessor.pca is not None:
        pca_components = int(preprocessor.pca.n_components_) if hasattr(preprocessor.pca, "n_components_") else None
        explained_variance = float(np.sum(preprocessor.pca.explained_variance_ratio_))

    return {
        "standardize": preprocessor.scaler is not None,
        "pca_enabled": preprocessor.pca is not None,
        "pca_components": pca_components,
        "pca_explained_variance": explained_variance,
    }


def run_model_selection(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: dict[str, Any],
    label_names: list[str] | None,
) -> ModelSelectionResult:
    seed = int(config.get("train", {}).get("seed", 42))
    primary_metric = str(config.get("ml_experiment", {}).get("model_selection", {}).get("primary_metric", "macro_f1")).lower()
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported ml_experiment.model_selection.primary_metric={primary_metric}. "
            f"Supported: {sorted(SUPPORTED_PRIMARY_METRICS)}"
        )

    selected = _resolve_selected_classifiers(config)
    transformed_train, preprocessor = fit_preprocessor(train_features, config, seed)
    transformed_val = transform_features(val_features, preprocessor)

    classifiers_config = config.get("ml_experiment", {}).get("classifiers", {})
    records: list[dict[str, Any]] = []

    best_score = float("-inf")
    best_classifier = ""
    best_hyperparams: dict[str, Any] = {}
    best_model = None
    best_metrics: dict[str, Any] = {}

    for classifier_name in selected:
        parameter_space = {
            key: value
            for key, value in (classifiers_config.get(classifier_name, {}) or {}).items()
            if key not in {"max_iter", "class_weight", "probability"}
        }
        parameter_grid = _build_param_grid(parameter_space)

        for hyperparams in parameter_grid:
            model = _build_classifier(
                classifier_name=classifier_name,
                hyperparams=hyperparams,
                config=config,
                seed=seed,
            )
            model.fit(transformed_train, train_labels)

            val_predictions = model.predict(transformed_val)
            val_scores = get_prediction_scores(model, transformed_val)
            val_metrics = compute_metrics_for_split(
                y_true=val_labels,
                y_pred=val_predictions,
                label_names=label_names,
                score_matrix=val_scores,
            )

            record = {
                "classifier": classifier_name,
                "hyperparams": json.dumps(hyperparams, ensure_ascii=False, sort_keys=True),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_auc_ovr": val_metrics.get("auc_ovr"),
            }
            for param_name, param_value in hyperparams.items():
                record[f"param_{param_name}"] = param_value
            records.append(record)

            score_value = val_metrics.get(primary_metric)
            if score_value is None:
                score_value = float("-inf")
            score_value = float(score_value)

            should_update = score_value > best_score
            if not should_update and np.isclose(score_value, best_score):
                current_best_macro = float(best_metrics.get("macro_f1", float("-inf"))) if best_metrics else float("-inf")
                should_update = float(val_metrics["macro_f1"]) > current_best_macro

            if should_update:
                best_score = score_value
                best_classifier = classifier_name
                best_hyperparams = dict(hyperparams)
                best_model = model
                best_metrics = val_metrics

    if best_model is None:
        raise RuntimeError("Model selection failed: no classifier candidates were evaluated.")

    return ModelSelectionResult(
        best_classifier=best_classifier,
        best_hyperparams=best_hyperparams,
        best_model=best_model,
        preprocessor=preprocessor,
        val_records=records,
        best_val_metrics=best_metrics,
    )
