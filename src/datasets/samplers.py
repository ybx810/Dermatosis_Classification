from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

VALID_SAMPLING_STRATEGIES = {"none", "weighted", "oversample"}


class RandomOversampleSampler(Sampler[int]):
    """Randomly oversample minority classes to a balanced class count per epoch."""

    def __init__(
        self,
        label_to_indices: Mapping[int, Sequence[int]],
        target_count_per_class: int,
        epoch_size: int,
        replacement: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.label_to_indices = {
            int(label_idx): [int(dataset_idx) for dataset_idx in dataset_indices]
            for label_idx, dataset_indices in label_to_indices.items()
        }
        self.target_count_per_class = int(target_count_per_class)
        self.epoch_size = int(epoch_size)
        self.replacement = bool(replacement)
        self._labels = sorted(self.label_to_indices.keys())
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed))

        if not self._labels:
            raise ValueError("Cannot build oversample sampler because no labels were found.")
        if self.target_count_per_class <= 0:
            raise ValueError(f"target_count_per_class must be positive, got: {target_count_per_class}")
        if self.epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive, got: {epoch_size}")

        for label_idx, dataset_indices in self.label_to_indices.items():
            if not dataset_indices:
                raise ValueError(
                    f"Cannot build oversample sampler because class index {label_idx} has no samples."
                )

        if not self.replacement:
            for label_idx, dataset_indices in self.label_to_indices.items():
                if len(dataset_indices) < self.target_count_per_class:
                    raise ValueError(
                        "oversample strategy requires replacement=true when a class is smaller than "
                        f"target_count_per_class. class_idx={label_idx}, class_count={len(dataset_indices)}, "
                        f"target_count_per_class={self.target_count_per_class}"
                    )

        self._balanced_epoch_size = self.target_count_per_class * len(self._labels)
        if not self.replacement and self.epoch_size > self._balanced_epoch_size:
            raise ValueError(
                "oversample strategy with replacement=false cannot produce epoch_size larger than "
                f"balanced epoch size ({self._balanced_epoch_size}). Got epoch_size={self.epoch_size}."
            )

    def __iter__(self):
        balanced_indices: list[int] = []
        for label_idx in self._labels:
            class_indices = self.label_to_indices[label_idx]
            balanced_indices.extend(class_indices)
            deficit = self.target_count_per_class - len(class_indices)
            if deficit > 0:
                draw_positions = torch.randint(
                    low=0,
                    high=len(class_indices),
                    size=(deficit,),
                    generator=self._generator,
                ).tolist()
                balanced_indices.extend(class_indices[int(position)] for position in draw_positions)

        if not balanced_indices:
            raise RuntimeError("oversample sampler generated an empty index list.")

        balanced_indices = _shuffle_indices(balanced_indices, generator=self._generator)

        if self.epoch_size < len(balanced_indices):
            selected_positions = torch.randperm(len(balanced_indices), generator=self._generator).tolist()
            output_indices = [balanced_indices[int(position)] for position in selected_positions[: self.epoch_size]]
        elif self.epoch_size > len(balanced_indices):
            if not self.replacement:
                raise RuntimeError(
                    "oversample sampler cannot expand epoch size without replacement. "
                    f"epoch_size={self.epoch_size}, balanced_size={len(balanced_indices)}"
                )
            extra_count = self.epoch_size - len(balanced_indices)
            extra_positions = torch.randint(
                low=0,
                high=len(balanced_indices),
                size=(extra_count,),
                generator=self._generator,
            ).tolist()
            output_indices = balanced_indices + [balanced_indices[int(position)] for position in extra_positions]
            output_indices = _shuffle_indices(output_indices, generator=self._generator)
        else:
            output_indices = balanced_indices

        return iter(output_indices)

    def __len__(self) -> int:
        return self.epoch_size


def _shuffle_indices(indices: Sequence[int], generator: torch.Generator) -> list[int]:
    if not indices:
        return []
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    return [int(indices[int(position)]) for position in permutation]


def _normalize_strategy(sampling_config: Mapping[str, Any]) -> str:
    strategy = str(sampling_config.get("strategy", "none")).lower().strip()
    if strategy not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(
            f"train.sampling.strategy must be one of {sorted(VALID_SAMPLING_STRATEGIES)}, got: {strategy}"
        )
    return strategy


def _resolve_epoch_size(epoch_size_config: Any, default_epoch_size: int) -> int:
    if epoch_size_config is None:
        return int(default_epoch_size)

    if isinstance(epoch_size_config, str):
        lowered = epoch_size_config.strip().lower()
        if lowered == "auto":
            return int(default_epoch_size)
        if lowered.isdigit():
            value = int(lowered)
            if value <= 0:
                raise ValueError(f"train.sampling.epoch_size must be a positive integer, got: {epoch_size_config}")
            return value
        raise ValueError(
            "train.sampling.epoch_size must be 'auto' or a positive integer, "
            f"got string value: {epoch_size_config}"
        )

    if isinstance(epoch_size_config, bool):
        raise ValueError(
            "train.sampling.epoch_size must be 'auto' or a positive integer, got boolean value."
        )

    try:
        value = int(epoch_size_config)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "train.sampling.epoch_size must be 'auto' or a positive integer, "
            f"got: {epoch_size_config}"
        ) from exc

    if value <= 0:
        raise ValueError(f"train.sampling.epoch_size must be a positive integer, got: {epoch_size_config}")
    return value


def extract_label_indices_from_dataset(dataset: Any) -> list[int]:
    if not hasattr(dataset, "samples"):
        raise ValueError("Dataset must provide a 'samples' attribute to build a train sampler.")
    if not hasattr(dataset, "label_mapping"):
        raise ValueError("Dataset must provide a 'label_mapping' attribute to build a train sampler.")

    samples = dataset.samples
    if "label" not in samples.columns:
        raise ValueError("Dataset samples must include a 'label' column to build a train sampler.")

    label_mapping = {str(label_name): int(label_idx) for label_name, label_idx in dict(dataset.label_mapping).items()}
    label_indices: list[int] = []
    for label_name in samples["label"].astype(str).tolist():
        if label_name not in label_mapping:
            raise KeyError(f"Label '{label_name}' is missing from dataset.label_mapping: {label_mapping}")
        label_indices.append(int(label_mapping[label_name]))

    if not label_indices:
        raise ValueError("Training dataset is empty; cannot build a train sampler.")
    return label_indices


def compute_label_counts(label_indices: Sequence[int]) -> dict[int, int]:
    counts = Counter(int(label_idx) for label_idx in label_indices)
    return {int(label_idx): int(counts[label_idx]) for label_idx in sorted(counts.keys())}


def _validate_non_zero_classes(
    class_counts: Mapping[int, int],
    label_mapping: Mapping[str, int],
) -> None:
    expected_class_indices = sorted({int(label_idx) for label_idx in label_mapping.values()})
    missing_class_indices = [label_idx for label_idx in expected_class_indices if int(class_counts.get(label_idx, 0)) <= 0]
    if not missing_class_indices:
        return

    index_to_label = {int(label_idx): str(label_name) for label_name, label_idx in label_mapping.items()}
    missing_descriptions = [
        f"{label_idx}({index_to_label.get(label_idx, 'unknown')})" for label_idx in missing_class_indices
    ]
    raise ValueError(
        "Found classes with zero samples in the current training CSV. "
        "Sampling rebalance requires every class in dataset.label_mapping to appear at least once. "
        f"Missing classes: {missing_descriptions}"
    )


def build_weighted_sampler(
    label_indices: Sequence[int],
    epoch_size: int,
    replacement: bool,
    seed: int,
) -> WeightedRandomSampler:
    class_counts = compute_label_counts(label_indices)
    sample_weights = [1.0 / float(class_counts[int(label_idx)]) for label_idx in label_indices]

    if not replacement and int(epoch_size) > len(label_indices):
        raise ValueError(
            "train.sampling.epoch_size cannot exceed dataset size when weighted sampling uses replacement=false. "
            f"Got epoch_size={epoch_size}, dataset_size={len(label_indices)}."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=int(epoch_size),
        replacement=bool(replacement),
        generator=generator,
    )


def build_random_oversample_sampler(
    label_indices: Sequence[int],
    epoch_size: int,
    replacement: bool,
    seed: int,
) -> RandomOversampleSampler:
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for dataset_idx, label_idx in enumerate(label_indices):
        label_to_indices[int(label_idx)].append(int(dataset_idx))

    class_counts = compute_label_counts(label_indices)
    target_count_per_class = max(class_counts.values()) if class_counts else 0
    return RandomOversampleSampler(
        label_to_indices=label_to_indices,
        target_count_per_class=target_count_per_class,
        epoch_size=int(epoch_size),
        replacement=bool(replacement),
        seed=int(seed),
    )


def summarize_sampling_plan(
    strategy: str,
    class_counts: Mapping[int, int],
    target_epoch_size: int,
    replacement: bool,
    source_num_samples: int,
    default_epoch_size: int,
    log_distribution: bool,
) -> dict[str, Any]:
    normalized_strategy = str(strategy).lower().strip()
    ordered_counts = {int(label_idx): int(class_counts[label_idx]) for label_idx in sorted(class_counts.keys())}

    summary: dict[str, Any] = {
        "enabled": normalized_strategy != "none",
        "strategy": normalized_strategy,
        "class_counts": ordered_counts,
        "target_epoch_size": int(target_epoch_size),
        "default_epoch_size": int(default_epoch_size),
        "source_num_samples": int(source_num_samples),
        "replacement": bool(replacement),
        "log_distribution": bool(log_distribution),
        "warnings": [],
    }

    log_lines = [
        f"Train sampling | strategy={normalized_strategy}",
        f"Train sampling | target epoch size={int(target_epoch_size)}",
        f"Train sampling | replacement={bool(replacement)}",
    ]
    if bool(log_distribution):
        log_lines.append(f"Train sampling | original class counts={ordered_counts}")
    summary["log_lines"] = log_lines
    return summary


def build_train_sampler(
    dataset: Any,
    sampling_config: Mapping[str, Any] | None,
    seed: int,
) -> tuple[Sampler[int] | WeightedRandomSampler | None, dict[str, Any]]:
    config = dict(sampling_config or {})
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return None, {
            "enabled": False,
            "strategy": "none",
            "class_counts": {},
            "target_epoch_size": None,
            "default_epoch_size": None,
            "source_num_samples": None,
            "replacement": bool(config.get("replacement", True)),
            "log_distribution": bool(config.get("log_distribution", True)),
            "warnings": [],
            "log_lines": ["Train sampling | strategy=none"],
        }

    strategy = _normalize_strategy(config)
    if strategy == "none":
        return None, {
            "enabled": False,
            "strategy": "none",
            "class_counts": {},
            "target_epoch_size": None,
            "default_epoch_size": None,
            "source_num_samples": None,
            "replacement": bool(config.get("replacement", True)),
            "log_distribution": bool(config.get("log_distribution", True)),
            "warnings": [],
            "log_lines": ["Train sampling | strategy=none"],
        }

    replacement = bool(config.get("replacement", True))
    log_distribution = bool(config.get("log_distribution", True))
    label_indices = extract_label_indices_from_dataset(dataset)
    class_counts = compute_label_counts(label_indices)
    _validate_non_zero_classes(class_counts, dataset.label_mapping)

    if strategy == "weighted":
        default_epoch_size = len(label_indices)
        epoch_size = _resolve_epoch_size(config.get("epoch_size", "auto"), default_epoch_size)
        sampler = build_weighted_sampler(
            label_indices=label_indices,
            epoch_size=epoch_size,
            replacement=replacement,
            seed=seed,
        )
    elif strategy == "oversample":
        max_class_count = max(class_counts.values())
        num_present_classes = len(class_counts)
        default_epoch_size = max_class_count * num_present_classes
        epoch_size = _resolve_epoch_size(config.get("epoch_size", "auto"), default_epoch_size)
        sampler = build_random_oversample_sampler(
            label_indices=label_indices,
            epoch_size=epoch_size,
            replacement=replacement,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unsupported train sampling strategy: {strategy}. "
            f"Expected one of {sorted(VALID_SAMPLING_STRATEGIES)}."
        )

    summary = summarize_sampling_plan(
        strategy=strategy,
        class_counts=class_counts,
        target_epoch_size=epoch_size,
        replacement=replacement,
        source_num_samples=len(label_indices),
        default_epoch_size=default_epoch_size,
        log_distribution=log_distribution,
    )

    if strategy == "oversample" and epoch_size < default_epoch_size:
        warning = (
            "train.sampling.epoch_size is smaller than auto oversample size. "
            f"epoch_size={epoch_size}, auto_oversample_size={default_epoch_size}. "
            "This intentionally reduces each epoch below full balanced oversampling."
        )
        summary["warnings"].append(warning)
        summary["log_lines"].append(f"Train sampling | warning={warning}")

    return sampler, summary
