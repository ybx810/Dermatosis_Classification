from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets import build_whole_image_dataloader
from src.ml.io import (
    PROJECT_ROOT,
    load_label_names,
    resolve_project_path,
    resolve_split_paths,
    save_feature_split,
    save_json,
    verify_feature_alignment,
)
from src.models import build_feature_extractor, extract_backbone_features


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_feature_settings(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml_experiment", {})
    source = str(ml_config.get("feature_source", "imagenet_pretrained")).lower()
    if source not in {"imagenet_pretrained", "checkpoint"}:
        raise ValueError(
            f"Unsupported ml_experiment.feature_source: {source}. "
            "Expected one of {'imagenet_pretrained', 'checkpoint'}."
        )

    model_config = config.get("model", {})
    backbone = str(ml_config.get("backbone") or model_config.get("name", "resnet18")).lower()

    checkpoint_path = None
    if source == "checkpoint":
        checkpoint_path = resolve_project_path(ml_config.get("checkpoint_path"), PROJECT_ROOT)
        if checkpoint_path is None:
            raise ValueError("ml_experiment.checkpoint_path is required when feature_source=checkpoint")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    extract_batch_size = ml_config.get("extract_batch_size")
    if extract_batch_size in (None, "", "null"):
        extract_batch_size = int(config.get("train", {}).get("batch_size", 16))
    else:
        extract_batch_size = int(extract_batch_size)

    use_mixed_precision = bool(ml_config.get("use_mixed_precision", True))
    save_features = bool(ml_config.get("save_features", True))
    reuse_saved_features = bool(ml_config.get("reuse_saved_features", False))

    return {
        "source": source,
        "backbone": backbone,
        "checkpoint_path": checkpoint_path,
        "extract_batch_size": extract_batch_size,
        "use_mixed_precision": use_mixed_precision,
        "save_features": save_features,
        "reuse_saved_features": reuse_saved_features,
    }


def _build_extraction_config(config: dict[str, Any], extract_batch_size: int) -> dict[str, Any]:
    extraction_config = deepcopy(config)
    whole_image_config = extraction_config.setdefault("whole_image", {})
    whole_image_config["batch_size"] = int(extract_batch_size)
    return extraction_config


def _feature_files_exist(feature_dir: Path) -> bool:
    return all((feature_dir / f"{split}_features.npz").exists() for split in ("train", "val", "test"))


@torch.no_grad()
def extract_features_for_split(
    model: torch.nn.Module,
    backbone_name: str,
    dataloader: torch.utils.data.DataLoader,
    split_name: str,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()

    features_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    source_images: list[str] = []
    label_names: list[str] = []
    image_paths: list[str] = []

    dataset = dataloader.dataset
    can_resolve_paths = hasattr(dataset, "_resolve_preferred_image_path")
    progress = tqdm(dataloader, desc=f"Extract[{split_name}]", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            embeddings = extract_backbone_features(model, backbone_name, images)

        embedding_array = embeddings.detach().float().cpu().numpy()
        label_array = labels.detach().cpu().numpy().astype(np.int64)

        batch_source_images = [str(item) for item in batch["source_image"]]
        batch_label_names = [str(item) for item in batch["label_name"]]

        features_chunks.append(embedding_array)
        labels_chunks.append(label_array)
        source_images.extend(batch_source_images)
        label_names.extend(batch_label_names)

        if can_resolve_paths:
            resolved_batch_paths: list[str] = []
            for source_image in batch_source_images:
                resolved_path, _ = dataset._resolve_preferred_image_path(source_image)
                resolved_batch_paths.append(str(resolved_path))
            image_paths.extend(resolved_batch_paths)

    if features_chunks:
        features = np.concatenate(features_chunks, axis=0).astype(np.float32)
        labels = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    else:
        features = np.empty((0, 0), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)

    split_data: dict[str, Any] = {
        "X": features,
        "y": labels,
        "source_image": source_images,
        "label_name": label_names,
    }
    if image_paths:
        split_data["image_path"] = image_paths

    verify_feature_alignment(split_data, split_name)
    return split_data


def extract_all_splits(
    config: dict[str, Any],
    output_dir: str | Path,
    run_tag: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = resolve_split_paths(config, project_root=PROJECT_ROOT)
    label_names = load_label_names(split_paths["label_mapping_path"], num_classes=config.get("data", {}).get("num_classes"))
    settings = resolve_feature_settings(config)

    feature_info_path = output_dir / "feature_info.json"
    if settings["reuse_saved_features"] and _feature_files_exist(output_dir) and feature_info_path.exists():
        feature_info = {
            "reused_existing_features": True,
            "feature_dir": str(output_dir),
            "feature_info_path": str(feature_info_path),
            "feature_source": settings["source"],
            "backbone": settings["backbone"],
            "checkpoint_path": str(settings["checkpoint_path"]) if settings["checkpoint_path"] else None,
            "run_tag": run_tag,
        }
        return feature_info

    device = select_device(config)
    use_amp = bool(settings["use_mixed_precision"] and device.type == "cuda")

    extraction_config = _build_extraction_config(config, settings["extract_batch_size"])
    model, resolved_backbone = build_feature_extractor(
        config=extraction_config,
        backbone_name=settings["backbone"],
        source=settings["source"],
        checkpoint_path=settings["checkpoint_path"],
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    split_outputs: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        csv_path = split_paths[f"{split_name}_csv"]
        dataloader = build_whole_image_dataloader(
            csv_file=csv_path,
            mode=split_name,
            config=extraction_config,
            label_mapping_path=split_paths["label_mapping_path"],
            project_root=PROJECT_ROOT,
            shuffle=False,
            drop_last=False,
        )

        split_data = extract_features_for_split(
            model=model,
            backbone_name=resolved_backbone,
            dataloader=dataloader,
            split_name=split_name,
            device=device,
            use_amp=use_amp,
        )

        expected_count = int(pd.read_csv(csv_path).shape[0])
        actual_count = int(split_data["X"].shape[0])
        if actual_count != expected_count:
            raise ValueError(
                f"{split_name} feature count mismatch: extracted {actual_count} rows but CSV has {expected_count} rows ({csv_path})"
            )

        feature_path = None
        metadata_path = None
        if settings["save_features"]:
            feature_path, metadata_path = save_feature_split(
                output_dir=output_dir,
                split_name=split_name,
                features=split_data["X"],
                labels=split_data["y"],
                source_images=split_data["source_image"],
                label_names=split_data["label_name"],
                image_paths=split_data.get("image_path"),
            )

        split_outputs[split_name] = {
            "num_samples": actual_count,
            "num_features": int(split_data["X"].shape[1]) if split_data["X"].ndim == 2 and actual_count > 0 else 0,
            "csv_path": str(csv_path),
            "feature_path": str(feature_path) if feature_path else None,
            "metadata_path": str(metadata_path) if metadata_path else None,
        }

    feature_info = {
        "feature_dir": str(output_dir),
        "run_tag": run_tag,
        "feature_source": settings["source"],
        "backbone": resolved_backbone,
        "checkpoint_path": str(settings["checkpoint_path"]) if settings["checkpoint_path"] else None,
        "extract_batch_size": int(settings["extract_batch_size"]),
        "use_mixed_precision": bool(use_amp),
        "device": str(device),
        "label_mapping_path": str(split_paths["label_mapping_path"]),
        "labels": label_names,
        "splits": split_outputs,
    }
    save_json(feature_info, feature_info_path)
    return feature_info
