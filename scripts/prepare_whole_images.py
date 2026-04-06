from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_CACHE_FORMATS = {"png", "jpg", "jpeg"}
SUPPORTED_INTERPOLATIONS = {"area", "bilinear"}
METADATA_COLUMNS = [
    "source_image",
    "cached_image_path",
    "label",
    "label_idx",
    "patient_id",
    "split",
    "original_width",
    "original_height",
    "cached_width",
    "cached_height",
    "target_size",
    "status",
    "error",
]


@dataclass(frozen=True)
class WholeImageCacheConfig:
    raw_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    output_dir: Path
    metadata_path: Path
    summary_path: Path
    image_size: int
    image_format: str
    overwrite: bool
    num_workers: int
    pad_value: int | float | tuple[int, ...] | tuple[float, ...]
    pad_position: str
    interpolation: str
    max_image_pixels: int | float | str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cached whole-image copies for whole-image training."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Deprecated alias for whole_image.image_size. If provided, it must match whole_image.image_size.",
    )
    parser.add_argument("--format", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pad-value", type=float, default=None)
    parser.add_argument(
        "--pad-position",
        choices=["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"],
        default=None,
    )
    parser.add_argument(
        "--interpolation",
        choices=sorted(SUPPORTED_INTERPOLATIONS),
        default=None,
    )
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(path_value: str | Path | None, default: str) -> Path:
    raw_path = Path(path_value or default)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def _configure_max_image_pixels(max_image_pixels: int | float | str | None) -> None:
    if max_image_pixels in (None, "", "null"):
        Image.MAX_IMAGE_PIXELS = None
        return
    Image.MAX_IMAGE_PIXELS = int(max_image_pixels)


def _resolve_image_size(args: argparse.Namespace, whole_image_cfg: dict[str, Any], cache_cfg: dict[str, Any]) -> int:
    image_size = int(whole_image_cfg.get("image_size", 1024))
    if image_size <= 0:
        raise ValueError("whole_image.image_size must be a positive integer.")

    if args.size is not None and int(args.size) != image_size:
        raise ValueError(
            "--size is deprecated and must match whole_image.image_size. "
            f"Got --size={int(args.size)} and whole_image.image_size={image_size}."
        )

    deprecated_cache_size = cache_cfg.get("size")
    if deprecated_cache_size not in (None, "", "null"):
        deprecated_cache_size = int(deprecated_cache_size)
        if deprecated_cache_size != image_size:
            raise ValueError(
                "whole_image.cache.size is deprecated and must match whole_image.image_size. "
                f"Got cache.size={deprecated_cache_size} and image_size={image_size}."
            )
        logging.warning(
            "whole_image.cache.size is deprecated. Using whole_image.image_size=%d as the only geometry size.",
            image_size,
        )

    return image_size


def build_config(args: argparse.Namespace) -> WholeImageCacheConfig:
    config_path = resolve_path(args.config, "configs/default.yaml")
    config = load_yaml(config_path) if config_path.exists() else {}

    data_cfg = config.get("data", {})
    whole_image_cfg = config.get("whole_image", {})
    cache_cfg = whole_image_cfg.get("cache", {})
    split_cfg = config.get("build_image_splits", {})

    raw_dir = resolve_path(args.raw_dir or data_cfg.get("raw_dir"), "data/raw")
    default_split_dir = split_cfg.get("output_dir") or data_cfg.get("split_dir") or "data/splits"
    train_csv = resolve_path(args.train_csv or whole_image_cfg.get("train_csv"), f"{default_split_dir}/train_images.csv")
    val_csv = resolve_path(args.val_csv or whole_image_cfg.get("val_csv"), f"{default_split_dir}/val_images.csv")
    test_csv = resolve_path(args.test_csv or whole_image_cfg.get("test_csv"), f"{default_split_dir}/test_images.csv")
    output_dir = resolve_path(args.output_dir or cache_cfg.get("dir"), "data/cache/whole_images")
    metadata_path = resolve_path(args.metadata_path or cache_cfg.get("metadata_path"), "data/metadata/whole_image_metadata.csv")
    summary_path = resolve_path(args.summary_path or cache_cfg.get("summary_path"), "data/metadata/whole_image_summary.json")

    image_size = _resolve_image_size(args, whole_image_cfg, cache_cfg)
    image_format = str(args.format or cache_cfg.get("format") or "png").lower()
    overwrite = args.overwrite if args.overwrite is not None else bool(cache_cfg.get("overwrite", False))
    num_workers = int(args.num_workers or cache_cfg.get("num_workers") or 4)
    pad_value = args.pad_value if args.pad_value is not None else whole_image_cfg.get("pad_value", 0)
    pad_position = str(args.pad_position or whole_image_cfg.get("pad_position", "center")).lower()
    interpolation = str(args.interpolation or whole_image_cfg.get("interpolation", "area")).lower()
    max_image_pixels = whole_image_cfg.get("max_image_pixels")

    if num_workers <= 0:
        raise ValueError("whole_image.cache.num_workers must be a positive integer.")
    if image_format not in SUPPORTED_CACHE_FORMATS:
        raise ValueError(
            f"Unsupported whole_image.cache.format: {image_format}. Expected one of {sorted(SUPPORTED_CACHE_FORMATS)}."
        )
    if interpolation not in SUPPORTED_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported whole_image.interpolation: {interpolation}. Expected one of {sorted(SUPPORTED_INTERPOLATIONS)}."
        )

    return WholeImageCacheConfig(
        raw_dir=raw_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=output_dir,
        metadata_path=metadata_path,
        summary_path=summary_path,
        image_size=image_size,
        image_format=image_format,
        overwrite=overwrite,
        num_workers=num_workers,
        pad_value=pad_value,
        pad_position=pad_position,
        interpolation=interpolation,
        max_image_pixels=max_image_pixels,
    )


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_source_records(config: WholeImageCacheConfig) -> list[dict[str, Any]]:
    split_frames: list[pd.DataFrame] = []
    for split_name, csv_path in (("train", config.train_csv), ("val", config.val_csv), ("test", config.test_csv)):
        if not csv_path.exists():
            raise FileNotFoundError(f"Whole-image split CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "source_image" not in frame.columns or "label" not in frame.columns:
            raise ValueError(f"Split CSV must contain source_image and label columns: {csv_path}")
        if "split" not in frame.columns:
            frame["split"] = split_name
        split_frames.append(frame)

    dataframe = pd.concat(split_frames, ignore_index=True)
    for column_name in ["source_image", "label", "label_idx", "patient_id", "split"]:
        if column_name not in dataframe.columns:
            dataframe[column_name] = ""
        dataframe[column_name] = dataframe[column_name].apply(_normalize_text)

    dataframe = dataframe[dataframe["source_image"].str.len() > 0].copy()
    dataframe = dataframe.drop_duplicates(subset="source_image", keep="first").reset_index(drop=True)
    dataframe = dataframe[["source_image", "label", "label_idx", "patient_id", "split"]]
    records = dataframe.to_dict(orient="records")
    logging.info("Preparing cache for %d unique source_image entries.", len(records))
    return records


def resolve_source_path(source_image: str) -> Path:
    candidate = Path(source_image)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _path_relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return None


def build_cached_output_path(source_image: str, source_path: Path, config: WholeImageCacheConfig) -> Path:
    relative_source = _path_relative_to(source_path, config.raw_dir)
    if relative_source is None:
        digest = hashlib.sha1(source_image.encode("utf-8")).hexdigest()[:12]
        relative_source = Path("external") / f"{source_path.stem}_{digest}{source_path.suffix}"
    size_dir = f"size_{config.image_size}"
    return (config.output_dir / size_dir / relative_source).with_suffix(f".{config.image_format}")


def path_to_project_string(path: Path) -> str:
    relative_path = _path_relative_to(path, PROJECT_ROOT)
    return relative_path.as_posix() if relative_path is not None else str(path)


def _resolve_interpolation(interpolation: str) -> int:
    interpolation_map = {
        "area": cv2.INTER_AREA,
        "bilinear": cv2.INTER_LINEAR,
    }
    interpolation_name = interpolation.lower()
    if interpolation_name not in interpolation_map:
        raise ValueError(
            f"Unsupported interpolation: {interpolation_name}. Expected one of {sorted(interpolation_map)}."
        )
    return interpolation_map[interpolation_name]


def _resolve_pad_value(pad_value: int | float | tuple[int, ...] | tuple[float, ...] | list[int] | list[float]) -> tuple[int, int, int]:
    if isinstance(pad_value, (list, tuple)):
        values = [int(round(float(value))) for value in pad_value]
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise ValueError("pad_value must be a scalar or a 3-element sequence.")
        return tuple(max(0, min(255, value)) for value in values)

    scalar = max(0, min(255, int(round(float(pad_value)))))
    return (scalar, scalar, scalar)


def compute_pad_offsets(
    target_size: int,
    resized_height: int,
    resized_width: int,
    pad_position: str,
    source_image: str,
) -> tuple[int, int]:
    max_top = max(target_size - resized_height, 0)
    max_left = max(target_size - resized_width, 0)
    position = pad_position.lower()

    if position == "center":
        return max_top // 2, max_left // 2
    if position == "top_left":
        return 0, 0
    if position == "top_right":
        return 0, max_left
    if position == "bottom_left":
        return max_top, 0
    if position == "bottom_right":
        return max_top, max_left
    if position == "random":
        seed = int(hashlib.sha1(source_image.encode("utf-8")).hexdigest()[:8], 16)
        generator = np.random.default_rng(seed)
        top = int(generator.integers(0, max_top + 1)) if max_top > 0 else 0
        left = int(generator.integers(0, max_left + 1)) if max_left > 0 else 0
        return top, left

    raise ValueError(
        f"Unsupported pad_position: {position}. Expected one of ['bottom_left', 'bottom_right', 'center', 'random', 'top_left', 'top_right']."
    )


def resize_and_pad_image(
    image: np.ndarray,
    source_image: str,
    target_size: int,
    interpolation: int,
    pad_value: tuple[int, int, int],
    pad_position: str,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape [H, W, 3], got {image.shape}")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = target_size / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    top, left = compute_pad_offsets(target_size, resized_height, resized_width, pad_position, source_image)
    canvas[top : top + resized_height, left : left + resized_width] = resized_image
    return canvas


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def save_cached_image(image: np.ndarray, output_path: Path, image_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image)
    if image_format in {"jpg", "jpeg"}:
        pil_image.save(output_path, format="JPEG", quality=95, subsampling=0)
        return
    pil_image.save(output_path, format="PNG")


def build_metadata_row(
    record: dict[str, Any],
    cached_image_path: str,
    original_size: tuple[int, int] | None,
    cached_size: tuple[int, int] | None,
    target_size: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    original_width, original_height = original_size if original_size is not None else ("", "")
    cached_width, cached_height = cached_size if cached_size is not None else ("", "")
    return {
        "source_image": _normalize_text(record.get("source_image")),
        "cached_image_path": cached_image_path,
        "label": _normalize_text(record.get("label")),
        "label_idx": _normalize_text(record.get("label_idx")),
        "patient_id": _normalize_text(record.get("patient_id")),
        "split": _normalize_text(record.get("split")),
        "original_width": original_width,
        "original_height": original_height,
        "cached_width": cached_width,
        "cached_height": cached_height,
        "target_size": int(target_size),
        "status": status,
        "error": error,
    }


def process_source_image(record: dict[str, Any], config: WholeImageCacheConfig) -> dict[str, Any]:
    source_image = _normalize_text(record.get("source_image"))
    source_path = resolve_source_path(source_image)
    cached_output_path = build_cached_output_path(source_image, source_path, config)
    cached_image_path = path_to_project_string(cached_output_path)

    if cached_output_path.exists() and not config.overwrite:
        try:
            cached_width, cached_height = load_image_size(cached_output_path)
            if cached_width != config.image_size or cached_height != config.image_size:
                return build_metadata_row(
                    record=record,
                    cached_image_path=cached_image_path,
                    original_size=None,
                    cached_size=(cached_width, cached_height),
                    target_size=config.image_size,
                    status="error",
                    error=(
                        "Existing cached whole-image has unexpected size "
                        f"{cached_width}x{cached_height}; expected {config.image_size}x{config.image_size}. "
                        "Re-run with --overwrite to rebuild it."
                    ),
                )

            original_width, original_height = load_image_size(source_path)
            return build_metadata_row(
                record=record,
                cached_image_path=cached_image_path,
                original_size=(original_width, original_height),
                cached_size=(cached_width, cached_height),
                target_size=config.image_size,
                status="skipped_existing",
            )
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            return build_metadata_row(
                record=record,
                cached_image_path=cached_image_path,
                original_size=None,
                cached_size=None,
                target_size=config.image_size,
                status="error",
                error=str(exc),
            )

    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            image_array = np.array(image)

        cached_array = resize_and_pad_image(
            image=image_array,
            source_image=source_image,
            target_size=config.image_size,
            interpolation=_resolve_interpolation(config.interpolation),
            pad_value=_resolve_pad_value(config.pad_value),
            pad_position=config.pad_position,
        )
        save_cached_image(cached_array, cached_output_path, config.image_format)
        cached_width, cached_height = load_image_size(cached_output_path)
        return build_metadata_row(
            record=record,
            cached_image_path=cached_image_path,
            original_size=(original_width, original_height),
            cached_size=(cached_width, cached_height),
            target_size=config.image_size,
            status="success",
        )
    except FileNotFoundError as exc:
        return build_metadata_row(
            record=record,
            cached_image_path=cached_image_path,
            original_size=None,
            cached_size=None,
            target_size=config.image_size,
            status="error",
            error=f"File not found: {exc}",
        )
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
        return build_metadata_row(
            record=record,
            cached_image_path=cached_image_path,
            original_size=None,
            cached_size=None,
            target_size=config.image_size,
            status="error",
            error=str(exc),
        )


def write_metadata(metadata_path: Path, rows: list[dict[str, Any]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    dataframe.sort_values(by=["label", "source_image"], inplace=True, ignore_index=True)
    dataframe.to_csv(metadata_path, index=False, encoding="utf-8")


def write_summary(summary_path: Path, config: WholeImageCacheConfig, rows: list[dict[str, Any]]) -> None:
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict() if rows else {}
    summary = {
        "raw_dir": str(config.raw_dir),
        "train_csv": str(config.train_csv),
        "val_csv": str(config.val_csv),
        "test_csv": str(config.test_csv),
        "output_dir": str(config.output_dir),
        "metadata_path": str(config.metadata_path),
        "image_size": int(config.image_size),
        "format": config.image_format,
        "overwrite": bool(config.overwrite),
        "num_workers": int(config.num_workers),
        "pad_position": config.pad_position,
        "pad_value": config.pad_value if isinstance(config.pad_value, (int, float)) else list(config.pad_value),
        "interpolation": config.interpolation,
        "max_image_pixels": config.max_image_pixels,
        "total_source_images": len(rows),
        "status_counts": counts,
        "success_count": int(counts.get("success", 0)),
        "skipped_existing_count": int(counts.get("skipped_existing", 0)),
        "error_count": int(counts.get("error", 0)),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_whole_images(config: WholeImageCacheConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _configure_max_image_pixels(config.max_image_pixels)
    records = load_source_records(config)
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [executor.submit(process_source_image, record, config) for record in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing whole images"):
            rows.append(future.result())

    write_metadata(config.metadata_path, rows)
    write_summary(config.summary_path, config, rows)

    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict() if rows else {}
    status_counts = {
        "success": int(counts.get("success", 0)),
        "skipped_existing": int(counts.get("skipped_existing", 0)),
        "error": int(counts.get("error", 0)),
    }
    return rows, status_counts


def main() -> None:
    args = parse_args()
    setup_logging()
    config = build_config(args)
    logging.info(
        "Whole-image cache config | image_size=%d format=%s interpolation=%s pad_position=%s overwrite=%s workers=%d",
        config.image_size,
        config.image_format,
        config.interpolation,
        config.pad_position,
        config.overwrite,
        config.num_workers,
    )
    rows, status_counts = prepare_whole_images(config)
    logging.info(
        "Whole-image cache completed | total=%d success=%d skipped=%d error=%d",
        len(rows),
        status_counts["success"],
        status_counts["skipped_existing"],
        status_counts["error"],
    )
    logging.info("Metadata written to %s", config.metadata_path)
    logging.info("Summary written to %s", config.summary_path)


if __name__ == "__main__":
    main()
