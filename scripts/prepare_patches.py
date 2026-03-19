from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
METADATA_COLUMNS = [
    "patch_path",
    "label",
    "source_image",
    "patch_row",
    "patch_col",
    "patient_id",
]
BLACK_BACKGROUND_REASON = "black_background"
LOW_INFORMATION_REASON = "low_information"


@dataclass
class PatchPrepConfig:
    raw_dir: Path
    output_dir: Path
    metadata_path: Path
    summary_path: Path | None
    patch_size: int = 512
    stride: int = 512
    edge_mode: str = "drop"
    save_format: str | None = None
    pad_value: int = 0
    patient_id_strategy: str = "auto"
    enable_patch_filter: bool = True
    black_pixel_threshold: int = 5
    max_black_ratio: float = 0.95
    min_std: float = 8.0


@dataclass
class SourceImageRecord:
    image_path: Path
    label: str
    patient_id: str


@dataclass
class PatchRecord:
    patch_path: str
    label: str
    source_image: str
    patch_row: int
    patch_col: int
    patient_id: str


@dataclass
class PatchFilterStats:
    total_candidate_patches: int = 0
    kept_patch_count: int = 0
    dropped_patch_count: int = 0
    dropped_by_black_background: int = 0
    dropped_by_low_information: int = 0
    dropped_by_multiple_rules: int = 0

    def record_candidate(self, reasons: list[str]) -> None:
        self.total_candidate_patches += 1
        if not reasons:
            self.kept_patch_count += 1
            return

        self.dropped_patch_count += 1
        unique_reasons = set(reasons)
        if BLACK_BACKGROUND_REASON in unique_reasons:
            self.dropped_by_black_background += 1
        if LOW_INFORMATION_REASON in unique_reasons:
            self.dropped_by_low_information += 1
        if len(unique_reasons) > 1:
            self.dropped_by_multiple_rules += 1

    def merge(self, other: "PatchFilterStats") -> None:
        self.total_candidate_patches += other.total_candidate_patches
        self.kept_patch_count += other.kept_patch_count
        self.dropped_patch_count += other.dropped_patch_count
        self.dropped_by_black_background += other.dropped_by_black_background
        self.dropped_by_low_information += other.dropped_by_low_information
        self.dropped_by_multiple_rules += other.dropped_by_multiple_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 512x512 patches from large medical images."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--edge-mode", choices=["drop", "pad"], default=None)
    parser.add_argument("--save-format", type=str, default=None)
    parser.add_argument("--pad-value", type=int, default=None)
    parser.add_argument("--enable-patch-filter", dest="enable_patch_filter", action="store_true")
    parser.add_argument("--disable-patch-filter", dest="enable_patch_filter", action="store_false")
    parser.set_defaults(enable_patch_filter=None)
    parser.add_argument("--black-pixel-threshold", type=int, default=None)
    parser.add_argument("--max-black-ratio", type=float, default=None)
    parser.add_argument("--min-std", type=float, default=None)
    parser.add_argument(
        "--patient-id-strategy",
        choices=["auto", "parent", "filename", "none"],
        default=None,
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path | None, default: str) -> Path:
    raw_path = Path(path_value or default)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def build_config(args: argparse.Namespace) -> PatchPrepConfig:
    config_path = resolve_path(args.config, "configs/default.yaml")
    config = load_yaml(config_path) if config_path.exists() else {}

    data_cfg = config.get("data", {})
    patch_cfg = config.get("prepare_patches", {})

    raw_dir = resolve_path(
        args.raw_dir or patch_cfg.get("raw_dir") or data_cfg.get("raw_dir"),
        "data/raw",
    )
    output_dir = resolve_path(
        args.output_dir
        or patch_cfg.get("output_dir")
        or patch_cfg.get("patch_output_dir")
        or "data/cache/patches",
        "data/cache/patches",
    )
    metadata_path = resolve_path(
        args.metadata_path or patch_cfg.get("metadata_path"),
        "data/metadata/patch_metadata.csv",
    )

    summary_path_value = args.summary_path
    if summary_path_value is None:
        summary_path_value = patch_cfg.get("summary_path")
    summary_path = resolve_path(summary_path_value, "data/metadata/patch_summary.json")

    patch_size = args.patch_size or patch_cfg.get("patch_size") or data_cfg.get("patch_size") or 512
    stride = args.stride or patch_cfg.get("stride") or patch_size
    edge_mode = args.edge_mode or patch_cfg.get("edge_mode") or "drop"
    save_format = args.save_format if args.save_format is not None else patch_cfg.get("save_format")
    pad_value = args.pad_value if args.pad_value is not None else patch_cfg.get("pad_value", 0)
    enable_patch_filter = (
        args.enable_patch_filter
        if args.enable_patch_filter is not None
        else patch_cfg.get("enable_patch_filter", True)
    )
    black_pixel_threshold = (
        args.black_pixel_threshold
        if args.black_pixel_threshold is not None
        else patch_cfg.get("black_pixel_threshold", 5)
    )
    max_black_ratio = (
        args.max_black_ratio
        if args.max_black_ratio is not None
        else patch_cfg.get("max_black_ratio", 0.95)
    )
    min_std = args.min_std if args.min_std is not None else patch_cfg.get("min_std", 8.0)
    patient_id_strategy = (
        args.patient_id_strategy
        or patch_cfg.get("patient_id_strategy")
        or "auto"
    )

    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if edge_mode not in {"drop", "pad"}:
        raise ValueError("edge_mode must be either 'drop' or 'pad'.")
    if not 0 <= int(black_pixel_threshold) <= 255:
        raise ValueError("black_pixel_threshold must be in the range [0, 255].")
    if not 0.0 <= float(max_black_ratio) <= 1.0:
        raise ValueError("max_black_ratio must be in the range [0.0, 1.0].")
    if float(min_std) < 0:
        raise ValueError("min_std must be non-negative.")

    return PatchPrepConfig(
        raw_dir=raw_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        summary_path=summary_path,
        patch_size=patch_size,
        stride=stride,
        edge_mode=edge_mode,
        save_format=save_format,
        pad_value=pad_value,
        patient_id_strategy=patient_id_strategy,
        enable_patch_filter=bool(enable_patch_filter),
        black_pixel_threshold=int(black_pixel_threshold),
        max_black_ratio=float(max_black_ratio),
        min_std=float(min_std),
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_images(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        logging.warning("Raw image directory does not exist: %s", raw_dir)
        return []

    image_paths = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_paths)


def infer_label(image_path: Path, raw_dir: Path) -> str:
    relative_path = image_path.relative_to(raw_dir)
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    if image_path.parent != raw_dir:
        return image_path.parent.name
    return "unknown"


def filename_patient_token(image_path: Path) -> str:
    tokens = [token for token in re.split(r"[_\-\s]+", image_path.stem) if token]
    return tokens[0] if tokens else ""


def infer_patient_id(
    image_path: Path,
    raw_dir: Path,
    label: str,
    strategy: str,
) -> str:
    if strategy == "none":
        return ""
    if strategy == "filename":
        return filename_patient_token(image_path)
    if strategy == "parent":
        return image_path.parent.name if image_path.parent != raw_dir else ""

    relative_path = image_path.relative_to(raw_dir)
    parent_parts = list(relative_path.parts[:-1])
    if len(parent_parts) >= 2:
        return parent_parts[1]
    if len(parent_parts) == 1 and parent_parts[0] != label:
        return parent_parts[0]
    return filename_patient_token(image_path)


def build_source_records(
    image_paths: Iterable[Path],
    raw_dir: Path,
    patient_id_strategy: str,
) -> list[SourceImageRecord]:
    records: list[SourceImageRecord] = []
    for image_path in image_paths:
        label = infer_label(image_path, raw_dir)
        patient_id = infer_patient_id(image_path, raw_dir, label, patient_id_strategy)
        records.append(
            SourceImageRecord(
                image_path=image_path,
                label=label,
                patient_id=patient_id,
            )
        )
    return records


def iter_positions(length: int, patch_size: int, stride: int, edge_mode: str) -> list[int]:
    if edge_mode == "drop":
        return [position for position in range(0, length, stride) if position + patch_size <= length]

    if length <= patch_size:
        return [0]
    return list(range(0, length, stride))


def iter_patch_grid(
    image_width: int,
    image_height: int,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Iterable[tuple[int, int, int, int]]:
    row_positions = iter_positions(image_height, patch_size, stride, edge_mode)
    col_positions = iter_positions(image_width, patch_size, stride, edge_mode)

    for row_index, top in enumerate(row_positions):
        for col_index, left in enumerate(col_positions):
            yield row_index, col_index, top, left


def make_fill_value(image_mode: str, pad_value: int) -> int | tuple[int, ...]:
    bands = Image.getmodebands(image_mode)
    if bands == 1:
        return pad_value
    return tuple([pad_value] * bands)


def extract_patch(
    image: Image.Image,
    left: int,
    top: int,
    patch_size: int,
    edge_mode: str,
    pad_value: int,
) -> Image.Image | None:
    right = min(left + patch_size, image.width)
    bottom = min(top + patch_size, image.height)
    patch = image.crop((left, top, right, bottom))

    if patch.size == (patch_size, patch_size):
        return patch
    if edge_mode == "drop":
        return None

    canvas = Image.new(image.mode, (patch_size, patch_size), color=make_fill_value(image.mode, pad_value))
    canvas.paste(patch, (0, 0))
    return canvas


def resolve_patch_suffix(source_path: Path, save_format: str | None) -> str:
    if not save_format:
        return source_path.suffix.lower()

    normalized = save_format.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def build_patch_output_path(
    source_path: Path,
    raw_dir: Path,
    output_dir: Path,
    patch_row: int,
    patch_col: int,
    save_format: str | None,
) -> Path:
    relative_source = source_path.relative_to(raw_dir)
    suffix = resolve_patch_suffix(source_path, save_format)
    patch_name = f"{source_path.stem}__r{patch_row:03d}_c{patch_col:03d}{suffix}"
    return output_dir / relative_source.parent / patch_name


def to_project_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def save_patch_image(patch: Image.Image, patch_path: Path) -> None:
    patch_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = patch_path.suffix.lower()
    image_to_save = patch
    save_kwargs: dict[str, int] = {}

    if suffix in {".jpg", ".jpeg"}:
        if image_to_save.mode not in {"RGB", "L"}:
            image_to_save = image_to_save.convert("RGB")
        save_kwargs["quality"] = 95

    image_to_save.save(patch_path, **save_kwargs)


def compute_black_pixel_ratio(patch: Image.Image, black_pixel_threshold: int) -> float:
    pixel_array = np.asarray(patch)
    if pixel_array.size == 0:
        return 1.0

    if pixel_array.ndim == 2:
        black_mask = pixel_array <= black_pixel_threshold
    else:
        pixel_channels = pixel_array[..., :3] if pixel_array.shape[-1] >= 3 else pixel_array
        black_mask = np.all(pixel_channels <= black_pixel_threshold, axis=-1)

    return float(np.mean(black_mask))


def compute_grayscale_std(patch: Image.Image) -> float:
    grayscale_array = np.asarray(patch.convert("L"), dtype=np.float32)
    if grayscale_array.size == 0:
        return 0.0
    return float(grayscale_array.std())


def get_invalid_patch_reasons(patch: Image.Image, config: PatchPrepConfig) -> list[str]:
    if not config.enable_patch_filter:
        return []

    reasons: list[str] = []
    black_ratio = compute_black_pixel_ratio(patch, config.black_pixel_threshold)
    if black_ratio > config.max_black_ratio:
        reasons.append(BLACK_BACKGROUND_REASON)

    grayscale_std = compute_grayscale_std(patch)
    if grayscale_std < config.min_std:
        reasons.append(LOW_INFORMATION_REASON)

    return reasons


def is_valid_patch(patch: Image.Image, config: PatchPrepConfig) -> tuple[bool, list[str]]:
    reasons = get_invalid_patch_reasons(patch, config)
    return len(reasons) == 0, reasons


def generate_patches_for_image(
    record: SourceImageRecord,
    config: PatchPrepConfig,
) -> tuple[list[PatchRecord], PatchFilterStats]:
    patch_records: list[PatchRecord] = []
    filter_stats = PatchFilterStats()

    with Image.open(record.image_path) as image:
        for patch_row, patch_col, top, left in iter_patch_grid(
            image_width=image.width,
            image_height=image.height,
            patch_size=config.patch_size,
            stride=config.stride,
            edge_mode=config.edge_mode,
        ):
            patch = extract_patch(
                image=image,
                left=left,
                top=top,
                patch_size=config.patch_size,
                edge_mode=config.edge_mode,
                pad_value=config.pad_value,
            )
            if patch is None:
                continue

            is_valid, invalid_reasons = is_valid_patch(patch, config)
            filter_stats.record_candidate(invalid_reasons)
            if not is_valid:
                continue

            patch_path = build_patch_output_path(
                source_path=record.image_path,
                raw_dir=config.raw_dir,
                output_dir=config.output_dir,
                patch_row=patch_row,
                patch_col=patch_col,
                save_format=config.save_format,
            )
            save_patch_image(patch, patch_path)
            patch_records.append(
                PatchRecord(
                    patch_path=to_project_relative(patch_path),
                    label=record.label,
                    source_image=to_project_relative(record.image_path),
                    patch_row=patch_row,
                    patch_col=patch_col,
                    patient_id=record.patient_id,
                )
            )

    return patch_records, filter_stats


def write_metadata(records: list[PatchRecord], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame(columns=METADATA_COLUMNS).to_csv(metadata_path, index=False)
        return

    dataframe = pd.DataFrame([asdict(record) for record in records])
    dataframe = dataframe.sort_values(["label", "source_image", "patch_row", "patch_col"])
    dataframe.to_csv(metadata_path, index=False)


def build_summary(
    source_records: list[SourceImageRecord],
    patch_records: list[PatchRecord],
    config: PatchPrepConfig,
    skipped_images: int,
    filter_stats: PatchFilterStats,
) -> dict:
    image_counter = Counter(record.label for record in source_records)
    patch_counter = Counter(record.label for record in patch_records)

    return {
        "raw_dir": to_project_relative(config.raw_dir),
        "output_dir": to_project_relative(config.output_dir),
        "metadata_path": to_project_relative(config.metadata_path),
        "patch_size": config.patch_size,
        "stride": config.stride,
        "edge_mode": config.edge_mode,
        "patch_filter": {
            "enabled": config.enable_patch_filter,
            "black_pixel_threshold": config.black_pixel_threshold,
            "max_black_ratio": config.max_black_ratio,
            "min_std": config.min_std,
        },
        "raw_image_count": len(source_records),
        "skipped_images": skipped_images,
        "images_per_class": dict(sorted(image_counter.items())),
        "patches_per_class": dict(sorted(patch_counter.items())),
        "total_candidate_patch_count": filter_stats.total_candidate_patches,
        "kept_patch_count": filter_stats.kept_patch_count,
        "dropped_patch_count": filter_stats.dropped_patch_count,
        "dropped_by_black_background": filter_stats.dropped_by_black_background,
        "dropped_by_low_information": filter_stats.dropped_by_low_information,
        "dropped_by_multiple_rules": filter_stats.dropped_by_multiple_rules,
        "total_patch_count": len(patch_records),
    }


def write_summary(summary: dict, summary_path: Path | None) -> None:
    if summary_path is None:
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def log_summary(summary: dict) -> None:
    logging.info("Raw image count: %s", summary["raw_image_count"])
    logging.info("Skipped images: %s", summary["skipped_images"])
    logging.info("Patch filter enabled: %s", summary["patch_filter"]["enabled"])
    logging.info("Total candidate patches: %s", summary["total_candidate_patch_count"])
    logging.info("Kept patches: %s", summary["kept_patch_count"])
    logging.info("Dropped patches: %s", summary["dropped_patch_count"])
    logging.info("Dropped by black background: %s", summary["dropped_by_black_background"])
    logging.info("Dropped by low information: %s", summary["dropped_by_low_information"])
    logging.info("Dropped by multiple rules: %s", summary["dropped_by_multiple_rules"])
    logging.info("Images per class: %s", summary["images_per_class"])
    logging.info("Patches per class: %s", summary["patches_per_class"])
    logging.info("Total patch count: %s", summary["total_patch_count"])


def run_patch_preparation(config: PatchPrepConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(config.raw_dir)
    source_records = build_source_records(
        image_paths=image_paths,
        raw_dir=config.raw_dir,
        patient_id_strategy=config.patient_id_strategy,
    )

    logging.info("Discovered %s raw images under %s", len(source_records), config.raw_dir)

    patch_records: list[PatchRecord] = []
    filter_stats = PatchFilterStats()
    skipped_images = 0
    for record in tqdm(source_records, desc="Extracting patches", unit="image"):
        try:
            image_patch_records, image_filter_stats = generate_patches_for_image(record, config)
            patch_records.extend(image_patch_records)
            filter_stats.merge(image_filter_stats)
        except Exception as exc:
            skipped_images += 1
            logging.warning("Failed to process %s: %s", record.image_path, exc)

    write_metadata(patch_records, config.metadata_path)
    summary = build_summary(source_records, patch_records, config, skipped_images, filter_stats)
    write_summary(summary, config.summary_path)
    log_summary(summary)
    logging.info("Patch metadata saved to %s", config.metadata_path)

    if config.summary_path is not None:
        logging.info("Patch summary saved to %s", config.summary_path)

    return summary


def main() -> None:
    setup_logging()
    args = parse_args()
    config = build_config(args)
    run_patch_preparation(config)


if __name__ == "__main__":
    main()
