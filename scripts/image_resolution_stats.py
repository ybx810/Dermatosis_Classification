from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageResolution:
    path: Path
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the images with the smallest and largest resolutions in a directory."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data/raw/16disease_full",
        help="Directory to scan recursively.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def iter_image_paths(root_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_resolution(path: Path) -> ImageResolution:
    with Image.open(path) as image:
        width, height = image.size
    return ImageResolution(path=path, width=width, height=height)


def format_resolution(item: ImageResolution, root_dir: Path) -> str:
    try:
        display_path = item.path.relative_to(root_dir)
    except ValueError:
        display_path = item.path
    return f"{item.width} x {item.height} ({item.pixels:,} pixels) | {display_path}"


def main() -> None:
    args = parse_args()
    setup_logging()

    root_dir = resolve_path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root_dir}")

    image_paths = iter_image_paths(root_dir)
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {root_dir}")

    readable_images: list[ImageResolution] = []
    failed = 0

    iterator = tqdm(image_paths, desc="Scanning images", unit="image", disable=args.no_progress)
    for image_path in iterator:
        try:
            item = read_resolution(image_path)
        except (OSError, UnidentifiedImageError) as exc:
            failed += 1
            logging.warning("Failed to read image %s: %s", image_path, exc)
            continue

        readable_images.append(item)

    if not readable_images:
        raise RuntimeError(f"Found {len(image_paths)} files, but none could be read as images.")
    if len(readable_images) < 2:
        raise RuntimeError("At least two readable images are required to report second smallest and second largest.")

    readable_images.sort(key=lambda item: (item.pixels, item.width, item.height, str(item.path)))
    smallest = readable_images[0]
    second_smallest = readable_images[1]
    second_largest = readable_images[-2]
    largest = readable_images[-1]

    print(f"Scanned images: {len(readable_images)}")
    if failed:
        print(f"Unreadable images: {failed}")
    print(f"Smallest resolution: {format_resolution(smallest, root_dir)}")
    print(f"Second smallest resolution: {format_resolution(second_smallest, root_dir)}")
    print(f"Second largest resolution: {format_resolution(second_largest, root_dir)}")
    print(f"Largest resolution: {format_resolution(largest, root_dir)}")


if __name__ == "__main__":
    main()
