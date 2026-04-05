# Large Medical Whole-Image Classification

This repository is a pure whole-image classification project.

The training, validation, and test sample unit is always one `source_image`.
The recommended data path is:

1. build image-level splits from raw whole images
2. prepare cached low-resolution whole-image copies offline
3. train a single-branch classifier on cached whole images
4. evaluate checkpoints with image-level metrics and per-class metrics

## Project Layout

```text
.
|-- configs/
|-- data/
|   |-- metadata/
|   |-- splits/
|   `-- cache/
|-- outputs/
|-- scripts/
`-- src/
    |-- datasets/
    |-- engine/
    |-- losses/
    |-- models/
    `-- utils/
```

## 1. Build Image Splits

Generate image-level train/val/test CSV files directly from the raw whole-image directory:

```bash
python scripts/build_image_splits.py --config configs/default.yaml
```

Outputs:

- `data/splits/train_images.csv`
- `data/splits/val_images.csv`
- `data/splits/test_images.csv`
- `data/splits/label_mapping.json`
- `data/splits/split_summary.json`

Each row corresponds to one `source_image`. The split builder keeps train/val/test mutually exclusive at image level.

## 2. Prepare Cached Whole Images

Prepare cached low-resolution whole-image copies before training:

```bash
python scripts/prepare_whole_images.py --config configs/default.yaml
```

Outputs:

- cached whole-image files under `data/cache/whole_images/size_<cache_size>/...`
- `data/metadata/whole_image_metadata.csv`
- `data/metadata/whole_image_summary.json`

Each cached whole image preserves aspect ratio, is resized to fit within `whole_image.cache.size`, and is padded to a fixed square canvas.

## 3. Train

The default config already targets whole-image training:

```yaml
task:
  mode: whole_image
```

Run training with:

```bash
python scripts/train.py --config configs/default.yaml
```

Behavior:

- one sample equals one `source_image`
- the dataset prefers `cached_image_path` from `whole_image_metadata.csv`
- if a cached file is missing, the dataset can fall back to the raw source image
- the model is a single-input single-output backbone classifier
- validation uses direct image-level metrics
- best model selection defaults to validation `macro_f1`

## 4. Test

Evaluate a checkpoint with the same whole-image config:

```bash
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<project>/<run>/best_model.pth
```

Test outputs are written to:

```text
outputs/<project>/<run>/test/
```

Files written by the test step:

- `metrics.json`
- `confusion_matrix.png`
- `per_class_metrics.csv`

`metrics.json` includes image-level accuracy, macro precision, macro recall, macro F1, optional multi-class AUC, confusion matrix, and per-class metrics.

## 5. Key Config

The whole-image config block looks like this:

```yaml
whole_image:
  train_csv: data/splits/train_images.csv
  val_csv: data/splits/val_images.csv
  test_csv: data/splits/test_images.csv
  image_size: 512
  interpolation: area
  pad_value: 0
  pad_position: center
  max_image_pixels: null
  cache:
    enabled: true
    dir: data/cache/whole_images
    metadata_path: data/metadata/whole_image_metadata.csv
    summary_path: data/metadata/whole_image_summary.json
    size: 1024
    format: png
    overwrite: false
    num_workers: 4
    use_cached_for_training: true
```

`image_size` is the final model input size.
`whole_image.cache.size` is the offline cached whole-image size.

## 6. Standard Workflow

```bash
python scripts/build_image_splits.py --config configs/default.yaml
python scripts/prepare_whole_images.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<project>/<run>/best_model.pth
```

## Install

```bash
pip install -r requirements.txt
```