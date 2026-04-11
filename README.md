# Large Medical Whole-Image Classification

This repository is a pure whole-image classification project.

The training, validation, and test sample unit is always one `source_image`.
The whole-image workflow is a single offline geometry preprocessing path:

1. build image-level splits from raw whole images
2. run `scripts/prepare_whole_images.py` to convert raw whole images into fixed-size square cached images
3. train a single-branch classifier directly on those cached whole images
4. optionally evaluate checkpoints with image-level metrics and per-class metrics

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

Generate image-level split CSV files directly from the raw whole-image directory:

```bash
python scripts/build_image_splits.py --config configs/default.yaml
```

`build_image_splits.mode` supports two modes:

- `single`: keep the original train/val/test split workflow
- `kfold`: generate stratified k-fold CSVs (default `n_splits=3`)

When `mode=kfold`, outputs include:

- `data/splits/cv3/fold_0_train_images.csv`
- `data/splits/cv3/fold_0_val_images.csv`
- `data/splits/cv3/fold_1_train_images.csv`
- `data/splits/cv3/fold_1_val_images.csv`
- `data/splits/cv3/fold_2_train_images.csv`
- `data/splits/cv3/fold_2_val_images.csv`
- `data/splits/cv3/cv_split_summary.json`

The script still writes one shared `data/splits/label_mapping.json` and keeps compatibility train/val/test CSVs under `data/splits/`.

## 2. Prepare Cached Whole Images (One-Time)

Prepare fixed-size cached whole-image copies before training:

```bash
python scripts/prepare_whole_images.py --config configs/default.yaml
```

Outputs:

- cached whole-image files under `data/cache/whole_images/size_<image_size>/...`
- `data/metadata/whole_image_metadata.csv`
- `data/metadata/whole_image_summary.json`

Behavior:

- raw whole images are converted to RGB
- the long side is scaled to `whole_image.image_size`
- the shorter side is padded to a square canvas using `whole_image.pad_position` and `whole_image.pad_value`
- the saved cached image size is always exactly `whole_image.image_size x whole_image.image_size`
- `whole_image.interpolation` controls the offline downsampling interpolation and supports `area` and `bilinear`

This offline script is the only place where whole-image resize and padding happen.
For 3-fold cross-validation, cache preparation is still done only once for the full dataset.

## 3. Train (Single Run)

Run a normal single training run with:

```bash
python scripts/train.py --config configs/default.yaml
```

Behavior:

- one sample equals one `source_image`
- the dataset reads `cached_image_path` from `whole_image_metadata.csv`
- training, validation, and testing do not repeat resize and padding in the dataloader
- transforms only apply flip augmentation on train, then normalize and convert to tensor
- when `whole_image.cache.enabled=true` and `whole_image.cache.use_cached_for_training=true`, cached images are required by default
- if a cached entry is missing and `whole_image.cache.allow_raw_fallback=false`, the dataset raises a clear error and stops
- the model is a single-input single-output backbone classifier
- validation uses direct image-level metrics
- best model selection defaults to validation `macro_f1`

## 4. 3-Fold Cross-Validation

Run 3-fold cross-validation training with:

```bash
python scripts/cross_validate.py --config configs/default.yaml
```

Cross-validation runner behavior:

- reads fold CSVs from `build_image_splits.folds_dir`
- sequentially trains `fold_0`, `fold_1`, `fold_2`
- for each fold, uses `fold_i_train_images.csv` as train and `fold_i_val_images.csv` as held-out eval
- reuses the same whole-image cache metadata and cached files (no fold-specific cache regeneration)

Outputs are written to:

```text
outputs/<project_name>/crossval/<timestamp>/
```

including:

- `fold_0/`, `fold_1/`, `fold_2/` run directories
- `per_fold_results.csv`
- `crossval_summary.json`
- `crossval_summary.csv`

`crossval_summary.json/csv` reports per-fold metrics and mean/std for:

- accuracy
- precision
- recall
- macro_f1
- auc_ovr

## 5. Test (Single Checkpoint)

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

## 6. Key Config

The split config block:

```yaml
build_image_splits:
  mode: single
  output_dir: data/splits
  folds_dir: data/splits/cv3
  label_mapping_path: data/splits/label_mapping.json
  summary_path: data/splits/split_summary.json
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  n_splits: 3
  seed: 42
```

The whole-image config block:

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
    format: png
    overwrite: false
    num_workers: 4
    use_cached_for_training: true
    allow_raw_fallback: false
```

`whole_image.image_size` is the single source of truth for both offline cached image size and model input size.
If an older config still contains `whole_image.cache.size`, it is treated as deprecated and must match `whole_image.image_size`.

## 7. Standard Workflows

Single-run workflow:

```bash
python scripts/build_image_splits.py --config configs/default.yaml
python scripts/prepare_whole_images.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<project>/<run>/best_model.pth
```

3-fold cross-validation workflow:

```bash
python scripts/build_image_splits.py --config configs/default.yaml
python scripts/prepare_whole_images.py --config configs/default.yaml
python scripts/cross_validate.py --config configs/default.yaml
```

Before running cross-validation, set `build_image_splits.mode: kfold` in `configs/default.yaml`.

## Install

```bash
pip install -r requirements.txt
```

