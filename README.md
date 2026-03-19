# Large Medical Image Patch Classification

This project is a modular PyTorch scaffold for medical image classification where very large raw images are first split into `512x512` patches, trained at patch level, and optionally evaluated again at image level by aggregating patch predictions back to the original source image.

## Project Goal

Many medical images are too large to feed directly into a GPU training pipeline. This repository is designed for a workflow like:

1. Read large raw images.
2. Split each image into `512x512` patches.
3. Filter out invalid patches such as mostly black background or low-information regions.
4. Build metadata and train/validation/test splits at the patch level.
5. Train a patch classification model on retained patches.
6. Evaluate predictions at patch level, image level, or both.
7. Extend the training pipeline later with class-imbalance strategies such as Focal Loss.

The current version intentionally focuses on a clean and extensible project structure rather than a full business implementation.

## Directory Layout

```text
.
|-- configs/              # YAML config files
|-- data/
|   |-- metadata/         # Source image index, patch index, labels, etc.
|   |-- splits/           # Train/val/test split definitions
|   `-- cache/            # Temporary cached files
|-- outputs/              # Checkpoints, logs, predictions, evaluation artifacts
|-- scripts/              # Entry-point scripts
`-- src/
    |-- datasets/         # Dataset and transform code
    |-- engine/           # Training and evaluation loop logic
    |-- losses/           # Loss functions such as Focal Loss
    |-- models/           # Model builders and heads
    `-- utils/            # Shared utilities
```

## Patch Preparation

The repository includes a patch preparation script for large raw medical images:

```bash
python scripts/prepare_patches.py --config configs/default.yaml
```

Example with custom stride, padded edges, and explicit filter thresholds:

```bash
python scripts/prepare_patches.py --config configs/default.yaml --stride 256 --edge-mode pad --black-pixel-threshold 5 --max-black-ratio 0.95 --min-std 8.0
```

If you need to keep all patches temporarily, you can disable filtering from the command line:

```bash
python scripts/prepare_patches.py --config configs/default.yaml --disable-patch-filter
```

Expected raw data layout:

```text
data/raw/
`-- class_name/
    `-- patient_id/
        `-- image_001.tif
```

The script will:

- scan `jpg`, `jpeg`, `png`, `tif`, and `tiff` files recursively
- infer labels from the first directory level under `raw_dir`
- try to infer `patient_id` from directory structure or file name
- split each source image into patches with the configured window and stride
- filter invalid patches before saving them
- save only retained patches to `data/cache/patches/`
- write metadata only for retained patches to `data/metadata/patch_metadata.csv`
- save patch summary statistics, including candidate, kept, and dropped counts

## Invalid Patch Filtering

`prepare_patches.py` supports an optional invalid-patch filter that runs immediately before each patch is saved.

Default rules:

- Black background filter: count pixels whose intensity is less than or equal to `black_pixel_threshold`; if their ratio is greater than `max_black_ratio`, the patch is dropped.
- Low-information filter: convert the patch to grayscale and compute pixel standard deviation; if it is lower than `min_std`, the patch is dropped.

The filter is compatible with both RGB and grayscale patches. When a patch matches multiple invalid conditions, it is dropped once, and the summary records both per-rule counts plus a separate `dropped_by_multiple_rules` count to avoid confusing totals.

### Patch Filter Config

These options live under `prepare_patches` in `configs/default.yaml` and can also be overridden from the CLI:

- `enable_patch_filter`: enable or disable invalid patch filtering.
- `black_pixel_threshold`: pixel values less than or equal to this threshold are treated as near-black.
- `max_black_ratio`: maximum allowed ratio of near-black pixels in one patch.
- `min_std`: minimum grayscale standard deviation required for a patch to be kept.

## Split Generation

After patch extraction, you can build grouped train/val/test splits from patch metadata:

```bash
python scripts/build_patch_splits.py --config configs/default.yaml
```

Example with explicit metadata path and a fixed seed:

```bash
python scripts/build_patch_splits.py --metadata-path data/metadata/patch_metadata.csv --output-dir data/splits --seed 42 --group-by auto
```

The split builder will:

- read patch-level metadata
- generate `data/splits/all_patches.csv`
- group samples by `patient_id` when available, otherwise by `source_image`
- save `train.csv`, `val.csv`, and `test.csv`
- save `label_mapping.json`
- print sample counts and class distributions for each split

## Patch-Level Training and Image-Level Evaluation

Training remains patch based. Each forward pass, loss computation, and optimizer step still uses individual `512x512` patches.

Validation and testing can now evaluate the same patch predictions in three modes:

- `patch`: report only patch-level metrics.
- `image`: aggregate all patches from the same `source_image` and report image-level metrics as the primary result.
- `both`: keep patch-level metrics as the primary validation/test result and also report image-level metrics.

Image-level ground truth is inferred from the shared label of all patches under the same `source_image`. If inconsistent labels are found inside one `source_image`, evaluation raises a clear error instead of silently mixing labels.

### Aggregation Strategies

Image-level prediction is computed from all patch predictions belonging to one `source_image`.

Supported aggregation strategies:

- `mean_prob`: average patch softmax probabilities, then take `argmax`.
- `majority_vote`: vote on patch predicted classes, with mean probability used as the tie breaker.
- `max_prob`: take the per-class maximum probability across patches, then take `argmax`.

Default recommendation:

- `aggregation: mean_prob`

### Evaluation Config

Add or edit the `evaluation` section in `configs/default.yaml`:

```yaml
evaluation:
  level: both
  aggregation: mean_prob
  report_patch_metrics: true
  report_image_metrics: true
```

Config fields:

- `level`: primary evaluation level used by validation/test summaries and checkpoint selection. Supported values are `patch`, `image`, and `both`.
- `aggregation`: image-level aggregation strategy. Supported values are `mean_prob`, `majority_vote`, and `max_prob`.
- `report_patch_metrics`: whether to keep patch-level metrics in the output payload.
- `report_image_metrics`: whether to keep image-level metrics in the output payload.

Recommended examples:

Patch-level only:

```yaml
evaluation:
  level: patch
  aggregation: mean_prob
  report_patch_metrics: true
  report_image_metrics: false
```

Image-level primary evaluation:

```yaml
evaluation:
  level: image
  aggregation: mean_prob
  report_patch_metrics: true
  report_image_metrics: true
```

### Validation and Test Outputs

During training:

- patch-level training remains unchanged
- `history.json` still records `train_loss`, `train_accuracy`, `train_macro_f1`, `val_loss`, `val_accuracy`, and `val_macro_f1`
- when image-level metrics are enabled, `history.json` also records fields such as `val_patch_accuracy`, `val_patch_macro_f1`, `val_image_accuracy`, and `val_image_macro_f1`
- `val_accuracy` and `val_macro_f1` follow the configured primary evaluation level

During test evaluation, the result directory can now contain:

- `metrics.json`: combined summary with the configured primary metrics plus nested patch/image details
- `patch_metrics.json`: patch-level metrics
- `image_metrics.json`: image-level metrics
- `patch_confusion_matrix.png`: patch-level confusion matrix
- `image_confusion_matrix.png`: image-level confusion matrix

## Running Image-Level Evaluation

Validation during training uses the `evaluation` section automatically:

```bash
python src/main.py --config configs/default.yaml
```

If you want image-level metrics to be the primary validation metric, set:

```yaml
evaluation:
  level: image
  aggregation: mean_prob
  report_patch_metrics: true
  report_image_metrics: true
```

A minimal Python example for checkpoint-based test evaluation is:

```python
from pathlib import Path

from src.engine.test import run_test_from_checkpoint
from src.utils.io import load_yaml

config = load_yaml("configs/default.yaml")
run_test_from_checkpoint(
    config=config,
    checkpoint_path=Path("outputs/large-medical-image-patch-classification/<run>/best_model.pth"),
    run_dir=Path("outputs/large-medical-image-patch-classification/<run>"),
)
```

The resulting test artifacts are written to:

```text
outputs/<project>/<run>/test/
```

## Recommended Regeneration Workflow

When you change patch filter settings, patch size, stride, or edge handling, regenerate both patches and splits so every downstream CSV matches the retained patch set.

1. Remove or archive the old patch cache, patch metadata, patch summary, and split CSV files if you no longer need them.
2. Re-run patch generation with the desired filter settings.
3. Re-run split generation from the new `patch_metadata.csv`.
4. Re-run training and evaluation with the desired `evaluation` config.

Typical command sequence:

```bash
python scripts/prepare_patches.py --config configs/default.yaml
python scripts/build_patch_splits.py --config configs/default.yaml
python src/main.py --config configs/default.yaml
```

## Suggested Workflow

1. Prepare patches with `scripts/prepare_patches.py`.
2. Build grouped split CSV files with `scripts/build_patch_splits.py`.
3. Train with `src/main.py` using patch-level inputs.
4. Inspect patch-level and image-level validation/test metrics depending on `evaluation.level`.
5. Extend the model and loss setup as needed.

## Install

```bash
pip install -r requirements.txt
```

## Notes

- Default patch size is `512`.
- Patch filtering is enabled by default for patch generation.
- Default evaluation uses `level: both` with `aggregation: mean_prob`.
- This scaffold uses Python and PyTorch only, with minimal supporting libraries.
- Focal Loss is reserved for the next phase and only has a placeholder module for now.
