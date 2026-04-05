# Large Medical Image Classification

This repository now supports exactly two training modes:

- `patch`: keep the original patch-based training flow and optional patch-to-image aggregated evaluation.
- `whole_image`: a pure whole-image baseline that reads one `source_image` per sample, downsamples it, and feeds it directly into a single backbone classifier.

The current codebase contains only the patch workflow and the whole-image baseline.

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

## 1. Prepare Patches

Patch extraction is still the first step for the patch workflow and also provides the metadata used to derive image-level splits:

```bash
python scripts/prepare_patches.py --config configs/default.yaml
```

The script scans raw images under `data.raw_dir`, cuts them into patches, filters invalid patches, and writes:

- `data/metadata/patch_metadata.csv`
- `data/metadata/patch_summary.json`
- retained patch files under `data/cache/patches/`

## 2. Build Splits

Build the split files after patch extraction:

```bash
python scripts/build_patch_splits.py --config configs/default.yaml
```

The split builder keeps `source_image` mutually exclusive across train/val/test and now exports both patch-level and image-level CSVs:

- Patch-level: `train.csv`, `val.csv`, `test.csv`
- Image-level: `train_images.csv`, `val_images.csv`, `test_images.csv`
- Shared label mapping: `label_mapping.json`
- Summary: `split_summary.json`

Patch split files keep their original format. Whole-image training reads the new `*_images.csv` files only.

## 3. Patch Mode

Use patch mode when you want the original workflow:

```yaml
task:
  mode: patch
```

Behavior in `patch` mode:

- training samples are individual patches from `train.csv`
- validation/test still use patch predictions
- image-level metrics are produced by aggregating patch predictions back to `source_image`
- existing loss, optimizer, scheduler, augmentation, and best-model logic stay on the original patch path

Run training:

```bash
python scripts/train.py --config configs/default.yaml
```

Patch evaluation behavior is controlled by the `evaluation` block in the config, for example:

```yaml
evaluation:
  level: both
  aggregation: mean_prob
  report_patch_metrics: true
  report_image_metrics: true
```

Supported aggregation strategies for patch mode are:

- `mean_prob`
- `majority_vote`
- `max_prob`

## 4. Whole-Image Baseline

Use whole-image mode for the simple baseline:

```yaml
task:
  mode: whole_image
```

Relevant config block:

```yaml
whole_image:
  image_size: 512
  resize_size: 544
  interpolation: bilinear
  train_csv: data/splits/train_images.csv
  val_csv: data/splits/val_images.csv
  test_csv: data/splits/test_images.csv
```

Behavior in `whole_image` mode:

- one sample equals one `source_image`
- the dataset reads raw whole images directly from `source_image`
- each image is resized, center-cropped, normalized, and fed into a single backbone classifier
- training uses the same loss / optimizer / scheduler framework as patch mode
- validation/test report direct image-level classification metrics only
- best model is selected by validation image-level `macro_f1`

Run training after switching `task.mode`:

```bash
python scripts/train.py --config configs/default.yaml
```

## 5. Test a Checkpoint

Evaluate a trained checkpoint with the same config mode used during training:

```bash
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<project>/<run>/best_model.pth
```

Outputs are written to:

```text
outputs/<project>/<run>/test/
```

Patch mode writes combined metrics and per-level artifacts when patch/image reporting is enabled. Whole-image mode writes image-level metrics and artifacts only.

## 6. Typical Workflow

```bash
python scripts/prepare_patches.py --config configs/default.yaml
python scripts/build_patch_splits.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
```

To switch from patch to whole-image baseline, keep the same project data and split generation pipeline, then only change:

```yaml
task:
  mode: whole_image
```

## Install

```bash
pip install -r requirements.txt
```