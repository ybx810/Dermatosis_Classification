# Large Medical Whole-Image Classification

This repository is a pure whole-image classification project.

The sample unit is always one `source_image`.
The whole-image workflow uses one offline geometry preprocessing path:

1. build image-level split CSVs from raw whole images
2. run `scripts/prepare_whole_images.py` once to generate fixed-size cached whole images
3. train/evaluate on cached whole images (no online resize/pad in dataloader)

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

## Split Modes

`build_image_splits.mode` supports:

- `single`: classic train/val/test split
- `kfold`: fixed independent test split first, then k-fold CV on remaining trainval pool

### kfold semantics (current)

In `mode=kfold`:

1. split full data into:
   - `fixed_test`
   - `trainval_pool`
2. run `n_splits` folds only on `trainval_pool`
3. for each fold:
   - val = one held-out fold from trainval pool
   - train = union of other folds
4. `fixed_test` never participates in fold train/val

Coverage constraints:

- fixed test must include all classes
- every fold val must include all classes
- for each label, image count must be at least `n_splits + 1`
  - for 3-fold, each class must have at least 4 images

## 1. Build Image Splits

```bash
python scripts/build_image_splits.py --config configs/default.yaml
```

### mode=single outputs

- `data/splits/train_images.csv`
- `data/splits/val_images.csv`
- `data/splits/test_images.csv`
- `data/splits/label_mapping.json`
- `data/splits/split_summary.json`

### mode=kfold outputs

- `data/splits/test_images.csv` (fixed independent test)
- `data/splits/trainval_pool_images.csv`
- `data/splits/cv3/fold_0_train_images.csv`
- `data/splits/cv3/fold_0_val_images.csv`
- `data/splits/cv3/fold_1_train_images.csv`
- `data/splits/cv3/fold_1_val_images.csv`
- `data/splits/cv3/fold_2_train_images.csv`
- `data/splits/cv3/fold_2_val_images.csv`
- `data/splits/split_summary.json`
- `data/splits/cv3/cv_split_summary.json`
- `data/splits/label_mapping.json`

## 2. Prepare Cached Whole Images (One-Time)

```bash
python scripts/prepare_whole_images.py --config configs/default.yaml
```

Outputs:

- cached images: `data/cache/whole_images/size_<image_size>/...`
- metadata: `data/metadata/whole_image_metadata.csv`
- summary: `data/metadata/whole_image_summary.json`

Notes:

- offline script is the only place doing resize + padding
- training/validation/testing do not redo geometry transforms
- in `mode=kfold`, cache preparation is still run once for full data (`trainval_pool + fixed_test`), not per fold

## 3. Train (Single Mode)

```bash
python scripts/train.py --config configs/default.yaml
```

## 4. Test One Checkpoint

```bash
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<project>/<run>/best_model.pth
```

Test artifacts are written under:

```text
outputs/<project>/<run>/test/
```

## 5. Cross-Validation Run (kfold Mode)

```bash
python scripts/cross_validate.py --config configs/default.yaml
```

Behavior:

- trains each fold sequentially with fold train/val CSVs
- picks best checkpoint by fold validation primary metric
- evaluates each fold best checkpoint on the same fixed `test_images.csv`
- writes both validation and fixed-test summaries
- defines the final reported experiment result as:
  - mean of 3 folds on fixed-test metrics (`test_accuracy/test_precision/test_recall/test_macro_f1/test_auc_ovr`)
- keeps fixed-test std across folds as stability reference

Cross-validation output root:

```text
outputs/<project_name>/crossval/<timestamp>/
```

Includes:

- `fold_0/`, `fold_1/`, `fold_2/`
- `per_fold_results.csv` (`val_*` and `test_*` metrics)
- `crossval_summary.json` (contains top-level `final_results`)
- `crossval_summary.csv`
- `final_results.csv`

Final-result priority for reporting:

1. `outputs/.../crossval/.../final_results.csv`
2. `outputs/.../crossval/.../crossval_summary.json` -> `final_results`

## Key Config

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
  keep_fixed_test_in_kfold: true
```

`mode=kfold` notes:

- `test_ratio` is used to create fixed test first
- fold construction is then applied to remaining trainval pool
- `train_ratio` and `val_ratio` are kept for compatibility but do not control fold split

## Standard Workflows

### Single workflow

```bash
python scripts/build_image_splits.py --config configs/default.yaml
python scripts/prepare_whole_images.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<project>/<run>/best_model.pth
```

### kfold workflow

```bash
python scripts/build_image_splits.py --config configs/default.yaml
python scripts/prepare_whole_images.py --config configs/default.yaml
python scripts/cross_validate.py --config configs/default.yaml
```

Before kfold workflow, set `build_image_splits.mode: kfold` (or use `--mode kfold`).

## Install

```bash
pip install -r requirements.txt
```
