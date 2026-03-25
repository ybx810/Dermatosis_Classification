from importlib import import_module

__all__ = [
    "SkinPatchDataset",
    "build_dataloader",
    "SkinImagePatchBagDataset",
    "build_dual_branch_dataloader",
    "dual_branch_collate_fn",
]


_MODULE_MAP = {
    "SkinPatchDataset": "src.datasets.skin_patch_dataset",
    "build_dataloader": "src.datasets.skin_patch_dataset",
    "SkinImagePatchBagDataset": "src.datasets.skin_image_patch_bag_dataset",
    "build_dual_branch_dataloader": "src.datasets.skin_image_patch_bag_dataset",
    "dual_branch_collate_fn": "src.datasets.skin_image_patch_bag_dataset",
}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        module = import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
