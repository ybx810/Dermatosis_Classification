from importlib import import_module

__all__ = [
    "SkinPatchDataset",
    "build_patch_dataloader",
    "build_dataloader",
    "SkinWholeImageDataset",
    "build_whole_image_dataloader",
]


_MODULE_MAP = {
    "SkinPatchDataset": "src.datasets.skin_patch_dataset",
    "build_patch_dataloader": "src.datasets.skin_patch_dataset",
    "build_dataloader": "src.datasets.skin_patch_dataset",
    "SkinWholeImageDataset": "src.datasets.skin_whole_image_dataset",
    "build_whole_image_dataloader": "src.datasets.skin_whole_image_dataset",
}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        module = import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")