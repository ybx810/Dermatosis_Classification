from importlib import import_module

__all__ = [
    "SkinMILDataset",
    "SkinPatchDataset",
    "build_dataloader",
    "build_mil_dataloader",
    "mil_collate_fn",
]


def __getattr__(name: str):
    if name in {"SkinPatchDataset", "build_dataloader"}:
        module = import_module("src.datasets.skin_patch_dataset")
        return getattr(module, name)
    if name in {"SkinMILDataset", "build_mil_dataloader", "mil_collate_fn"}:
        module = import_module("src.datasets.skin_mil_dataset")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
