from importlib import import_module

__all__ = [
    "SkinPatchDataset",
    "build_dataloader",
]


def __getattr__(name: str):
    if name in {"SkinPatchDataset", "build_dataloader"}:
        module = import_module("src.datasets.skin_patch_dataset")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
