from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PatchClassificationDataset(Dataset):
    """Patch-level classification dataset driven by a metadata table."""

    def __init__(self, metadata_file: str | Path, transform: Any = None) -> None:
        self.metadata_file = Path(metadata_file)
        self.transform = transform
        self.samples = pd.read_csv(self.metadata_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return {
            "image": image,
            "label": int(row["label"]),
            "image_path": row["image_path"],
        }
