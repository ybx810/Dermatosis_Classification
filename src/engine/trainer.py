from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.train_one_epoch import train_one_epoch
from src.engine.validate import validate


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = 0.0


class Trainer:
    """Thin wrapper around epoch-level train and validation functions."""

    def __init__(
        self,
        model: Any,
        criterion: Any,
        optimizer: Any,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.state = TrainState()

    def train_one_epoch(self, train_loader: Any, epoch: int, **kwargs: Any) -> dict[str, float]:
        return train_one_epoch(
            model=self.model,
            dataloader=train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            epoch=epoch,
            **kwargs,
        )

    def validate(self, valid_loader: Any, epoch: int, **kwargs: Any) -> dict[str, float]:
        return validate(
            model=self.model,
            dataloader=valid_loader,
            criterion=self.criterion,
            device=self.device,
            epoch=epoch,
            **kwargs,
        )
