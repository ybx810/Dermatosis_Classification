from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MIL_BACKBONES = {"resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"}


def _resolve_weights(backbone_name: str, pretrained: bool):
    if not pretrained:
        return None

    if backbone_name == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    if backbone_name == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    if backbone_name == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT
    if backbone_name == "convnext_tiny":
        return models.ConvNeXt_Tiny_Weights.DEFAULT
    raise ValueError(
        f"Unsupported MIL backbone: {backbone_name}. "
        "Currently supported: resnet18, resnet50, efficientnet_b0, convnext_tiny."
    )


def _build_backbone_encoder(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    weights = _resolve_weights(backbone_name, pretrained)

    if backbone_name == "resnet18":
        backbone = models.resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
    elif backbone_name == "resnet50":
        backbone = models.resnet50(weights=weights)
        feature_dim = int(backbone.fc.in_features)
    elif backbone_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=weights)
        feature_dim = int(backbone.classifier[-1].in_features)
    elif backbone_name == "convnext_tiny":
        backbone = models.convnext_tiny(weights=weights)
        feature_dim = int(backbone.classifier[-1].in_features)
    else:
        raise ValueError(
            f"Unsupported MIL backbone: {backbone_name}. "
            "Currently supported: resnet18, resnet50, efficientnet_b0, convnext_tiny."
        )

    encoder = nn.Sequential(*list(backbone.children())[:-1])
    return encoder, feature_dim


class AttentionMILModel(nn.Module):
    """Attention-based MIL classifier for variable-sized bags of image patches."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        embedding_dim: int = 256,
        attention_hidden_dim: int = 128,
        dropout: float = 0.0,
        return_attention: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got: {embedding_dim}")
        if attention_hidden_dim <= 0:
            raise ValueError(f"attention_hidden_dim must be positive, got: {attention_hidden_dim}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0], got: {dropout}")

        self.backbone_name = str(backbone_name).lower()
        self.num_classes = int(num_classes)
        self.return_attention = bool(return_attention)

        self.encoder, feature_dim = _build_backbone_encoder(self.backbone_name, pretrained)
        self.feature_dim = int(feature_dim)
        self.embedding_dim = int(embedding_dim)

        self.instance_projection = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

    def _normalize_bag_inputs(self, bags: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(bags, torch.Tensor):
            if bags.ndim == 5:
                return [bag for bag in bags]
            if bags.ndim == 4:
                return [bags]
            raise ValueError(
                f"MIL model expects a 4D bag tensor or 5D batch tensor, got shape {tuple(bags.shape)}"
            )

        if not isinstance(bags, Sequence) or len(bags) == 0:
            raise ValueError("MIL model expects a non-empty sequence of bag tensors.")
        return list(bags)

    def encode_instances(self, bag_images: torch.Tensor) -> torch.Tensor:
        if bag_images.ndim != 4:
            raise ValueError(
                f"Each MIL bag must have shape [num_instances, C, H, W], got {tuple(bag_images.shape)}"
            )
        if bag_images.size(0) == 0:
            raise ValueError("MIL bag must contain at least one patch instance.")

        features = self.encoder(bag_images)
        return torch.flatten(features, start_dim=1)

    def forward(
        self,
        bags: torch.Tensor | Sequence[torch.Tensor],
        return_attention: bool | None = None,
        return_bag_embedding: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        bag_tensors = self._normalize_bag_inputs(bags)
        should_return_attention = self.return_attention if return_attention is None else bool(return_attention)

        bag_logits: list[torch.Tensor] = []
        bag_embeddings: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []

        for bag_images in bag_tensors:
            instance_features = self.encode_instances(bag_images)
            instance_embeddings = self.instance_projection(instance_features)

            scores = self.attention(instance_embeddings).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            bag_embedding = torch.sum(weights.unsqueeze(-1) * instance_embeddings, dim=0)
            logits = self.classifier(bag_embedding)

            bag_logits.append(logits)
            bag_embeddings.append(bag_embedding)
            attention_weights.append(weights)

        logits_tensor = torch.stack(bag_logits, dim=0)
        if not should_return_attention and not return_bag_embedding:
            return logits_tensor

        output: dict[str, Any] = {"logits": logits_tensor}
        if should_return_attention:
            output["attention_weights"] = attention_weights
        if return_bag_embedding:
            output["bag_embedding"] = torch.stack(bag_embeddings, dim=0)
        return output
