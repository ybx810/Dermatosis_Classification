from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_DUAL_BRANCH_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")
SUPPORTED_PATCH_POOLING = {"mean", "max"}
SUPPORTED_FUSIONS = {"concat", "weighted_sum", "cross_attention"}


class TorchvisionFeatureExtractor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.kind: str

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = int(model.fc.in_features)
            self.kind = "resnet"
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = int(model.fc.in_features)
            self.kind = "resnet"
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            self.features = model.features
            self.pool = model.avgpool
            self.feature_dim = int(model.classifier[-1].in_features)
            self.kind = "efficientnet"
        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            self.features = model.features
            self.pool = model.avgpool
            self.post_pool = nn.Sequential(model.classifier[0], nn.Flatten(1))
            self.feature_dim = int(model.classifier[-1].in_features)
            self.kind = "convnext"
        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                f"Currently supported: {', '.join(SUPPORTED_DUAL_BRANCH_BACKBONES)}."
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.kind == "resnet":
            features = self.encoder(images)
            return torch.flatten(features, start_dim=1)
        if self.kind == "efficientnet":
            features = self.features(images)
            pooled = self.pool(features)
            return torch.flatten(pooled, start_dim=1)
        if self.kind == "convnext":
            features = self.features(images)
            pooled = self.pool(features)
            return self.post_pool(pooled)
        raise RuntimeError(f"Unsupported extractor kind: {self.kind}")


class DualBranchImagePatchClassifier(nn.Module):
    def __init__(
        self,
        whole_backbone: str,
        patch_backbone: str,
        num_classes: int,
        pretrained: bool = True,
        patch_pooling: str = "mean",
        fusion: str = "concat",
        embedding_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_pooling = str(patch_pooling).lower()
        self.fusion = str(fusion).lower()
        if self.patch_pooling not in SUPPORTED_PATCH_POOLING:
            raise ValueError(
                f"Unsupported dual_branch.patch_pooling: {self.patch_pooling}. "
                f"Expected one of {sorted(SUPPORTED_PATCH_POOLING)}."
            )
        if self.fusion not in SUPPORTED_FUSIONS:
            raise ValueError(
                f"Unsupported dual_branch.fusion: {self.fusion}. Expected one of {sorted(SUPPORTED_FUSIONS)}."
            )
        if embedding_dim <= 0 or fusion_hidden_dim <= 0:
            raise ValueError("embedding_dim and fusion_hidden_dim must be positive integers.")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dual_branch.dropout must be in [0.0, 1.0], got: {dropout}")

        self.whole_encoder = TorchvisionFeatureExtractor(whole_backbone, pretrained=pretrained)
        self.patch_encoder = TorchvisionFeatureExtractor(patch_backbone, pretrained=pretrained)
        self.whole_projection = nn.Linear(self.whole_encoder.feature_dim, embedding_dim)
        self.patch_projection = nn.Linear(self.patch_encoder.feature_dim, embedding_dim)
        self.feature_dropout = nn.Dropout(float(dropout))

        if self.fusion == "concat":
            fused_dim = embedding_dim * 2
            self.classifier = self._build_classifier(fused_dim, fusion_hidden_dim, num_classes, float(dropout))
        elif self.fusion == "weighted_sum":
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_hidden_dim, 1),
            )
            self.classifier = self._build_classifier(embedding_dim, fusion_hidden_dim, num_classes, float(dropout))
        else:
            resolved_heads = self._resolve_attention_heads(embedding_dim, int(attention_heads))
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=resolved_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(embedding_dim)
            self.classifier = self._build_classifier(embedding_dim * 2, fusion_hidden_dim, num_classes, float(dropout))

    def forward(self, whole_images: torch.Tensor, patch_images: Sequence[torch.Tensor]) -> torch.Tensor:
        if whole_images.ndim != 4:
            raise ValueError(f"whole_images must have shape [B, C, H, W], got: {tuple(whole_images.shape)}")
        if not isinstance(patch_images, (list, tuple)):
            raise TypeError("patch_images must be provided as a list or tuple of tensors.")
        if len(patch_images) != int(whole_images.size(0)):
            raise ValueError(
                "The number of patch groups must match the batch size of whole_images: "
                f"{len(patch_images)} vs {whole_images.size(0)}."
            )

        whole_features = self.whole_encoder(whole_images)
        patch_feature_groups = self._encode_patch_groups(patch_images)

        if self.fusion == "concat":
            return self._forward_concat(whole_features, patch_feature_groups)
        if self.fusion == "weighted_sum":
            return self._forward_weighted_sum(whole_features, patch_feature_groups)
        if self.fusion == "cross_attention":
            return self._forward_cross_attention(whole_features, patch_feature_groups)
        raise RuntimeError(f"Unsupported fusion strategy: {self.fusion}")

    def _encode_patch_groups(self, patch_images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        counts: list[int] = []
        flat_patches: list[torch.Tensor] = []
        for bag_index, patch_tensor in enumerate(patch_images):
            if patch_tensor.ndim != 4:
                raise ValueError(
                    f"Each patch tensor must have shape [N, C, H, W], got {tuple(patch_tensor.shape)} at batch index {bag_index}."
                )
            if int(patch_tensor.size(0)) == 0:
                raise ValueError(f"patch_images[{bag_index}] is empty. Each source_image must contribute at least one patch.")
            counts.append(int(patch_tensor.size(0)))
            flat_patches.append(patch_tensor)

        all_patches = torch.cat(flat_patches, dim=0)
        all_features = self.patch_encoder(all_patches)
        return list(torch.split(all_features, counts, dim=0))

    def _pool_patch_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        if self.patch_pooling == "mean":
            return patch_features.mean(dim=0)
        if self.patch_pooling == "max":
            return patch_features.max(dim=0).values
        raise RuntimeError(f"Unsupported patch pooling strategy: {self.patch_pooling}")

    def _forward_concat(
        self,
        whole_features: torch.Tensor,
        patch_feature_groups: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        whole_embeddings = self.feature_dropout(self.whole_projection(whole_features))
        pooled_patch_features = torch.stack([self._pool_patch_features(features) for features in patch_feature_groups], dim=0)
        patch_embeddings = self.feature_dropout(self.patch_projection(pooled_patch_features))
        fused = torch.cat([whole_embeddings, patch_embeddings], dim=1)
        return self.classifier(fused)

    def _forward_weighted_sum(
        self,
        whole_features: torch.Tensor,
        patch_feature_groups: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        whole_embeddings = self.feature_dropout(self.whole_projection(whole_features))
        pooled_patch_features = torch.stack([self._pool_patch_features(features) for features in patch_feature_groups], dim=0)
        patch_embeddings = self.feature_dropout(self.patch_projection(pooled_patch_features))
        gate_input = torch.cat([whole_embeddings, patch_embeddings], dim=1)
        alpha = torch.sigmoid(self.gate(gate_input))
        fused = alpha * whole_embeddings + (1.0 - alpha) * patch_embeddings
        return self.classifier(fused)

    def _forward_cross_attention(
        self,
        whole_features: torch.Tensor,
        patch_feature_groups: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        whole_embeddings = self.feature_dropout(self.whole_projection(whole_features))
        fused_features: list[torch.Tensor] = []
        for index, patch_features in enumerate(patch_feature_groups):
            query = whole_embeddings[index : index + 1].unsqueeze(1)
            patch_tokens = self.feature_dropout(self.patch_projection(patch_features)).unsqueeze(0)
            attended_tokens, _ = self.cross_attention(query=query, key=patch_tokens, value=patch_tokens, need_weights=False)
            attended = attended_tokens.squeeze(1)
            refined = self.attention_norm(attended + whole_embeddings[index : index + 1])
            fused_features.append(torch.cat([whole_embeddings[index : index + 1], refined], dim=1))
        fused = torch.cat(fused_features, dim=0)
        return self.classifier(fused)

    @staticmethod
    def _build_classifier(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _resolve_attention_heads(embedding_dim: int, requested_heads: int) -> int:
        resolved_heads = max(1, int(requested_heads))
        while resolved_heads > 1 and embedding_dim % resolved_heads != 0:
            resolved_heads -= 1
        return resolved_heads
