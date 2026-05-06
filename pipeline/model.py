"""
DINOv2 model loading and inference helpers.

Inference is split into two stages so features can be cached independently:

  forward_features(tiles) -> float32 array [N, 768]   (CLS-token embeddings)
  forward_head(features)  -> float32 array [N, 7806]  (raw logits)

Both stages are batched to respect GPU memory limits.
"""

from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_tile_transform(tile_size: int):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((tile_size, tile_size)),   # tiles should already be tile_size x tile_size
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DINOv2Classifier(nn.Module):
    """Thin wrapper around a timm DINOv2 model exposing feature/head split."""

    def __init__(self, model_name: str, num_classes: int, checkpoint_path: str):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
        )
        self.backbone.eval()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled CLS-token embeddings, shape [B, embed_dim]."""
        feats = self.backbone.forward_features(x)          # [B, N_tokens, D]
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # [B, D]
        return pooled

    def get_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the linear classification head to pooled features."""
        return self.backbone.head(features)                # [B, num_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_logits(self.get_features(x))


def load_model(model_name: str, num_classes: int, checkpoint_path: str) -> DINOv2Classifier:
    model = DINOv2Classifier(model_name, num_classes, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def tiles_to_tensor(tiles: List[Image.Image], tile_size: int = 518) -> torch.Tensor:
    """Convert a list of PIL tiles to a batched float tensor [N, 3, H, W]."""
    transform = _build_tile_transform(tile_size)
    return torch.stack([transform(t) for t in tiles])  # [N, 3, H, W]


@torch.no_grad()
def extract_features_batched(
    model: DINOv2Classifier,
    tiles: List[Image.Image],
    tile_size: int = 518,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run the backbone on tiles in mini-batches.
    Returns a float32 numpy array of shape [N, embed_dim].
    """
    device = next(model.parameters()).device
    transform = _build_tile_transform(tile_size)
    results = []

    for start in range(0, len(tiles), batch_size):
        batch = torch.stack([transform(t) for t in tiles[start : start + batch_size]])
        batch = batch.to(device)
        feats = model.get_features(batch)          # [B, D]
        results.append(feats.cpu().numpy())

    return np.concatenate(results, axis=0)         # [N, D]


@torch.no_grad()
def features_to_logits_batched(
    model: DINOv2Classifier,
    features: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Apply the classification head to pre-computed features.
    Returns float32 numpy array of shape [N, num_classes].
    """
    device = next(model.parameters()).device
    feat_tensor = torch.from_numpy(features)
    results = []

    for start in range(0, len(feat_tensor), batch_size):
        batch = feat_tensor[start : start + batch_size].to(device)
        logits = model.get_logits(batch)
        results.append(logits.cpu().numpy())

    return np.concatenate(results, axis=0)         # [N, num_classes]
