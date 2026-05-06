"""
Feature extraction with disk caching.

Cache layout (one file per image per scale):

  {cache_dir}/features/{image_stem}_s{scale}.npy   shape [S*S, embed_dim]
  {cache_dir}/logits/{image_stem}_s{scale}.npy     shape [S*S, num_classes]

Both levels are cached.  If a logits file exists it is loaded directly,
skipping the feature-extraction and head steps entirely.  If only features
are cached, only the cheap linear head is re-run.

If only features are cached, only the cheap linear head is re-run.
Separating the two levels lets you reload features without re-running the backbone
when experimenting with different heads or aggregation variants.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .compression import jpeg_compress
from .model import (
    DINOv2Classifier,
    extract_features_batched,
    features_to_logits_batched,
)
from .tiling import extract_tiles


def _feat_path(cache_dir: str, stem: str, scale: int) -> Path:
    return Path(cache_dir) / "features" / f"{stem}_s{scale}.npy"


def _logit_path(cache_dir: str, stem: str, scale: int) -> Path:
    return Path(cache_dir) / "logits" / f"{stem}_s{scale}.npy"


def get_logits_for_scale(
    img_path: Path,
    stem: str,
    scale: int,
    model: DINOv2Classifier,
    cache_dir: str,
    tile_size: int = 518,
    batch_size: int = 32,
    use_jpeg_compression: bool = True,
    jpeg_quality: int = 85,
    jpeg_subsampling: str = "4:2:2",
) -> np.ndarray:
    """
    Return logits of shape [S*S, num_classes] for one image at one scale.

    The result is loaded from cache if available; otherwise tiles are extracted,
    (optionally JPEG-compressed), features extracted, head applied, and both
    intermediate arrays saved.
    """
    lp = _logit_path(cache_dir, stem, scale)
    if lp.exists():
        return np.load(lp)

    fp = _feat_path(cache_dir, stem, scale)

    if fp.exists():
        features = np.load(fp)
    else:
        # Extract tiles for this scale
        img = Image.open(img_path).convert("RGB")
        tiles: List[Image.Image] = extract_tiles(img, scale, tile_size)

        # JPEG round-trip at tile level (paper 1 domain alignment).
        # Applied even for .jpg source files - the round-trip standardises
        # compression artifacts regardless of the original encoding.
        if use_jpeg_compression:
            tiles = [jpeg_compress(t, jpeg_quality, jpeg_subsampling) for t in tiles]

        features = extract_features_batched(model, tiles, tile_size, batch_size)

        fp.parent.mkdir(parents=True, exist_ok=True)
        np.save(fp, features)

    logits = features_to_logits_batched(model, features, batch_size=256)

    lp.parent.mkdir(parents=True, exist_ok=True)
    np.save(lp, logits)

    return logits


def get_all_logits(
    img: Image.Image,
    stem: str,
    scales: List[int],
    model: DINOv2Classifier,
    cache_dir: str,
    tile_size: int = 518,
    batch_size: int = 32,
    use_jpeg_compression: bool = True,
    jpeg_quality: int = 85,
    jpeg_subsampling: str = "4:2:2",
) -> List[np.ndarray]:
    """
    Return a list of logit arrays, one per scale, in the order given by *scales*.
    Each array has shape [S*S, num_classes].
    """
    return [
        get_logits_for_scale(
            img,
            stem,
            scale,
            model,
            cache_dir,
            tile_size,
            batch_size,
            use_jpeg_compression,
            jpeg_quality,
            jpeg_subsampling,
        )
        for scale in scales
    ]


def load_cached_logits_scale1(
    stems: List[str],
    cache_dir: str,
) -> Optional[np.ndarray]:
    """
    Load scale-1 logits for all *stems* from cache.
    Returns array of shape [N, num_classes], or None if any file is missing.
    """
    rows = []
    for stem in stems:
        p = _logit_path(cache_dir, stem, scale=1)
        if not p.exists():
            return None
        rows.append(np.load(p))  # each is [1, num_classes]
    return np.concatenate(rows, axis=0)  # [N, num_classes]
