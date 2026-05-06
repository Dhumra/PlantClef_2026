"""
Tile aggregation strategies.

Supported methods (selectable via PipelineConfig.aggregation):
  "max"       - maximum per species across all tiles (paper 1 default).
  "mean"      - arithmetic mean across all tiles.
  "topk_mean" - average of the top-k tile values per species
                (k set by PipelineConfig.topk_mean_k).
  "vote"      - each tile votes for its top vote_k species; score is the
                fraction of tiles that included each species in their vote.

The Bayesian prior is constant across tiles, so it is applied after aggregation.
"""

import numpy as np


def aggregate_max(tile_probs: np.ndarray) -> np.ndarray:
    """Shape [T, C] -> [C], element-wise maximum over tiles."""
    return tile_probs.max(axis=0)


def aggregate_mean(tile_probs: np.ndarray) -> np.ndarray:
    """Shape [T, C] -> [C], arithmetic mean over tiles."""
    return tile_probs.mean(axis=0)


def aggregate_topk_mean(tile_probs: np.ndarray, k: int) -> np.ndarray:
    """
    Shape [T, C] -> [C].
    For each species, average the k highest tile probabilities.
    When k >= T this is equivalent to the plain mean.
    """
    k = min(k, tile_probs.shape[0])
    # Partial sort: get top-k along tile axis without full sort
    top_k = np.partition(tile_probs, -k, axis=0)[-k:]   # [k, C]
    return top_k.mean(axis=0)


def aggregate_vote(tile_probs: np.ndarray, vote_k: int) -> np.ndarray:
    """
    Shape [T, C] -> [C].
    Each tile votes for its top vote_k species. The image-level score is the
    fraction of tiles that voted for each species, in [0, 1].
    """
    T, C = tile_probs.shape
    k = min(vote_k, C)
    top_idx = np.argpartition(tile_probs, -k, axis=1)[:, -k:]   # [T, k]
    counts = np.zeros(C, dtype=np.float32)
    np.add.at(counts, top_idx.ravel(), 1)
    return counts / float(T)


def aggregate(
    tile_logits: np.ndarray,
    method: str = "max",
    topk_mean_k: int = 5,
    vote_k: int = 5,
) -> np.ndarray:
    """
    Convert raw logits [T, C] to an aggregated image-level score [C].
    Softmax is applied per tile before aggregation.
    """
    # Numerically stable softmax along the class axis
    shifted = tile_logits - tile_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    tile_probs = exp / exp.sum(axis=1, keepdims=True)   # [T, C]

    if method == "max":
        return aggregate_max(tile_probs)
    if method == "mean":
        return aggregate_mean(tile_probs)
    if method == "topk_mean":
        return aggregate_topk_mean(tile_probs, topk_mean_k)
    if method == "vote":
        return aggregate_vote(tile_probs, vote_k)

    raise ValueError(f"Unknown aggregation method: {method!r}. "
                     "Choose 'max', 'mean', 'topk_mean', or 'vote'.")
