"""
Visualise the per-image aggregated score distribution (ranks 1-10).

For each of the 2105 test images the pipeline aggregates tile logits into a
single [7806]-dim probability vector.  This script sorts that vector
descending and plots how the average score decays from rank 1 to rank N,
along with the inter-quartile range across images.

Useful for:
  - Explaining why top-k=3 is optimal (steep drop-off after rank 3).
  - Comparing aggregation methods (max vs mean) in one figure.

Requirements: all requested scales must be cached (logits/ sub-dir).

Usage
-----
    python scripts/plot_score_distribution.py
    python scripts/plot_score_distribution.py \\
        --cache-dir cache_nojpeg \\
        --aggregation max mean \\
        --output output/score_dist_nojpeg.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.aggregation import aggregate
from pipeline.config import PipelineConfig
from pipeline.features import _logit_path


# helpers

def _image_stems(images_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return [p.stem for p in sorted(Path(images_dir).iterdir()) if p.suffix in exts]


def _load_top_scores(stems, scales, cache_dir, agg_method, top_n):
    """Return float32 array [N, top_n]: sorted scores for each image."""
    rows = []
    missing = []
    for stem in stems:
        scale_logits = []
        for scale in scales:
            lp = _logit_path(cache_dir, stem, scale)
            if not lp.exists():
                missing.append(str(lp))
                break
            scale_logits.append(np.load(lp))   # [S*S, C]
        else:
            tile_logits = np.concatenate(scale_logits, axis=0)  # [T, C]
            scores = aggregate(tile_logits, agg_method)          # [C]
            rows.append(np.sort(scores)[::-1][:top_n])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} logit file(s) missing from {cache_dir!r}.\n"
            f"First: {missing[0]}"
        )
    return np.array(rows, dtype=np.float32)   # [N, top_n]


# plot

# Colour cycle for multiple aggregation methods
_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]


def _plot(ax, top_scores, agg_label, color, rank_offset=0.0, bar_width=0.35):
    """Draw one set of bars (mean) + shaded IQR for one aggregation method."""
    top_n = top_scores.shape[1]
    ranks = np.arange(1, top_n + 1)

    means = top_scores.mean(axis=0)
    p25   = np.percentile(top_scores, 25, axis=0)
    p75   = np.percentile(top_scores, 75, axis=0)
    stds  = top_scores.std(axis=0)

    x = ranks + rank_offset
    ax.bar(x, means, width=bar_width, color=color, alpha=0.75, label=agg_label)
    ax.errorbar(x, means, yerr=stds, fmt="none", color=color,
                capsize=3, linewidth=1.2, alpha=0.9)

    return means, p25, p75


def make_figure(results, top_n, scales, cache_dir, n_images):
    """
    results: list of (agg_label, top_scores [N, top_n])
    """
    n_methods = len(results)
    bar_width  = 0.7 / max(n_methods, 1)
    offsets    = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: absolute scores
    ax = axes[0]
    for (label, scores), color, offset in zip(results, _COLORS, offsets):
        _plot(ax, scores, label, color, rank_offset=offset, bar_width=bar_width)

    ax.set_xticks(np.arange(1, top_n + 1))
    ax.set_xlabel("Rank")
    ax.set_ylabel("Aggregated softmax probability")
    ax.set_title("Mean score by rank  (error bars = +/-1 std)")
    ax.legend()
    ax.set_xlim(0.3, top_n + 0.7)

    # right: score normalised to rank-1 per image (fall-off shape)
    ax = axes[1]
    for (label, scores), color, offset in zip(results, _COLORS, offsets):
        # Normalise each image's vector by its rank-1 score, then average
        rank1 = scores[:, 0:1].clip(min=1e-9)
        normed = scores / rank1                 # [N, top_n], rank-1 = 1.0 always
        means_n  = normed.mean(axis=0)
        p25_n    = np.percentile(normed, 25, axis=0)
        p75_n    = np.percentile(normed, 75, axis=0)

        x = np.arange(1, top_n + 1) + offset
        ax.bar(x, means_n, width=bar_width, color=color, alpha=0.75, label=label)
        ax.fill_between(x, p25_n, p75_n, alpha=0.15, color=color)

    ax.set_xticks(np.arange(1, top_n + 1))
    ax.set_xlabel("Rank")
    ax.set_ylabel("Score / rank-1 score  (per image)")
    ax.set_title("Relative score fall-off  (IQR shaded)")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.set_xlim(0.3, top_n + 0.7)

    scale_str = "+".join(str(s) for s in scales)
    fig.suptitle(
        f"Score distribution - scales [{scale_str}]  |  "
        f"cache: {Path(cache_dir).name}  |  N={n_images} images",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# CLI

def main():
    defaults = PipelineConfig()

    p = argparse.ArgumentParser(
        description="Plot per-image aggregated score distribution (ranks 1-N)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir",  default=defaults.images_dir)
    p.add_argument("--cache-dir",   default=defaults.cache_dir)
    p.add_argument("--scales",      type=int, nargs="+", default=[6, 5, 4, 3, 2, 1])
    p.add_argument(
        "--aggregation",
        nargs="+",
        choices=["max", "mean", "topk_mean", "vote"],
        default=["max", "mean"],
        help="One or more aggregation methods to compare.",
    )
    p.add_argument("--top-n",  type=int, default=10,
                   help="Number of ranks to show (x-axis cap).")
    p.add_argument("--output", default="output/score_distribution.png")
    a = p.parse_args()

    stems = _image_stems(a.images_dir)
    if not stems:
        raise FileNotFoundError(f"No images found in {a.images_dir!r}")
    print(f"Found {len(stems)} images.  Scales: {a.scales}  "
          f"({sum(s**2 for s in a.scales)} tiles/image)")

    results = []
    for agg in a.aggregation:
        print(f"Loading + aggregating ({agg})...")
        scores = _load_top_scores(stems, a.scales, a.cache_dir, agg, a.top_n)
        results.append((agg, scores))

        means = scores.mean(axis=0)
        print(f"  {'Rank':>6}  {'Mean':>8}  {'Std':>8}")
        for r, m in enumerate(means, 1):
            print(f"  {r:>6}  {m:>8.5f}  {scores[:, r-1].std():>8.5f}")
        print(f"  rank-3 / rank-1 ratio: {means[2] / means[0]:.3f}")
        if a.top_n >= 10:
            print(f"  rank-10 / rank-1 ratio: {means[9] / means[0]:.3f}")
        print()

    fig = make_figure(results, a.top_n, a.scales, a.cache_dir, len(stems))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
