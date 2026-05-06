"""
Visualize scale-1 CLS-token embeddings via PaCMAP.

For each test image we cached a [1, 768] embedding at scale=1 (the whole image
pass through DINOv2).  This script stacks them into [N, 768], runs PaCMAP for
2-D visualization, and renders points colored by:

  - region prefix (from filename)         - which sites cluster together
  - paper 2's hardcoded 3-cluster mapping - reproduce their visual clustering
  - optional KMeans(k) on PaCMAP output   - data-driven clusters for comparison

Requires:
    pip install pacmap matplotlib scikit-learn

Usage:
    python scripts/visualize_features_pacmap.py
    python scripts/visualize_features_pacmap.py --kmeans 5
    python scripts/visualize_features_pacmap.py --output features_pacmap.png
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from pipeline.prior import REGION_PREFIXES, REGION_TO_CLUSTER, get_cluster


def load_scale1_features(cache_dir: str, images_dir: str):
    feat_dir = Path(cache_dir) / "features"
    img_paths = sorted(
        p
        for p in Path(images_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    stems, vecs = [], []
    for p in img_paths:
        f = feat_dir / f"{p.stem}_s1.npy"
        if not f.exists():
            print(f"WARN: missing {f}, skipping")
            continue
        a = np.load(f)
        if a.shape != (1, 768):
            print(f"WARN: unexpected shape {a.shape} for {f}, skipping")
            continue
        stems.append(p.stem)
        vecs.append(a[0])
    X = np.stack(vecs, axis=0).astype(np.float32)
    print(f"Loaded {X.shape[0]} feature vectors of dim {X.shape[1]}")
    return stems, X


def get_region_prefix(stem: str) -> str:
    """Return the longest matching region prefix, or 'OTHER'."""
    for prefix in sorted(REGION_PREFIXES, key=len, reverse=True):
        if stem.startswith(prefix):
            return prefix
    return "OTHER"


def main():
    cfg = PipelineConfig()

    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=cfg.cache_dir)
    p.add_argument("--images-dir", default=cfg.images_dir)
    p.add_argument(
        "--kmeans",
        type=int,
        default=0,
        help="If > 0, also run KMeans(k) on the embeddings and plot.",
    )
    p.add_argument("--output", default="features_pacmap.png")
    p.add_argument(
        "--n-neighbors", type=int, default=10, help="PaCMAP n_neighbors parameter."
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    # Lazy imports - pacmap/matplotlib may not be available everywhere
    import pacmap
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})

    stems, X = load_scale1_features(a.cache_dir, a.images_dir)

    # PaCMAP reduction
    print(f"Running PaCMAP (n_neighbors={a.n_neighbors}, seed={a.seed})...")
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=a.n_neighbors,
        random_state=a.seed,
    )
    Y = reducer.fit_transform(X, init="pca")
    print(f"PaCMAP done. Output shape: {Y.shape}")

    # Categorical labels
    region_labels = [get_region_prefix(s) for s in stems]
    cluster_labels = [get_cluster(s) for s in stems]

    # Region count summary
    print("\nRegion distribution:")
    for region, n in sorted(Counter(region_labels).items(), key=lambda x: -x[1]):
        cid = REGION_TO_CLUSTER.get(region, "?")
        print(f"  {region:20s}  n={n:5d}  paper2 cluster={cid}")

    # Plot
    n_panels = 2 + (1 if a.kmeans > 0 else 0)
    _w = 7.0 if n_panels == 2 else 3.5 * n_panels
    _h = 4.0 if n_panels == 2 else 5.0
    fig, axes = plt.subplots(1, n_panels, figsize=(_w, _h))
    if n_panels == 1:
        axes = [axes]

    # Panel (a): by region
    unique_regions = sorted(set(region_labels))
    cmap1 = plt.cm.get_cmap("tab20", len(unique_regions))
    for i, region in enumerate(unique_regions):
        mask = np.array([r == region for r in region_labels])
        axes[0].scatter(
            Y[mask, 0],
            Y[mask, 1],
            c=[cmap1(i)],
            s=8,
            alpha=0.7,
            label=region,
        )
    axes[0].text(0.02, 0.97, "(a)", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    axes[0].legend(
        fontsize=7, ncol=3, markerscale=2, framealpha=0.8,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.6, handletextpad=0.3,
    )
    axes[0].set_xlabel("PaCMAP-1")
    axes[0].set_ylabel("PaCMAP-2")

    # Panel (b): by paper 2 cluster
    cmap2 = plt.cm.get_cmap("Set1", 3)
    for cid in sorted(set(cluster_labels)):
        mask = np.array([c == cid for c in cluster_labels])
        axes[1].scatter(
            Y[mask, 0],
            Y[mask, 1],
            c=[cmap2(cid)],
            s=8,
            alpha=0.7,
            label=f"Cluster {cid}",
        )
    axes[1].text(0.02, 0.97, "(b)", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    axes[1].legend(
        fontsize=7, ncol=3, markerscale=2, framealpha=0.8,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.6, handletextpad=0.3,
    )
    axes[1].set_xlabel("PaCMAP-1")
    axes[1].set_ylabel("PaCMAP-2")

    # Panel 3: KMeans
    if a.kmeans > 0:
        from sklearn.cluster import KMeans

        print(f"\nRunning KMeans(k={a.kmeans}) on raw 768-d features...")
        km = KMeans(n_clusters=a.kmeans, random_state=a.seed, n_init=10)
        km_labels = km.fit_predict(X)
        cmap3 = plt.cm.get_cmap("tab10", a.kmeans)
        for c in range(a.kmeans):
            mask = km_labels == c
            axes[2].scatter(
                Y[mask, 0],
                Y[mask, 1],
                c=[cmap3(c)],
                s=8,
                alpha=0.7,
                label=f"k={c} (n={mask.sum()})",
            )
        axes[2].text(0.02, 0.97, "(c)", transform=axes[2].transAxes,
                     fontsize=9, fontweight="bold", va="top")
        axes[2].legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
        axes[2].set_xlabel("PaCMAP-1")
        axes[2].set_ylabel("PaCMAP-2")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)   # room for the region legend (5 rows) below panel (a)
    save_kwargs = {"bbox_inches": "tight"}
    if Path(a.output).suffix.lower() != ".pdf":
        save_kwargs["dpi"] = 150
    fig.savefig(a.output, **save_kwargs)
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
