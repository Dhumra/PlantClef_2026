"""
Cluster-based Bayesian prior (paper 2 post-processing).

Two cluster-assignment strategies are supported:

1. Filename-prefix (paper 2 original)
   Assignment is derived from the image filename, which encodes the survey
   region. No embedding lookup needed at inference time.

2. Data-driven k-means (our extension)
   Assignment is derived by running k-means on DINOv2 scale-1 CLS-token
   embeddings of the test set (2105 images x 768 dims). The cluster
   assignments and prior vectors are pre-computed once and saved to disk.

   To build priors for a sweep of k values (no GPU needed, ~1-3 min total):
       python -m pipeline.prior --cache-dir cache --k 3 5 7 10

   Then pass the results to the pipeline:
       python -m pipeline.run_pipeline \\
           --prior-data-path cache/priors_k5.npy \\
           --prior-assignments-path cache/assignments_k5.json \\
           ...

Prior computation: P(y | cluster c) is the mean softmax of scale-1 logits
over all images in cluster c.

prior_data_path format: .npy file of shape [k, num_classes], float32.
prior_assignments_path format: .json mapping image stem -> cluster_id (int).

Application:
  p_weighted = p * prior[cluster_id] ** strength
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax  # numerically stable

# Cluster prefix mapping (from paper 2 analysis)

REGION_PREFIXES: List[str] = [
    "2024-CEV3",
    "CBN-can",
    "CBN-PdlC",
    "CBN-Pla",
    "CBN-Pyr",
    "GUARDEN-AMB",
    "GUARDEN-CBNMed",
    "LISAH-BOU",
    "LISAH-BVD",
    "LISAH-JAS",
    "LISAH-PEC",
    "OPTMix",
    "RNNB",
]

# Dominant cluster per region prefix (paper 2, clustering notebook)
REGION_TO_CLUSTER: Dict[str, int] = {
    "GUARDEN-CBNMed": 0,
    "RNNB":           0,
    "LISAH-BOU":      0,
    "OPTMix":         0,
    "LISAH-BVD":      0,
    "GUARDEN-AMB":    0,
    "LISAH-PEC":      0,
    "LISAH-JAS":      0,
    "CBN-Pyr":        0,
    "2024-CEV3":      0,
    "CBN-PdlC":       1,
    "CBN-can":        1,
    "CBN-Pla":        2,
}

NUM_CLUSTERS = 3
DEFAULT_CLUSTER = 0   # fallback for unrecognised prefixes

# Build a single regex that matches any known prefix at the start of the name
_PREFIX_PATTERN = re.compile(
    "^(" + "|".join(re.escape(p) for p in sorted(REGION_PREFIXES, key=len, reverse=True)) + ")"
)


def get_cluster(image_name: str) -> int:
    """Return cluster id (0, 1, or 2) from filename prefix (paper 2 original)."""
    m = _PREFIX_PATTERN.match(image_name)
    if m:
        return REGION_TO_CLUSTER.get(m.group(1), DEFAULT_CLUSTER)
    return DEFAULT_CLUSTER


# Prior loading / computation

def load_prior_from_file(path: str) -> Dict[int, np.ndarray]:
    """Load cluster priors from a .npy file (shape [k, num_classes])."""
    arr = np.load(path)
    return {cid: arr[cid] for cid in range(len(arr))}


def compute_prior_from_logits(
    stems: List[str],
    scale1_logits: np.ndarray,   # [N, num_classes]
) -> Dict[int, np.ndarray]:
    """
    Derive P(y | cluster) by averaging softmax probabilities within each cluster.
    Uses paper 2 filename-prefix cluster assignment.
    """
    num_classes = scale1_logits.shape[1]
    cluster_sums   = np.zeros((NUM_CLUSTERS, num_classes), dtype=np.float64)
    cluster_counts = np.zeros(NUM_CLUSTERS, dtype=np.int64)

    probs = softmax(scale1_logits, axis=1).astype(np.float64)  # [N, C]

    for i, stem in enumerate(stems):
        cid = get_cluster(stem)
        cluster_sums[cid]   += probs[i]
        cluster_counts[cid] += 1

    priors: Dict[int, np.ndarray] = {}
    for cid in range(NUM_CLUSTERS):
        if cluster_counts[cid] > 0:
            priors[cid] = (cluster_sums[cid] / cluster_counts[cid]).astype(np.float32)
        else:
            priors[cid] = np.ones(num_classes, dtype=np.float32) / num_classes

    return priors


def save_priors(priors: Dict[int, np.ndarray], cache_dir: str) -> None:
    """Persist filename-prefix priors as {cache_dir}/priors.npy, shape [3, num_classes]."""
    arr = np.stack([priors[cid] for cid in range(NUM_CLUSTERS)])
    out = Path(cache_dir) / "priors.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)


def apply_prior(
    image_probs: np.ndarray,        # [num_classes]
    image_name: str,
    priors: Dict[int, np.ndarray],  # {cluster_id: [num_classes]}
    strength: float = 1.0,
    cluster_assignments: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Multiply image-level probabilities by the cluster prior.

    strength controls how hard the prior is applied:
      1.0 -> image_probs * prior
      0.5 -> image_probs * sqrt(prior)
      0.0 -> image_probs  (no-op)

    cluster_assignments, if provided, overrides the filename-prefix lookup with
    a pre-computed {stem: cluster_id} dict (e.g. from build_kmeans_priors).
    """
    if strength < 0.0:
        raise ValueError("prior strength must be non-negative.")
    if strength == 0.0:
        return image_probs

    if cluster_assignments is not None:
        cid = cluster_assignments.get(image_name, DEFAULT_CLUSTER)
    else:
        cid = get_cluster(image_name)

    prior = priors.get(cid, priors.get(DEFAULT_CLUSTER))
    return image_probs * np.power(prior, strength)


# Data-driven k-means priors

def _load_scale1_array(stems: List[str], cache_dir: str, kind: str) -> np.ndarray:
    """Load and concatenate per-image scale-1 arrays from cache (features or logits)."""
    rows = []
    for stem in stems:
        p = Path(cache_dir) / kind / f"{stem}_s1.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Scale-1 {kind} not cached: {p}\n"
                "Run the overnight extraction script first."
            )
        rows.append(np.load(p))   # [1, D]
    return np.concatenate(rows, axis=0)   # [N, D]


def build_kmeans_priors(
    stems: List[str],
    cache_dir: str,
    k: int,
    n_init: int = 10,
    random_state: int = 42,
) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
    """
    Cluster test images with k-means on scale-1 DINOv2 features, then compute
    P(y | cluster) = mean softmax of scale-1 logits within each cluster.

    Inputs (must already be cached by the overnight extraction script):
      {cache_dir}/features/{stem}_s1.npy  - shape [1, 768]
      {cache_dir}/logits/{stem}_s1.npy   - shape [1, 7806]

    Returns:
      priors:      {cluster_id: float32 [num_classes]}
      assignments: {stem: cluster_id}   for every image in stems

    No GPU needed. Pure CPU + sklearn.
    """
    from sklearn.cluster import KMeans

    print(f"  Loading scale-1 features ({len(stems)} images)...")
    features = _load_scale1_array(stems, cache_dir, "features")  # [N, 768]

    print(f"  Loading scale-1 logits...")
    logits = _load_scale1_array(stems, cache_dir, "logits")       # [N, 7806]

    print(f"  Running KMeans(k={k}, n_init={n_init})...")
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(features)   # [N]

    assignments: Dict[str, int] = {stem: int(labels[i]) for i, stem in enumerate(stems)}

    probs = softmax(logits, axis=1).astype(np.float64)  # [N, C]
    num_classes = logits.shape[1]
    cluster_sums   = np.zeros((k, num_classes), dtype=np.float64)
    cluster_counts = np.zeros(k, dtype=np.int64)
    for i in range(len(stems)):
        cid = int(labels[i])
        cluster_sums[cid]   += probs[i]
        cluster_counts[cid] += 1

    priors: Dict[int, np.ndarray] = {}
    for cid in range(k):
        if cluster_counts[cid] > 0:
            priors[cid] = (cluster_sums[cid] / cluster_counts[cid]).astype(np.float32)
        else:
            priors[cid] = np.ones(num_classes, dtype=np.float32) / num_classes

    return priors, assignments


def save_kmeans_priors(
    priors: Dict[int, np.ndarray],
    assignments: Dict[str, int],
    cache_dir: str,
    k: int,
) -> Tuple[str, str]:
    """
    Save k-means priors and assignments to disk.

    Writes:
      {cache_dir}/priors_k{k}.npy         - float32 [k, num_classes]
      {cache_dir}/assignments_k{k}.json   - {stem: cluster_id}

    Returns the two file paths as strings.
    """
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)

    prior_path = base / f"priors_k{k}.npy"
    assign_path = base / f"assignments_k{k}.json"

    arr = np.stack([priors[cid] for cid in range(k)])   # [k, num_classes]
    np.save(prior_path, arr)
    with open(assign_path, "w") as f:
        json.dump(assignments, f)

    return str(prior_path), str(assign_path)


def load_kmeans_priors(
    cache_dir: str,
    k: int,
) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
    """Load pre-built k-means priors and assignments from disk."""
    arr = np.load(Path(cache_dir) / f"priors_k{k}.npy")   # [k, num_classes]
    priors = {cid: arr[cid] for cid in range(len(arr))}
    with open(Path(cache_dir) / f"assignments_k{k}.json") as f:
        raw = json.load(f)
    assignments = {stem: int(cid) for stem, cid in raw.items()}
    return priors, assignments


def load_assignments_from_file(path: str) -> Dict[str, int]:
    """Load a {stem: cluster_id} assignments JSON written by save_kmeans_priors."""
    with open(path) as f:
        raw = json.load(f)
    return {stem: int(cid) for stem, cid in raw.items()}


# Per-region-prefix priors (one cluster per survey region)

# Stable index for each known prefix (alphabetical order matches REGION_PREFIXES).
_PREFIX_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(REGION_PREFIXES)}
NUM_PREFIX_CLUSTERS: int = len(REGION_PREFIXES)  # 13


def build_prefix_priors(
    stems: List[str],
    cache_dir: str,
) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
    """
    Build one prior vector per known survey region (13 clusters).

    Assignment is purely by filename prefix - identical to get_cluster() but at
    finer granularity: each of the 13 prefixes in REGION_PREFIXES gets its own
    cluster rather than being merged into 3 groups.

    Stems not matching any known prefix fall back to cluster 0 ("2024-CEV3").

    Only {cache_dir}/logits/{stem}_s1.npy is needed (no features file).
    """
    logits = _load_scale1_array(stems, cache_dir, "logits")  # [N, 7806]

    assignments: Dict[str, int] = {}
    for stem in stems:
        m = _PREFIX_PATTERN.match(stem)
        assignments[stem] = _PREFIX_TO_IDX.get(m.group(1), 0) if m else 0

    probs = softmax(logits, axis=1).astype(np.float64)  # [N, C]
    num_classes = logits.shape[1]
    cluster_sums   = np.zeros((NUM_PREFIX_CLUSTERS, num_classes), dtype=np.float64)
    cluster_counts = np.zeros(NUM_PREFIX_CLUSTERS, dtype=np.int64)
    for i, stem in enumerate(stems):
        cid = assignments[stem]
        cluster_sums[cid]   += probs[i]
        cluster_counts[cid] += 1

    priors: Dict[int, np.ndarray] = {}
    for cid in range(NUM_PREFIX_CLUSTERS):
        if cluster_counts[cid] > 0:
            priors[cid] = (cluster_sums[cid] / cluster_counts[cid]).astype(np.float32)
        else:
            priors[cid] = np.ones(num_classes, dtype=np.float32) / num_classes

    return priors, assignments


def save_prefix_priors(
    priors: Dict[int, np.ndarray],
    assignments: Dict[str, int],
    cache_dir: str,
) -> Tuple[str, str]:
    """Save per-prefix priors to {cache_dir}/priors_prefix.npy + assignments_prefix.json."""
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    arr = np.stack([priors[cid] for cid in range(NUM_PREFIX_CLUSTERS)])
    prior_path  = base / "priors_prefix.npy"
    assign_path = base / "assignments_prefix.json"
    np.save(prior_path, arr)
    with open(assign_path, "w") as f:
        json.dump(assignments, f)
    return str(prior_path), str(assign_path)


# CLI - build k-means priors

def main() -> None:
    """
    Pre-compute data-driven k-means priors for a sweep of k values.

    Speed: no GPU needed (pure CPU + sklearn).
    On the HPC cluster the dominant cost is loading ~4200 small .npy files
    from NFS (features + logits for each image).  Expect roughly:
      - File I/O:  ~30-90s  (2105 images x 2 files, NFS latency)
      - KMeans:   ~2-5s per k value  (2105 x 768 is tiny for sklearn)
      Total: ~1-3 minutes for k in {3, 5, 7, 10} combined.

    Example:
        python -m pipeline.prior --cache-dir cache --k 3 5 7 10
    """
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    p = argparse.ArgumentParser(
        description="Build data-driven k-means cluster priors for PlantCLEF 2026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", default="data/test/images",
                   help="Directory of test images (used only to read stems).")
    p.add_argument("--cache-dir", default="cache",
                   help="Cache directory containing features/ and logits/ sub-dirs.")
    p.add_argument(
        "--k", type=int, nargs="+", default=[3, 5, 7, 10],
        help="One or more k-means cluster counts to sweep.",
    )
    p.add_argument(
        "--prefix", action="store_true",
        help="Also build per-region-prefix priors (one cluster per survey region, 13 total). "
             "Writes priors_prefix.npy and assignments_prefix.json.",
    )
    p.add_argument("--n-init", type=int, default=10,
                   help="KMeans n_init (number of restarts).")
    p.add_argument("--random-state", type=int, default=42)
    a = p.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    stems = [f.stem for f in sorted(Path(a.images_dir).iterdir()) if f.suffix in exts]
    if not stems:
        raise FileNotFoundError(f"No images found in {a.images_dir!r}")
    print(f"Found {len(stems)} test images.\n")

    t_global = time.time()

    if a.prefix:
        print(f"{'='*50}")
        print("prefix (one cluster per survey region)")
        t0 = time.time()
        priors, assignments = build_prefix_priors(stems, a.cache_dir)
        prior_path, assign_path = save_prefix_priors(priors, assignments, a.cache_dir)
        counts = {}
        for cid in assignments.values():
            counts[cid] = counts.get(cid, 0) + 1
        sizes = [(REGION_PREFIXES[i], counts.get(i, 0)) for i in range(NUM_PREFIX_CLUSTERS)]
        print(f"  Region sizes: {sizes}")
        print(f"  Saved: {prior_path}")
        print(f"  Saved: {assign_path}")
        print(f"  Done in {time.time() - t0:.1f}s")

    for k in a.k:
        print(f"{'='*50}")
        print(f"k = {k}")
        t0 = time.time()
        priors, assignments = build_kmeans_priors(
            stems, a.cache_dir, k, n_init=a.n_init, random_state=a.random_state,
        )
        prior_path, assign_path = save_kmeans_priors(priors, assignments, a.cache_dir, k)

        counts = {}
        for cid in assignments.values():
            counts[cid] = counts.get(cid, 0) + 1
        sizes = [counts.get(i, 0) for i in range(k)]
        print(f"  Cluster sizes: {sizes}")
        print(f"  Saved: {prior_path}")
        print(f"  Saved: {assign_path}")
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nAll done in {time.time() - t_global:.1f}s total.")
    print(
        "\nTo use in the pipeline:\n"
        "  python -m pipeline.run_pipeline \\\n"
        f"      --kmeans-k prefix 3 5 7 10 \\\n"
        "      ..."
    )


if __name__ == "__main__":
    main()
