"""
Run the inference pipeline with logit-level aggregation per base quadrat.

Difference from pipeline.run_pipeline:
  Each base quadrat (multiple images of the same plot at different dates) is
  treated as one prediction unit. We average image-level probabilities across
  all images of the same quadrat BEFORE applying prior/geo/top-K, so the soft
  distribution over species - not just the top-K species IDs - is combined.

Pipeline:
  1. Load cached logits (no GPU needed).
  2. For each image: tile-level softmax -> tile aggregation (max/mean/topk_mean)
                    -> image-level probabilities [num_classes].
  3. Group images by base_quadrat_id (regex on filename, see scripts/quadrat_aggregate.py).
  4. Combine image-level probs per quadrat (mean by default).
  5. Apply Bayesian cluster prior (cluster determined by filename prefix -
     same for every image of a given quadrat).
  6. Apply geographical filter.
  7. Top-K species per quadrat; assigned to every image in that quadrat.

Usage:
  python scripts/run_pipeline_quadrat.py \\
      --scales 6 5 4 3 2 1 --aggregation max \\
      --top-k 25 \\
      --output output/submission_quadrat_logit.csv
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running as  python scripts/run_pipeline_quadrat.py  from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.aggregation import aggregate
from pipeline.config import PipelineConfig
from pipeline.features import _logit_path
from pipeline.geo_filter import apply_geo_filter, load_or_build_geo_mask
from pipeline.prior import apply_prior, compute_prior_from_logits, load_prior_from_file
from pipeline.submission import load_class_names, write_submission
from scripts.quadrat_aggregate import base_quadrat_id


def _image_paths(images_dir: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir!r}")
    return paths


def _load_image_probs(
    stem: str, scales: List[int], cache_dir: str,
    aggregation: str, topk_mean_k: int, vote_k: int,
) -> np.ndarray:
    """Load cached tile logits for one image, aggregate to [num_classes] probs."""
    tile_logits = []
    for scale in scales:
        lp = _logit_path(cache_dir, stem, scale)
        if not lp.exists():
            raise FileNotFoundError(
                f"Missing cached logits: {lp}. "
                f"Run the extraction job first."
            )
        tile_logits.append(np.load(lp))    # [S*S, C]
    tile_logits = np.concatenate(tile_logits, axis=0)   # [total_tiles, C]
    return aggregate(tile_logits, aggregation, topk_mean_k, vote_k)   # [C]


def main():
    defaults = PipelineConfig()

    p = argparse.ArgumentParser(
        description="Quadrat-level logit aggregation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", default=defaults.images_dir)
    p.add_argument("--cache-dir",  default=defaults.cache_dir)
    p.add_argument("--class-mapping-file", default=defaults.class_mapping_file)
    p.add_argument("--output", required=True)

    p.add_argument("--scales", type=int, nargs="+", default=defaults.scales)
    p.add_argument("--aggregation",
                   choices=["max", "mean", "topk_mean", "vote"], default=defaults.aggregation)
    p.add_argument("--topk-mean-k", type=int, default=defaults.topk_mean_k)
    p.add_argument("--vote-k", type=int, default=defaults.vote_k,
                   help="Top-k species each tile votes for (vote aggregation).")

    p.add_argument("--quadrat-combine",
                   choices=["mean", "max"], default="mean",
                   help="How to combine image-level probs across images of the same quadrat.")

    # Post-processing toggles
    p.add_argument("--no-bayesian-prior", dest="use_bayesian_prior",
                   action="store_false", default=True)
    p.add_argument("--prior-data-path", default=defaults.prior_data_path)
    p.add_argument("--no-geo-filter", dest="use_geo_filter",
                   action="store_false", default=True)
    p.add_argument("--training-metadata-csv", default=defaults.training_metadata_csv)

    p.add_argument("--top-k",     type=int,   default=defaults.top_k)
    p.add_argument("--min-score", type=float, default=defaults.min_score)

    a = p.parse_args()

    t0 = time.time()
    img_paths = _image_paths(a.images_dir)
    stems = [p.stem for p in img_paths]
    print(f"Found {len(img_paths)} test images.")

    # Ensure scale-1 is loaded if we need to compute the prior on the fly
    run_scales = list(a.scales)
    extra_scale1 = False
    if a.use_bayesian_prior and a.prior_data_path is None and 1 not in run_scales:
        run_scales = run_scales + [1]
        extra_scale1 = True

    # Load cached logits per image, aggregate to image-level probs
    print(f"Loading cached logits and aggregating per image (scales={run_scales}, "
          f"agg={a.aggregation})...")
    image_probs: Dict[str, np.ndarray] = {}
    scale1_logits_for_prior = []   # only used if computing prior on the fly

    for stem in stems:
        image_probs[stem] = _load_image_probs(
            stem,
            scales=[s for s in run_scales if s != 1] if extra_scale1 else run_scales,
            cache_dir=a.cache_dir,
            aggregation=a.aggregation,
            topk_mean_k=a.topk_mean_k,
            vote_k=a.vote_k,
        )
        if a.use_bayesian_prior and a.prior_data_path is None:
            lp = _logit_path(a.cache_dir, stem, 1)
            scale1_logits_for_prior.append(np.load(lp))     # [1, C]

    # Group by base quadrat
    quadrat_groups: Dict[str, List[str]] = defaultdict(list)
    for stem in stems:
        quadrat_groups[base_quadrat_id(stem)].append(stem)
    print(f"Grouped {len(stems)} images into {len(quadrat_groups)} base quadrats "
          f"(avg {len(stems) / len(quadrat_groups):.2f} images/quadrat).")

    # Combine image probs per quadrat
    quadrat_probs: Dict[str, np.ndarray] = {}
    for base, group_stems in quadrat_groups.items():
        stack = np.stack([image_probs[s] for s in group_stems], axis=0)   # [G, C]
        if a.quadrat_combine == "mean":
            quadrat_probs[base] = stack.mean(axis=0)
        else:  # max
            quadrat_probs[base] = stack.max(axis=0)

    # Bayesian prior
    priors = None
    if a.use_bayesian_prior:
        if a.prior_data_path:
            print(f"Loading priors from {a.prior_data_path}")
            priors = load_prior_from_file(a.prior_data_path)
        else:
            scale1_logits = np.concatenate(scale1_logits_for_prior, axis=0)   # [N, C]
            print("Computing Bayesian priors from scale-1 logits...")
            priors = compute_prior_from_logits(stems, scale1_logits)

    # Geo filter
    geo_mask = None
    if a.use_geo_filter:
        geo_mask = load_or_build_geo_mask(
            a.training_metadata_csv,
            defaults.num_classes,
            a.cache_dir,
            defaults.geo_ref_lat,
            defaults.geo_ref_lon,
            defaults.geo_country_boxes,
        )

    # Per-quadrat post-processing -> top-K -> broadcast to all images
    class_ids = load_class_names(a.class_mapping_file)
    results: Dict[str, List[int]] = {}

    for base, group_stems in quadrat_groups.items():
        probs = quadrat_probs[base].copy()
        # Cluster is determined from filename prefix, identical for all images
        # in the quadrat; use the first image's stem as representative.
        if priors is not None:
            probs = apply_prior(probs, group_stems[0], priors)
        if geo_mask is not None:
            probs = apply_geo_filter(probs, geo_mask)

        order = np.argsort(probs)[::-1]
        top_ids = [
            class_ids[i] for i in order if probs[i] >= a.min_score
        ][: a.top_k]

        for stem in group_stems:
            results[stem] = top_ids

    write_submission(results, a.output)
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
