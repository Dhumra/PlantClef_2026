"""
Unified PlantCLEF 2026 inference pipeline.

Combines:
  - Paper 1 pre-processing: JPEG tile compression + multi-scale tiling
  - Paper 2 post-processing: Bayesian prior + geographical species filter

Every stage is optional and controlled by CLI flags, enabling ablation studies
without re-running expensive feature extraction (all intermediate results are
cached to disk).

Example invocations
-------------------
# Full pipeline (paper1 + paper2):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_full.csv

# Multi-scale tiling only (no prior, no geo filter):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_tiling_only.csv \\
    --no-bayesian-prior --no-geo-filter

# Paper2 baseline (4x4 tiling + prior + geo filter):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_paper2.csv \\
    --scales 4 --aggregation mean

# Scale-1 sanity check (no tiling, no extras):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_baseline.csv \\
    --scales 1 --no-jpeg --no-bayesian-prior --no-geo-filter

# Sweep aggregation x prior_strength x top_k in one run:
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output "output/sub_{agg}_ps{ps}_k{k}.csv" \\
    --aggregation max mean \\
    --prior-strength 1.0 0.5 0.0 \\
    --top-k 5 7 9 11 13
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .aggregation import aggregate
from .config import PipelineConfig
from .features import _logit_path, get_all_logits, load_cached_logits_scale1
from .geo_filter import apply_geo_filter, load_or_build_geo_mask
from .model import load_model
from .prior import (
    apply_prior,
    compute_prior_from_logits,
    load_assignments_from_file,
    load_prior_from_file,
    save_priors,
)
from .submission import load_class_names, write_submission

# helpers


def _image_paths(images_dir: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in sorted(Path(images_dir).iterdir()) if p.suffix in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir!r}")
    return paths


def _ensure_scale1_in_scales(scales: List[int]) -> List[int]:
    """Add scale=1 if needed for prior computation; return augmented list."""
    if 1 not in scales:
        return scales + [1]
    return scales


def _all_logits_cached(stems: List[str], scales: List[int], cache_dir: str) -> bool:
    """Return True when every requested image/scale logit file exists."""
    return all(
        _logit_path(cache_dir, stem, scale).exists()
        for stem in stems
        for scale in scales
    )


# pipeline


def _load_logits_and_metadata(cfg: PipelineConfig):
    """Load/extract logits and build shared metadata.

    Returns (stems, all_logits_per_image, run_scales, extra_scale1, priors, geo_mask, class_ids)

    all_logits_per_image[i][j] is the [S*S, C] logit array for image i at run_scales[j].
    extra_scale1 is True when scale-1 was appended solely for prior computation and
    should be excluded from aggregation.

    This is the expensive step (model load + GPU inference). Call it once and reuse
    the returned arrays across all (aggregation, prior_strength, top_k) combinations.
    """
    t0 = time.time()
    img_paths = _image_paths(cfg.images_dir)
    stems = [p.stem for p in img_paths]
    print(f"Found {len(img_paths)} test images.")

    run_scales = list(cfg.scales)
    extra_scale1 = False
    if cfg.use_bayesian_prior and cfg.prior_data_path is None and 1 not in run_scales:
        run_scales = _ensure_scale1_in_scales(run_scales)
        extra_scale1 = True

    model = None
    if _all_logits_cached(stems, run_scales, cfg.cache_dir):
        print("All requested logits found in cache; skipping model load.")
    else:
        print("Loading model...")
        model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
    class_ids = load_class_names(cfg.class_mapping_file)

    geo_mask: Optional[np.ndarray] = None
    if cfg.use_geo_filter:
        geo_mask = load_or_build_geo_mask(
            cfg.training_metadata_csv,
            cfg.num_classes,
            cfg.cache_dir,
            cfg.geo_ref_lat,
            cfg.geo_ref_lon,
            cfg.geo_country_boxes,
        )

    jpeg_desc = (
        f"JPEG {cfg.jpeg_subsampling} q{cfg.jpeg_quality}"
        if cfg.use_jpeg_compression
        else "no JPEG compression"
    )
    print(f"Extracting features for scales {run_scales} ({jpeg_desc})...")

    all_logits_per_image: List[List[np.ndarray]] = []
    for idx, (img_path, stem) in enumerate(zip(img_paths, stems)):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(img_paths)}]  {stem}  ({elapsed:.0f}s elapsed)")

        scale_logits = get_all_logits(
            img_path,
            stem,
            run_scales,
            model,
            cfg.cache_dir,
            cfg.tile_size,
            cfg.batch_size,
            cfg.use_jpeg_compression,
            cfg.jpeg_quality,
            cfg.jpeg_subsampling,
        )
        all_logits_per_image.append(scale_logits)

    priors = None
    if cfg.use_bayesian_prior:
        if cfg.prior_data_path:
            print(f"Loading pre-computed priors from {cfg.prior_data_path}")
            priors = load_prior_from_file(cfg.prior_data_path)
        else:
            s1_idx = run_scales.index(1)
            scale1_logits = np.concatenate(
                [img_logits[s1_idx] for img_logits in all_logits_per_image], axis=0
            )
            print("Computing Bayesian priors from scale-1 predictions...")
            priors = compute_prior_from_logits(stems, scale1_logits)
            save_priors(priors, cfg.cache_dir)

    cluster_assignments: Optional[Dict[str, int]] = None
    if cfg.prior_assignments_path:
        print(f"Loading cluster assignments from {cfg.prior_assignments_path}")
        cluster_assignments = load_assignments_from_file(cfg.prior_assignments_path)

    print(f"Logit loading done in {time.time() - t0:.1f}s.")
    return (
        stems,
        all_logits_per_image,
        run_scales,
        extra_scale1,
        priors,
        cluster_assignments,
        geo_mask,
        class_ids,
    )


def _do_aggregate(
    stems: List[str],
    all_logits_per_image: List[List[np.ndarray]],
    run_scales: List[int],
    extra_scale1: bool,
    aggregation: str,
    topk_mean_k: int,
    vote_k: int,
) -> Dict[str, np.ndarray]:
    """Aggregate tile logits into one probability vector per image.

    Cheap (pure numpy). Safe to call multiple times with different aggregation
    methods on the same all_logits_per_image without re-loading logits.
    """
    agg_probs: Dict[str, np.ndarray] = {}
    for stem, img_scale_logits in zip(stems, all_logits_per_image):
        if extra_scale1:
            selected = [
                logits for s, logits in zip(run_scales, img_scale_logits) if s != 1
            ]
        else:
            selected = img_scale_logits
        tile_logits = np.concatenate(selected, axis=0)
        agg_probs[stem] = aggregate(tile_logits, aggregation, topk_mean_k, vote_k)
    return agg_probs


def _finalize(
    stems: List[str],
    agg_probs: Dict[str, np.ndarray],
    priors,
    prior_strength: float,
    geo_mask: Optional[np.ndarray],
    class_ids: list,
    min_score: float,
    top_k: int,
    cluster_assignments: Optional[Dict[str, int]] = None,
) -> Dict[str, List[int]]:
    """Apply prior + geo filter + top-K for one (prior_strength, top_k) setting."""
    results: Dict[str, List[int]] = {}
    for stem in stems:
        probs = agg_probs[stem].copy()
        if priors is not None:
            probs = apply_prior(
                probs, stem, priors, prior_strength, cluster_assignments
            )
        if geo_mask is not None:
            probs = apply_geo_filter(probs, geo_mask)
        order = np.argsort(probs)[::-1]
        results[stem] = [class_ids[i] for i in order if probs[i] >= min_score][:top_k]
    return results


def run(cfg: PipelineConfig) -> Dict[str, List[int]]:
    t0 = time.time()
    (
        stems,
        all_logits_per_image,
        run_scales,
        extra_scale1,
        priors,
        cluster_assignments,
        geo_mask,
        class_ids,
    ) = _load_logits_and_metadata(cfg)
    agg_probs = _do_aggregate(
        stems,
        all_logits_per_image,
        run_scales,
        extra_scale1,
        cfg.aggregation,
        cfg.topk_mean_k,
        cfg.vote_k,
    )
    results = _finalize(
        stems,
        agg_probs,
        priors,
        cfg.prior_strength,
        geo_mask,
        class_ids,
        cfg.min_score,
        cfg.top_k,
        cluster_assignments=cluster_assignments,
    )
    total = time.time() - t0
    print(
        f"Done. {len(results)} images processed in {total:.1f}s "
        f"({total / len(results):.2f}s/image)."
    )
    return results


# CLI


def _kmeans_k_type(value: str) -> Union[str, int]:
    """Argparse type for --kmeans-k: accepts 'prefix' or a positive integer."""
    if value == "prefix":
        return "prefix"
    try:
        k = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected integer or 'prefix', got {value!r}")
    if k < 2:
        raise argparse.ArgumentTypeError(f"k must be >= 2, got {k}")
    return k


def _parse_args() -> PipelineConfig:
    defaults = PipelineConfig()

    p = argparse.ArgumentParser(
        description="PlantCLEF 2026 unified inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--images-dir", default=defaults.images_dir)
    p.add_argument(
        "--output",
        default=str(Path(defaults.output_dir) / "submission.csv"),
        help=(
            "Path of the output submission CSV. Supports {agg}, {ps}, {k} placeholders "
            "for sweeps, e.g. output/sub_{agg}_ps{ps}_k{k}.csv. "
            "When sweeping multiple values without placeholders, suffixes are auto-appended."
        ),
    )
    p.add_argument(
        "--cache-dir",
        default=defaults.cache_dir,
        help="Directory for cached features and logits.",
    )

    # Model
    p.add_argument("--model-path", default=defaults.model_path)
    p.add_argument("--class-mapping-file", default=defaults.class_mapping_file)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)

    # Pre-processing (paper 1)
    p.add_argument(
        "--jpeg", dest="use_jpeg_compression", action="store_true", default=True
    )
    p.add_argument("--no-jpeg", dest="use_jpeg_compression", action="store_false")
    p.add_argument("--jpeg-quality", type=int, default=defaults.jpeg_quality)
    p.add_argument(
        "--jpeg-subsampling",
        default=defaults.jpeg_subsampling,
        choices=["4:4:4", "4:2:2", "4:2:0", "4:1:1"],
        help=(
            "Chroma subsampling for the JPEG round-trip. "
            "Paper 1 best: '4:2:2' at quality 85, or '4:1:1' at quality 94 "
            "(the latter requires:  pip install PyTurboJPEG)."
        ),
    )
    p.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=defaults.scales,
        help="Tiling scales, e.g. --scales 6 5 4 3 2 1  or  --scales 4  or  --scales 1",
    )

    # Aggregation
    p.add_argument(
        "--aggregation",
        choices=["max", "mean", "topk_mean", "vote"],
        nargs="+",
        default=[defaults.aggregation],
        help=(
            "One or more tile aggregation methods. When multiple are given, writes one "
            "CSV per method. Use {agg} in --output as a template, or names are "
            "auto-generated as <stem>_agg<method>.csv."
        ),
    )
    p.add_argument(
        "--topk-mean-k",
        type=int,
        default=defaults.topk_mean_k,
        help="k for topk_mean aggregation.",
    )
    p.add_argument(
        "--vote-k",
        type=int,
        default=defaults.vote_k,
        help="Top-k species each tile votes for (vote aggregation).",
    )

    # Post-processing (paper 2)
    p.add_argument(
        "--bayesian-prior", dest="use_bayesian_prior", action="store_true", default=True
    )
    p.add_argument(
        "--no-bayesian-prior", dest="use_bayesian_prior", action="store_false"
    )
    p.add_argument(
        "--prior-data-path",
        default=defaults.prior_data_path,
        help="Pre-computed cluster priors .npy file, shape [k, num_classes]. "
        "If omitted, priors are computed from scale-1 predictions. "
        "Ignored when --kmeans-k is given.",
    )
    p.add_argument(
        "--prior-assignments-path",
        default=defaults.prior_assignments_path,
        help="JSON file mapping image stem -> cluster_id, written by "
        "`python -m pipeline.prior`. Ignored when --kmeans-k is given.",
    )
    p.add_argument(
        "--kmeans-k",
        type=_kmeans_k_type,
        nargs="+",
        default=None,
        help="One or more cluster options: integers for data-driven k-means "
        "(reads {cache_dir}/priors_k{{K}}.npy + assignments_k{{K}}.json), "
        "or the special value 'prefix' for one cluster per survey region "
        "(reads priors_prefix.npy + assignments_prefix.json). "
        "All files must be pre-built with `python -m pipeline.prior`. "
        "Overrides --prior-data-path / --prior-assignments-path. "
        "Use {km} in --output, e.g. output/sub_{km}_ps{ps}_k{k}.csv.",
    )
    p.add_argument(
        "--prior-strength",
        type=float,
        nargs="+",
        default=[defaults.prior_strength],
        help="One or more prior strength values (exponent on prior). "
        "1.0 = full prior, 0.5 = square-root dampening, 0.0 = no prior effect. "
        "When multiple are given, writes one CSV per value. The pipeline runs once.",
    )

    p.add_argument(
        "--geo-filter", dest="use_geo_filter", action="store_true", default=True
    )
    p.add_argument("--no-geo-filter", dest="use_geo_filter", action="store_false")
    p.add_argument("--training-metadata-csv", default=defaults.training_metadata_csv)

    # Submission
    p.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[defaults.top_k],
        help="One or more K values. When multiple are given, writes one CSV per K "
        "(use {k} in --output as a template, e.g. output/sub_{k}.csv, or names "
        "are auto-generated as <stem>_topk<K>.csv). The pipeline runs once.",
    )
    p.add_argument("--min-score", type=float, default=defaults.min_score)

    a = p.parse_args()
    aggregation_list = list(dict.fromkeys(a.aggregation))  # deduplicate, preserve order
    top_k_list = sorted(set(a.top_k))  # ascending
    prior_strength_list = sorted(set(a.prior_strength), reverse=True)  # high to low
    # None    = paper 2 original (3 hardcoded filename-prefix clusters)
    # "prefix" = one cluster per survey region (13 clusters, filename-based)
    # int     = data-driven k-means with k clusters
    if a.kmeans_k:
        seen: set = set()
        raw_km: List[Union[str, int]] = [
            v for v in a.kmeans_k if not (v in seen or seen.add(v))
        ]
        # Sort: "prefix" first, then integers ascending
        kmeans_k_list: List[Union[str, int]] = sorted(
            raw_km, key=lambda v: (-1 if v == "prefix" else int(v))
        )
    else:
        kmeans_k_list = [None]

    cfg = PipelineConfig(
        images_dir=a.images_dir,
        output_dir=str(Path(a.output).parent),
        cache_dir=a.cache_dir,
        model_path=a.model_path,
        class_mapping_file=a.class_mapping_file,
        batch_size=a.batch_size,
        use_jpeg_compression=a.use_jpeg_compression,
        jpeg_quality=a.jpeg_quality,
        jpeg_subsampling=a.jpeg_subsampling,
        scales=a.scales,
        aggregation=aggregation_list[0],  # placeholder; main() sweeps the list
        topk_mean_k=a.topk_mean_k,
        vote_k=a.vote_k,
        use_bayesian_prior=a.use_bayesian_prior,
        prior_strength=prior_strength_list[0],  # placeholder; main() sweeps the list
        prior_data_path=a.prior_data_path,
        prior_assignments_path=a.prior_assignments_path,
        use_geo_filter=a.use_geo_filter,
        training_metadata_csv=a.training_metadata_csv,
        top_k=max(top_k_list),  # placeholder; main() slices to each k
        min_score=a.min_score,
    )
    # Stored for main() - not part of PipelineConfig dataclass
    cfg._output_csv = a.output
    cfg._aggregation_list = aggregation_list
    cfg._top_k_list = top_k_list
    cfg._prior_strength_list = prior_strength_list
    cfg._kmeans_k_list = kmeans_k_list
    return cfg


def _format_output_path(
    template: str,
    km: Optional[int],
    agg: str,
    ps: float,
    k: int,
    multi_km: bool,
    multi_agg: bool,
    multi_ps: bool,
    multi_k: bool,
) -> str:
    """Return output path for one (kmeans_k, aggregation, prior_strength, top_k) combo.

    Placeholders in template: {km}, {agg}, {ps}, {k}.
    {km} is the k-means cluster count, or 'prefix' for the filename-prefix baseline.
    If any placeholder is present all four are substituted.
    Otherwise auto-append suffixes only for dimensions with multiple values.
    Single-value runs with no placeholders return the template unchanged.
    """
    ps_str = f"{ps:g}"
    km_str = "paper2" if km is None else str(km)
    if any(p in template for p in ("{km}", "{agg}", "{ps}", "{k}")):
        return template.format(km=km_str, agg=agg, ps=ps_str, k=k)
    path = Path(template)
    parts = []
    if multi_km:
        parts.append(f"km{km_str}")
    if multi_agg:
        parts.append(f"agg{agg}")
    if multi_ps:
        parts.append(f"ps{ps_str}")
    if multi_k:
        parts.append(f"topk{k}")
    if parts:
        return str(path.parent / f"{path.stem}_{'_'.join(parts)}{path.suffix}")
    return template


def main() -> None:
    cfg = _parse_args()
    kmeans_k_list: List[Optional[int]] = cfg._kmeans_k_list
    aggregation_list: List[str] = cfg._aggregation_list
    prior_strength_list: List[float] = cfg._prior_strength_list
    top_k_list: List[int] = cfg._top_k_list

    t0 = time.time()
    (
        stems,
        all_logits_per_image,
        run_scales,
        extra_scale1,
        default_priors,
        default_assignments,
        geo_mask,
        class_ids,
    ) = _load_logits_and_metadata(cfg)

    multi_km = len(kmeans_k_list) > 1
    multi_agg = len(aggregation_list) > 1
    multi_ps = len(prior_strength_list) > 1
    multi_k = len(top_k_list) > 1
    max_k = max(top_k_list)

    for km_k in kmeans_k_list:
        if km_k is None:
            loop_priors = default_priors
            loop_assignments = default_assignments
        elif km_k == "prefix":
            print("Loading per-region-prefix priors...")
            loop_priors = load_prior_from_file(
                str(Path(cfg.cache_dir) / "priors_prefix.npy")
            )
            loop_assignments = load_assignments_from_file(
                str(Path(cfg.cache_dir) / "assignments_prefix.json")
            )
        else:
            print(f"Loading data-driven priors (k={km_k})...")
            loop_priors = load_prior_from_file(
                str(Path(cfg.cache_dir) / f"priors_k{km_k}.npy")
            )
            loop_assignments = load_assignments_from_file(
                str(Path(cfg.cache_dir) / f"assignments_k{km_k}.json")
            )

        for agg in aggregation_list:
            print(f"Aggregating ({agg})...")
            agg_probs = _do_aggregate(
                stems,
                all_logits_per_image,
                run_scales,
                extra_scale1,
                agg,
                cfg.topk_mean_k,
                cfg.vote_k,
            )
            for ps in prior_strength_list:
                results = _finalize(
                    stems,
                    agg_probs,
                    loop_priors,
                    ps,
                    geo_mask,
                    class_ids,
                    cfg.min_score,
                    max_k,
                    cluster_assignments=loop_assignments,
                )
                for k in top_k_list:
                    sliced = {stem: species[:k] for stem, species in results.items()}
                    out_path = _format_output_path(
                        cfg._output_csv,
                        km_k,
                        agg,
                        ps,
                        k,
                        multi_km,
                        multi_agg,
                        multi_ps,
                        multi_k,
                    )
                    write_submission(sliced, out_path)
                    if multi_km or multi_agg or multi_ps or multi_k:
                        print(
                            f"Written km={km_k} agg={agg} ps={ps} top-{k}: {out_path}"
                        )

    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
