"""
Overnight feature/logit extraction for PlantCLEF 2026.

Caches to disk:
  cache/features/{stem}_s{scale}.npy   [S*S, 768]   DINOv2 CLS embeddings
  cache/logits/{stem}_s{scale}.npy     [S*S, 7806]  raw classifier logits
  cache/priors.npy                     [3, 7806]    cluster Bayesian priors
  cache/geo_mask.npy                   [7806]       geographical species filter

Once this job finishes, every ablation run (any scale subset, any aggregation,
prior on/off, geo on/off) reads from disk and completes in minutes without GPU.

Memory profile
--------------
  GPU VRAM : ~4.4 GB peak (batch=32, float32)  - fits V100-16GB
  CPU RAM  : < 2 GB regardless of test-set size
             (only scale-1 logits [N x 7806] are held for prior computation;
              all other scale logits are written to disk and discarded)

Usage
-----
  python scripts/extract_overnight.py              # all defaults
  python scripts/extract_overnight.py --skip-geo   # skip metadata CSV step
  python scripts/extract_overnight.py --dry-run    # count work, no GPU
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running as  python scripts/extract_overnight.py  from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from pipeline.features import get_logits_for_scale
from pipeline.geo_filter import load_or_build_geo_mask
from pipeline.model import load_model
from pipeline.prior import compute_prior_from_logits, save_priors
from pipeline.submission import load_class_names


# helpers

def _image_paths(images_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir!r}")
    return paths


def _count_pending(stems, scales, cache_dir):
    """Count (image, scale) pairs whose logits are not yet cached."""
    from pipeline.features import _logit_path
    pending = sum(
        1
        for stem in stems
        for scale in scales
        if not _logit_path(cache_dir, stem, scale).exists()
    )
    return pending


def _fmt_eta(seconds):
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# extraction loop

def extract(cfg: PipelineConfig, skip_geo: bool, dry_run: bool):
    img_paths = _image_paths(cfg.images_dir)
    stems = [p.stem for p in img_paths]
    n_images = len(img_paths)
    n_total_pairs = n_images * len(cfg.scales)

    print(f"Images:  {n_images}")
    print(f"Scales:  {cfg.scales}  ({sum(s**2 for s in cfg.scales)} tiles/image)")
    print(f"JPEG:    {'4:2:2 quality=85' if cfg.use_jpeg_compression else 'disabled'}")
    print(f"Cache:   {cfg.cache_dir}")

    pending = _count_pending(stems, cfg.scales, cfg.cache_dir)
    already_done = n_total_pairs - pending
    print(f"Pending: {pending}/{n_total_pairs} (image, scale) pairs "
          f"({already_done} already cached)")

    if dry_run:
        print("\n[dry-run] Exiting without GPU work.")
        return

    if pending == 0:
        print("\nAll logits already cached - nothing to extract.")
    else:
        # Load model
        print("\nLoading model...")
        model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
        import torch
        device = next(model.parameters()).device
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
            print(f"Device:  {torch.cuda.get_device_name(0)}  ({total_vram} MB VRAM)")
        else:
            print(f"Device:  {device}  (no GPU - will be slow)")

        # Accumulate scale-1 logits for prior computation (small: N x 7806 x 4 bytes).
        # All other scale logits are written to disk and discarded immediately.
        scale1_logits_list = []
        t_start = time.time()
        images_done = 0
        pairs_done = 0

        for idx, (img_path, stem) in enumerate(zip(img_paths, stems)):
            img = Image.open(img_path).convert("RGB")

            for scale in cfg.scales:
                logits = get_logits_for_scale(
                    img, stem, scale, model, cfg.cache_dir,
                    tile_size=cfg.tile_size,
                    batch_size=cfg.batch_size,
                    use_jpeg_compression=cfg.use_jpeg_compression,
                    jpeg_quality=cfg.jpeg_quality,
                    jpeg_subsampling=cfg.jpeg_subsampling,
                )  # [S*S, 7806] - written to disk inside the call

                if scale == 1:
                    scale1_logits_list.append(logits)  # [1, 7806]

                pairs_done += 1

            del img
            images_done += 1

            # Progress line every 10 images or on the first
            if images_done == 1 or images_done % 10 == 0 or images_done == n_images:
                elapsed = time.time() - t_start
                rate = images_done / elapsed if elapsed > 0 else 0
                eta_s = (n_images - images_done) / rate if rate > 0 else 0
                print(f"  [{images_done:>{len(str(n_images))}}/{n_images}]  "
                      f"{stem}  "
                      f"({elapsed:.0f}s elapsed, ETA {_fmt_eta(eta_s)})")

        elapsed_total = time.time() - t_start
        print(f"\nExtraction done: {images_done} images in {elapsed_total:.1f}s "
              f"({elapsed_total / images_done:.2f}s/image).")

        # VRAM peak report
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() // 1024**2
            print(f"Peak GPU VRAM used: {peak_mb} MB")

        # Cluster priors
        print("\nComputing cluster priors from scale-1 logits...")
        scale1_logits = np.concatenate(scale1_logits_list, axis=0)  # [N, 7806]
        priors = compute_prior_from_logits(stems, scale1_logits)
        save_priors(priors, cfg.cache_dir)
        print(f"Priors saved: {Path(cfg.cache_dir) / 'priors.npy'}  "
              f"shape={np.stack(list(priors.values())).shape}")

    # Geo mask
    if skip_geo:
        print("\n[skip-geo] Skipping geographical filter build.")
    elif not Path(cfg.training_metadata_csv).exists():
        print(f"\n[skip-geo] Metadata CSV not found: {cfg.training_metadata_csv}")
    else:
        print("\nBuilding geographical species filter...")
        t0 = time.time()
        mask = load_or_build_geo_mask(
            cfg.training_metadata_csv,
            cfg.num_classes,
            cfg.cache_dir,
            cfg.geo_ref_lat,
            cfg.geo_ref_lon,
            cfg.geo_country_boxes,
        )
        print(f"Geo mask done in {time.time()-t0:.1f}s  "
              f"({int(mask.sum())}/{cfg.num_classes} species retained).")

    print("\nAll artifacts cached. Tomorrow's pipeline runs read from disk only.")


# CLI

def main():
    defaults = PipelineConfig()

    p = argparse.ArgumentParser(
        description="Overnight feature extraction for PlantCLEF 2026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir",           default=defaults.images_dir)
    p.add_argument("--model-path",           default=defaults.model_path)
    p.add_argument("--class-mapping-file",   default=defaults.class_mapping_file)
    p.add_argument("--cache-dir",            default=defaults.cache_dir)
    p.add_argument("--training-metadata-csv", default=defaults.training_metadata_csv)
    p.add_argument("--batch-size",  type=int, default=defaults.batch_size)
    p.add_argument("--jpeg",    dest="use_jpeg_compression", action="store_true",  default=True,
                   help="Apply JPEG 4:2:2 q85 round-trip per tile (default).")
    p.add_argument("--no-jpeg", dest="use_jpeg_compression", action="store_false",
                   help="Skip JPEG round-trip - raw tiles. Use a separate --cache-dir.")
    p.add_argument("--skip-geo",    action="store_true",
                   help="Skip geo mask build (if metadata CSV is unavailable).")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print what would be done without running the GPU.")
    a = p.parse_args()

    cfg = PipelineConfig(
        images_dir            = a.images_dir,
        model_path            = a.model_path,
        class_mapping_file    = a.class_mapping_file,
        cache_dir             = a.cache_dir,
        training_metadata_csv = a.training_metadata_csv,
        batch_size            = a.batch_size,
        scales                = [6, 5, 4, 3, 2, 1],
        use_jpeg_compression  = a.use_jpeg_compression,
        jpeg_quality          = 85,
        jpeg_subsampling      = "4:2:2",
    )

    print("=" * 60)
    print("PlantCLEF 2026 - overnight feature extraction")
    print("=" * 60)
    extract(cfg, skip_geo=a.skip_geo, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
