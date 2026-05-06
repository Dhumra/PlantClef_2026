# PlantCLEF 2026 — Combined Inference Pipeline

CS 682 Final Project, Spring 2026  
PlantCLEF 2026 @ LifeCLEF / CVPR-FGVC

This repository implements a combined inference pipeline for the PlantCLEF 2026
quadrat-level plant species identification challenge. It integrates and extends
the two strongest PlantCLEF 2025 submissions:

- **Paper 1** — Espitalier et al., _"Pre-processing for multi-scale tiling"_, CLEF 2025 ([paper 238](https://ceur-ws.org/Vol-4038/paper_238.pdf))
- **Paper 2** — Gustineli et al., _"Post-processing with Bayesian priors and geographic filtering"_, CLEF 2025 ([paper 242](https://ceur-ws.org/Vol-4038/paper_242.pdf))

Our contributions on top of these baselines:

1. **Unified Python re-implementation** — both paper's approaches in a single modular pipeline.
2. **Data-driven clustering** — K-Means on DINOv2 CLS embeddings as an alternative to Paper 2's fixed 3-cluster region mapping; also a finer 13-cluster per-region variant.
3. **Prior strength annealing** — a `prior_strength` scalar to soften the Bayesian prior.
4. **Multi-sweep CLI** — a single command can sweep multiple aggregation methods, prior types, prior strengths, and top-K values, loading the GPU cache only once.

---

## Results (Kaggle Public Leaderboard)

| Configuration             | Aggregation | Prior         | Prior strength | Top-K | Public F1  |
| ------------------------- | ----------- | ------------- | -------------- | ----- | ---------- |
| Paper 1 (reference)       | max         | —             | —              | —     | 0.3523     |
| Paper 2 (reference)       | mean        | region-3      | 1.0            | —     | 0.3160     |
| Ours (JPEG + multi-scale) | max         | region-3      | 0.5            | 3     | **0.3741** |
| Ours (JPEG + multi-scale) | max         | per-region-13 | 0.5            | 3     | **0.3806** |
| Ours (JPEG + multi-scale) | mean        | region-3      | 1.0            | 3     | 0.36657     |

---

## Repository Structure

```
pipeline/          Core inference modules (imported by all scripts)
scripts/           Extraction, ablation, and visualization scripts
results/           Result plotting scripts and Kaggle F1 data
README.md
```

---

## Setup

### Requirements

```bash
pip install torch torchvision timm
pip install numpy pandas pillow scikit-learn
pip install pacmap matplotlib          # for visualization scripts
pip install PyTurboJPEG                # optional: for 4:1:1 JPEG subsampling
```

### Data layout

Data and models can be found in the official Kaggle challenge website: [PlantCLEF 2026](https://www.kaggle.com/competitions/plantclef-2026/data)

```
data/
  test/images/                    # 2,105 test quadrat images
  train_singleplant/
    PlantCLEF2024_single_plant_training_metadata.csv

pretrained_models/
  vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/
    model_best.pth.tar
  class_mapping.txt
```

---

## Running the Pipeline

### Step 1 — Extract features and logits (GPU required)

This step runs the DINOv2 backbone and classification head over all 91 tiles
per image and caches the results to disk. It only needs to run once per
pre-processing configuration.

```bash
# Default: JPEG quality 85, 4:2:2 subsampling, all 6 scales
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/submission.csv

# No JPEG compression (keep in a separate cache)
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --no-jpeg --cache-dir cache_nojpeg \
    --output output/submission_nojpeg.csv
```

On HPC, use the provided sbatch scripts:

```bash
sbatch scripts/sbatch_extract.sh          # JPEG cache
sbatch scripts/sbatch_extract_nojpeg.sh   # no-JPEG cache
```

### Step 2 — Pre-compute data-driven priors (CPU, ~1 min)

Required only when using `--kmeans-k` with integer values or `prefix`.

```bash
# Paper 2's 3-cluster mapping and the 13-cluster per-region variant
python -m pipeline.prior --cache-dir cache --prefix --k 3 5 7 10

# Same for the no-JPEG cache
python -m pipeline.prior --cache-dir cache_nojpeg --prefix --k 3 5 7 10
```

### Step 3 — Run post-processing and generate submissions

Once features and logits are cached, all post-processing flags can be swept
cheaply (no GPU needed).

```bash
# Single run — full pipeline default
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/submission.csv

# Ablation sweep: two aggregation methods × two prior types × two top-K
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/sweep_{km}_{agg}_ps{ps}_topk{k}.csv \
    --aggregation max mean \
    --kmeans-k prefix \
    --prior-strength 1.0 0.5 \
    --top-k 3 9
```

### Key CLI flags

| Flag                  | Default             | Description                                                                                                                      |
| --------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--aggregation`       | `max`               | One or more of `max`, `mean`, `topk_mean`, `vote`                                                                                |
| `--kmeans-k`          | _(paper 2 default)_ | One or more of: integer k (data-driven K-Means), `prefix` (13-cluster per-region), or omit for Paper 2's fixed 3-cluster mapping |
| `--prior-strength`    | `0.5`               | Exponent on `P(y\|cluster)` before multiplying scores (1.0 = full prior)                                                         |
| `--top-k`             | `9`                 | Maximum species predictions per image                                                                                            |
| `--scales`            | `6 5 4 3 2 1`       | Tiling scales; 91 tiles total                                                                                                    |
| `--no-jpeg`           | —                   | Disable JPEG tile round-trip                                                                                                     |
| `--jpeg-quality`      | `85`                | JPEG quality (1–95)                                                                                                              |
| `--no-bayesian-prior` | —                   | Disable Bayesian prior reweighting                                                                                               |
| `--no-geo-filter`     | —                   | Disable geographic species filter                                                                                                |
| `--cache-dir`         | `cache/`            | Root for cached features and logits                                                                                              |
| `--output`            | —                   | Output CSV path; supports `{km}`, `{agg}`, `{ps}`, `{k}` template variables                                                      |

---

## Pipeline Overview

```
Test images
    │
    ▼
[A] JPEG tile compression          ← Paper 1
    │
    ▼
[B] Multi-scale tiling (91 tiles)  ← Paper 1
    │
    ▼
[C] DINOv2 feature extraction      ← both papers
    │  cached: cache/features/{stem}_s{scale}.npy  [S², 768]
    ▼
[D] Classification head            ← both papers
    │  cached: cache/logits/{stem}_s{scale}.npy    [S², 7806]
    ▼
[E] Tile aggregation (max / mean)  ← Paper 1
    │
    ▼
[F] Bayesian prior reweighting     ← Paper 2 (extended)
    │
    ▼
[G] Geographic species filter      ← Paper 2
    │
    ▼
[H] Top-K selection → CSV
```

See `pipeline/PIPELINE.md` for a detailed description of each stage including
all design decisions, implementation notes, and configuration flags.

---

## Visualization Scripts

```bash
# PaCMAP 2-D embedding of scale-1 CLS features, coloured by region / cluster
python scripts/visualize_features_pacmap.py --output output/features_pacmap.pdf

# Score fall-off: mean aggregated probability at ranks 1–10
python scripts/plot_score_distribution.py \
    --aggregation max mean --output output/score_distribution.png

# Multi-scale tiling grid overlaid on a sample image (one file per scale)
python scripts/visualize_tiling_scales.py --output output/tiling_scale{s}.png

# Show extracted tiles for one image
python scripts/visualize_tiles.py
```

---

## Module Reference

| Module                     | Responsibility                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `pipeline/config.py`       | `PipelineConfig` dataclass — all flags with defaults                                |
| `pipeline/compression.py`  | JPEG tile round-trip                                                                |
| `pipeline/tiling.py`       | Multi-scale resize, center-crop, tile extraction                                    |
| `pipeline/model.py`        | DINOv2 backbone + classification head; batched inference                            |
| `pipeline/features.py`     | Two-level disk cache; orchestrates tiling → compression → model                     |
| `pipeline/aggregation.py`  | `aggregate(tile_logits, method)` — max / mean / topk_mean / vote                    |
| `pipeline/prior.py`        | Cluster assignment; prior computation (paper 2, K-Means, per-region); `apply_prior` |
| `pipeline/geo_filter.py`   | Geographic mask from training metadata; `apply_geo_filter`                          |
| `pipeline/submission.py`   | Formats and writes the output CSV                                                   |
| `pipeline/run_pipeline.py` | CLI entry point; orchestrates all stages                                            |
