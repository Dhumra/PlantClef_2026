# PlantCLEF 2026 — Combined Inference Pipeline

This document describes the unified inference pipeline that combines the two
strongest submissions from PlantCLEF 2025. Each component is attributed to
the paper it was drawn from.

- **Paper 1** — Espitalier et al., _"Pre-processing for multi-scale tiling"_,
  CLEF 2025 Working Notes ([paper 238](https://ceur-ws.org/Vol-4038/paper_238.pdf)).
  Source: `paper1_preprocessing/` (Rust).
- **Paper 2** — Gustineli et al., _"Post-processing with Bayesian priors and
  geographic filtering"_, CLEF 2025 Working Notes ([paper 242](https://ceur-ws.org/Vol-4038/paper_242.pdf)).
  Source: `paper2_postprocessing/` (Python / PySpark).

---

## 1. Overview

The pipeline is a sequential, modular chain. Every stage is independently
switchable for ablation studies; intermediate results are cached to disk so
that expensive GPU computation is never repeated unnecessarily.

```
Test images
    │
    ▼
[A] JPEG tile compression          ← Paper 1
    │
    ▼
[B] Multi-scale tiling             ← Paper 1
    │
    ▼
[C] DINOv2 feature extraction      ← both papers (same model)
    │  (cached: features/*.npy)
    ▼
[D] Classification head            ← both papers (same head)
    │  (cached: logits/*.npy)
    ▼
[E] Tile aggregation               ← Paper 1 (max-pool); method selectable
    │
    ▼
[F] Bayesian prior reweighting     ← Paper 2
    │
    ▼
[G] Geographical species filter    ← Paper 2
    │
    ▼
[H] Top-K selection → submission CSV
```

---

## 2. Shared Model

Both papers use the **same pre-trained model**:

| Property          | Value                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------- |
| Architecture      | ViT-Base/14 with 4 register tokens (DINOv2-reg4)                                       |
| Pre-training      | LVD-142M (DINOv2 self-supervised)                                                      |
| Fine-tuning       | PlantCLEF 2024 single-plant training set (end-to-end)                                  |
| Output classes    | 7,806 plant species                                                                    |
| Feature dimension | 768 (CLS-token embedding)                                                              |
| Checkpoint        | `vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar` |

Inference is split into two stages to enable selective caching:

1. **Backbone** — `forward_features(tile)` → CLS-token embedding, shape `[768]`.
2. **Head** — `head(embedding)` → raw logits, shape `[7806]`.

---

## 3. Stage A — JPEG Tile Compression

**Source: Paper 1**

### Motivation

The PlantCLEF 2024 training images were taken by standard digital cameras and
stored as JPEG files. JPEG encoding introduces characteristic block-boundary
artifacts at 8-pixel intervals. The test images may have been re-compressed
at different settings or with different software, creating a domain shift.
Paper 1 discovered that applying a JPEG round-trip to every test tile before
inference reduces this gap and significantly improves accuracy.

### Why the round-trip is necessary even for .jpg test images

The test set images are delivered as `.jpg` files. One might assume the
round-trip is therefore redundant. It is not. When Pillow opens a `.jpg`, it
decodes the file to raw RGB pixels; at that point the original encoder's
quality setting and subsampling scheme are gone — their only trace is already
baked into the pixel values. The purpose of the round-trip is to **replace**
those original artifacts with a _standardised_ set (fixed quality, fixed
subsampling) that matches the training data distribution. A test image
encoded at quality 95 with 4:4:4 subsampling would give the model very
different pixels than the training images without the round-trip.

### Implementation

After extracting a 518×518 tile (see Stage B), the tile is encoded to JPEG
with specified quality and chroma subsampling, then decoded back. The
round-trip is done at the **tile level**, not the full-image level, because
JPEG quantisation artifacts occur on 8×8-pixel MCU blocks; compressing before
tiling would displace those block boundaries relative to each 518×518 tile.

```
tile (PIL Image, 518×518)
    → encode to JPEG (quality Q, subsampling S)
    → decode back to PIL Image
    → pass to backbone
```

### Subsampling

Paper 1 reports two equally performing configurations:

| Config      | Quality | Subsampling | Description                      |
| ----------- | ------- | ----------- | -------------------------------- |
| A (default) | 85      | **4:2:2**   | Cb/Cr at ½ horizontal resolution |
| B           | 94      | **4:1:1**   | Cb/Cr at ¼ horizontal resolution |

**Pillow subsampling support** (verified with Pillow 11.3 + libjpeg):

| Pillow `subsampling=` | Mode produced                                          |
| --------------------- | ------------------------------------------------------ |
| `0`                   | 4:4:4 (no chroma subsampling)                          |
| `1`                   | **4:2:2** ✓                                            |
| `2`                   | **4:2:0** — half horizontal _and_ half vertical chroma |
| `-1` (auto)           | 4:2:0 regardless of quality                            |

> **Important:** Pillow's `subsampling=2` produces **4:2:0**, not 4:1:1.
> The two modes have the same number of chroma samples (25 % of luma) but
> arranged differently: 4:2:0 subsamples both horizontally and vertically,
> while 4:1:1 subsamples only horizontally (by a factor of 4).

To use 4:1:1, pass `--jpeg-subsampling 4:1:1`. This routes through
**PyTurboJPEG** (a thin wrapper around libjpeg-turbo with full sampling-factor
control). Install it with `pip install PyTurboJPEG` before using this option.

**Configuration flags:**

| Flag                   | Default   | Effect                                                            |
| ---------------------- | --------- | ----------------------------------------------------------------- |
| `use_jpeg_compression` | `True`    | Enable/disable the round-trip                                     |
| `jpeg_quality`         | `85`      | JPEG quality factor (1–95)                                        |
| `jpeg_subsampling`     | `"4:2:2"` | Chroma subsampling: `"4:4:4"`, `"4:2:2"`, `"4:2:0"`, or `"4:1:1"` |

> **Paper 1 implementation note:** The Rust code uses ImageMagick (`magick_rust`)
> for the JPEG round-trip via `wand.write_image_blob("jpeg")`. ImageMagick's
> default subsampling at quality 85 is 4:2:2, consistent with this pipeline's
> default. The Python implementation uses Pillow (libjpeg) for 4:2:2 / 4:2:0
> and PyTurboJPEG (libjpeg-turbo) for 4:1:1.

---

## 4. Stage B — Multi-Scale Tiling

**Source: Paper 1**

### Motivation

A single field plot photograph contains dozens of plant species at varying
depths and scales. A global 518×518 resize loses fine-grained texture
discriminating nearby species. Paper 1's central contribution is to process
each image at multiple scales simultaneously, running the classifier on every
tile and aggregating the evidence.

### Geometry

For scale `S`, the following steps prepare the input image:

1. **Resize** — The image is Lanczos-resized so that its **short side** equals
   `S × 518` pixels. The long side is scaled proportionally.
2. **Center crop** — The resized image is cropped to exactly
   `(S × 518) × (S × 518)` pixels, discarding equal margins from both sides of
   the long dimension.
3. **Tile extraction** — The square is divided into `S²` non-overlapping
   tiles of 518×518 pixels, traversed in row-major order.

| Scale S   | Image prepared | Tiles extracted |
| --------- | -------------- | --------------- |
| 6         | 3108 × 3108 px | 36 tiles        |
| 5         | 2590 × 2590 px | 25 tiles        |
| 4         | 2072 × 2072 px | 16 tiles        |
| 3         | 1554 × 1554 px | 9 tiles         |
| 2         | 1036 × 1036 px | 4 tiles         |
| 1         | 518 × 518 px   | 1 tile          |
| **Total** |                | **91 tiles**    |

Scale 1 acts as a whole-image baseline: the full image is simply resized and
center-cropped to 518×518 — no subdivision.

**Configuration flag:**

| Flag     | Default              | Effect                                   |
| -------- | -------------------- | ---------------------------------------- |
| `scales` | `[6, 5, 4, 3, 2, 1]` | Which scales to run; any subset is valid |

Common ablation configurations:

| `scales`             | Description                      |
| -------------------- | -------------------------------- |
| `[6, 5, 4, 3, 2, 1]` | Full multi-scale (Paper 1 best)  |
| `[4]`                | Single-scale 4×4 (Paper 2 style) |
| `[1]`                | No tiling, whole-image baseline  |

---

## 5. Stage C — Feature Extraction (Cached)

**Source: Both papers (identical model)**

The DINOv2 backbone is applied to every tile to produce a 768-dimensional
CLS-token embedding. Because this is the most expensive step (full ViT-B
forward pass), results are saved to disk immediately after processing each
image at each scale.

**Cache layout:**

```
{cache_dir}/
  features/
    {image_stem}_s6.npy   # shape [36, 768], float32
    {image_stem}_s5.npy   # shape [25, 768], float32
    ...
    {image_stem}_s1.npy   # shape [ 1, 768], float32
```

On subsequent runs, the cache is checked first; if a file exists the backbone
is skipped entirely. Use a different `cache_dir` when changing Stage A
settings (JPEG quality), since they affect the pixel values seen by the
backbone.

---

## 6. Stage D — Classification Head (Cached)

**Source: Both papers (identical head)**

The linear classification head maps each 768-dim embedding to 7,806 raw
logits. The head is cheap relative to the backbone (one matrix multiply), but
its output is also cached so that post-processing ablations (stages F and G)
can be run instantly without touching the GPU.

**Cache layout:**

```
{cache_dir}/
  logits/
    {image_stem}_s6.npy   # shape [36, 7806], float32
    ...
```

---

## 7. Stage E — Tile Aggregation

**Source: Paper 1 (max-pool); method extended in this pipeline**

After applying the classification head to all tiles, the 91 logit vectors for
each image are combined into a single image-level probability vector.

### Softmax

Raw logits from all tiles are first converted to probabilities via softmax,
applied independently per tile:

```
p_t[j] = exp(logit_t[j]) / Σ_k exp(logit_t[k])
```

### Aggregation methods

| Method      | Formula                | Description                                                                                              |
| ----------- | ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `max`       | `max_t p_t[j]`         | Maximum over tiles per species. Selects the single most confident tile as evidence. **Paper 1 default.** |
| `mean`      | `mean_t p_t[j]`        | Arithmetic mean over all tiles. More robust to outlier tiles; may dilute sparse species.                 |
| `topk_mean` | `mean of top-k p_t[j]` | Average of the k highest tile probabilities per species. Interpolates between max and mean.              |

**Configuration flags:**

| Flag          | Default | Effect             |
| ------------- | ------- | ------------------ |
| `aggregation` | `"max"` | Aggregation method |
| `topk_mean_k` | `5`     | k for `topk_mean`  |

### Note on prior application order

The Bayesian prior (Stage F) is a vector that is constant across all tiles of
one image, so it commutes with all three aggregation methods:

```
agg_method( p_t · prior ) = agg_method( p_t ) · prior
```

We therefore apply the prior **after** aggregation for efficiency, performing
one multiplication instead of 91.

---

## 8. Stage F — Bayesian Prior Reweighting

**Source: Paper 2**

### Motivation

The PlantCLEF 2025 test set images are drawn from specific geographic sites in
Southern France and adjacent regions. Each site has a characteristic plant
community: coastal Mediterranean sites share species that rarely appear in
highland Pyrenean sites. Paper 2 exploits this structure by computing a
per-cluster empirical prior `P(y | cluster)` and using it to up-weight species
that are statistically common in the cluster to which the query image belongs.

### Cluster assignment

Paper 2 applied PaCMAP dimensionality reduction followed by KMeans (k=3) to
the DINOv2 CLS-token embeddings of all 2,105 test images, then identified the
dominant cluster for each geographic site prefix. The resulting mapping is
fixed and requires no embedding step at inference:

| Cluster | Region prefixes                                                         |
| ------- | ----------------------------------------------------------------------- |
| 0       | GUARDEN-CBNMed, RNNB, LISAH-\*, OPTMix, CBN-Pyr, GUARDEN-AMB, 2024-CEV3 |
| 1       | CBN-PdlC, CBN-can                                                       |
| 2       | CBN-Pla                                                                 |

The cluster is read directly from the image filename (e.g.,
`CBN-Pla-B2-20190723` → cluster 2). Any unrecognised prefix defaults to
cluster 0.

### Prior computation

`P(y | cluster c)` is the **element-wise average of softmax(logits)** over all
test images in cluster `c`, computed from scale-1 (whole-image) predictions:

```
P(y | c) = (1 / |C_c|) · Σ_{i ∈ C_c} softmax( logits_i )
```

Because scale-1 logits are cached as part of the multi-scale run, the prior is
computed automatically at the end of Stage D if no pre-computed file is
provided. The result is saved to `{cache_dir}/priors.npy` for reuse.

Alternatively, a pre-computed prior can be supplied via `--prior-data-path`.
The file format is a `.npy` array of shape `[NUM_CLUSTERS, num_classes]` where
row index equals cluster id (row 0 = cluster 0, row 1 = cluster 1, …).
This is a flat numpy array rather than a structured format because the number
of clusters is fixed (3) and all rows have the same length (7,806), making a
2-D float32 array the simplest and most consistent representation.

### Application

Image-level probabilities `p` (shape `[7806]`) from Stage E are multiplied
element-wise by the cluster prior:

```
p_weighted[j] = p[j] · P(y_j | cluster(image))
```

The result is not re-normalised; only the relative ranking matters for Top-K
selection.

**Configuration flags:**

| Flag                 | Default | Effect                                                           |
| -------------------- | ------- | ---------------------------------------------------------------- |
| `use_bayesian_prior` | `True`  | Enable/disable prior reweighting                                 |
| `prior_data_path`    | `None`  | Path to pre-computed priors file; if `None`, computed from cache |

---

## 9. Stage G — Geographical Species Filter

**Source: Paper 2**

### Motivation

The DINOv2 model was trained on a global plant dataset (PlantCLEF 2024) and
can predict species from any continent. For a test set known to cover Southern
Europe, predictions of, say, _Eucalyptus regnans_ (native to Tasmania) or
_Opuntia_ spp. (native to the Americas) should be suppressed. Paper 2 applies
a hard filter that removes any species with no known occurrence in the target
region from the final prediction list.

### Filter construction

The filter is derived once from the PlantCLEF 2024 single-plant training
metadata (`PlantCLEF2024_single_plant_training_metadata.csv`, semicolon
delimited). The relevant columns are `species_id`, `latitude`, `longitude`.

**Algorithm:**

1. For each species, collect all training observations that have valid
   latitude/longitude coordinates.
2. Compute the squared Euclidean distance from each observation to the
   **reference point** (default: 44°N, 4°E — Southern France).
3. Select the **nearest observation** per species (minimum distance).
4. Check whether that nearest observation falls within the bounding box of any
   of the four target countries:

| Country     | Latitude range    | Longitude range  |
| ----------- | ----------------- | ---------------- |
| France      | 41.3° N – 51.1° N | 5.2° W – 9.6° E  |
| Spain       | 35.9° N – 43.8° N | 9.3° W – 4.3° E  |
| Italy       | 35.5° N – 47.1° N | 6.6° E – 18.5° E |
| Switzerland | 45.8° N – 47.8° N | 5.9° E – 10.5° E |

5. Species whose nearest observation does **not** fall in any of these boxes
   are excluded (probability set to 0). Species absent from the training
   metadata are kept (conservative assumption).

The result is a boolean mask of shape `[7806]` cached to
`{cache_dir}/geo_mask.npy`.

> **Design note:** Paper 2's actual code uses pre-fetched GBIF country-count
> JSON files rather than training metadata coordinates. The approach
> implemented here — distance-from-reference-point using training coordinates
> — matches the description in the paper text and avoids a dependency on the
> GBIF API. The country set (France, Spain, Italy, Switzerland) is Paper 2's
> choice; the PlantCLEF competition description mentions _"Pyrenean and
> Mediterranean"_ regions without specifying exact countries.

**Configuration flags:**

| Flag                          | Default        | Effect                                         |
| ----------------------------- | -------------- | ---------------------------------------------- |
| `use_geo_filter`              | `True`         | Enable/disable the filter                      |
| `training_metadata_csv`       | (see config)   | Path to the metadata CSV                       |
| `geo_ref_lat` / `geo_ref_lon` | `44.0` / `4.0` | Reference point                                |
| `geo_country_boxes`           | (see config)   | List of `[lat_min, lat_max, lon_min, lon_max]` |

---

## 10. Stage H — Top-K Selection and Submission Formatting

**Source: Both papers (identical format)**

After all post-processing, the final image-level probability vector is sorted
in descending order. Species whose probability exceeds `min_score` are
selected, up to a maximum of `top_k`.

The submission CSV follows the PlantCLEF 2025/2026 format:

```csv
quadrat_id,species_ids
"CBN-PdlC-A6-20130807","[1397475, 1741661, 1395190]"
"CBN-Pla-B2-20190723","[1395807, 1397463, 1741880]"
```

**Configuration flags:**

| Flag        | Default | Effect                              |
| ----------- | ------- | ----------------------------------- |
| `top_k`     | `15`    | Maximum number of species per image |
| `min_score` | `0.01`  | Minimum probability threshold       |

---

## 11. Caching Strategy

All slow operations write their results to `{cache_dir}/` immediately after
completion, keyed by image stem and scale. On re-runs, each stage checks for
its output file before computing.

```
{cache_dir}/
  features/          Stage C — backbone embeddings [S², 768]
  logits/            Stage D — classification logits [S², 7806]
  geo_mask.npy       Stage G — valid-species boolean mask [7806]
  priors.parquet     Stage F — cluster priors {cluster_id, prior_probabilities}
```

**Important:** `cache_dir` encodes the pre-processing configuration implicitly.
If you change `use_jpeg_compression` or `jpeg_quality`, point to a **different**
`cache_dir` so the new run does not load stale features computed from different
pixel values.

Recommended naming convention:

```bash
--cache-dir cache_jpeg85     # JPEG quality 85 (default)
--cache-dir cache_jpeg70     # JPEG quality 70
--cache-dir cache_nojpeg     # no JPEG compression
```

---

## 12. Ablation Configurations

The table below lists suggested configurations for systematic ablation studies.

| Run name        | `scales`        | `use_jpeg` | `aggregation` | `use_prior` | `use_geo` | Notes                       |
| --------------- | --------------- | ---------- | ------------- | ----------- | --------- | --------------------------- |
| `baseline_1x1`  | `[1]`           | ✗          | `max`         | ✗           | ✗         | Single-image, no extras     |
| `paper2_style`  | `[4]`           | ✗          | `mean`        | ✓           | ✓         | Reproduces Paper 2 approach |
| `paper1_tiling` | `[6,5,4,3,2,1]` | ✓          | `max`         | ✗           | ✗         | Paper 1 contribution only   |
| `combined_full` | `[6,5,4,3,2,1]` | ✓          | `max`         | ✓           | ✓         | Full pipeline (default)     |
| `no_jpeg`       | `[6,5,4,3,2,1]` | ✗          | `max`         | ✓           | ✓         | Ablate JPEG compression     |
| `no_prior`      | `[6,5,4,3,2,1]` | ✓          | `max`         | ✗           | ✓         | Ablate Bayesian prior       |
| `no_geo`        | `[6,5,4,3,2,1]` | ✓          | `max`         | ✓           | ✗         | Ablate geo filter           |
| `mean_agg`      | `[6,5,4,3,2,1]` | ✓          | `mean`        | ✓           | ✓         | Mean instead of max-pool    |
| `topk5_agg`     | `[6,5,4,3,2,1]` | ✓          | `topk_mean`   | ✓           | ✓         | Top-5 mean aggregation      |

Example commands:

```bash
# Full pipeline
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/combined_full.csv

# Paper 1 tiling only (no post-processing)
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/paper1_tiling.csv \
    --no-bayesian-prior --no-geo-filter

# Paper 2 style (4×4 tiling, mean aggregation, both post-processing stages)
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/paper2_style.csv \
    --scales 4 --aggregation mean --no-jpeg

# Ablate JPEG compression (use a separate cache to avoid mixing features)
python -m pipeline.run_pipeline \
    --images-dir data/test/images \
    --output output/no_jpeg.csv \
    --no-jpeg --cache-dir cache_nojpeg
```

---

## 13. Pipeline Module Reference

| Module            | Responsibility                                                               |
| ----------------- | ---------------------------------------------------------------------------- |
| `config.py`       | `PipelineConfig` dataclass — single source of truth for all flags            |
| `compression.py`  | JPEG round-trip (`jpeg_compress`)                                            |
| `tiling.py`       | `extract_tiles(img, scale)`, `resize_and_center_crop`                        |
| `model.py`        | `DINOv2Classifier`, `extract_features_batched`, `features_to_logits_batched` |
| `features.py`     | Two-level disk cache; calls `tiling`, `compression`, `model`                 |
| `aggregation.py`  | `aggregate(tile_logits, method)` — max / mean / topk_mean                    |
| `prior.py`        | Cluster assignment from filename; prior computation and application          |
| `geo_filter.py`   | Geo mask construction from training metadata; `apply_geo_filter`             |
| `submission.py`   | `write_submission` — formats and writes the output CSV                       |
| `run_pipeline.py` | CLI entry point; orchestrates all stages                                     |
