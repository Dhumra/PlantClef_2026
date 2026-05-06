from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PipelineConfig:
    # I/O
    images_dir: str = str(PROJECT_ROOT / "data" / "test" / "images")
    output_dir: str = str(PROJECT_ROOT / "output")
    # Use different cache_dir values for different pre-processing configs,
    # e.g. "cache_jpeg85" vs "cache_nojpeg".
    cache_dir: str = str(PROJECT_ROOT / "cache")

    # Model
    model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    model_path: str = str(
        PROJECT_ROOT
        / "pretrained_models"
        / "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
        / "model_best.pth.tar"
    )
    class_mapping_file: str = str(
        PROJECT_ROOT / "pretrained_models" / "class_mapping.txt"
    )
    num_classes: int = 7806
    tile_size: int = 518
    # Number of tiles to forward in one GPU batch across all scales.
    batch_size: int = 32

    # Pre-processing (paper 1)
    # Apply a JPEG round-trip on every 518x518 tile before inference to align
    # test-domain compression artifacts with training data. Necessary even for
    # .jpg test images because their original encoding settings are unknown.
    use_jpeg_compression: bool = True
    jpeg_quality: int = 85
    # Chroma subsampling for the JPEG round-trip.
    # Paper 1 best: "4:2:2" at quality 85 (Pillow native, default here)
    #              "4:1:1" at quality 94 (requires: pip install PyTurboJPEG)
    # Also accepted: "4:4:4" (no subsampling), "4:2:0" (Pillow native).
    # NOTE: Pillow subsampling=2 produces 4:2:0, not 4:1:1.
    jpeg_subsampling: str = "4:2:2"

    # Tiling scales. Each entry S produces S*S non-overlapping 518x518 tiles.
    # [6, 5, 4, 3, 2, 1] = 91 tiles total (paper 1 full multi-scale).
    # [4] = 16 tiles (paper 2 single-scale baseline).
    # [1] = 1 tile (no tiling baseline).
    scales: List[int] = field(default_factory=lambda: [6, 5, 4, 3, 2, 1])

    # Aggregation
    # How to combine probabilities across all tiles of an image.
    #   "max"       - take the maximum per species (paper 1 default)
    #   "mean"      - average across all tiles
    #   "topk_mean" - average of the top-k tile values per species (use topk_mean_k)
    #   "vote"      - each tile votes for its top-vote_k species; score = fraction of tiles
    aggregation: Literal["max", "mean", "topk_mean", "vote"] = "max"
    topk_mean_k: int = 5  # only used when aggregation == "topk_mean"
    vote_k: int = 5  # only used when aggregation == "vote"

    # Post-processing (paper 2)
    # Bayesian prior: multiply image-level probabilities by P(y | cluster).
    # If prior_data_path is None and use_bayesian_prior is True, the prior is
    # computed on-the-fly from scale-1 predictions cached in cache_dir.
    use_bayesian_prior: bool = True
    # Exponent applied to P(y | cluster) before multiplying image scores.
    # 1.0 keeps the original prior; smaller values make it softer.
    prior_strength: float = 0.5
    # Path to a .npy file of shape [k, num_classes] with cluster prior vectors.
    # Leave None to compute from the scale-1 tile cache.
    prior_data_path: Optional[str] = None
    # Path to a {stem: cluster_id} JSON written by python -m pipeline.prior.
    # If set, overrides the filename-prefix cluster assignment from paper 2.
    # Must be paired with a matching prior_data_path (same k, same run).
    prior_assignments_path: Optional[str] = None

    # Geographical filter: zero out species with no known observation in the
    # target region. The valid species set is derived from training metadata.
    use_geo_filter: bool = True
    training_metadata_csv: str = str(
        PROJECT_ROOT
        / "data"
        / "train_singleplant"
        / "PlantCLEF2024_single_plant_training_metadata.csv"
    )
    # Reference point for the test set: Southern France.
    geo_ref_lat: float = 44.0
    geo_ref_lon: float = 4.0
    # Approximate bounding boxes of target countries [lat_min, lat_max, lon_min, lon_max].
    geo_country_boxes: List[List[float]] = field(
        default_factory=lambda: [
            [41.3, 51.1, -5.2, 9.6],  # France
            [35.9, 43.8, -9.3, 4.3],  # Spain
            [35.5, 47.1, 6.6, 18.5],  # Italy
            [45.8, 47.8, 5.9, 10.5],  # Switzerland
        ]
    )

    # Submission
    top_k: int = 9
    min_score: float = 0.0
