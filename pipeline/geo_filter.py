"""
Geographical species filter (paper 2 post-processing).

Method:
  1. Load training metadata; columns used: species_id, latitude, longitude.
  2. For each species, find the observation nearest to the reference point
     (default: 44N 4E, Southern France) using squared Euclidean distance.
  3. Keep species whose nearest observation falls inside a target country
     bounding box (France / Spain / Italy / Switzerland).

Result: a boolean mask of shape [num_classes], cached to disk.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _in_any_box(lat: float, lon: float, boxes: List[List[float]]) -> bool:
    """Return True if (lat, lon) falls inside at least one bounding box.
    Each box is [lat_min, lat_max, lon_min, lon_max].
    """
    for lat_min, lat_max, lon_min, lon_max in boxes:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


def build_geo_mask(
    metadata_csv: str,
    num_classes: int,
    ref_lat: float = 44.0,
    ref_lon: float = 4.0,
    country_boxes: List[List[float]] = None,
) -> np.ndarray:
    """
    Compute a boolean mask of shape [num_classes] where True means the species
    has at least one training observation near the target region.

    Species absent from training metadata are kept (mask=True).
    """
    if country_boxes is None:
        country_boxes = [
            [41.3, 51.1, -5.2,  9.6],   # France
            [35.9, 43.8, -9.3,  4.3],   # Spain
            [35.5, 47.1,  6.6, 18.5],   # Italy
            [45.8, 47.8,  5.9, 10.5],   # Switzerland
        ]

    df = pd.read_csv(metadata_csv, sep=";", usecols=["species_id", "latitude", "longitude"])
    df = df.dropna(subset=["latitude", "longitude"])
    df["species_id"] = df["species_id"].astype(int)

    # Squared distance from reference point (no sqrt needed - only need argmin)
    df["dist2"] = (df["latitude"] - ref_lat) ** 2 + (df["longitude"] - ref_lon) ** 2

    # For each species, pick the observation closest to the reference point
    nearest = df.loc[df.groupby("species_id")["dist2"].idxmin()]

    # Check which nearest observations fall inside a target country
    in_region = nearest.apply(
        lambda row: _in_any_box(row["latitude"], row["longitude"], country_boxes),
        axis=1,
    )
    valid_ids = set(nearest.loc[in_region, "species_id"].tolist())

    # Build the mask; default True for species absent from metadata
    mask = np.ones(num_classes, dtype=bool)
    all_ids_in_metadata = set(nearest["species_id"].tolist())
    for sid in all_ids_in_metadata:
        if sid < num_classes:
            mask[sid] = sid in valid_ids

    return mask


def load_or_build_geo_mask(
    metadata_csv: str,
    num_classes: int,
    cache_dir: str,
    ref_lat: float = 44.0,
    ref_lon: float = 4.0,
    country_boxes: List[List[float]] = None,
) -> np.ndarray:
    """Load the geo mask from cache, or compute and cache it."""
    cache_path = Path(cache_dir) / "geo_mask.npy"
    if cache_path.exists():
        return np.load(cache_path)

    print("Building geo mask from training metadata (runs once)...")
    mask = build_geo_mask(metadata_csv, num_classes, ref_lat, ref_lon, country_boxes)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mask)
    n_valid = int(mask.sum())
    print(f"Geo mask built: {n_valid}/{num_classes} species retained.")
    return mask


def apply_geo_filter(probs: np.ndarray, geo_mask: np.ndarray) -> np.ndarray:
    """Zero out species not present in the target region."""
    result = probs.copy()
    result[~geo_mask] = 0.0
    return result
