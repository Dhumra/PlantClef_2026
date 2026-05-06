"""
Submission file formatting.

Output CSV format matches PlantCLEF 2025/2026:
  quadrat_id,species_ids
  "CBN-PdlC-A6-20130807","[1397475, 1741661, 1395190]"
"""

import csv
from pathlib import Path
from typing import Dict, List


def load_class_names(class_mapping_file: str) -> List[int]:
    """
    Return a list where index i is the integer species_id for class i.
    The file has one species_id per line (matching the model's output order).
    """
    with open(class_mapping_file) as f:
        return [int(line.strip()) for line in f if line.strip()]


def format_species_ids(species_ids: List[int]) -> str:
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def write_submission(
    results: Dict[str, List[int]],  # image_stem -> list of species_ids
    output_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["quadrat_id", "species_ids"])
        for stem, ids in sorted(results.items()):
            writer.writerow([stem, format_species_ids(ids)])
    print(f"Submission written to {output_path}  ({len(results)} rows)")
