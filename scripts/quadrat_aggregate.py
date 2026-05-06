"""
Post-process a submission CSV by aggregating predictions across multiple
images of the same base quadrat (Paper 2's strategy).

The same physical plot is photographed at multiple dates; combining predictions
across time captures more of the species roster (recall boost).

Usage:
    python scripts/quadrat_aggregate.py <input.csv> <output.csv> [--top-k 25]

Also reports the count of unique base quadrats vs total images, so you can
see how many test images share a quadrat.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

# Regex patterns from paper2_postprocessing/plantclef/classification/aggregation.py:21
PATTERNS = [
    (re.compile(r"^(CBN-.*?-.*?)-\d{8}$"), 1),  # CBN
    (re.compile(r"^(GUARDEN-.*?-.*?)-.*$"), 1),  # GUARDEN
    (re.compile(r"^(LISAH-.*?)-\d{8}$"), 1),  # LISAH
    (re.compile(r"^(OPTMix-\d+)-.*$"), 1),  # OPTMix
    (re.compile(r"^(RNNB-\d+-\d+)-\d{8}$"), 1),  # RNNB
    (re.compile(r"^(2024-CEV3)-\d{8}$"), 1),  # 2024-CEV3 (guess)
]


def base_quadrat_id(quadrat_id: str) -> str:
    for pat, idx in PATTERNS:
        m = pat.match(quadrat_id)
        if m:
            return m.group(idx)
    return quadrat_id  # fallback if no pattern matches


def load_submission(path: str):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            quadrat_id, species_str = row[0], row[1]
            # species_str is like "[1397475, 1741661, 1395190]"
            ids = [
                int(s.strip()) for s in species_str.strip("[]").split(",") if s.strip()
            ]
            rows.append((quadrat_id, ids))
    return rows


def write_submission(rows, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["quadrat_id", "species_ids"])
        for q, ids in rows:
            writer.writerow([q, "[" + ", ".join(str(s) for s in ids) + "]"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Take top-K species (by frequency across images of the same quadrat).",
    )
    p.add_argument(
        "--strategy",
        choices=["topk", "union"],
        default="topk",
        help="topk: top-K by frequency. union: all unique species across images.",
    )
    args = p.parse_args()

    rows = load_submission(args.input)
    print(f"Input: {len(rows)} image-level predictions")

    # Group by base quadrat
    groups = defaultdict(list)
    for quadrat_id, ids in rows:
        base = base_quadrat_id(quadrat_id)
        groups[base].append((quadrat_id, ids))

    print(f"Unique base quadrats: {len(groups)}")
    print(f"Avg images per quadrat: {len(rows) / len(groups):.2f}")

    # Distribution of images per quadrat
    sizes = [len(v) for v in groups.values()]
    size_counts = Counter(sizes)
    print(f"Images-per-quadrat distribution: {dict(sorted(size_counts.items()))}")

    if len(groups) == len(rows):
        print("\nAll quadrat IDs are unique - no aggregation possible.")
        print(
            "Either the regex patterns don't match this test set, or the test set has"
        )
        print("only one image per quadrat. Inspect a few quadrat IDs:")
        for q, _ in rows[:5]:
            print(f"  {q!r}  ->  base = {base_quadrat_id(q)!r}")
        return

    # Aggregate per quadrat
    new_rows = []
    for quadrat_id, ids in rows:
        base = base_quadrat_id(quadrat_id)
        all_preds = [s for (_, sids) in groups[base] for s in sids]

        if args.strategy == "union":
            # Sorted by frequency across the quadrat's images
            counts = Counter(all_preds)
            agg_ids = [s for s, _ in counts.most_common()]
        else:  # topk
            counts = Counter(all_preds)
            agg_ids = [s for s, _ in counts.most_common(args.top_k)]

        new_rows.append((quadrat_id, agg_ids))

    write_submission(new_rows, args.output)
    print(f"\nWritten {len(new_rows)} rows to {args.output}")
    print(f"Strategy: {args.strategy}, top_k: {args.top_k}")


if __name__ == "__main__":
    main()
