"""
Figure 2: Top-k sensitivity curves.

Usage (run from project root):
    python results/plot_topk_sensitivity.py           # saves PNG
    python results/plot_topk_sensitivity.py --pdf     # saves PDF (for LaTeX / poster)
    python results/plot_topk_sensitivity.py --pdf --output-dir figures
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_PATH = os.path.join(os.path.dirname(__file__), "public_f1.json")

# Color families: PaCMAP prior = blues, Region prior = oranges
PACMAP_DARK = "#1f6fbf"  # mean, strength=1.0
PACMAP_LIGHT = "#74b3e8"  # max, strength=0.5
REGION_DARK = "#d95f02"  # mean, strength=1.0
REGION_LIGHT = "#f5a44a"  # max, strength=0.5
# Baselines
BASE1_COLOR = "#2ca02c"  # Paper 1  (green)
BASE2_COLOR = "#9467bd"  # Paper 2  (purple)


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_curves(data: dict) -> list[dict]:
    jpeg = data["with_jpeg_recomp"]
    return [
        {
            "y": jpeg["prior_paper2"]["agg_mean_ps1p0"],
            "label": "PaCMAP prior, mean agg. (strength=1.0)",
            "color": PACMAP_LIGHT,
            "linestyle": "--",
            "marker": "o",
        },
        {
            "y": jpeg["prior_paper2"]["agg_max_ps0p5"],
            "label": "PaCMAP prior, max agg. (strength=0.5)",
            "color": PACMAP_DARK,
            "linestyle": "-",
            "marker": "s",
        },
        {
            "y": jpeg["prior_prefix"]["agg_mean_ps1p0"],
            "label": "Region prior, mean agg. (strength=1.0)",
            "color": REGION_LIGHT,
            "linestyle": "--",
            "marker": "o",
        },
        {
            "y": jpeg["prior_prefix"]["agg_max_ps0p5"],
            "label": "Region prior, max agg. (strength=0.5)",
            "color": REGION_DARK,
            "linestyle": "-",
            "marker": "s",
        },
    ]


def plot(save_pdf: bool = False, output_dir: str = "results") -> None:
    data = load_data(DATA_PATH)
    top_k = data["with_jpeg_recomp"]["top_k_vals"]
    curves = build_curves(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for c in curves:
        ax.plot(
            top_k,
            c["y"],
            label=c["label"],
            color=c["color"],
            linestyle=c["linestyle"],
            marker=c["marker"],
            linewidth=1.8,
            markersize=5,
            zorder=3,
        )

    # Baseline horizontal lines
    b1 = data["baselines"]["paper1"]["best_public"]
    b2 = data["baselines"]["paper2"]["best_public"]
    ax.axhline(
        b1,
        color=BASE1_COLOR,
        linestyle=":",
        linewidth=1.6,
        label=f"Paper 1 baseline ({b1:.4f})",
        zorder=2,
    )
    ax.axhline(
        b2,
        color=BASE2_COLOR,
        linestyle=":",
        linewidth=1.6,
        label=f"Paper 2 baseline ({b2:.4f})",
        zorder=2,
    )

    ax.set_xlabel("Top-k Predictions per Image", fontsize=11)
    ax.set_ylabel("F1 Score (public leaderboard)", fontsize=11)
    ax.set_title(
        "Effect of Prior Source and Aggregation on Top-k F1", fontsize=12, pad=10
    )
    ax.set_xticks(top_k)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    if save_pdf:
        out = os.path.join(output_dir, "topk_sensitivity.pdf")
        plt.savefig(out, format="pdf", bbox_inches="tight")
    else:
        out = os.path.join(output_dir, "topk_sensitivity.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")

    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true", help="Save as PDF instead of PNG")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write the figure (default: results/)",
    )
    args = parser.parse_args()
    plot(save_pdf=args.pdf, output_dir=args.output_dir)
