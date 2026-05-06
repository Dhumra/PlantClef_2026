"""
Figure: Sample images - train-test domain gap and within-class variation.

Layout:
    [ test image (quadrat)  |  NxN grid of training images (same species) ]

Uses manual axis positioning (fig.add_axes) so gaps are equal in physical
inches regardless of figure aspect ratio - no GridSpec wspace distortion.

Usage (from project root):
    python results/plot_sample_images.py              # 2x2 grid, PNG
    python results/plot_sample_images.py --pdf        # 2x2 grid, PDF
    python results/plot_sample_images.py --grid 3     # 3x3 grid (poster)
    python results/plot_sample_images.py --grid 3 --pdf --output figures/overview.pdf
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SHARE_DIR = Path(__file__).parent.parent / "share"

TEST_IMAGE = SHARE_DIR / "sample_testimg_2024-CEV3-20240602.jpg"

# All 9 available images for Centaurea melitensis L. (species 1355885).
# First 4 are used for the 2x2 grid; all 9 for the 3x3 grid.
TRAIN_IMAGES = [
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "00f86c7468a2335ab7c35c17485771486feb4477.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "0af691f92b4a2765a74a6e5223c6302755584b4e.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "0b77abbb02fdcd846cdf055eb6d58959739eaa02.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "0d458a7358b7453791930f8c6d15876a9a8091dd.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "00fd494dcfafce980afe89f47f2f3794a2245c5f.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "0b1a5c8eb323b12acd8392531b042e6168e5615c.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "0c718115dd64b51af0d4443ac648f88095819e41.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "0d647a3bef5ad767e5553283470cd3f2c4367724.jpg",
    SHARE_DIR
    / "sample_trainimg_1355885"
    / "1a28fe6b730b26c997f4c9dee4d8701ab6311ebb.jpg",
]


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def load_img(path: Path) -> np.ndarray:
    return np.array(center_crop_square(Image.open(path).convert("RGB")))


def plot(grid: int = 2, save_pdf: bool = False, output: str | None = None) -> None:
    assert grid in (2, 3), "--grid must be 2 or 3"
    plt.rcParams.update({"font.size": 9})

    n = grid  # nxn training grid
    n_train = n * n

    test_img = load_img(TEST_IMAGE)
    train_imgs = [load_img(p) for p in TRAIN_IMAGES[:n_train]]

    # Layout geometry (all in inches)
    FIG_W = 7.0
    MARGIN = 0.08  # left / right / bottom margin
    SEP = 0.12  # gap between test group and train group
    GAP = 0.04  # gap between individual train images (equal in both axes)
    TEXT_H = 0.40  # height reserved above image area for two lines of labels

    img_w_in = (FIG_W - 2 * MARGIN - SEP) / 2  # width of each group
    ti_w_in = (img_w_in - (n - 1) * GAP) / n  # individual train cell width
    img_h_in = n * ti_w_in + (n - 1) * GAP  # == img_w_in
    FIG_H = MARGIN + img_h_in + TEXT_H

    # Convert to figure fractions
    def fx(v):
        return v / FIG_W

    def fy(v):
        return v / FIG_H

    test_x0 = fx(MARGIN)
    test_y0 = fy(MARGIN)
    test_w = fx(img_w_in)
    test_h = fy(img_h_in)

    train_x0 = fx(MARGIN + img_w_in + SEP)
    ti_w = fx(ti_w_in)
    ti_h = fy(ti_w_in)  # square cells; fy(GAP) == fx(GAP) * (FIG_W/FIG_H)
    gap_x = fx(GAP)
    gap_y = fy(GAP)

    TOP_IMG = test_y0 + test_h  # top of image area in figure fracs

    # Label y-positions (va="bottom" means text sits above these y values)
    pad_y = fy(0.04)
    line2_h = 8 / 72 / FIG_H  # 8 pt in figure fracs
    line1_h = 9 / 72 / FIG_H  # 9 pt
    LINE2_Y = TOP_IMG + pad_y  # bottom of species name
    LINE1_Y = LINE2_Y + line2_h + pad_y  # bottom of bold group label

    cx_test = test_x0 + test_w / 2
    cx_train = train_x0 + test_w / 2  # same width as test group

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Test image
    ax = fig.add_axes([test_x0, test_y0, test_w, test_h])
    ax.imshow(test_img, aspect="auto", interpolation="antialiased")
    ax.axis("off")

    # Training images (row-major: top-left to top-right to ... to bottom-right)
    for idx, img in enumerate(train_imgs):
        row = idx // n
        col = idx % n
        x0 = train_x0 + col * (ti_w + gap_x)
        y0 = test_y0 + (n - 1 - row) * (ti_h + gap_y)  # top row first
        ax = fig.add_axes([x0, y0, ti_w, ti_h])
        ax.imshow(img, aspect="auto", interpolation="antialiased")
        ax.axis("off")

    # Labels
    test_filename = TEST_IMAGE.name.replace("sample_testimg_", "")

    fig.text(
        cx_test,
        LINE1_Y,
        "Test image",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(cx_test, LINE2_Y, test_filename, ha="center", va="bottom", fontsize=8)
    fig.text(
        cx_train,
        LINE1_Y,
        "Training images",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        cx_train,
        LINE2_Y,
        r"($\it{Centaurea\ melitensis}$ L.)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Thin vertical separator
    sep_x = fx(MARGIN + img_w_in + SEP / 2)
    fig.add_artist(
        plt.Line2D(
            [sep_x, sep_x],
            [test_y0, TOP_IMG],
            transform=fig.transFigure,
            color="#bbbbbb",
            linewidth=0.8,
            zorder=10,
        )
    )

    # Save
    if output is None:
        ext = "pdf" if save_pdf else "png"
        output = f"results/sample_images_{n}x{n}.{ext}"

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_kwargs = {"bbox_inches": "tight", "dpi": 300}
    fig.savefig(output, format="pdf" if save_pdf else "png", **save_kwargs)
    print(f"Saved: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid",
        type=int,
        default=2,
        choices=[2, 3],
        help="Training image grid size: 2 (2x2, paper) or 3 (3x3, poster)",
    )
    p.add_argument("--pdf", action="store_true", help="Save as PDF instead of PNG")
    p.add_argument(
        "--output",
        default=None,
        help="Output path (default: results/sample_images_NxN.{png|pdf})",
    )
    args = p.parse_args()
    plot(grid=args.grid, save_pdf=args.pdf, output=args.output)
