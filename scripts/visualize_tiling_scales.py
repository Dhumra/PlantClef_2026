"""
Draw multi-scale tiling grids on a test image.

For each scale S in [6, 5, 4, 3, 2, 1] the pipeline resizes the image to
(518*S)x(518*S) and cuts it into S*S non-overlapping 518x518 tiles.
This script renders the resulting SxS grid on the original image aspect ratio
so the split boundaries are visually clear.

Saves one file per scale, e.g. output/tiling_scale6.png, ..., output/tiling_scale1.png.

Usage:
    python scripts/visualize_tiling_scales.py
    python scripts/visualize_tiling_scales.py --image share/sample_testimg_2024-CEV3-20240602.jpg
    python scripts/visualize_tiling_scales.py --output output/tiling_scale{s}.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

SCALES = [6, 5, 4, 3, 2, 1]
LINE_COLOR = "red"
LINE_WIDTH = 20
DPI = 80


def save_grid(img_array, scale: int, out_path: Path):
    h, w = img_array.shape[:2]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / DPI, h / DPI)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img_array)
    for i in range(1, scale):
        ax.axvline(x=w * i / scale, color=LINE_COLOR, linewidth=LINE_WIDTH)
        ax.axhline(y=h * i / scale, color=LINE_COLOR, linewidth=LINE_WIDTH)

    save_kwargs = {}
    if out_path.suffix.lower() != ".pdf":
        save_kwargs["dpi"] = DPI
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    proj_root = Path(__file__).parent.parent
    default_img = str(proj_root / "share" / "sample_testimg_2024-CEV3-20240602.jpg")

    p = argparse.ArgumentParser()
    p.add_argument("--image", default=default_img)
    p.add_argument(
        "--output",
        default="output/tiling_scale{s}.png",
        help="Output path template; {s} is replaced by the scale value.",
    )
    p.add_argument("--scales", type=int, nargs="+", default=SCALES)
    a = p.parse_args()

    img = Image.open(a.image).convert("RGB")
    img_array = np.asarray(img)
    print(f"Image: {Path(a.image).name}  {img.width}x{img.height}")

    out_template = Path(a.output)
    out_template.parent.mkdir(parents=True, exist_ok=True)

    for scale in a.scales:
        out = Path(str(out_template).replace("{s}", str(scale)))
        save_grid(img_array, scale, out)


if __name__ == "__main__":
    main()
