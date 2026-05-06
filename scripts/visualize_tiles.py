"""
Visualize the actual tiles fed into the ViT model - diagnostic for confirming
the tiling pipeline is doing what we think it is.

For a given image, renders two grids side-by-side:
  - Paper 1 / our pipeline: short-side resize -> center-crop square -> S*S tiles
  - Paper 2 reference:      raw image -> grid_size*grid_size rectangular tiles -> squash to 518x518

Also shows the JPEG round-trip applied per tile (4:2:2 q85) so you can compare
to the un-compressed version.

Usage:
    python scripts/visualize_tiles.py <image_path> --scale 4
    python scripts/visualize_tiles.py data/test/images/CBN-PdlC-A6-20130807.jpg \\
        --scale 4 --output tile_check.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.compression import jpeg_compress
from pipeline.tiling import extract_tiles


# Paper 2's tile splitter (mirror of paper2_postprocessing/.../workflow.py)
def paper2_split_into_grid(image: Image.Image, grid_size: int):
    w, h = image.size
    grid_w, grid_h = w // grid_size, h // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            left, upper = i * grid_w, j * grid_h
            tile = image.crop((left, upper, left + grid_w, upper + grid_h))
            tile = tile.resize((518, 518), Image.BICUBIC)   # timm transform default
            tiles.append(tile)
    return tiles


def paste_grid(tiles, grid_size, cell_size=200, gap=4, label=None):
    """Render tiles as a labelled grid."""
    n = grid_size
    W = n * cell_size + (n + 1) * gap
    H = n * cell_size + (n + 1) * gap + (40 if label else 0)
    canvas = Image.new("RGB", (W, H), (240, 240, 240))
    if label:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 10), label, fill=(0, 0, 0))
    for i in range(n):
        for j in range(n):
            tile = tiles[i * n + j].resize((cell_size, cell_size), Image.BICUBIC)
            x = gap + j * (cell_size + gap)
            y = (40 if label else 0) + gap + i * (cell_size + gap)
            canvas.paste(tile, (x, y))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to a test image.")
    p.add_argument("--scale", type=int, default=4,
                   help="Tiling scale (S*S tiles).")
    p.add_argument("--output", default="tile_visualization.png")
    p.add_argument("--with-jpeg", action="store_true",
                   help="Also render JPEG-round-tripped tiles (4:2:2 q85).")
    args = p.parse_args()

    img = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image}")
    print(f"Original size: {img.size}  (W x H)")

    # Paper 1 / our pipeline
    p1_tiles = extract_tiles(img, args.scale, tile_size=518)
    print(f"Paper 1 tiles: {len(p1_tiles)} x 518x518 (after resize+center-crop+tile)")

    # Paper 2 reference
    p2_tiles = paper2_split_into_grid(img, args.scale)
    print(f"Paper 2 tiles: {len(p2_tiles)} x 518x518 (after raw grid split + per-tile resize)")

    panels = [
        paste_grid(p1_tiles, args.scale,
                   label=f"Paper 1 (ours): scale={args.scale}, center-cropped square"),
        paste_grid(p2_tiles, args.scale,
                   label=f"Paper 2: grid={args.scale}x{args.scale}, raw split + squash"),
    ]

    if args.with_jpeg:
        p1_jpeg = [jpeg_compress(t, 85, "4:2:2") for t in p1_tiles]
        panels.append(paste_grid(
            p1_jpeg, args.scale,
            label=f"Paper 1 + JPEG round-trip 4:2:2 q85 (what model actually sees)"))

    # Stack panels vertically
    W = max(p.width for p in panels)
    H = sum(p.height for p in panels) + 10 * (len(panels) - 1)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + 10

    canvas.save(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
