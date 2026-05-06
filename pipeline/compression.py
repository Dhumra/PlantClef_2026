"""
JPEG round-trip compression (paper 1 pre-processing).

Applying the same round-trip to each 518x518 test tile aligns the test
distribution with training data. The round-trip is necessary even for .jpg
test images: PIL decodes them to raw RGB, discarding the original encoding
artifacts; the round-trip re-introduces standardised artifacts matching the
training set.

Paper 1 best configurations:
  - 4:2:2 at quality 85 (Pillow native, default)
  - 4:1:1 at quality 94 (requires PyTurboJPEG; Pillow subsampling=2 gives
                          4:2:0, which is different from 4:1:1)

Pillow integer mapping: 0 -> 4:4:4, 1 -> 4:2:2, 2 -> 4:2:0
"""

import io
from typing import Union

import numpy as np
from PIL import Image

# Maps human-readable subsampling strings to Pillow's integer constants.
# "4:1:1" is absent because it cannot be encoded by Pillow (see module doc).
_PILLOW_SUBSAMPLING: dict = {
    "4:4:4": 0,
    "4:2:2": 1,
    "4:2:0": 2,
}


def _compress_via_pillow(img: Image.Image, quality: int, subsampling: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).copy()


def _compress_411(img: Image.Image, quality: int) -> Image.Image:
    """Encode with 4:1:1 subsampling via PyTurboJPEG (libjpeg-turbo)."""
    try:
        from turbojpeg import TurboJPEG, TJSAMP_411  # type: ignore
    except ImportError:
        raise RuntimeError(
            "4:1:1 subsampling requires PyTurboJPEG.  "
            "Install it with:  pip install PyTurboJPEG"
        )
    jpeg = TurboJPEG()
    # TurboJPEG.encode expects a uint8 numpy array in BGR order
    arr = np.array(img.convert("RGB"))[:, :, ::-1]   # RGB to BGR
    data = jpeg.encode(arr, quality=quality, jpeg_subsample=TJSAMP_411)
    return Image.open(io.BytesIO(data)).copy()


def jpeg_compress(
    img: Image.Image,
    quality: int = 85,
    subsampling: Union[str, int] = "4:2:2",
) -> Image.Image:
    """
    Encode img to JPEG and decode back to a PIL Image.

    Parameters
    ----------
    img         : Source tile (PIL Image; converted to RGB internally).
    quality     : JPEG quality factor, 1-95.
    subsampling : Chroma subsampling mode.
                  Accepts a string ("4:4:4", "4:2:2", "4:2:0", "4:1:1") or a
                  Pillow integer (0=4:4:4, 1=4:2:2, 2=4:2:0).
                  "4:1:1" requires PyTurboJPEG to be installed.
    """
    img = img.convert("RGB")

    if isinstance(subsampling, int):
        return _compress_via_pillow(img, quality, subsampling)

    if subsampling == "4:1:1":
        return _compress_411(img, quality)

    pillow_sub = _PILLOW_SUBSAMPLING.get(subsampling)
    if pillow_sub is None:
        raise ValueError(
            f"Unknown subsampling {subsampling!r}.  "
            f"Valid values: {list(_PILLOW_SUBSAMPLING)} + '4:1:1'."
        )
    return _compress_via_pillow(img, quality, pillow_sub)
