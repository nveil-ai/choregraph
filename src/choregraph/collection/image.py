# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Image processing transforms for choregraph.

Provides functions to extract pixel data from images into DataFrames
for use with standard chart marks (Mode 2 processing).

All transforms accept a ``PIL.Image.Image`` as first argument, which is
what Kedro's ``pillow.ImageDataset`` returns when loaded from the catalog.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def image_to_dataframe(
    image: Image.Image,
    color_format: str = "rgba",
    max_pixels: int = 500_000,
    sample_step: Optional[int] = None,
) -> pd.DataFrame:
    """Convert an image to a long-form DataFrame of pixel values.

    Args:
        image: PIL Image object (loaded by Kedro's pillow.ImageDataset).
        color_format: ``"rgba"`` produces one row per pixel per channel
            (r, g, b, a).  ``"gray"`` produces one row per pixel with a
            single ``gray`` channel (luminance).
        max_pixels: Maximum number of pixels to include. If the image exceeds
            this count, it is subsampled automatically.
        sample_step: Explicit step size for subsampling. Overrides *max_pixels*
            when provided.

    Returns:
        DataFrame with columns ``x``, ``y``, ``channel``, ``value``.
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        from PIL import Image  # noqa: F401
    grayscale = color_format.lower() == "gray"
    img = image.convert("L" if grayscale else "RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2] if arr.ndim >= 2 else (arr.shape[0], 1)

    if sample_step is None:
        total = h * w
        sample_step = max(1, int((total / max_pixels) ** 0.5)) if total > max_pixels else 1

    ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
    sampled = arr[0:h:sample_step, 0:w:sample_step]
    n_pixels = xs.size

    if grayscale:
        return pd.DataFrame({
            "x": xs.ravel(),
            "y": ys.ravel(),
            "channel": "gray",
            "value": sampled.ravel(),
        })

    channels = ["r", "g", "b", "a"]
    flat_x = np.tile(xs.ravel(), len(channels))
    flat_y = np.tile(ys.ravel(), len(channels))
    flat_ch = np.repeat(channels, n_pixels)
    flat_val = np.concatenate([sampled[:, :, i].ravel() for i in range(4)])

    return pd.DataFrame({
        "x": flat_x,
        "y": flat_y,
        "channel": flat_ch,
        "value": flat_val,
    })


def extract_channel(
    image: Image.Image,
    channel: str = "gray",
    max_pixels: int = 500_000,
    sample_step: Optional[int] = None,
) -> pd.DataFrame:
    """Extract a single colour channel from an image.

    Args:
        image: PIL Image object.
        channel: One of ``"r"``, ``"g"``, ``"b"``, ``"a"``, or ``"gray"``.
        max_pixels: Maximum number of pixels.
        sample_step: Explicit subsampling step.

    Returns:
        DataFrame with columns ``x``, ``y``, ``channel``, ``value``.
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        from PIL import Image  # noqa: F401
    channel = channel.lower()
    if channel == "gray":
        arr = np.array(image.convert("L"))
    else:
        arr = np.array(image.convert("RGBA"))
        idx = {"r": 0, "g": 1, "b": 2, "a": 3}[channel]
        arr = arr[:, :, idx]

    h, w = arr.shape[:2]
    if sample_step is None:
        total = h * w
        sample_step = max(1, int((total / max_pixels) ** 0.5)) if total > max_pixels else 1

    ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
    sampled = arr[0:h:sample_step, 0:w:sample_step]

    return pd.DataFrame({
        "x": xs.ravel(),
        "y": ys.ravel(),
        "channel": channel,
        "value": sampled.ravel(),
    })


def image_metadata(image: Image.Image) -> pd.DataFrame:
    """Return basic metadata about an image.

    Args:
        image: PIL Image object.

    Returns:
        Single-row DataFrame with columns ``width``, ``height``,
        ``channels``, ``format``, ``mode``.
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        from PIL import Image  # noqa: F401
    w, h = image.size
    n_channels = len(image.getbands())
    fmt = image.format or "UNKNOWN"
    mode = image.mode

    return pd.DataFrame([{
        "width": w,
        "height": h,
        "channels": n_channels,
        "format": fmt,
        "mode": mode,
    }])
