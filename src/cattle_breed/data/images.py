"""Image loading and resizing (Task 1).

Every image in the project passes through here, so all datasets end up in the same form:
an RGB uint8 array, upright, letterboxed to a square, then centre- or randomly cropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from cattle_breed.config import CACHE_SIZE, IMAGE_SUFFIXES, IMG_SIZE


class ImageLoadError(Exception):
    """Raised when a file cannot be decoded as an image."""


def is_image_file(path: str | Path) -> bool:
    """True for .jpg/.jpeg/.png/.webp files (any case), excluding macOS '._' resource-fork files."""
    p = Path(path)
    return p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as an (H, W, 3) uint8 RGB array with its EXIF orientation applied."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF, not pixels
            return np.asarray(img.convert("RGB"), dtype=np.uint8)  # greyscale, RGBA, palette -> RGB
    except (OSError, UnidentifiedImageError) as err:
        raise ImageLoadError(f"cannot read image {path}: {err}") from err


def letterbox(img: np.ndarray, size: int = CACHE_SIZE) -> np.ndarray:
    """Fit the image inside a size x size square without distorting it, padding with black."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = np.asarray(resized)
    return out


def center_crop(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Cut the central size x size region (used for evaluation and the demo)."""
    h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]
