"""Shared pytest fixtures. Tests use small synthetic images, never the real datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_image(tmp_path: Path):
    """Factory: make_image("images/SKU/0.jpg", size=(W, H), mode="RGB", color=(r, g, b), noise=False).

    Paths are relative to tmp_path; parent folders are created. The file format follows the
    suffix, so use .png for RGBA images. noise=True fills the image with seeded random pixels.
    """

    def _make(name: str = "img.jpg", size=(320, 200), mode: str = "RGB", color=(200, 30, 30),
              noise: bool = False, seed: int = 0, exif_orientation: int | None = None) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if noise:
            rng = np.random.default_rng(seed)
            arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
            img = Image.fromarray(arr).convert(mode)
        else:
            fill = color[0] if mode in ("L", "P") else (*color, 128) if mode == "RGBA" else color
            img = Image.new(mode, size, fill)
        kwargs = {}
        if exif_orientation is not None:
            exif = Image.Exif()
            exif[0x0112] = exif_orientation
            kwargs["exif"] = exif
        img.save(path, **kwargs)
        return path

    return _make
