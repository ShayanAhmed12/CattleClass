"""Turning raw dataset files into model-ready arrays.

    images   load_image, letterbox, center_crop      every image ends up RGB, upright, 240 x 240
    index    (Task 2) one table of path, animal, label for each dataset
    clean    (Task 4) remove duplicate photos
    folds    (Task 4) split CID animals into 5 folds
    cache    (Task 5) save all images of a dataset as one .npz file
    corrupt  (Task 10) blurred, darkened and occluded copies for the robustness test
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CACHE_SIZE, CLASSES, CLASS_TO_IDX, DATA_DIR, IMAGE_SUFFIXES, IMG_SIZE, RESULTS_DIR, to_canonical
from .download import EXPECTED


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


def build_index(dataset: str) -> pd.DataFrame:
    """Index the supported images in a downloaded dataset in stable path order."""
    root = (DATA_DIR / EXPECTED[dataset]["root"]).resolve()
    cid_labels = {}
    if dataset == "cid":
        with (root.parent / "dataset.csv").open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                sku = "".join(row["sku"].split())
                label = to_canonical(row["breed"], dataset)
                if sku in cid_labels and label != cid_labels[sku]:
                    raise ValueError(f"Conflicting CID breed labels for {sku}")
                cid_labels[sku] = label

    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        animal = "".join(folder.name.split())
        label = cid_labels.get(animal) if dataset == "cid" else to_canonical(folder.name, dataset)
        if label is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and is_image_file(path):
                filename = str(path)
                rows.append((filename, dataset, animal if dataset == "cid" else filename,
                             label, CLASS_TO_IDX[label]))
    return pd.DataFrame(rows, columns=["path", "dataset", "animal", "label", "label_idx"])


def save_cid_samples(index: pd.DataFrame) -> Path:
    """Write a local grid with the views of one representative animal per class."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 11), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(CLASSES), 4, squeeze=False)
    for breed, row in zip(CLASSES, axes):
        images = index[index["label"] == breed]
        for ax in row:
            ax.set_axis_off()
        if images.empty:
            continue
        animal = images.iloc[0]["animal"]
        for ax, path in zip(row, images[images["animal"] == animal]["path"]):
            ax.imshow(load_image(path))
            ax.set_title(Path(path).stem)
        row[0].text(-0.05, 0.5, breed, transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")
    path = RESULTS_DIR / "figures" / "cid_samples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path
