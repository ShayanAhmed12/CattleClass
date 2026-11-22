"""Index, clean, split and cache photos. Apply test corruptions to cached images."""
import csv
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from . import config
from .config import CACHE_SIZE, CLASSES, CLASS_TO_IDX, CORRUPTIONS, DATA_DIR, DUPLICATE_DISTANCE, N_FOLDS, RESULTS_DIR, SEED, VAL_FRAC, to_canonical
from .download import CID_SPEC, is_image_file


class ImageLoadError(Exception):
    """Raised when a file cannot be decoded as an image."""


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as an (H, W, 3) uint8 RGB array with its EXIF orientation applied."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF, not pixels
            return np.asarray(img.convert("RGB"), dtype=np.uint8)  # greyscale, RGBA, palette -> RGB
    except (OSError, Image.DecompressionBombError) as err:
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


def center_crop_batch(images):
    """Return a view of the centered model-sized region in every cached photo."""
    height, width = images.shape[1:3]
    size = config.IMG_SIZE
    top, left = (height - size) // 2, (width - size) // 2
    return images[:, top:top + size, left:left + size]


def require_cid(dataset):
    if dataset != "cid":
        raise ValueError("CattleClass uses CID, which provides breed, age and weight labels.")


def build_index(dataset: str) -> pd.DataFrame:
    """Index CID photos by original SKU and canonical breed in stable path order."""
    require_cid(dataset)
    root = (DATA_DIR / CID_SPEC["root"]).resolve()
    labels = {}
    with (root.parent / "dataset.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            sku = "".join(row["sku"].split())
            label = to_canonical(row["breed"])
            if sku in labels and label != labels[sku]:
                raise ValueError(f"Conflicting CID breed labels for {sku}")
            labels[sku] = label

    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        animal = "".join(folder.name.split())
        label = labels.get(animal)
        if label is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and is_image_file(path):
                rows.append((str(path), dataset, animal, label, CLASS_TO_IDX[label]))
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


def hash_images(paths):
    """Hash decodable photos in path order; the cache reports unreadable files."""
    import imagehash

    hashes = {}
    for path in paths:
        try:
            image = load_image(path)
        except ImageLoadError:
            continue
        hashes[path] = int(str(imagehash.phash(Image.fromarray(image))), 16)
    return hashes


def find_duplicates(paths, *, hashes=None):
    """Return all matching (earlier, later) pairs in input order."""
    if hashes is None:
        hashes = hash_images(paths)
    earlier = []
    pairs = []
    for path, fingerprint in hashes.items():
        pairs.extend((previous, path) for previous, value in earlier
                     if (fingerprint ^ value).bit_count() <= DUPLICATE_DISTANCE)
        earlier.append((path, fingerprint))
    return pairs


def clean_cid_index(index, duplicates):
    """Merge linked animal IDs before dropping duplicate CID photos."""
    animals = index.set_index('path')['animal'].to_dict()
    parents = {animal: animal for animal in animals.values()}

    def representative(animal):
        while parents[animal] != animal:
            parents[animal] = parents[parents[animal]]
            animal = parents[animal]
        return animal

    copies = set()
    for earlier, later in duplicates:
        first, second = sorted((representative(animals[earlier]), representative(animals[later])))
        parents[second] = first
        copies.add(later)
    clean = index.loc[~index['path'].isin(copies)].copy()
    clean['animal'] = clean['animal'].map(representative)
    return clean.reset_index(drop=True)


def make_folds(cid_index):
    """Assign each CID photo its reproducible animal-grouped test fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.empty(len(cid_index), dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(cid_index, cid_index['label_idx'], cid_index['animal'])):
        folds[test] = fold
    return pd.Series(folds, index=cid_index.index, name='fold')


def split_validation(train_index):
    """Stratify the 15% validation holdout by each animal's most frequent label."""
    from sklearn.model_selection import train_test_split

    animals = train_index.groupby('animal')['label_idx'].agg(lambda labels: labels.mode().iloc[0])
    _, val_animals = train_test_split(animals.index.to_numpy(), test_size=VAL_FRAC, random_state=SEED,
                                     stratify=animals)
    held_out = train_index['animal'].isin(val_animals)
    return train_index.loc[~held_out], train_index.loc[held_out]


def build_cache(index, *, return_index=False):
    """Decode and letterbox the usable rows, preserving index order and label alignment."""
    images = np.empty((len(index), CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
    positions = []
    for position, path in enumerate(index['path']):
        try:
            images[len(positions)] = letterbox(load_image(path))
        except ImageLoadError as error:
            print(error, file=sys.stderr)
            continue
        positions.append(position)
    usable = index.iloc[positions].reset_index(drop=True)
    images = images[:len(positions)]
    labels = usable['label_idx'].to_numpy(dtype=np.int64)
    return (images, labels, usable) if return_index else (images, labels)


@contextmanager
def atomic_output(path):
    """Stage an output beside its destination and replace it after a successful write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_csv(path, fields, rows):
    """Write a complete CSV before replacing an existing result."""
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def portable_index(index):
    """Store dataset paths relative to DATA_DIR for use on a different machine."""
    stored = index.copy()
    stored['path'] = stored['path'].map(lambda path: Path(path).relative_to(DATA_DIR.resolve()).as_posix())
    return stored


def save_cache(dataset, index, images, labels):
    """Write the processed index and image cache, including the row identities."""
    require_cid(dataset)
    root = DATA_DIR / 'processed'
    stored = portable_index(index)
    with atomic_output(root / f'{dataset}.npz') as temporary:
        np.savez_compressed(temporary, images=images, labels=labels, paths=stored['path'].to_numpy(dtype=str),
                            folds=stored['fold'].to_numpy() if 'fold' in stored else np.empty(0, dtype=int))
    with atomic_output(root / f'{dataset}.csv') as temporary:
        stored.to_csv(temporary, index=False)


def load_cache(dataset):
    """Load a matching processed index and cache, restoring absolute source paths."""
    require_cid(dataset)
    root = DATA_DIR / 'processed'
    index = pd.read_csv(root / f'{dataset}.csv', dtype={'path': str, 'animal': str})
    with np.load(root / f'{dataset}.npz', allow_pickle=False) as cache:
        images, labels = cache['images'], cache['labels']
        folds = index['fold'].to_numpy() if 'fold' in index else []
        if (len(images) != len(index) or not np.array_equal(index['path'], cache['paths'])
                or not np.array_equal(index['label_idx'], labels) or not np.array_equal(folds, cache['folds'])):
            raise ValueError(f'{dataset} processed index does not match its cache. Rerun prepare_data.py.')
    index['path'] = index['path'].map(lambda path: str(DATA_DIR.resolve() / path))
    return index, images, labels


def corrupt(images, kind, severity):
    """Apply a reproducible corruption to an RGB uint8 batch without modifying it."""
    try:
        strength = CORRUPTIONS[kind][severity]
    except KeyError as error:
        raise ValueError(f'Unknown corruption or severity: {kind!r}, {severity!r}') from error
    if kind == 'dark':
        return np.rint(images * strength).astype(np.uint8)
    result = images.copy()
    if kind == 'blur':
        blur = ImageFilter.GaussianBlur(radius=strength)
        for i, image in enumerate(images):
            result[i] = np.asarray(Image.fromarray(image).filter(blur))
        return result

    height, width = images.shape[1:3]
    side = min(height, width, max(1, round((strength * height * width) ** 0.5)))
    rng = np.random.default_rng(SEED)
    for image in result:
        top = rng.integers(0, height - side + 1)
        left = rng.integers(0, width - side + 1)
        image[top:top + side, left:left + side] = 128
    return result
