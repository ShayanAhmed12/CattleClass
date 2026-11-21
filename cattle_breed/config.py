"""Project settings for data, training and evaluation (see docs/report.md)."""
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Big files (datasets, caches, trained models) go under WORK_DIR, never into git.
# On your PC that is the project folder. On Colab it is Google Drive, so data and models survive
# a disconnect, while archives are downloaded to Colab's fast temporary disk.
# CATTLE_WORK_DIR overrides it anywhere.
ON_COLAB = Path("/content/drive/MyDrive").is_dir()
WORK_DIR = Path(os.environ.get("CATTLE_WORK_DIR",
                               "/content/drive/MyDrive/cattle-breed" if ON_COLAB else PROJECT_ROOT))
DATA_DIR = WORK_DIR / "data"  # data/raw (datasets) and data/processed (index + image cache)
MODELS_DIR = WORK_DIR / "models"
DOWNLOAD_DIR = Path("/content/downloads") if ON_COLAB else DATA_DIR / "raw" / "_downloads"  # archives
RESULTS_DIR = PROJECT_ROOT / "results"  # published tables and local plots

SEED = 42

# --------------------------------------------------------------------------- classes
# Fixed order: model output column i is CLASSES[i].
CLASSES = ("Holstein cross", "Local", "Red Chittagong", "Sahiwal", "Sindhi")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# The CID CSV spells Holstein cross as HOSTINE_CROSS.
LABEL_MAP = {
    "hostinecross": "Holstein cross",
    "local": "Local",
    "redchittagong": "Red Chittagong",
    "sahiwal": "Sahiwal",
    "sindhi": "Sindhi",
}


def normalise_label(raw):
    """Collapse the CID source spelling to lowercase letters."""
    return re.sub(r"[^a-z]", "", str(raw).lower())


def to_canonical(raw):
    """Return the supported CID breed name, or None."""
    return LABEL_MAP.get(normalise_label(raw))


# --------------------------------------------------------------------------- images
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
CACHE_SIZE = 240  # every image is letterboxed (aspect kept, black padding) to 240 x 240 and cached
IMG_SIZE = 224  # model input: random 224 crop in training, centre crop otherwise

# --------------------------------------------------------------------------- evaluation protocol
N_FOLDS = 5  # CID: stratified 5-fold cross-validation, split by animal
VAL_FRAC = 0.15  # share of each training fold's animals held out for early stopping
DUPLICATE_DISTANCE = 4  # perceptual-hash distance at or below which two images count as the same photo

# --------------------------------------------------------------------------- training
BATCH_SIZE = 32
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1
HEAD_LR, HEAD_EPOCHS, HEAD_PATIENCE = 1e-3, 10, 3  # stage 1: backbone frozen, train the new head (E1)
FT_LR, FT_EPOCHS, FT_PATIENCE = 1e-5, 20, 4  # stage 2: fine-tune the whole network (E2)

# Augmentation. No hue or saturation changes: coat colour is a breed cue.
AUG_ROTATION = 0.02  # fraction of a full turn, about +/- 7 degrees
AUG_BRIGHTNESS = 0.1
AUG_CONTRAST = 0.1

# --------------------------------------------------------------------------- robustness test (E5)
# corruption -> {severity: strength}
CORRUPTIONS = {
    "blur": {1: 2.0, 2: 4.0},  # Gaussian blur radius in pixels
    "dark": {1: 0.5, 2: 0.3},  # pixel values multiplied by this
    "occlude": {1: 0.125, 2: 0.25},  # share of the image covered by one grey square
}
