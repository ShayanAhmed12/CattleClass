"""Project-wide constants. Every other module reads its fixed values from here.

Change a value here only together with an entry in docs/decision_log.md, because
results in results/experiments.csv depend on these values.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# On Colab these two are set to folders on Google Drive (see notebooks/run_all_colab.ipynb).
DATA_ROOT = Path(os.environ.get("CATTLE_DATA_ROOT", PROJECT_ROOT / "data"))
ARTIFACTS_ROOT = Path(os.environ.get("CATTLE_ARTIFACTS_ROOT", PROJECT_ROOT / "artifacts"))
# Image caches (T7). On Colab, copy them to local disk and point CATTLE_CACHE_DIR there for speed.
CACHE_DIR = Path(os.environ.get("CATTLE_CACHE_DIR", DATA_ROOT / "processed"))
RESULTS_ROOT = PROJECT_ROOT / "results"  # small files only; committed to git
CURATION_ROOT = PROJECT_ROOT / "docs" / "curation"

# --------------------------------------------------------------------------- reproducibility
SEED = 42

# --------------------------------------------------------------------------- classes
# Index order is fixed. Model output column i is CLASSES[i].
CLASSES: tuple[str, ...] = ("Holstein cross", "Local", "Red Chittagong", "Sahiwal", "Sindhi")
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# Raw labels from every dataset, normalised with normalise_label(), mapped to a canonical class.
# Raw labels that are not listed (PABNA_BREED, BRAHMA, MIR_KADIM, Jersey, Gir, ...) are dropped.
LABEL_MAP: dict[str, str] = {
    # CID (dataset.csv "breed" column; HOSTINE_CROSS is misspelt in the source CSV)
    "hostinecross": "Holstein cross",
    "local": "Local",
    "redchittagong": "Red Chittagong",
    "sahiwal": "Sahiwal",
    "sindhi": "Sindhi",
    # Mendeley Bangladeshi cow breed dataset (folder names)
    "holsteinfriesian": "Holstein cross",
    "localdeshi": "Local",
    # Kaggle Indian Bovine Breeds (folder names); Holstein_Friesian and Sahiwal are covered above
    "redsindhi": "Sindhi",
}

DATASETS = ("cid", "mendeley", "kaggle")
# Classes each dataset can contain after mapping. External sets are scored on these.
DATASET_CLASSES: dict[str, tuple[str, ...]] = {
    "cid": CLASSES,
    "mendeley": ("Holstein cross", "Local", "Sahiwal"),
    "kaggle": ("Holstein cross", "Sahiwal", "Sindhi"),
}


def normalise_label(raw: str) -> str:
    """Lower-case and keep letters only: 'HOSTINE_CROSS' -> 'hostinecross', 'Local Deshi' -> 'localdeshi'."""
    return re.sub(r"[^a-z]", "", str(raw).lower())


def to_canonical(raw: str, dataset: str | None = None) -> str | None:
    """Canonical class name for a raw label, or None if the label is not used.

    If `dataset` is given, labels outside DATASET_CLASSES[dataset] also return None.
    """
    name = LABEL_MAP.get(normalise_label(raw))
    if name is not None and dataset is not None and name not in DATASET_CLASSES[dataset]:
        return None
    return name


# --------------------------------------------------------------------------- images
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
CACHE_SIZE = 240  # images are letterboxed (aspect kept, black padding) to 240x240 and cached
IMG_SIZE = 224  # model input; random 224 crop when training, centre crop otherwise

# --------------------------------------------------------------------------- splits
N_FOLDS = 5
INNER_VAL_FRAC = 0.15  # share of training-fold animals held out for early stopping


# --------------------------------------------------------------------------- training
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    dropout: float = 0.3
    label_smoothing: float = 0.1
    head_lr: float = 1e-3  # stage 1: backbone frozen, only the classifier head trains (E1)
    head_epochs: int = 10
    head_patience: int = 3
    ft_lr: float = 1e-5  # stage 2: whole network fine-tuned, BatchNorm kept in inference mode (E2)
    ft_epochs: int = 20
    ft_patience: int = 4


# Augmentation strengths (keras preprocessing layers). No hue or saturation changes:
# coat colour is a breed cue.
AUG_ROTATION = 0.02  # fraction of a full turn, about +/- 7 degrees
AUG_BRIGHTNESS = 0.1
AUG_CONTRAST = 0.1

# --------------------------------------------------------------------------- engineered test set (E5)
# kind -> {severity: parameter}
CORRUPTIONS: dict[str, dict[int, float]] = {
    "blur": {1: 2.0, 2: 4.0},  # Gaussian blur radius in pixels (at 240 px)
    "dark": {1: 0.5, 2: 0.3},  # pixel values multiplied by this factor
    "occlude": {1: 0.125, 2: 0.25},  # area fraction of one grey (127) square patch
}
OCCLUDE_VALUE = 127

# --------------------------------------------------------------------------- curation and duplicates
DEDUP_MAX_DISTANCE = 4  # perceptual-hash Hamming distance at or below which two images are duplicates
CURATION_DECISIONS = ("keep", "drop")
CURATION_REASONS = (
    "DRAWING",  # illustration, render or painting, not a photo
    "COLLAGE",  # several photos or animals combined into one image
    "TEXT_WATERMARK",  # text or watermark covers a large part of the animal
    "NOT_VISIBLE",  # animal too small, cut off or hidden to judge the breed
    "OTHER_SPECIES",  # buffalo, goat, toy, etc.
    "WRONG_BREED",  # clearly not the labelled breed
    "UNCERTAIN_BREED",  # cannot confirm the label from the image
    "PERSON",  # identifiable face, number plate, ID card or private document
    "DUPLICATE",  # same photo appears earlier in the set or in CID
    "CORRUPT",  # file cannot be decoded
)

# Cause codes for the error-analysis gallery (Task 15).
ERROR_CAUSES = (
    "SIMILAR_BREED", "POSE", "OCCLUSION", "LIGHTING", "BACKGROUND",
    "CALF", "SEX", "LABEL_NOISE", "OTHER",
)

# --------------------------------------------------------------------------- results registry
RESULT_COLUMNS = (
    "date", "commit", "experiment", "dataset", "scoring", "n_images",
    "accuracy", "balanced_accuracy", "macro_f1", "macro_f1_std", "notes",
)
