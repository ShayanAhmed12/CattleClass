"""Classification metrics and experiment results for cattle breed models."""

import csv
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from . import config, data

_RESULT_KEY = ("experiment", "dataset", "scoring")
_SCORE_FIELDS = ("n_images", "accuracy", "balanced_accuracy", "macro_f1")
_RESULT_FIELDS = (*_RESULT_KEY, *_SCORE_FIELDS, "macro_f1_std", "date")


def score(y_true, y_pred):
    """Score canonical class indices, averaging only over true-label classes.

    Per-class F1 is keyed by breed name. Both axes of the confusion matrix
    follow config.CLASSES, including classes absent from the ground truth.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not y_true.size:
        raise ValueError("Expected matching nonempty vectors of true and predicted labels.")

    labels = np.arange(len(config.CLASSES))
    if not (np.isin(y_true, labels).all() and np.isin(y_pred, labels).all()):
        raise ValueError("Labels must be canonical class indices.")
    present = np.unique(y_true)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n_images": len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=present, average="macro")),
        "macro_f1": float(f1[np.isin(labels, present)].mean()),
        "per_class_f1": {name: float(value) for name, value in zip(config.CLASSES, f1)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def average_fold_scores(folds):
    """Average rates over folds and sum the number of test photos."""
    averages = {key: float(np.mean([fold[key] for fold in folds]))
                for key in ('accuracy', 'balanced_accuracy', 'macro_f1')}
    return {**averages, 'n_images': sum(fold['n_images'] for fold in folds),
            'macro_f1_std': float(np.std([fold['macro_f1'] for fold in folds]))}


def save_result(experiment, dataset, scoring, scores, *, path=None):
    """Record one summary row, replacing the same experiment's row on rerun."""
    path = config.RESULTS_DIR / "metrics.csv" if path is None else Path(path)
    identity = (experiment, dataset, scoring)
    result = dict(zip(_RESULT_KEY, identity))
    result.update({field: str(scores[field]) for field in _SCORE_FIELDS})
    result["macro_f1_std"] = str(scores.get("macro_f1_std", ""))
    result["date"] = date.today().isoformat()

    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))

    matches = [i for i, row in enumerate(rows) if tuple(row[key] for key in _RESULT_KEY) == identity]
    if matches:
        first, *duplicates = matches
        previous = rows[first]
        if all(previous[field] == result[field] for field in _RESULT_FIELDS if field != "date"):
            result["date"] = previous["date"]
        rows[first] = result
        for i in reversed(duplicates):
            del rows[i]
    else:
        rows.append(result)

    data.save_csv(path, _RESULT_FIELDS, rows)
