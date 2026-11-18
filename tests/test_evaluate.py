import csv
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate
from cattle_breed import config

HOLSTEIN, LOCAL, RED_CHITTAGONG, SAHIWAL, SINDHI = range(len(config.CLASSES))


class Classifier:
    def __init__(self, predictions, seen):
        self.predictions = predictions
        self.seen = seen

    def predict(self, images, *, batch_size, verbose):
        assert images.shape[1:] == (config.IMG_SIZE, config.IMG_SIZE, 3)
        assert images.dtype == np.uint8
        assert batch_size == config.BATCH_SIZE
        codes = images[:, 0, 0, 0]
        self.seen.append(codes.tolist())
        probs = np.zeros((len(images), len(config.CLASSES)))
        for row, code in enumerate(codes):
            first, second = self.predictions[int(code)]
            probs[row, [first, second]] = [0.8, 0.2]
        return probs


def cached_examples(codes, labels, **columns):
    images = np.zeros((len(codes), config.CACHE_SIZE, config.CACHE_SIZE, 3), dtype=np.uint8)
    border = (config.CACHE_SIZE - config.IMG_SIZE) // 2
    images[:, border:-border, border:-border] = np.asarray(codes)[:, None, None, None]
    index = pd.DataFrame({"path": [f"photo_{code}.jpg" for code in codes], **columns})
    return index, images, np.asarray(labels)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def setup_evaluation(monkeypatch, tmp_path, *, cid=None):
    caches = {
        "mendeley": cached_examples([10, 20], [LOCAL, SAHIWAL]),
        "kaggle": cached_examples([30, 40], [SINDHI, HOLSTEIN]),
        "cid": cid,
    }
    loads = []
    seen = {}
    predictions = {
        10: (RED_CHITTAGONG, LOCAL),
        20: (SAHIWAL, HOLSTEIN),
        30: (LOCAL, SINDHI),
        40: (HOLSTEIN, SAHIWAL),
        50: (LOCAL, SINDHI),
        60: (SINDHI, SAHIWAL),
        51: (SINDHI, LOCAL),
        61: (SINDHI, SAHIWAL),
    }

    @contextmanager
    def atomic_output(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        yield path

    def load_cache(dataset):
        loads.append(dataset)
        assert caches[dataset] is not None
        return caches[dataset]

    def load_model(path):
        assert path.parent == tmp_path / "models"
        assert path.name not in seen
        seen[path.name] = []
        return Classifier(predictions, seen[path.name])

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(evaluate.data, "load_cache", load_cache, raising=False)
    monkeypatch.setattr(evaluate.data, "atomic_output", atomic_output, raising=False)
    monkeypatch.setattr(evaluate, "load_model", load_model)
    return loads, seen


def test_default_scores_both_external_protocols_and_writes_confusions(monkeypatch, tmp_path):
    loads, seen = setup_evaluation(monkeypatch, tmp_path)

    evaluate.main([])

    assert loads == ["mendeley", "kaggle"]
    assert seen == {"final.keras": [[10, 20], [30, 40]]}
    rows = read_csv(config.RESULTS_DIR / "metrics.csv")
    assert {(row["experiment"], row["dataset"], row["scoring"], float(row["macro_f1"]))
            for row in rows} == {
        ("E3", "mendeley", "5-way", 0.5),
        ("E3", "mendeley", "restricted", 1.0),
        ("E4", "kaggle", "5-way", 0.5),
        ("E4", "kaggle", "restricted", 1.0),
    }
    figures = config.RESULTS_DIR / "figures"
    for dataset in ("mendeley", "kaggle"):
        for protocol in ("5-way", "restricted"):
            assert (figures / f"{dataset}_{protocol}_confusion.png").stat().st_size > 0
    per_class = read_csv(config.RESULTS_DIR / "per_class_f1.csv")
    assert {row["breed"] for row in per_class
            if row["dataset"] == "mendeley" and row["scoring"] == "restricted"} == set(
        config.DATASET_CLASSES["mendeley"]
    )


def test_robustness_scores_cid_fold_models_only_on_their_held_out_images(monkeypatch, tmp_path):
    cid = cached_examples([50, 60], [LOCAL, SINDHI], fold=[0, 1])
    loads, seen = setup_evaluation(monkeypatch, tmp_path, cid=cid)
    monkeypatch.setattr(config, "N_FOLDS", 2)
    monkeypatch.setattr(config, "CORRUPTIONS", {"dark": {1: 0.5}})
    corrupted = []

    def corrupt(images, kind, severity):
        assert (kind, severity) == ("dark", 1)
        corrupted.append(images[:, 0, 0, 0].tolist())
        return np.where(images >= 50, images + 1, images)

    monkeypatch.setattr(evaluate.data, "corrupt", corrupt, raising=False)
    evaluate.main(["--robustness"])

    assert loads == ["mendeley", "kaggle", "cid"]
    assert seen == {
        "final.keras": [[10, 20], [10, 20], [30, 40], [30, 40]],
        "fold_0.keras": [[50], [51]],
        "fold_1.keras": [[60], [61]],
    }
    assert corrupted == [[10, 20], [30, 40], [50], [60]]
    rows = [row for row in read_csv(config.RESULTS_DIR / "metrics.csv") if row["experiment"] == "E5"]
    assert {(row["dataset"], row["scoring"]) for row in rows} == {
        ("cid", "5-way/dark-1"),
        ("mendeley", "5-way/dark-1"), ("mendeley", "restricted/dark-1"),
        ("kaggle", "5-way/dark-1"), ("kaggle", "restricted/dark-1"),
    }
    cid_result = next(row for row in rows if row["dataset"] == "cid")
    assert float(cid_result["macro_f1"]) == 0.5
    assert float(cid_result["macro_f1_std"]) == 0.5
    assert int(cid_result["n_images"]) == 2
    assert (config.RESULTS_DIR / "figures" / "robustness.png").stat().st_size > 0


def test_figures_use_predicted_breeds_for_gradcam_and_preserve_error_notes(monkeypatch, tmp_path):
    setup_evaluation(monkeypatch, tmp_path)
    calls = []

    def gradcam(model, image, class_idx):
        calls.append((image[0, 0, 0], class_idx))
        return np.ones((config.IMG_SIZE, config.IMG_SIZE))

    monkeypatch.setattr(evaluate, "gradcam", gradcam)
    evaluate.main(["--figures"])

    assert set(calls) == {(10, RED_CHITTAGONG), (20, SAHIWAL), (30, LOCAL), (40, HOLSTEIN)}
    local = config.RESULTS_DIR / "figures" / "local"
    for name in ("error_gallery.png", "gradcam.png"):
        assert (local / name).stat().st_size > 0
    annotations = local / "error_annotations.csv"
    rows = read_csv(annotations)
    assert {Path(row["path"]).name for row in rows} == {"photo_10.jpg", "photo_30.jpg"}
    edited = rows[0]
    edited["likely_cause"] = "pose"
    with annotations.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(edited))
        writer.writeheader()
        writer.writerows(rows)

    setup_evaluation(monkeypatch, tmp_path)
    evaluate.main(["--figures"])
    assert read_csv(annotations) == rows
