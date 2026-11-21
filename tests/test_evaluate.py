import csv

import numpy as np
import pandas as pd
import pytest

import evaluate
import train
from cattle_breed import config


def setup_cid(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "N_FOLDS", 2)
    monkeypatch.setattr(config, "CORRUPTIONS", {"dark": {1: 0.5}})
    index = pd.DataFrame({"path": ["one.jpg", "two.jpg"], "fold": [0, 1]})
    images = np.stack([np.full((240, 240, 3), code, np.uint8) for code in (10, 20)])
    labels = np.array([1, 4])
    loads = []
    seen = {}

    def load_cache(dataset):
        loads.append(dataset)
        assert dataset == "cid"
        return index, images, labels

    class Classifier:
        def __init__(self, fold):
            self.fold = fold

        def predict(self, batch, **kwargs):
            codes = batch[:, 0, 0, 0].tolist()
            seen.setdefault(self.fold, []).extend(codes)
            # The first clean image is correct. The second is an incorrect Local prediction.
            return np.eye(5)[[1 if code in {10, 20} else 4 for code in codes]]

    monkeypatch.setattr(evaluate.data, "load_cache", load_cache)
    monkeypatch.setattr(evaluate.data, "corrupt", lambda images, kind, severity: images + 1)
    monkeypatch.setattr(evaluate, "load_classifier", lambda path: Classifier(int(path.stem.split("_")[1])))
    monkeypatch.setattr(train, "training_id", lambda *args: "verified")
    monkeypatch.setattr(train, "read_fold", lambda *args: {"verified": True})
    return loads, seen


def test_default_uses_only_cid_held_out_folds(monkeypatch, tmp_path):
    loads, seen = setup_cid(monkeypatch, tmp_path)
    evaluate.main([])
    assert loads == ["cid"]
    assert seen == {0: [10], 1: [20]}
    assert (config.RESULTS_DIR / "figures" / "cid_5-way_confusion.png").is_file()


def test_robustness_writes_only_cid_results(monkeypatch, tmp_path):
    loads, seen = setup_cid(monkeypatch, tmp_path)
    evaluate.main(["--robustness"])
    assert loads == ["cid"] and seen == {0: [10, 11], 1: [20, 21]}
    with (config.RESULTS_DIR / "metrics.csv").open() as source:
        row, = csv.DictReader(source)
    assert (row["dataset"], row["scoring"], float(row["macro_f1"])) == ("cid", "5-way/dark-1", 0.5)
    assert (config.RESULTS_DIR / "figures" / "robustness.png").is_file()


def test_gradcam_uses_each_examples_held_out_model(monkeypatch, tmp_path):
    setup_cid(monkeypatch, tmp_path)
    calls = []

    def gradcam(classifier, image, label):
        calls.append((classifier.fold, image[0, 0, 0], label))
        return np.ones(image.shape[:2])

    monkeypatch.setattr(evaluate, "gradcam", gradcam)
    evaluate.main(["--figures"])
    assert set(calls) == {(0, 10, 1), (1, 20, 1)}
    local = config.RESULTS_DIR / "figures" / "local"
    assert (local / "gradcam.png").is_file()
    notes = local / "error_annotations.csv"
    with notes.open() as source:
        row, = csv.DictReader(source)
    assert row["path"] == "two.jpg"
    row["likely_cause"] = "pose"
    with notes.open("w") as target:
        writer = csv.DictWriter(target, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    evaluate.main(["--figures"])
    with notes.open() as source:
        assert list(csv.DictReader(source)) == [row]


def test_unverified_cid_fold_blocks_evaluation(monkeypatch, tmp_path):
    setup_cid(monkeypatch, tmp_path)
    monkeypatch.setattr(train, "read_fold", lambda root, fold, run_id, size: None if fold == 1 else {})
    with pytest.raises(ValueError, match="CID fold 1"):
        evaluate.main([])
