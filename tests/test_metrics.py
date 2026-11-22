import csv

import numpy as np
import pytest

from cattle_breed import metrics


def test_score_perfect_predictions():
    scores = metrics.score([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

    assert scores["n_images"] == 5
    assert scores["accuracy"] == 1.0
    assert scores["balanced_accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert all(f1 == 1.0 for f1 in scores["per_class_f1"].values())
    np.testing.assert_array_equal(scores["confusion_matrix"], np.eye(5, dtype=int))


def test_score_averages_only_classes_in_ground_truth():
    scores = metrics.score([0, 0, 1, 1], [0, 3, 0, 1])

    assert scores["accuracy"] == 0.5
    assert scores["balanced_accuracy"] == 0.5
    assert scores["macro_f1"] == pytest.approx(7 / 12)
    assert scores["per_class_f1"]["Holstein cross"] == 0.5
    assert scores["per_class_f1"]["Local"] == pytest.approx(2 / 3)
    assert scores["confusion_matrix"][0, 3] == 1
    assert scores["confusion_matrix"].sum() == 4
    assert metrics.score([1, 3], [1, 3])["macro_f1"] == 1.0


def test_save_result_appends_and_updates_by_experiment_dataset_and_scoring(tmp_path):
    path = tmp_path / "results" / "metrics.csv"
    scores = {
        "n_images": 4,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "macro_f1": 0.6,
    }
    metrics.save_result("E5", "cid", "5-way", scores, path=path)
    metrics.save_result("E5", "cid", "5-way/blur-1", scores, path=path)
    metrics.save_result(
        "E5", "cid", "5-way", {**scores, "macro_f1": 0.7, "macro_f1_std": 0.01}, path=path
    )

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == [
            "experiment", "dataset", "scoring", "n_images", "accuracy",
            "balanced_accuracy", "macro_f1", "macro_f1_std", "date",
        ]
        first, second = reader

    assert (first["experiment"], first["dataset"], first["scoring"]) == (
        "E5", "cid", "5-way"
    )
    assert first["n_images"] == "4"
    assert first["accuracy"] == first["balanced_accuracy"] == "0.5"
    assert first["macro_f1"] == "0.7"
    assert first["macro_f1_std"] == "0.01"
    assert first["date"]
    assert second["scoring"] == "5-way/blur-1"
    assert second["macro_f1"] == "0.6"
    assert second["macro_f1_std"] == ""
