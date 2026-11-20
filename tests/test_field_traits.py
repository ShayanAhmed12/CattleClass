import numpy as np
import pandas as pd
import pytest

from cattle_breed import field_traits


def test_field_partitions_reject_shared_animals_and_test_sites(tmp_path):
    def rows(animal, site):
        return pd.DataFrame({"animal_id": [animal], "site_id": [site], "path": [str(tmp_path / animal)]})

    with pytest.raises(ValueError, match="animal"):
        field_traits.check_partitions(rows("a", "one"), rows("a", "two"))
    with pytest.raises(ValueError, match="site"):
        field_traits.check_partitions(rows("a", "one"), rows("b", "one"), rows("c", "one"))
    field_traits.check_partitions(rows("a", "one"), rows("b", "one"), rows("c", "two"))


def test_conformal_quantile_uses_finite_sample_index_and_support():
    assert field_traits.calibration_cutoff(np.arange(1, 11), .1) == 10
    with pytest.raises(ValueError, match="calibration"):
        field_traits.calibration_cutoff([.1, .2], .1)


def test_field_report_counts_abstention_and_unverified_labels_in_separate_denominators():
    frame = pd.DataFrame({
        "target": [100.0, 210.0, 300.0, np.nan],
        "eligible": [True, True, False, True],
        "site_id": ["held"] * 4,
    })
    policy = {"max_interval_width": 30, "min_display_rate": .9, "min_coverage": .8,
              "min_reference_count": 2, "max_mae_kg": 20}
    results = field_traits.field_report(frame, [100, 180, 300, 400], 10, "weight", policy)
    assert results["attempts"] == 4
    assert results["references"] == 3
    assert results["display_rate"] == .75
    assert results["coverage_displayed"] == .5
    assert results["covered_per_reference"] == pytest.approx(1 / 3)
    assert results["approved"] is False


def test_demo_accepts_only_an_independently_approved_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import json
    from cattle_breed import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    release = tmp_path / "traits" / "approved"
    release.mkdir(parents=True)
    path = release / "program-ordinary-weight.json"
    candidate = tmp_path / "traits" / "field" / ("a" * 64) / "candidate.keras"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"weights")
    metadata = {"cohort": "program", "task": "weight", "capture": "ordinary",
                "study_id": "a" * 64, "preprocessing": "cattleclass-center-v1",
                "stage": "research", "field_approved": False, "field_report": {"approved": True},
                "model_sha256": hashlib.sha256(b"weights").hexdigest(),
                "policy": {"max_interval_width": 50}, "cutoff": 10}
    path.write_text(json.dumps(metadata))
    assert field_traits.release_for("program", "ordinary", "weight") is None
    path.write_text(json.dumps({**metadata, "stage": "field", "field_approved": True}))
    assert field_traits.release_for("program", "ordinary", "weight")[0] == candidate
    candidate.write_bytes(b"changed")
    assert field_traits.release_for("program", "ordinary", "weight") is None


def test_blank_attempts_do_not_create_a_false_image_overlap():
    def rows(animal, site):
        return pd.DataFrame({"animal_id": [animal], "site_id": [site], "path": [""]})

    field_traits.check_partitions(rows("a", "one"), rows("b", "one"), rows("c", "two"))


def test_controlled_demo_keeps_breed_when_trait_unapproved(monkeypatch):
    import demo

    monkeypatch.setattr(demo, "image_batch", lambda _: np.zeros((1, 224, 224, 3)))
    monkeypatch.setattr(demo, "predict_breed", lambda _: {"Local": .8})
    monkeypatch.setattr(field_traits, "release_for", lambda *args: None)
    assert demo.predict_field("cow.jpg", True, cohort="program", capture="ordinary") == (
        {"Local": .8}, "Age estimate unavailable", "Weight estimate unavailable",
    )
    assert demo.predict_field("cow.jpg", False, cohort="program", capture="ordinary") == (
        {"Local": .8}, "Confirm the capture protocol", "Confirm the capture protocol",
    )


@pytest.mark.parametrize("task", ["weight", "age"])
def test_synthetic_field_workflow_fits_locks_scores_and_withholds(tmp_path, monkeypatch, task):
    import hashlib
    import json
    from PIL import Image
    from tensorflow import keras
    from cattle_breed import config, traits

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(traits, "_backbone", lambda: keras.Sequential([
        keras.Input((224, 224, 3)), keras.layers.Rescaling(1 / 255),
        keras.layers.GlobalAveragePooling2D(),
    ]))
    for split_index, split in enumerate(field_traits.PARTITIONS):
        attempts = []
        for i in range(12):
            animal = f"{split}-{i}"
            path = f"{animal}.png"
            Image.new("RGB", (250, 250), (i * 17, 25 * split_index, 200)).save(tmp_path / path)
            attempts.append({"animal_id": animal, "site_id": "held" if split == "test" else "known",
                             "session_id": animal, "path": path, "capture": "ordinary", "quality_ok": "yes",
                             "photo_date": "2026-11-20T09:00:00Z", "weighed_at": "2026-11-20T09:05:00Z",
                             "weight_kg": str(100 + i * 15), "weight_source": "scale", "weight_equipment_id": "scale1",
                             "birth_date": "2024-06-01" if i % 2 else "2025-11-20", "birth_source": "record"})
        pd.DataFrame(attempts).to_csv(tmp_path / f"{split}.csv", index=False)
    spec = {
        "cohort": "synthetic", "intended_use": "synthetic test", "task": task, "capture": "ordinary",
        "development": "development.csv", "calibration": "calibration.csv", "test": "test.csv",
        "test_manifest_sha256": hashlib.sha256((tmp_path / "test.csv").read_bytes()).hexdigest(),
        "test_media_sha256": field_traits.test_media_checksum(tmp_path / "test.csv", "ordinary"),
        "age_edges_months": [0, 18, 36],
        "policy": {"max_interval_width": 300, "max_mae_kg": 150, "max_age_bands": 2,
                   "min_balanced_accuracy": .1, "min_per_band": 1, "min_reference_rate": .9,
                   "min_display_rate": .5, "min_coverage": .5, "min_reference_count": 99},
    }
    study = tmp_path / "field.json"
    study.write_text(json.dumps(spec))
    field_traits.train_field(study)
    report = field_traits.evaluate_field(study)
    assert report["attempts"] == report["references"] == 12
    assert report["approved"] is False
    assert field_traits.release_for("synthetic", "ordinary", task) is None
    with pytest.raises(ValueError, match="evaluated test cohort"):
        field_traits.train_field(study)
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(test_csv.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        field_traits.evaluate_field(study)


def test_reserved_media_binding_changes_when_two_photo_contents_swap(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"first")
    b.write_bytes(b"second")
    manifest = tmp_path / "test.csv"
    pd.DataFrame({"path": ["a.jpg", "b.jpg"], "capture": ["ordinary"] * 2}).to_csv(manifest, index=False)
    original = field_traits.test_media_checksum(manifest, "ordinary")
    a.write_bytes(b"second")
    b.write_bytes(b"first")
    assert field_traits.test_media_checksum(manifest, "ordinary") != original


def test_reserved_animals_cannot_be_reused_after_changing_the_policy(tmp_path, monkeypatch):
    from cattle_breed import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    field_traits.reserve_test_animals(["animal-1"], "weight", "ordinary", "locked-study")
    field_traits.reserve_test_animals(["animal-1"], "weight", "ordinary", "locked-study")
    with pytest.raises(ValueError, match="previously"):
        field_traits.reserve_test_animals(["animal-1", "animal-2"], "weight", "ordinary", "changed-policy")
    field_traits.reserve_test_animals(["animal-1"], "age", "ordinary", "another-task")


def test_displayed_weight_uses_the_rounded_range_for_gate_and_coverage():
    frame = pd.DataFrame({"target": [90.0], "eligible": [True], "site_id": ["held"]})
    policy = {"max_interval_width": 20, "min_display_rate": 1, "min_coverage": 1,
              "min_reference_count": 1, "max_mae_kg": 20}
    assert field_traits.field_report(frame, [100.4], 10, "weight", policy)["displayed"] == 0
    policy["max_interval_width"] = 21
    assert field_traits.field_report(frame, [100.4], 10, "weight", policy)["coverage_displayed"] == 1


def test_subgroup_approval_requires_every_declared_group():
    frame = pd.DataFrame({"target": [100.0, 100.0, 200.0], "eligible": [True, True, True],
                          "site_id": ["held"] * 3, "sex": ["male", "male", "female"]})
    policy = {"max_interval_width": 30, "min_display_rate": 1, "min_coverage": 1,
              "min_reference_count": 2, "max_mae_kg": 10}
    groups = field_traits.subgroup_reports(
        frame, np.array([100, 100, 200]), 10, "weight", policy,
        {"sex": ["male", "female", "unknown"]}, bands=None,
    )
    assert groups["sex"]["male"]["approved"] is True
    assert not groups["sex"]["female"]["approved"]
    assert groups["sex"]["unknown"] == {"attempts": 0, "approved": False}
