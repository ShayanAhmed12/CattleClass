import numpy as np
import pandas as pd
import pytest

from cattle_breed import traits


def test_join_keeps_original_labels_and_distinguishes_valid_age_from_eligibility(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("sku,age_in_year,weight_in_kg\nAB 1,2,180\nAB2,3,\n", encoding="utf-8")
    index = pd.DataFrame({
        "path": ["CID/AB2/a.jpg", "CID/AB1/b.jpg"],
        "animal": ["linked", "linked"],
        "fold": [1, 1],
    })
    labels = traits.join_cid_targets(index, metadata)
    assert labels["sku"].tolist() == ["AB2", "AB1"]
    assert labels["age_months"].tolist() == [36, 24]
    assert labels["age_valid"].tolist() == [True, True]
    assert labels["age_eligible"].tolist() == [False, True]
    assert labels["weight_valid"].tolist() == [False, True]
    assert labels["group_id"].tolist() == ["linked", "linked"]


def test_join_rejects_conflicts_and_groups_crossing_folds(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("sku,age_in_year,weight_in_kg\nA B,2,180\nAB,2,200\n", encoding="utf-8")
    index = pd.DataFrame({"path": ["AB/a.jpg"], "animal": ["g"], "fold": [0]})
    with pytest.raises(ValueError, match="Conflicting CID trait labels"):
        traits.join_cid_targets(index, path)
    path.write_text("sku,age_in_year,weight_in_kg\nAB,2,180\n", encoding="utf-8")
    split = pd.concat([index, index.assign(path="AB/b.jpg", fold=1)], ignore_index=True)
    with pytest.raises(ValueError, match="cross folds"):
        traits.join_cid_targets(split, path)


def test_baselines_only_fit_development_and_representatives_are_deterministic():
    index = pd.DataFrame({
        "path": ["C/z.jpg", "B/a.jpg", "A/a.jpg", "C/a.jpg"],
        "sku": ["C", "B", "A", "C"],
        "fold": [1, 0, 0, 1],
        "weight_kg": [900, 200, 100, 900],
        "breed": ["Local", "Sahiwal", "Local", "Local"],
        "predicted_breed": ["Sahiwal", "Local", "Local", "Sahiwal"],
    })
    cases = traits.representatives(index)
    assert cases["path"].tolist() == ["A/a.jpg", "B/a.jpg", "C/a.jpg"]
    fitted = traits.weight_baselines(cases.query("fold == 0"), cases.query("fold == 1"))
    assert fitted == {"global_median": [150], "recorded_breed_median": [100], "predicted_breed_median": [200]}


def test_trait_metrics_have_separate_units_and_expose_missing_age_support():
    result = traits.score_traits([24, 24, 30], [24, 30, 24], task="age")
    assert result["balanced_accuracy"] == 0.25
    assert result["mae_months"] == 4
    assert traits.score_traits([24], [24], task="age")["balanced_accuracy"] is None
    result = traits.score_traits([100, 200], [105, 170], task="weight")
    assert result["mae_kg"] == 17.5
    assert result["bias_kg"] == -12.5
    assert result["within_10_percent"] == 0.5
    assert traits.score_traits([], [], task="weight") == {"n": 0}
    with pytest.raises(ValueError, match="finite"):
        traits.score_traits([100], [np.nan], task="weight")


def test_quarantine_disagreements_only_for_identical_source_photo(tmp_path):
    paths = [tmp_path / sku / "a.jpg" for sku in ("A", "B", "C")]
    for path, contents in zip(paths, (b"same", b"same", b"different")):
        path.parent.mkdir()
        path.write_bytes(contents)
    labels = pd.DataFrame({"sku": ["A", "B", "C"], "age_months": [24, 24, 30],
                           "weight_kg": [125, 150, 175], "weight_valid": [True] * 3,
                           "age_eligible": [True] * 3})
    result = traits.quarantine_identical(labels, paths)
    assert result["weight_eligible"].tolist() == [False, False, True]
    assert result["age_eligible"].tolist() == [True, True, True]


def test_grouped_inner_validation_has_both_bands_and_equal_animal_weights():
    from cattle_breed import trait_models

    rows = pd.DataFrame({
        "sku": ["A", "A", "B", "C", "D", "E", "F", "G", "H"],
        "group_id": ["1", "1", "2", "3", "4", "5", "6", "7", "8"],
        "age_months": [24, 24, 24, 24, 24, 30, 30, 30, 30],
    })
    train, val = trait_models.inner_split(rows, task="age")
    assert set(train.group_id).isdisjoint(val.group_id)
    assert set(train.age_months) == set(val.age_months) == {24, 30}
    weights = trait_models.animal_weights(rows)
    totals = pd.Series(weights).groupby(rows.sku).sum()
    np.testing.assert_allclose(totals, totals.iloc[0])


@pytest.mark.parametrize("task,target", [("age", [24, 30] * 4), ("weight", [150, 275] * 4)])
def test_frozen_keras_head_fits_and_reloads(tmp_path, task, target):
    from tensorflow import keras
    from cattle_breed import trait_models

    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    model, transform = trait_models.fit_head(features, target, np.ones(8), task=task, epochs=2)
    encoded = np.asarray(model(features, training=False))
    predicted = trait_models.decode_predictions(encoded, transform, task=task)
    artifact = tmp_path / "head.keras"
    model.save(artifact)
    reloaded = keras.models.load_model(artifact, compile=False)
    np.testing.assert_allclose(reloaded(features, training=False), encoded)
    assert predicted.shape == (8,)
    assert np.isfinite(predicted).all() and (predicted > 0).all()


def test_group_bootstrap_preserves_paired_errors_and_age_classes():
    weights = traits.paired_improvement(
        [100, 100, 200], [150, 150, 250], [125, 125, 225], ["g1", "g1", "g2"], task="weight",
    )
    assert weights == {"lower": 25., "upper": 25., "estimate": 25.}
    age = traits.paired_improvement([24, 30], [24, 24], [24, 30], ["g1", "g2"], task="age")
    assert age == {"lower": .5, "upper": .5, "estimate": .5}


def test_fold_records_reject_stale_or_misaligned_predictions(tmp_path):
    path = tmp_path / "fold.json"
    rows = pd.DataFrame({"path": ["A/a.jpg", "B/b.jpg"]}, index=[2, 7])
    traits.save_fold(path, "study1", rows, {"selected": [110, 130]}, {"selected": "ridge"})
    result = traits.read_fold(path, "study1", rows)
    assert result["predictions"]["selected"] == [110, 130]
    assert traits.read_fold(path, "study2", rows) is None
    assert traits.read_fold(path, "study1", rows.iloc[::-1]) is None
    path.write_text(path.read_text().replace("110", "999"))
    assert traits.read_fold(path, "study1", rows) is None


def test_ridge_candidate_is_serializable_and_retains_log_kg(tmp_path):
    from tensorflow import keras
    from cattle_breed import trait_release

    features = np.arange(32, dtype=np.float32).reshape(8, 4)
    targets = np.linspace(100, 250, 8)
    head = trait_release.fit_weight(features, targets, np.ones(8), alpha=10)
    path = tmp_path / "candidate.keras"
    head.save(path)
    reloaded = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(reloaded(features), head(features))
    assert np.all(np.exp(np.asarray(reloaded(features))) > 0)


def test_nonfinite_age_logits_are_not_decoded_as_a_valid_class():
    from cattle_breed import trait_models

    with pytest.raises(ValueError, match="finite"):
        trait_models.decode_predictions([[float("nan"), 0]], None, task="age")
