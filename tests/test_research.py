import numpy as np
import pandas as pd
import pytest

from cattle_breed import config, data


def test_preparation_reuses_locked_folds_and_rejects_changed_index(tmp_path, monkeypatch):
    from prepare_data import locked_index

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    first = str(tmp_path / "one.jpg")
    second = str(tmp_path / "two.jpg")
    index = pd.DataFrame([(first, "cid", "a", "Local", 1), (second, "cid", "b", "Local", 1)],
                         columns=["path", "dataset", "animal", "label", "label_idx"])
    path = tmp_path / "folds.csv"
    data.portable_index(index).assign(fold=[3, 1]).to_csv(path, index=False)
    assert locked_index(index, path).fold.tolist() == [3, 1]
    with pytest.raises(ValueError, match="locked CID manifest"):
        locked_index(index.assign(animal=["a", "c"]), path)


def test_joint_score_uses_the_same_representative_photo_and_counts_missing_predictions():
    from cattle_breed.research import joint_score

    rows = pd.DataFrame({
        "sku": ["a", "a", "b", "c", "d", "e"], "path": ["a-2", "a-1", "b-1", "c-1", "d-1", "e-1"],
        "breed": ["Local"] * 6, "predicted_breed": ["Sindhi", *(["Local"] * 5)],
        "age_months": [24, 24, 24, 30, 36, 30], "age_prediction": [30, 27, 26.5, np.nan, 30, 26.5],
        "weight_kg": [200] * 6, "weight_prediction": [100, 220, 221, 200, 200, 200],
        "age_eligible": [True, True, True, True, False, True], "weight_eligible": [True] * 6,
    })
    result = joint_score(rows)
    assert result == {"animals": 5, "eligible": 4, "excluded_references": 1, "all_outputs": 3,
                      "joint_correct": 1, "joint_accuracy": 0.25,
                      "age_tolerance_months": 3.0, "weight_relative_tolerance": 0.1}


def test_research_demo_upload_shows_three_outputs(tmp_path, monkeypatch):
    import demo
    from cattle_breed import research
    from PIL import Image

    photo = tmp_path / "cow.png"
    Image.new("RGB", (400, 300), "brown").save(photo)

    def classifier(batch, training=False):
        assert batch.shape == (1, config.IMG_SIZE, config.IMG_SIZE, 3)
        return {"breed": [[0.1, 0.6, 0.1, 0.1, 0.1]], "age_months": [[27.4]],
                "log_kg": [[np.log(202)]]}

    monkeypatch.setattr(research, "load_preview", lambda: (classifier, {"benchmark": {"weight_mae_kg": 22.5}}))
    interface = demo.create_research_demo()
    breed, age, weight = interface(str(photo))
    assert breed["label"] == "Local"
    assert "27.4 months" in age
    assert "202 kg" in weight and "22.5 kg" in weight


def test_research_demo_reports_missing_bundle_and_unreadable_photo(tmp_path, monkeypatch):
    import demo
    import gradio as gr
    from gradio.exceptions import ComponentProcessingError
    from PIL import Image

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    picture = tmp_path / "valid.jpg"
    Image.new("RGB", (64, 64), "brown").save(picture)
    invalid = tmp_path / "invalid.jpg"
    invalid.write_text("not an image")
    interface = demo.create_research_demo()
    with pytest.raises(ComponentProcessingError, match="cannot identify image"):
        interface(str(invalid))
    with pytest.raises(gr.Error, match="cattle_breed.research"):
        interface(str(picture))


def test_numeric_age_fold_uses_every_positive_training_age_and_excludes_held_out_animals():
    from cattle_breed.research import _age_predictions

    records = pd.DataFrame({
        "fold": [0, 0, 1, 1], "sku": ["a", "b", "c", "d"], "path": ["a", "b", "c", "d"],
        "age_months": [8.0, 17.0, 24.0, 38.0], "age_eligible": [True] * 4,
    })
    inputs = np.asarray([[0.0], [1.0], [0.2], [0.8]])
    predicted, baseline = _age_predictions(records, inputs)
    assert baseline.to_list() == [31.0, 31.0, 12.5, 12.5]
    assert predicted.between(8, 38).all()
    assert not predicted.isin([24.0, 30.0]).any()
