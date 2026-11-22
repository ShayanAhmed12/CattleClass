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
        "sku": ["a", "a", "b", "c", "d"], "path": ["a-2", "a-1", "b-1", "c-1", "d-1"],
        "breed": ["Local"] * 5, "predicted_breed": ["Sindhi", "Local", "Local", "Local", "Local"],
        "age_months": [24, 24, 24, 30, 36], "age_prediction": [30, 24, 24, np.nan, 30],
        "weight_kg": [200] * 5, "weight_prediction": [100, 220, 221, 200, 200],
        "age_eligible": [True, True, True, True, False], "weight_eligible": [True] * 5,
    })
    result = joint_score(rows)
    assert result == {"animals": 4, "eligible": 3, "excluded_references": 1, "all_outputs": 2,
                      "joint_correct": 1, "joint_accuracy": 1 / 3, "weight_relative_tolerance": 0.1}


def test_research_demo_upload_shows_three_outputs(tmp_path, monkeypatch):
    import demo
    from cattle_breed import research
    from PIL import Image

    photo = tmp_path / "cow.png"
    Image.new("RGB", (400, 300), "brown").save(photo)

    def classifier(batch, training=False):
        assert batch.shape == (1, config.IMG_SIZE, config.IMG_SIZE, 3)
        return {"breed": [[0.1, 0.6, 0.1, 0.1, 0.1]], "age": [[0.2, 0.8]],
                "log_kg": [[np.log(202)]]}

    monkeypatch.setattr(research, "load_preview", lambda: (classifier, {"benchmark": {"weight_mae_kg": 22.5}}))
    interface = demo.create_research_demo()
    breed, age, weight = interface(str(photo))
    assert breed["label"] == "Local"
    assert "30 months" in age
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
