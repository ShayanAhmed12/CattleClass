"""The Gradio demo uses synthetic images and model outputs in these tests."""
import sys
from types import SimpleNamespace

import gradio as gr
import numpy as np
import pytest

import demo
from cattle_breed import config


def test_gradio_upload_crops_and_shows_three_ranked_probabilities(monkeypatch, make_image):
    photo = make_image("cow.png", size=(240, 120), color=(200, 30, 10))
    model_paths = []

    def classify(batch, *, training):
        assert training is False
        assert batch.dtype == np.uint8
        expected = np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        expected[:, 52:172, :, :] = (200, 30, 10)
        np.testing.assert_array_equal(batch, expected)
        return np.array([[0.11, 0.40, 0.14, 0.05, 0.30]])

    def load_model(path):
        model_paths.append(path)
        return classify

    monkeypatch.setattr(demo, "_load_model", load_model)
    app = demo.create_demo()
    assert model_paths == []

    result = app(str(photo))

    assert model_paths == [config.MODELS_DIR / "final.keras"]
    assert result["label"] == "Local"
    assert result["confidences"] == [
        {"label": "Local", "confidence": 0.40},
        {"label": "Sindhi", "confidence": 0.30},
        {"label": "Red Chittagong", "confidence": 0.14},
    ]


def test_missing_model_explains_how_to_create_it(monkeypatch, make_image, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    with pytest.raises(gr.Error, match=r"final\.keras.*python train\.py --final"):
        demo.predict(make_image("cow.png"))


def test_model_loader_caches_inference_only_model(monkeypatch, tmp_path):
    path = tmp_path / "final.keras"
    path.touch()
    loaded_model = object()
    load_calls = []

    def load_model(filename, *, compile):
        load_calls.append((filename, compile))
        return loaded_model

    keras = SimpleNamespace(models=SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(keras=keras))
    monkeypatch.setitem(sys.modules, "cattle_breed.model", SimpleNamespace())

    assert demo._load_model(path) is loaded_model
    assert demo._load_model(path) is loaded_model
    assert load_calls == [(path, False)]


@pytest.mark.parametrize(("argv", "share"), [([], False), (["--share"], True)])
def test_main_shares_only_when_requested(monkeypatch, argv, share):
    launches = []

    def launch(*, share):
        launches.append(share)

    monkeypatch.setattr(demo, "create_demo", lambda: SimpleNamespace(launch=launch))
    monkeypatch.setenv("GRADIO_SHARE", "true")
    demo.main(argv)
    assert launches == [share]
