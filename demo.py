"""Upload a cattle photo and see the classifier's three most likely breeds."""
import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from cattle_breed import config, data


@lru_cache(maxsize=1)
def _load_model(path: Path):
    if not path.is_file():
        raise gr.Error(f"No trained model found at {path}. Run `python train.py --final` first.")

    from cattle_breed import model  # Registers the custom backbone before deserialization.
    from tensorflow import keras

    return keras.models.load_model(path, compile=False)


def predict(photo_path: str | Path | None) -> dict[str, float]:
    if photo_path is None:
        raise gr.Error("Upload a cattle photo first.")
    try:
        rgb = data.load_image(photo_path)
    except data.ImageLoadError as error:
        raise gr.Error("Upload a readable image.") from error

    image = data.center_crop(data.letterbox(rgb, config.CACHE_SIZE), config.IMG_SIZE)
    classifier = _load_model(config.MODELS_DIR / "final.keras")
    probabilities = np.asarray(classifier(image[None, ...], training=False))[0]
    top_indices = np.argsort(-probabilities, kind="stable")[:3]
    return {config.CLASSES[index]: float(probabilities[index]) for index in top_indices}


def create_demo() -> gr.Interface:
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Cattle photo", sources=["upload"], type="filepath", image_mode=None),
        outputs=gr.Label(label="Breed probabilities"),
        title="Cattle Breed Classifier",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link for Colab.")
    args = parser.parse_args(argv)
    create_demo().launch(share=args.share)


if __name__ == "__main__":
    main()
