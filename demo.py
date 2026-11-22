"""Upload a cattle photo and see the classifier's three most likely breeds."""
import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from cattle_breed import config, data, load_classifier

BREED_SCOPE = (
    f"Supported breeds are {', '.join(config.CLASSES)}. "
    "Cattle outside these breeds can receive high scores."
)


@lru_cache(maxsize=1)
def _load_model(path: Path):
    if not path.is_file():
        raise gr.Error(f"No trained model found at {path}. Run `python train.py --final` first.")

    return load_classifier(path)


def image_batch(photo_path):
    if photo_path is None:
        raise gr.Error("Upload a cattle photo first.")
    try:
        rgb = data.load_image(photo_path)
    except data.ImageLoadError as error:
        raise gr.Error("Upload a readable image.") from error

    image = data.letterbox(rgb)
    return data.center_crop_batch(image[None, ...])


def breed_scores(probabilities):
    top_indices = np.argsort(-probabilities, kind="stable")[:3]
    return {config.CLASSES[index]: float(probabilities[index]) for index in top_indices}


def predict_breed(batch):
    classifier = _load_model(config.MODELS_DIR / "final.keras")
    return breed_scores(np.asarray(classifier(batch, training=False))[0])


def predict(photo_path: str | Path | None) -> dict[str, float]:
    return predict_breed(image_batch(photo_path))


def predict_field(photo_path, capture_confirmed, *, cohort, capture):
    from cattle_breed import field_traits

    batch = image_batch(photo_path)
    breed = predict_breed(batch)
    if not capture_confirmed:
        message = "Confirm the capture protocol"
        return breed, message, message
    results = []
    for task in ("age", "weight"):
        approval = field_traits.release_for(cohort, capture, task)
        estimate = None
        if approval:
            try:
                estimate = field_traits.describe_prediction(approval, batch, task)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                pass
        results.append(estimate or f"{task.capitalize()} estimate unavailable")
    return breed, *results


def create_field_demo(cohort, capture) -> gr.Interface:
    def run(photo_path, capture_confirmed):
        return predict_field(photo_path, capture_confirmed, cohort=cohort, capture=capture)

    return gr.Interface(
        fn=run,
        inputs=[
            gr.Image(label="Cattle photo", sources=["upload"], type="filepath", image_mode=None),
            gr.Checkbox(label=f"Operator confirmed the enrolled animal and {capture} capture protocol"),
        ],
        outputs=[gr.Label(label="Supported breed scores"), gr.Textbox(label="Age"), gr.Textbox(label="Weight")],
        title="CattleClass field assessment",
        description=BREED_SCOPE,
    )


def create_demo() -> gr.Interface:
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Cattle photo", sources=["upload"], type="filepath", image_mode=None),
        outputs=gr.Label(label="Supported breed scores"),
        title="Cattle Breed Classifier",
        description=BREED_SCOPE,
    )


def create_research_demo() -> gr.Interface:
    from cattle_breed import research, traits

    load_bundle = lru_cache(maxsize=1)(research.load_preview)

    def run(photo_path):
        batch = image_batch(photo_path)
        try:
            classifier, record = load_bundle()
            output = classifier(batch, training=False)
            breed = np.asarray(output["breed"])[0]
            age = np.asarray(output["age"])[0]
            with np.errstate(over="ignore", invalid="ignore"):
                kg = float(np.exp(np.asarray(output["log_kg"])[0, 0]))
            error = float(record["benchmark"]["weight_mae_kg"])
            if (breed.shape != (len(config.CLASSES),) or age.shape != (len(traits.AGE_BANDS),)
                    or not np.isfinite(np.r_[breed, age, kg, error]).all() or kg <= 0 or error < 0):
                raise ValueError("Invalid research outputs.")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise gr.Error("Research model unavailable. Run `python -m cattle_breed.research`.") from exc
        return (breed_scores(breed), f"{traits.AGE_BANDS[int(age.argmax())]} months (CID label)",
                f"About {kg:.0f} kg · average held-out error {error:.1f} kg on CID")

    return gr.Interface(
        fn=run,
        inputs=gr.Image(label="Cattle photo", sources=["upload"], type="filepath", image_mode=None),
        outputs=[gr.Label(label="Supported breed scores"), gr.Textbox(label="Research age label"),
                 gr.Textbox(label="Research weight estimate")],
        title="CattleClass research prototype",
        description=BREED_SCOPE + " Research outputs use CID's recorded labels. "
        "Age options are 24 and 30 months. Weight and age accuracy on independent farms remains unverified.",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link for Colab.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--traits", action="store_true", help="enable the local controlled field interface")
    mode.add_argument("--research", action="store_true", help="show breed, age and weight from the CID research bundle")
    parser.add_argument("--traits-cohort", help="field cohort assigned by the operator")
    parser.add_argument("--traits-capture", choices=("ordinary", "side", "side_marker"), default="ordinary")
    args = parser.parse_args(argv)
    if args.traits:
        if not args.traits_cohort or args.share:
            parser.error("--traits requires an enrolled cohort and a local session")
        create_field_demo(args.traits_cohort, args.traits_capture).launch(share=False)
    else:
        if args.traits_cohort:
            parser.error("--traits-cohort requires --traits")
        (create_research_demo() if args.research else create_demo()).launch(share=args.share)


if __name__ == "__main__":
    main()
