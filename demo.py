"""Run the breed, research or controlled field photo demo."""
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

PAGE_STYLE = """
.gradio-container { max-width: 1080px !important; margin-inline: auto !important; padding-inline: 24px !important; }
.gradio-container h1 { text-align: left !important; margin: 28px 0 8px; }
#photo { min-width: 0 !important; }
@media (max-width: 550px) {
    .gradio-container { padding-inline: 16px !important; }
    .gradio-container h1 { margin-top: 14px; }
}
"""


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


def photo_interface(fn, title, action, description, *, estimates=False, confirmation=None, about=BREED_SCOPE):
    """Share the same upload, actions and result widgets across the three modes."""
    inputs = [gr.Image(label="Cattle photo", sources=["upload"], type="filepath", image_mode=None,
                       height=300, elem_id="photo")]
    if confirmation:
        inputs.append(gr.Checkbox(label=confirmation))
    outputs = [gr.Label(label="Breed scores")]
    if estimates:
        outputs.extend([gr.Textbox(label="Age estimate"), gr.Textbox(label="Weight estimate")])
    app = gr.Interface(
        fn=fn, inputs=inputs, outputs=outputs, title=title, description=description,
        submit_btn=action, clear_btn="Reset", flagging_mode="never", analytics_enabled=False,
    )
    with app, gr.Accordion("About these estimates", open=False):
        gr.Markdown(about)
    return app


def create_field_demo(cohort, capture) -> gr.Interface:
    def run(photo_path, capture_confirmed):
        return predict_field(photo_path, capture_confirmed, cohort=cohort, capture=capture)

    return photo_interface(run, "CattleClass field assessment", "Assess photo",
                           "Upload a photo of an enrolled animal.", estimates=True,
                           confirmation=f"I confirmed the animal and the {capture} capture protocol",
                           about=BREED_SCOPE + " Age and weight appear when field approval is available.")


def create_demo() -> gr.Interface:
    return photo_interface(predict, "CattleClass breed classifier", "Identify breed",
                           "Upload a clear photo of one animal to see its three highest breed scores.")


def create_research_demo() -> gr.Interface:
    from cattle_breed import research

    load_bundle = lru_cache(maxsize=1)(research.load_preview)

    def run(photo_path):
        batch = image_batch(photo_path)
        try:
            classifier, record = load_bundle()
            breed, age, kg = research.predict_preview(classifier, batch)
            error = float(record["benchmark"]["weight_mae_kg"])
            if not np.isfinite(error) or error < 0:
                raise ValueError("Invalid benchmark error.")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise gr.Error("Research model unavailable. Run `python -m cattle_breed.research`.") from exc
        return (breed_scores(breed), f"About {age:.1f} months",
                f"About {kg:.0f} kg · average held-out error {error:.1f} kg on CID")

    return photo_interface(
        run, "CattleClass", "Estimate traits",
        "Upload a clear side photo of one animal to estimate breed, age, and weight. "
        "Age and weight are research estimates from CID.", estimates=True,
        about=BREED_SCOPE + "\n\nCID age records span 24–36 months, mostly at 24 and 30 months. "
        "The displayed weight error is the CID average. Age and weight accuracy on other farms remains unverified.",
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
        app = create_field_demo(args.traits_cohort, args.traits_capture)
    else:
        if args.traits_cohort:
            parser.error("--traits-cohort requires --traits")
        app = create_research_demo() if args.research else create_demo()
    theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="stone", font="system-ui")
    theme.set(body_background_fill="#f6f7f2", block_label_text_color="#16644c",
              button_primary_background_fill="#16644c",
              button_primary_background_fill_hover="#104f3b", button_primary_text_color="white")
    app.launch(share=args.share, theme=theme, css=PAGE_STYLE, footer_links=[])


if __name__ == "__main__":
    main()
