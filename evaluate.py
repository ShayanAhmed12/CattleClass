"""Evaluate the breed models on held-out CID animals."""
import argparse
import csv
from pathlib import Path

import numpy as np

from cattle_breed import config, data, load_classifier, metrics

def gradcam(classifier, image, class_idx):
    from cattle_breed import model

    return model.gradcam(classifier, image, class_idx)


def predict(classifier, images):
    return classifier.predict(images, batch_size=config.BATCH_SIZE, verbose=0)


def figure(*, figsize):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(plot)
    return plot


def save_figure(plot, path):
    with data.atomic_output(path) as temporary:
        plot.savefig(temporary, dpi=150)


def plot_confusion(matrix, directory):
    plot = figure(figsize=(6, 5))
    ax = plot.subplots()
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_title("Cid (5-way)")
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ticks = range(len(config.CLASSES))
    ax.set_xticks(ticks, config.CLASSES, rotation=35, ha="right")
    ax.set_yticks(ticks, config.CLASSES)
    for (row, column), count in np.ndenumerate(matrix):
        ax.text(column, row, f"{count:d}", ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    plot.colorbar(heatmap, ax=ax, label="Photos")
    save_figure(plot, directory / "cid_5-way_confusion.png")


def gallery_examples(fold, paths, images, labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ranked = np.argsort(-confidence, kind="stable")
    matches = predictions[ranked] == labels[ranked]
    positions = np.concatenate((ranked[matches][:3], ranked[~matches][:3]))
    return [
        (fold, paths[i], images[i].copy(), labels[i], predictions[i], confidence[i])
        for i in positions
    ]


def plot_robustness(scores, directory):
    plot = figure(figsize=(4 * len(config.CORRUPTIONS), 4))
    axes = plot.subplots(1, len(config.CORRUPTIONS), sharey=True, squeeze=False)[0]
    for ax, (kind, settings) in zip(axes, config.CORRUPTIONS.items()):
        severities = sorted(settings)
        keys = ["clean", *(f"{kind}-{severity}" for severity in severities)]
        ax.plot(range(len(keys)), [scores[key]["macro_f1"] for key in keys], marker="o")
        ax.set_xticks(range(len(keys)), ["Clean", *(f"Severity {i}" for i in severities)])
        ax.set_title(kind.capitalize())
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("CID macro-F1")
    save_figure(plot, directory / "robustness.png")


def save_galleries(examples, directory):
    ranked = sorted(examples, key=lambda example: -example[-1])
    correct = [example for example in ranked if example[3] == example[4]][:3]
    incorrect = [example for example in ranked if example[3] != example[4]][:3]
    examples = sorted(correct + incorrect, key=lambda example: example[0])
    gallery = figure(figsize=(13, 3.2))
    cam = figure(figsize=(13, 3.2))
    gallery.suptitle("Held-out CID predictions")
    cam.suptitle("Grad-CAM for the predicted breed")
    gallery_axes = gallery.subplots(1, 6).ravel()
    cam_axes = cam.subplots(1, 6).ravel()
    for ax in [*gallery_axes, *cam_axes]:
        ax.set_axis_off()
    current_fold = None
    for i, (fold, path, image, actual, predicted, confidence) in enumerate(examples):
        if fold != current_fold:
            classifier = load_classifier(config.MODELS_DIR / f"fold_{fold}.keras")
            current_fold = fold
        title = (f"Fold {fold} · {'Correct' if actual == predicted else 'Error'}\n"
                 f"True: {config.CLASSES[actual]}\n"
                 f"Pred: {config.CLASSES[predicted]} ({confidence:.0%})")
        for ax in (gallery_axes[i], cam_axes[i]):
            ax.imshow(image)
            ax.set_title(title, fontsize=8)
        heatmap = gradcam(classifier, image, int(predicted))
        cam_axes[i].imshow(heatmap, cmap="magma", alpha=0.45 * heatmap, vmin=0, vmax=1)
    save_figure(gallery, directory / "error_gallery.png")
    save_figure(cam, directory / "gradcam.png")

    notes_path = directory / "error_annotations.csv"
    fields = ("dataset", "path", "true_breed", "predicted_breed", "likely_cause")
    notes = {}
    if notes_path.exists():
        with notes_path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if row["dataset"] == "cid":
                    notes[tuple(row[field] for field in fields[:-1])] = row["likely_cause"]
    for _, path, _, actual, predicted, _ in examples:
        if actual != predicted:
            notes.setdefault(("cid", str(path), config.CLASSES[actual], config.CLASSES[predicted]), "")
    data.save_csv(notes_path, fields, (dict(zip(fields, (*key, cause))) for key, cause in notes.items()))


def verify_checkpoints(index, images):
    """Check every held-out model against the current prepared CID photos."""
    from train import read_fold, training_id

    folds = index["fold"].value_counts()
    if set(folds.index) != set(range(config.N_FOLDS)):
        raise ValueError("CID cache must contain every held-out fold. Rerun prepare_data.py.")
    run_id = training_id(index, images, 1.0)
    for fold, count in folds.items():
        if read_fold(config.MODELS_DIR, int(fold), run_id, int(count)) is None:
            raise ValueError(f"CID fold {fold} has no matching checkpoint. Rerun python train.py.")


def evaluate_cid(corruptions, directory, *, include_gallery):
    index, cached, labels = data.load_cache("cid")
    verify_checkpoints(index, cached)
    scores = {"clean": []}
    scores.update({f"{kind}-{severity}": [] for kind, severity in corruptions})
    examples = []
    for fold in range(config.N_FOLDS):
        selection = index["fold"].to_numpy() == fold
        images = data.center_crop_batch(cached[selection]).copy()
        fold_labels = labels[selection]
        classifier = load_classifier(config.MODELS_DIR / f"fold_{fold}.keras")
        probabilities = predict(classifier, images)
        scores["clean"].append(metrics.score(fold_labels, probabilities.argmax(axis=1)))
        if include_gallery:
            examples.extend(gallery_examples(fold, index.loc[selection, "path"].tolist(),
                                             images, fold_labels, probabilities))
        for kind, severity in corruptions:
            guess = predict(classifier, data.corrupt(images, kind, severity)).argmax(axis=1)
            scores[f"{kind}-{severity}"].append(metrics.score(fold_labels, guess))
        del classifier

    confusion = np.sum([fold["confusion_matrix"] for fold in scores["clean"]], axis=0)
    plot_confusion(confusion, directory)
    summaries = {setting: metrics.average_fold_scores(folds) for setting, folds in scores.items()}
    for kind, severity in corruptions:
        key = f"{kind}-{severity}"
        metrics.save_result("E5", "cid", f"5-way/{key}", summaries[key])
    print(f"CID held-out macro-F1: {summaries['clean']['macro_f1']:.3f}")
    if corruptions:
        plot_robustness(summaries, directory)
    if include_gallery:
        save_galleries(examples, directory / "local")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness", action="store_true", help="also evaluate corrupted images")
    parser.add_argument("--figures", action="store_true", help="also generate local error and Grad-CAM galleries")
    parser.add_argument("--traits", action="store_true", help="score saved CID age and weight predictions")
    parser.add_argument("--traits-data", type=Path, help="private field study JSON")
    args = parser.parse_args(argv)
    if args.traits_data and not args.traits:
        parser.error("--traits-data requires --traits")
    if args.traits:
        if args.robustness or args.figures:
            parser.error("--traits cannot be combined with --robustness or --figures")
        if args.traits_data:
            from cattle_breed import field_traits

            field_traits.evaluate_field(args.traits_data)
        else:
            from cattle_breed import traits

            traits.evaluate_cid_benchmark()
        return

    corruptions = [(kind, severity) for kind, settings in config.CORRUPTIONS.items()
                   for severity in settings] if args.robustness else []
    evaluate_cid(corruptions, config.RESULTS_DIR / "figures", include_gallery=args.figures)


if __name__ == "__main__":
    main()
