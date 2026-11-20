"""Evaluate trained cattle breed models on cached, reviewed datasets."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cattle_breed import config, data, load_classifier, metrics

EXTERNAL_EXPERIMENTS = {"mendeley": "E3", "kaggle": "E4"}


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


def plot_confusion(dataset, scoring, matrix, breeds, directory):
    class_indices = [config.CLASS_TO_IDX[breed] for breed in breeds]
    matrix = matrix[np.ix_(class_indices, class_indices)]
    plot = figure(figsize=(6, 5))
    ax = plot.subplots()
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_title(f"{dataset.capitalize()} ({scoring})")
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ticks = range(len(breeds))
    ax.set_xticks(ticks, breeds, rotation=35, ha="right")
    ax.set_yticks(ticks, breeds)
    for (row, column), count in np.ndenumerate(matrix):
        ax.text(column, row, f"{count:d}", ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    plot.colorbar(heatmap, ax=ax, label="Photos")
    save_figure(plot, directory / f"{dataset}_{scoring}_confusion.png")


def external_scores(dataset, labels, probs):
    allowed = config.DATASET_CLASSES[dataset]
    return {
        "5-way": metrics.score(labels, probs.argmax(axis=1)),
        "restricted": metrics.score(labels, metrics.restrict(probs, allowed)),
    }


def gallery_examples(dataset, index, images, labels, probs):
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    ranked = np.argsort(-confidence, kind="stable")
    matches = predictions[ranked] == labels[ranked]
    positions = np.concatenate((ranked[matches][:3], ranked[~matches][:3]))
    return [
        (dataset, index.iloc[i]["path"], images[i].copy(), labels[i], predictions[i], confidence[i])
        for i in positions
    ]


def evaluate_external(classifier, dataset, experiment, corruptions, figures_dir, include_gallery):
    index, cached, labels = data.load_cache(dataset)
    images = data.center_crop_batch(cached).copy()
    del cached
    probs = predict(classifier, images)
    clean = external_scores(dataset, labels, probs)
    per_class_rows = []
    for scoring, scores in clean.items():
        breeds = config.CLASSES if scoring == "5-way" else config.DATASET_CLASSES[dataset]
        metrics.save_result(experiment, dataset, scoring, scores)
        plot_confusion(dataset, scoring, scores["confusion_matrix"], breeds, figures_dir)
        per_class_rows.extend(
            {"experiment": experiment, "dataset": dataset, "scoring": scoring,
             "breed": breed, "f1": scores["per_class_f1"][breed]}
            for breed in breeds
        )

    robustness = {"clean": clean["5-way"]["macro_f1"]}
    for kind, severity in corruptions:
        scores_by_protocol = external_scores(
            dataset, labels, predict(classifier, data.corrupt(images, kind, severity))
        )
        setting = f"{kind}-{severity}"
        for scoring, scores in scores_by_protocol.items():
            metrics.save_result("E5", dataset, f"{scoring}/{setting}", scores)
        robustness[setting] = scores_by_protocol["5-way"]["macro_f1"]
    examples = gallery_examples(dataset, index, images, labels, probs) if include_gallery else []
    return per_class_rows, robustness, examples


def evaluate_cid(corruptions, figures_dir):
    index, cached, labels = data.load_cache("cid")
    fold_ids = index["fold"].to_numpy()
    if set(fold_ids) != set(range(config.N_FOLDS)):
        raise ValueError("CID cache must contain every held-out fold. Rerun prepare_data.py.")
    scores = {"clean": []}
    scores.update({f"{kind}-{severity}": [] for kind, severity in corruptions})
    for fold in range(config.N_FOLDS):
        selection = fold_ids == fold
        images = data.center_crop_batch(cached[selection]).copy()
        fold_labels = labels[selection]
        classifier = load_classifier(config.MODELS_DIR / f"fold_{fold}.keras")
        scores["clean"].append(metrics.score(fold_labels, predict(classifier, images).argmax(axis=1)))
        for kind, severity in corruptions:
            probs = predict(classifier, data.corrupt(images, kind, severity))
            scores[f"{kind}-{severity}"].append(metrics.score(fold_labels, probs.argmax(axis=1)))

    summaries = {setting: metrics.average_fold_scores(folds) for setting, folds in scores.items()}
    confusion = np.sum([fold["confusion_matrix"] for fold in scores["clean"]], axis=0)
    plot_confusion("cid", "5-way", confusion, config.CLASSES, figures_dir)
    for setting, summary in summaries.items():
        if setting != "clean":
            metrics.save_result("E5", "cid", f"5-way/{setting}", summary)
    return {setting: summary["macro_f1"] for setting, summary in summaries.items()}


def plot_robustness(results, figures_dir):
    plot = figure(figsize=(4 * len(config.CORRUPTIONS), 4))
    axes = plot.subplots(1, len(config.CORRUPTIONS), sharey=True, squeeze=False)[0]
    for ax, (kind, settings) in zip(axes, config.CORRUPTIONS.items()):
        severities = sorted(settings)
        keys = ["clean", *(f"{kind}-{severity}" for severity in severities)]
        for dataset, scores in results.items():
            label = "CID" if dataset == "cid" else dataset.capitalize()
            ax.plot(range(len(keys)), [scores[key] for key in keys], marker="o", label=label)
        ax.set_xticks(range(len(keys)), ["Clean", *(f"Severity {i}" for i in severities)])
        ax.set_title(kind.capitalize())
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("5-way macro-F1")
    axes[-1].legend()
    save_figure(plot, figures_dir / "robustness.png")


def save_galleries(classifier, examples, directory):
    gallery = figure(figsize=(15, 5.8))
    cam = figure(figsize=(15, 5.8))
    gallery.suptitle("External test predictions")
    cam.suptitle("Grad-CAM for the predicted breed")
    gallery_axes = gallery.subplots(2, 6).ravel()
    cam_axes = cam.subplots(2, 6).ravel()
    for ax in [*gallery_axes, *cam_axes]:
        ax.set_axis_off()
    for i, (dataset, path, image, actual, predicted, confidence) in enumerate(examples):
        title = (f"{dataset.capitalize()} · {'Correct' if actual == predicted else 'Error'}\n"
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
                notes[tuple(row[field] for field in fields[:-1])] = row["likely_cause"]
    for dataset, path, _, actual, predicted, _ in examples:
        if actual != predicted:
            notes.setdefault((dataset, str(path), config.CLASSES[actual], config.CLASSES[predicted]), "")
    data.save_csv(notes_path, fields, (dict(zip(fields, (*key, cause))) for key, cause in notes.items()))


def verify_checkpoints(*, include_folds):
    """Use only model artifacts produced for the current prepared CID data."""
    from cattle_breed.download import sha256
    from train import read_fold, training_id

    root = config.MODELS_DIR
    final = root / "final.keras"
    try:
        record = json.loads((root / "final.json").read_text(encoding="utf-8"))
        checksum = sha256(final)
    except (OSError, ValueError) as error:
        raise ValueError("Finish `python train.py --final` before evaluating.") from error
    index, images, _ = data.load_cache("cid")
    run_id = training_id(index, images, 1.0)
    if record.get("run_id") != run_id or record.get("model_hash") != checksum:
        raise ValueError("Final checkpoint does not match the current CID cache. Rerun training.")
    if include_folds:
        folds = index["fold"].value_counts()
        if any(fold not in folds or read_fold(root, fold, run_id, int(folds[fold])) is None
               for fold in range(config.N_FOLDS)):
            raise ValueError("E5 requires all matching CID test-fold checkpoints. Rerun training.")


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

    verify_checkpoints(include_folds=args.robustness)
    directory = config.RESULTS_DIR / "figures"
    corruptions = [(kind, severity) for kind, settings in config.CORRUPTIONS.items()
                   for severity in settings] if args.robustness else []
    classifier = load_classifier(config.MODELS_DIR / "final.keras")
    per_class = []
    robustness = {}
    examples = []
    for dataset, experiment in EXTERNAL_EXPERIMENTS.items():
        rows, robustness[dataset], selected = evaluate_external(
            classifier, dataset, experiment, corruptions, directory, args.figures
        )
        per_class.extend(rows)
        examples.extend(selected)
    data.save_csv(
        config.RESULTS_DIR / "per_class_f1.csv",
        ("experiment", "dataset", "scoring", "breed", "f1"),
        per_class,
    )

    if args.figures:
        save_galleries(classifier, examples, directory / "local")
    if args.robustness:
        del classifier
        robustness["cid"] = evaluate_cid(corruptions, directory)
        plot_robustness(robustness, directory)


if __name__ == "__main__":
    main()
