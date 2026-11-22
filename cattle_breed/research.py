"""Evaluate and export the CID-only three-output research prototype."""
import json

import numpy as np
import pandas as pd

from . import config, data, load_classifier, metrics, trait_models, trait_release, traits
from .download import sha256


def joint_score(rows):
    """Score the same representative photo for all claimed outputs."""
    animals = traits.representatives(rows)
    eligible = animals[animals.age_eligible & animals.weight_eligible]
    weight = eligible.weight_prediction.to_numpy()
    available = (eligible.predicted_breed.isin(config.CLASSES)
                 & eligible.age_prediction.isin(traits.AGE_BANDS) & np.isfinite(weight) & (weight > 0))
    correct = (available & eligible.predicted_breed.eq(eligible.breed)
               & eligible.age_prediction.eq(eligible.age_months)
               & eligible.weight_prediction.sub(eligible.weight_kg).abs().le(0.10 * eligible.weight_kg))
    return {"animals": len(animals), "eligible": len(eligible), "excluded_references": len(animals) - len(eligible),
            "all_outputs": int(available.sum()), "joint_correct": int(correct.sum()),
            "joint_accuracy": float(correct.mean()) if len(eligible) else None,
            "weight_relative_tolerance": 0.10}


def _held_out(rows, study_id):
    # Each prediction artifact binds to the original, ordered row indices of its held-out fold.
    outputs = {}
    settings = {}
    for task in ("age", "weight"):
        predictions, choices = [], []
        for fold in range(config.N_FOLDS):
            _, held_out = traits._fold_cases(rows, fold, task)
            payload = traits.read_fold(config.MODELS_DIR / "traits" / f"{task}-{fold}.json", study_id, held_out)
            if payload is None:
                raise ValueError(f"Run python train.py --traits to verify the {task} predictions for fold {fold}.")
            predictions.append(pd.Series(payload["predictions"]["selected"], index=held_out.index))
            choices.append(payload["selection"])
        outputs[f"{task}_prediction"] = pd.concat(predictions)
        settings[task] = choices
    return rows.assign(**outputs), settings


def preview_path():
    return config.MODELS_DIR / "traits" / "cid-research.keras"


def load_preview():
    path = preview_path()
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (record.get("stage") != "cid_research" or record.get("field_approved") is not False
            or record.get("breeds") != list(config.CLASSES) or record.get("age_months") != list(traits.AGE_BANDS)
            or record.get("sha256") != sha256(path)):
        raise ValueError("The CID research prototype is incompatible or does not match its checksum.")
    return load_classifier(path), record


def _export(rows, pixels, breed_id, settings, report):
    from tensorflow import keras

    classifier_path = config.MODELS_DIR / "final.keras"
    entry = json.loads(classifier_path.with_suffix(".json").read_text(encoding="utf-8"))
    if entry.get("run_id") != breed_id or entry.get("model_hash") != sha256(classifier_path):
        raise ValueError("Run python train.py --final for the locked CID folds.")
    choices = settings["weight"]
    if any(choice["selected"] != "frozen_ridge" for choice in choices):
        raise ValueError("The locked CID prototype expects log-weight Ridge on every fold.")
    alpha = float(np.median([choice["settings"]["frozen_ridge"] for choice in choices]))
    epochs = int(np.median([choice["settings"]["frozen_keras"] for choice in settings["age"]]))

    features = traits.frozen_embeddings(pixels, breed_id, config.MODELS_DIR / "traits/cid-features.npz")
    age, weight = rows[rows.age_eligible], rows[rows.weight_eligible]
    age_head, _ = trait_models.fit_head(features[age.index], age.age_months, trait_models.animal_weights(age),
                                       task="age", epochs=epochs)
    weight_head = trait_release.fit_weight(features[weight.index], weight.weight_kg,
                                          trait_models.animal_weights(weight), alpha=alpha)
    classifier, backbone = load_classifier(classifier_path), traits._backbone()
    inputs = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    features = backbone(inputs, training=False)
    model = keras.Model(inputs, {
        "breed": classifier(inputs, training=False),
        "age": keras.layers.Softmax()(age_head(features, training=False)),
        "log_kg": weight_head(features, training=False),
    })
    # Cover different photographs while keeping the reload test small.
    batch = data.center_crop_batch(pixels[np.linspace(0, len(pixels) - 1, 5, dtype=int)])
    path = preview_path()
    with data.atomic_output(path) as temporary:
        model.save(temporary)
        restored = load_classifier(temporary)
        after = restored(batch, training=False)
        for key, value in model(batch, training=False).items():
            np.testing.assert_allclose(after[key], value, rtol=1e-5, atol=1e-6)
    traits.write_summary(path.with_suffix(".json"), {
        "stage": "cid_research", "field_approved": False, "sha256": sha256(path),
        "breeds": list(config.CLASSES), "age_months": list(traits.AGE_BANDS),
        "ridge_alpha": alpha, "age_epochs": epochs, "benchmark": report,
    })
    print(f"Saved verified research prototype to {path}", flush=True)


def main():
    rows, pixels, breed_id, study_id = traits._study()
    scored, settings = _held_out(rows, study_id)
    report = joint_score(scored)
    cases = traits.representatives(scored)
    cases = cases[cases.age_eligible & cases.weight_eligible]
    breed = metrics.score(cases.breed.map(config.CLASS_TO_IDX), cases.predicted_breed.map(config.CLASS_TO_IDX))
    age = traits.score_traits(cases.age_months, cases.age_prediction, task="age")
    weight = traits.score_traits(cases.weight_kg, cases.weight_prediction, task="weight")
    report.update(study_id=study_id, breed_macro_f1=breed["macro_f1"],
                  age_balanced_accuracy=age["balanced_accuracy"],
                  age_recall_by_band=dict(zip(map(str, traits.AGE_BANDS), age["recall"])),
                  weight_mae_kg=weight["mae_kg"])
    _export(rows, pixels, breed_id, settings, report)
    traits.write_summary(config.RESULTS_DIR / "cid_joint.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
