"""Evaluate and export the CID-only three-output research prototype."""
import json

import numpy as np
import pandas as pd

from . import config, data, load_classifier, metrics, trait_models, trait_release, traits
from .download import sha256

AGE_RIDGE_ALPHA = 100.0
AGE_TOLERANCE_MONTHS = 3.0


def joint_score(rows):
    """Score the same representative photo for all claimed outputs."""
    animals = traits.representatives(rows)
    eligible = animals[animals.age_eligible & animals.weight_eligible]
    age = eligible.age_prediction.to_numpy()
    weight = eligible.weight_prediction.to_numpy()
    available = (eligible.predicted_breed.isin(config.CLASSES)
                 & np.isfinite(age) & (age > 0) & np.isfinite(weight) & (weight > 0))
    correct = (available & eligible.predicted_breed.eq(eligible.breed)
               & eligible.age_prediction.sub(eligible.age_months).abs().le(AGE_TOLERANCE_MONTHS)
               & eligible.weight_prediction.sub(eligible.weight_kg).abs().le(0.10 * eligible.weight_kg))
    return {"animals": len(animals), "eligible": len(eligible), "excluded_references": len(animals) - len(eligible),
            "all_outputs": int(available.sum()), "joint_correct": int(correct.sum()),
            "joint_accuracy": float(correct.mean()) if len(eligible) else None,
            "age_tolerance_months": AGE_TOLERANCE_MONTHS, "weight_relative_tolerance": 0.10}


def _weight_predictions(rows, study_id):
    predictions, choices = [], []
    for fold in range(config.N_FOLDS):
        _, held_out = traits._fold_cases(rows, fold, "weight")
        path = config.MODELS_DIR / "traits" / f"weight-{fold}.json"
        payload = traits.read_fold(path, study_id, held_out)
        if payload is None:
            raise ValueError(f"Run python train.py --traits to verify the weight predictions for fold {fold}.")
        predictions.append(pd.Series(payload["predictions"]["selected"], index=held_out.index))
        choices.append(payload["selection"])
    return pd.concat(predictions), choices


def _fit_age(rows, features):
    return trait_release.fit_positive_target(features[rows.index], rows.age_months,
                                             trait_models.animal_weights(rows), alpha=AGE_RIDGE_ALPHA)


def _age_predictions(rows, features):
    eligible = rows[rows.age_eligible]
    predictions, baselines = [], []
    for fold, held_out in eligible.groupby("fold"):
        development = eligible[eligible.fold.ne(fold)]
        head = _fit_age(development, features)
        age = np.exp(np.asarray(head(features[held_out.index], training=False)).ravel())
        predictions.append(pd.Series(age, index=held_out.index))
        baselines.append(pd.Series(traits.representatives(development).age_months.median(), index=held_out.index))
    return pd.concat(predictions).sort_index(), pd.concat(baselines).sort_index()


def _age_scores(cases):
    error = cases.age_prediction.sub(cases.age_months)
    mae = error.abs()
    return {
        "mae_months": float(mae.mean()),
        "median_baseline_mae_months": float(cases.age_baseline.sub(cases.age_months).abs().mean()),
        "within_tolerance": float(mae.le(AGE_TOLERANCE_MONTHS).mean()),
        "observed_months": [float(cases.age_months.min()), float(cases.age_months.max())],
        "predicted_months": [float(cases.age_prediction.min()), float(cases.age_prediction.max())],
        "by_recorded_age": {f"{age:g}": {"animals": len(part), "mae_months": float(mae.loc[part.index].mean())}
                            for age, part in cases.groupby("age_months")},
    }


def preview_path():
    return config.MODELS_DIR / "traits" / "cid-research.keras"


def load_preview():
    path = preview_path()
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (record.get("stage") != "cid_research" or record.get("field_approved") is not False
            or record.get("breeds") != list(config.CLASSES) or record.get("age_output") != "months"
            or record.get("sha256") != sha256(path)):
        raise ValueError("The CID research prototype is incompatible or does not match its checksum.")
    return load_classifier(path), record


def _export(rows, pixels, features, breed_id, weight_choices, report):
    from tensorflow import keras

    classifier_path = config.MODELS_DIR / "final.keras"
    entry = json.loads(classifier_path.with_suffix(".json").read_text(encoding="utf-8"))
    if entry.get("run_id") != breed_id or entry.get("model_hash") != sha256(classifier_path):
        raise ValueError("Run python train.py --final for the locked CID folds.")
    if any(choice["selected"] != "frozen_ridge" for choice in weight_choices):
        raise ValueError("The locked CID prototype expects log-weight Ridge on every fold.")
    alpha = float(np.median([choice["settings"]["frozen_ridge"] for choice in weight_choices]))

    age, weight = rows[rows.age_eligible], rows[rows.weight_eligible]
    age_head = _fit_age(age, features)
    weight_head = trait_release.fit_weight(features[weight.index], weight.weight_kg,
                                          trait_models.animal_weights(weight), alpha=alpha)
    classifier, backbone = load_classifier(classifier_path), traits._backbone()
    inputs = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    features = backbone(inputs, training=False)
    model = keras.Model(inputs, {
        "breed": classifier(inputs, training=False),
        "age_months": keras.layers.Activation("exponential")(age_head(features, training=False)),
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
        "breeds": list(config.CLASSES), "age_output": "months",
        "ridge_alpha": alpha, "age_ridge_alpha": AGE_RIDGE_ALPHA, "benchmark": report,
    })
    print(f"Saved verified research prototype to {path}", flush=True)


def main():
    rows, pixels, breed_id, study_id = traits._study()
    rows = traits.quarantine_identical(rows.assign(age_eligible=rows.age_valid), data.build_index("cid").path)
    features = traits.frozen_embeddings(pixels, breed_id, config.MODELS_DIR / "traits/cid-features.npz")
    age, baseline = _age_predictions(rows, features)
    weight, weight_choices = _weight_predictions(rows, study_id)
    scored = rows.assign(age_prediction=age, age_baseline=baseline, weight_prediction=weight)
    report = joint_score(scored)
    cases = traits.representatives(scored)
    cases = cases[cases.age_eligible & cases.weight_eligible]
    breed = metrics.score(cases.breed.map(config.CLASS_TO_IDX), cases.predicted_breed.map(config.CLASS_TO_IDX))
    weight = traits.score_traits(cases.weight_kg, cases.weight_prediction, task="weight")
    report.update(study_id=study_id, breed_macro_f1=breed["macro_f1"],
                  age=_age_scores(cases), weight_mae_kg=weight["mae_kg"])
    _export(rows, pixels, features, breed_id, weight_choices, report)
    traits.write_summary(config.RESULTS_DIR / "cid_joint.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
