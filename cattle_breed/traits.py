"""Grouped CID research benchmarks for recorded age bands and weight."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data
from .download import sha256

AGE_BANDS = (24, 30)


def _sku(path):
    return "".join(Path(path).parent.name.split())


def _positive(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values) & (values > 0))


def join_cid_targets(index, metadata_csv):
    """Join by original SKU while preserving the cache order and split groups."""
    raw = pd.read_csv(metadata_csv, dtype=str, encoding="utf-8-sig")
    labels = pd.DataFrame({
        "sku": raw["sku"].str.replace(r"\s+", "", regex=True),
        "age_months": _positive(raw["age_in_year"]) * 12,
        "weight_kg": _positive(raw["weight_in_kg"]),
    }).drop_duplicates()
    if labels["sku"].duplicated().any():
        raise ValueError("Conflicting CID trait labels for a normalized SKU.")
    if index.path.duplicated().any():
        raise ValueError("Duplicate photo paths in the CID cache.")
    aligned = index.rename(columns={"animal": "group_id"}).copy()
    aligned["sku"] = aligned.path.map(_sku)
    if aligned[["group_id", "fold"]].isna().any().any():
        raise ValueError("CID traits require a group and fold for every photo.")
    for key in ("sku", "group_id"):
        if aligned.groupby(key).fold.nunique().gt(1).any():
            raise ValueError("CID animals or duplicate-linked groups cross folds.")
    unknown = set(aligned.sku) - set(labels.sku)
    if unknown:
        raise ValueError(f"Missing CID metadata for {len(unknown)} indexed SKUs.")
    aligned = aligned.join(labels.set_index("sku"), on="sku", validate="many_to_one")
    aligned["age_valid"] = aligned.age_months.notna()
    aligned["age_eligible"] = aligned.age_months.isin(AGE_BANDS)
    aligned["weight_valid"] = aligned.weight_kg.notna()
    aligned["weight_eligible"] = aligned.weight_valid
    return aligned


def quarantine_identical(labels, raw_paths):
    """Exclude conflicting targets when original photos have identical bytes."""
    aligned = labels.copy()
    animals = aligned.drop_duplicates("sku").set_index("sku")
    same_image = defaultdict(set)
    for path in raw_paths:
        sku = _sku(path)
        if sku in animals.index:
            same_image[sha256(path)].add(sku)
    for target, eligibility in (("age_months", "age_eligible"), ("weight_kg", "weight_eligible")):
        excluded = set()
        for skus in same_image.values():
            if len(skus) > 1 and animals.loc[sorted(skus), target].nunique() > 1:
                excluded.update(skus)
        if eligibility not in aligned:
            aligned[eligibility] = aligned[target].notna()
        aligned.loc[aligned.sku.isin(excluded), eligibility] = False
    return aligned


def representatives(index):
    """Choose one photo per source animal by a fixed path ordering."""
    return index.sort_values("path", kind="stable").drop_duplicates("sku")


def weight_baselines(development, test):
    median = float(development.weight_kg.median())
    by_breed = development.groupby("breed").weight_kg.median()
    return {
        "global_median": [median] * len(test),
        "recorded_breed_median": test.breed.map(by_breed).fillna(median).tolist(),
        "predicted_breed_median": test.predicted_breed.map(by_breed).fillna(median).tolist(),
    }


def score_traits(y_true, y_pred, *, task):
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if task not in {"age", "weight"}:
        raise ValueError(f"Unknown trait: {task}")
    if (true.ndim != 1 or true.shape != pred.shape or
            not (np.isfinite(true).all() and np.isfinite(pred).all())):
        raise ValueError("Trait targets and predictions must be aligned, finite vectors.")
    if not true.size:
        return {"n": 0}
    error = pred - true
    absolute = np.abs(error)
    if task == "weight":
        if (true <= 0).any() or (pred <= 0).any():
            raise ValueError("Weight must be positive.")
        return {"n": len(true), "mae_kg": float(absolute.mean()),
                "rmse_kg": float(np.sqrt(np.mean(error ** 2))),
                "median_error_kg": float(np.median(absolute)), "bias_kg": float(error.mean()),
                "within_10_percent": float(np.mean(absolute <= 0.1 * true))}
    if not (np.isin(true, AGE_BANDS).all() and np.isin(pred, AGE_BANDS).all()):
        raise ValueError("CID age scoring only supports 24- and 30-month labels.")
    confusion = [[int(((true == actual) & (pred == guess)).sum()) for guess in AGE_BANDS]
                 for actual in AGE_BANDS]
    support = [sum(row) for row in confusion]
    recall = [row[i] / support[i] if support[i] else None for i, row in enumerate(confusion)]
    return {"n": len(true), "mae_months": float(absolute.mean()),
            "accuracy": float(np.mean(true == pred)), "support": support,
            "recall": recall, "confusion": confusion,
            "balanced_accuracy": sum(recall) / 2 if all(support) else None}


def _breed_inputs():
    from train import read_fold, training_id

    index, pixels, _ = data.load_cache("cid")
    run_id = training_id(index, pixels, 1.0)
    predicted = pd.Series(index=index.index, dtype=object)
    if set(index.fold) != set(range(config.N_FOLDS)):
        raise ValueError("CID traits require all five prepared folds.")
    for fold, held_out in index.groupby("fold"):
        checked = read_fold(config.MODELS_DIR, int(fold), run_id, len(held_out))
        if checked is None:
            raise ValueError("Run python train.py to create all verified CID breed folds.")
        predicted.loc[held_out.index] = [config.CLASSES[i] for i in checked["e2"].argmax(axis=1)]
    return index.assign(breed=index.label, predicted_breed=predicted), pixels, run_id


def _backbone(weights="imagenet"):
    from tensorflow import keras

    return keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        pooling="avg",
    )


def frozen_embeddings(pixels, input_id, path):
    """Cache deterministic ImageNet features, reusable across all development folds."""
    from tensorflow import keras

    protocol = f"efficientnetb0-imagenet-center-avg-v1:{keras.__version__}:{input_id}"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as stored:
                features = stored["features"]
                if (stored["protocol"].item() == protocol and features.shape == (len(pixels), 1280)
                        and np.isfinite(features).all()):
                    return features
        except (OSError, ValueError, KeyError):
            pass
    backbone = _backbone()
    features = np.concatenate([
        np.asarray(backbone(data.center_crop_batch(pixels[i:i + config.BATCH_SIZE]), training=False))
        for i in range(0, len(pixels), config.BATCH_SIZE)
    ])
    with data.atomic_output(path) as temporary:
        np.savez(temporary, protocol=protocol, features=features)
    return features


def paired_improvement(y_true, baseline, candidate, groups, *, task, iterations=2000):
    """Bootstrap paired score differences by duplicate-linked group."""
    truth, baseline, candidate = map(np.asarray, (y_true, baseline, candidate))
    _, group_ids = np.unique(groups, return_inverse=True)
    multiplicity = np.eye(group_ids.max() + 1, dtype=np.int8)[group_ids]
    if task == "weight":
        change = np.abs(truth - baseline) - np.abs(truth - candidate)
        totals = (multiplicity.T @ change)[:, None]
        support = multiplicity.sum(axis=0)[:, None]
    else:
        change = (candidate == truth).astype(int) - (baseline == truth)
        totals = np.stack([multiplicity.T @ (change * (truth == band)) for band in AGE_BANDS], axis=1)
        support = np.stack([multiplicity.T @ (truth == band).astype(int) for band in AGE_BANDS], axis=1)
    if (support.sum(axis=0) == 0).any():
        return None
    rng = np.random.default_rng(config.SEED)
    drawn = rng.integers(0, len(totals), size=(iterations, len(totals)))
    denominator = support[drawn].sum(axis=1)
    valid = (denominator > 0).all(axis=1)
    means = (totals[drawn[valid]].sum(axis=1) / denominator[valid]).mean(axis=1)
    if not len(means):
        return None
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {"lower": float(lower), "upper": float(upper),
            "estimate": float((totals.sum(axis=0) / support.sum(axis=0)).mean())}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_fold(path, study_id, rows, predictions, selection):
    payload = {
        "study_id": study_id,
        "rows": rows.index.tolist(),
        "selection": selection,
        "predictions": {method: np.asarray(values, dtype=float).tolist()
                        for method, values in predictions.items()},
    }
    write_summary(path, {"payload": payload, "sha256": _digest(payload)})


def read_fold(path, study_id, rows):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        payload = record["payload"]
        if (record["sha256"] != _digest(payload) or payload["study_id"] != study_id
                or payload["rows"] != rows.index.tolist() or "selected" not in payload["predictions"]):
            return None
        if any(np.asarray(values).shape != (len(rows),) or not np.isfinite(values).all()
               for values in payload["predictions"].values()):
            return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return payload


def write_summary(path, summary):
    with data.atomic_output(path) as temporary:
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _study():
    from tensorflow import keras
    from . import trait_models

    index, pixels, breed_id = _breed_inputs()
    metadata = config.DATA_DIR / "raw/cid/dataset.csv"
    raw_paths = data.build_index("cid").path
    aligned = quarantine_identical(join_cid_targets(index, metadata), raw_paths)
    study_id = _digest({
        "breed_id": breed_id, "metadata": sha256(metadata), "keras": keras.__version__,
        "protocol_files": [sha256(Path(__file__)), sha256(Path(trait_models.__file__))],
        "eligible": aligned[["age_eligible", "weight_eligible"]].to_numpy().tolist(),
        "breed_predictions": index.predicted_breed.tolist(),
    })
    return aligned, pixels, breed_id, study_id


def _fold_cases(aligned, fold, task):
    eligible = aligned[f"{task}_eligible"]
    development = aligned[eligible & aligned.fold.ne(fold)]
    # For age, retain 36-month photos only to measure the forced binary output separately.
    report = eligible | aligned.age_months.eq(36) if task == "age" else eligible
    held_out = aligned[report & aligned.fold.eq(fold)]
    return development, held_out


def _naive_predictions(development, holdout, task):
    fit = representatives(development)
    if task == "weight":
        return weight_baselines(fit, holdout)
    # Restricted to the two observed labels. The majority is the median band.
    baseline = int(fit.age_months.mode().iloc[0])
    return {"median_age_band": [baseline] * len(holdout)}


def run_cid_benchmark():
    from . import trait_models

    root = config.MODELS_DIR / "traits"
    aligned, pixels, breed_id, study_id = _study()
    features = frozen_embeddings(pixels, breed_id, root / "cid-features.npz")
    del pixels
    print(f"CID traits study {study_id[:12]}: {aligned.sku.nunique()} SKUs", flush=True)
    for task in ("weight", "age"):
        for fold in range(config.N_FOLDS):
            development, held_out = _fold_cases(aligned, fold, task)
            artifact = root / f"{task}-{fold}.json"
            if read_fold(artifact, study_id, held_out) is not None:
                print(f"{task} fold {fold}: verified saved predictions", flush=True)
                continue
            outputs, settings, selected = trait_models.candidate_predictions(
                features, development, held_out, task=task,
            )
            outputs.update(_naive_predictions(development, held_out, task))
            save_fold(artifact, study_id, held_out, outputs, {"selected": selected, "settings": settings})
            print(f"{task} fold {fold}: {selected}, settings {settings}", flush=True)


def _gate(cases, task, overall):
    target = cases["weight_kg" if task == "weight" else "age_months"]
    baseline_names = ("global_median", "predicted_breed_median") if task == "weight" else ("median_age_band",)
    comparisons = {
        name: paired_improvement(target, cases[name], cases.selected, cases.group_id, task=task)
        for name in baseline_names
    }
    if task == "weight":
        threshold = 0.90 * min(overall[name]["mae_kg"] for name in baseline_names)
        passes = overall["selected"]["mae_kg"] <= threshold
    else:
        threshold = 0.55
        passes = overall["selected"]["balanced_accuracy"] >= threshold
    return {"passes_cid_gate": bool(passes and all(ci and ci["lower"] > 0 for ci in comparisons.values())),
            "threshold": threshold, "paired_baseline_improvement": comparisons}


def _method_scores(frame, methods, task):
    target = frame["weight_kg" if task == "weight" else "age_months"]
    return {method: score_traits(target, frame[method], task=task) for method in methods}


def _sensitivity(all_views, cases, task):
    target_col = "weight_kg" if task == "weight" else "age_months"
    conflicting = cases.groupby("group_id")[target_col].transform("nunique").gt(1)
    per_case_error = all_views.selected.sub(all_views[target_col]).abs()
    equally_weighted_mae = float(per_case_error.groupby(all_views.sku).mean().mean())
    return {
        "all_view_equal_sku_mae": equally_weighted_mae,
        "exclude_conflicting_groups": score_traits(cases.loc[~conflicting, target_col],
                                                   cases.loc[~conflicting, "selected"], task=task),
    }


def evaluate_cid_benchmark():
    root = config.MODELS_DIR / "traits"
    aligned, pixels, _, study_id = _study()
    del pixels
    summary = {"study_id": study_id, "photos": len(aligned), "source_skus": aligned.sku.nunique(),
               "groups": aligned.group_id.nunique(), "tasks": {}}
    csv_rows = []
    for task in ("weight", "age"):
        frames = []
        folded = {}
        selections = {}
        age_36_guesses = []
        for fold in range(config.N_FOLDS):
            _, held_out = _fold_cases(aligned, fold, task)
            payload = read_fold(root / f"{task}-{fold}.json", study_id, held_out)
            if payload is None:
                raise ValueError(f"Run python train.py --traits for verified {task} fold {fold}.")
            methods = sorted(payload["predictions"])
            predictions = held_out.assign(**payload["predictions"])
            if task == "age":
                age_36_guesses.extend(representatives(predictions[predictions.age_months.eq(36)]).selected)
            eligible = predictions[predictions[f"{task}_eligible"]]
            frames.append(eligible)
            folded[fold] = _method_scores(representatives(eligible), methods, task)
            selections[fold] = payload["selection"]

        all_views = pd.concat(frames)
        cases = representatives(all_views)
        overall = _method_scores(cases, methods, task)
        result = {"pooled": overall, "folds": folded, "inner_selection": selections,
                  "sensitivity": _sensitivity(all_views, cases, task), **_gate(cases, task, overall)}
        if task == "age":
            result["descriptive_36_month_cases"] = {
                str(band): age_36_guesses.count(band) for band in AGE_BANDS
            }
        summary["tasks"][task] = result
        for fold, scores in {"pooled": overall, **{str(k): v for k, v in folded.items()}}.items():
            for method, values in scores.items():
                for metric, value in values.items():
                    if value is not None and isinstance(value, (int, float)):
                        csv_rows.append(dict(task=task, fold=fold, method=method, metric=metric, value=value))
        print(f"{task}: {overall['selected']}, CID gate passed: {result['passes_cid_gate']}", flush=True)
    fields = ("task", "fold", "method", "metric", "value")
    data.save_csv(config.RESULTS_DIR / "traits_metrics.csv", fields, csv_rows)
    write_summary(config.RESULTS_DIR / "traits_evaluation.json", summary)
    return summary
