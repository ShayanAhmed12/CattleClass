"""Research checks for animal crops, age support, and breed abstention."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from . import public_weight as public
from cattle_breed import config, data, trait_release, traits
from cattle_breed.download import sha256

DETECTOR_SHA256 = "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
ALPHA = 10
KNOWN_COVERAGE = 0.90
UNKNOWN_REJECTION = 0.80


def detector(path):
    import cv2
    import onnxruntime
    from rtmlib import YOLOX

    if sha256(path) != DETECTOR_SHA256:
        raise ValueError("Use the pinned YOLOX-tiny weights listed in docs/traits.md.")
    cv2.setNumThreads(1)
    onnxruntime.set_default_logger_severity(3)
    model = YOLOX(str(path), (416, 416), det_mode="multiclass", score_thr=0.7, backend="opencv")
    return model


def crop_image(model, rgb):
    boxes, labels = model(rgb[:, :, ::-1].copy())
    cows = np.asarray(boxes).reshape(-1, 4)[np.asarray(labels).reshape(-1) == 19]
    if len(cows) != 1:
        return None
    x0, y0, x1, y1 = cows[0].astype(int)
    x0, x1 = np.clip([x0, x1], 0, rgb.shape[1])
    y0, y1 = np.clip([y0, y1], 0, rgb.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return data.letterbox(rgb[y0:y1, x0:x1])


def encode(backbone, images):
    if not len(images):
        return np.empty((0, backbone.output_shape[-1]), dtype=np.float32)
    return np.asarray(backbone(data.center_crop_batch(np.stack(images)), training=False))


def image_features(backbone, detect, photos, *, full_needed):
    kept, full, cropped = [], [], []
    for start in range(0, len(photos), 16):
        rgbs = (load() for load in photos[start:start + 16])
        batch = [(rgb, crop_image(detect, rgb)) for rgb in rgbs]
        kept.extend(crop is not None for _, crop in batch)
        if full_needed:
            full.append(encode(backbone, [data.letterbox(rgb) for rgb, _ in batch]))
        cropped.append(encode(backbone, [crop for _, crop in batch if crop is not None]))
    return np.asarray(kept), np.concatenate(full) if full_needed else None, np.concatenate(cropped)


def predictions(development, test, targets):
    model = trait_release.fit_weight(development, targets, np.ones(len(targets)), alpha=ALPHA)
    return np.exp(model(test, training=False).numpy().ravel())


def compare_weights(frame, full, cropped, median, *, group_column):
    truth = frame.weight_kg.to_numpy()
    paired = traits.paired_improvement(truth, full, cropped, frame[group_column], task="weight") if len(truth) else None
    return {
        "full_frame": traits.score_traits(truth, full, task="weight"),
        "animal_crop": traits.score_traits(truth, cropped, task="weight"),
        "training_median": traits.score_traits(truth, median, task="weight"),
        "mae_improvement_crop_vs_full_kg_95_ci": paired,
    }


def cid_weights(cases, full, crop):
    truth = cases.weight_kg.to_numpy()
    folds = cases.fold.to_numpy()
    output = np.empty((3, len(cases)))
    for fold in sorted(set(folds)):
        fit, hold = folds != fold, folds == fold
        for target, features in zip(output[:2], (full, crop)):
            target[hold] = predictions(features[fit], features[hold], truth[fit])
        output[2, hold] = np.median(truth[fit])
    return compare_weights(cases, *output, group_column="group_id")


def nearest_distance(reference, photos):
    return 1 - (normalize(photos) @ normalize(reference).T).max(axis=1)


def accepts(reference, calibration, test):
    scores = nearest_distance(reference, calibration)
    rank = min(len(scores) - 1, int(np.ceil((len(scores) + 1) * KNOWN_COVERAGE)) - 1)
    threshold = np.sort(scores)[rank]
    return nearest_distance(reference, test) <= threshold


def breed_check(cases, full, angus):
    folds, labels = cases.fold.to_numpy(), cases.breed.to_numpy()
    known = []
    missed = {breed: [] for breed in config.CLASSES}
    for fold in range(config.N_FOLDS):
        hold = folds == fold
        cal = folds == (fold + 1) % config.N_FOLDS
        bank = ~(hold | cal)
        known.extend(accepts(full[bank], full[cal], full[hold]))
        for breed in config.CLASSES:
            other = labels != breed
            if (hold & ~other).any():
                accepted = accepts(full[bank & other], full[cal & other], full[hold & ~other])
                missed[breed].extend(~accepted)
    cal = folds == config.N_FOLDS - 1
    external = ~accepts(full[~cal], full[cal], angus)
    rejected = {breed: {"n": len(values), "rejection_rate": float(np.mean(values))}
                for breed, values in {**missed, "Aberdeen Angus": external}.items()}
    known_rate = float(np.mean(known))
    passes = known_rate >= KNOWN_COVERAGE and all(
        value["rejection_rate"] >= UNKNOWN_REJECTION for value in rejected.values()
    )
    return {"known_n": len(known), "known_acceptance": known_rate, "unknown": rejected,
            "known_target": KNOWN_COVERAGE, "unknown_target_per_breed": UNKNOWN_REJECTION,
            "research_gate_passed": passes}


def public_features(backbone, detect):
    import pandas as pd

    tree, records, frames = public.paired_source()
    reference = pd.DataFrame({"frame": frames, "weight_kg": [records[frame] for frame in frames]})
    for view in public.VIEWS:
        paths = [public.image_path(frame, view) for frame in frames]
        # Download at most one batch at a time. Raw photographs stay in memory.
        def load(path):
            return public.load_rgb(tree, path)

        keep, full, crop = [], [], []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(paths), 16):
                rgb = list(pool.map(load, paths[start:start + 16]))
                photos = [lambda image=image: image for image in rgb]
                mask, encoded, boxes = image_features(backbone, detect, photos, full_needed=True)
                keep.extend(mask)
                full.append(encoded)
                crop.append(boxes)
        yield view, reference, np.asarray(keep), np.concatenate(full), np.concatenate(crop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector", type=Path, help="pinned YOLOX-tiny ONNX file")
    args = parser.parse_args()
    detect = detector(args.detector)
    aligned, pixels, run_id, study_id = traits._study()
    features = traits.frozen_embeddings(pixels, run_id, config.MODELS_DIR / "traits/cid-features.npz")
    del pixels
    cases = traits.representatives(aligned)
    weights = cases[cases.weight_eligible]
    age_counts = [
        {"recorded_months": float(month), "animals": len(group),
         "duplicate_linked_groups": int(group.group_id.nunique())}
        for month, group in cases[cases.age_valid].groupby("age_months")
    ]
    backbone = traits._backbone()
    loaders = [lambda file=path: data.load_image(file) for path in weights.path]
    keep, _, crop = image_features(backbone, detect, loaders, full_needed=False)
    study = weights[keep]
    full = features[study.index]
    summary = {
        "cid_study_id": study_id, "yolox_sha256": DETECTOR_SHA256, "ridge_alpha": ALPHA,
        "cid_age_labels": age_counts, "cid_animals_without_age": int((~cases.age_valid).sum()),
        "cid_crop": {"attempted": len(weights), "retained": len(study),
                     **cid_weights(study, full, crop)},
        "cowdatabase2": {"revision": public.REVISION, "field_approved": False},
    }
    print(f"CID crops ready: {len(study)}/{len(weights)}", flush=True)
    truth = study.weight_kg.to_numpy()
    for view, reference, mask, public_full, public_crop in public_features(backbone, detect):
        comparison = reference[mask]
        if len(comparison):
            full_guess = predictions(full, public_full[mask], truth)
            crop_guess = predictions(crop, public_crop, truth)
        else:
            full_guess = crop_guess = np.empty(0)
        median = np.full(len(comparison), np.median(truth))
        summary["cowdatabase2"][view] = {
            "attempted": len(reference), "retained": len(comparison),
            **compare_weights(comparison, full_guess, crop_guess, median, group_column="frame"),
        }
        if view == "left":
            summary["breed_abstention"] = breed_check(cases, features[cases.index], public_full)
        print(f"CowDatabase2 {view} crops ready: {len(comparison)}/{len(reference)}", flush=True)
    traits.write_summary(config.RESULTS_DIR / "capture_diagnostics.json", summary)


if __name__ == "__main__":
    main()
