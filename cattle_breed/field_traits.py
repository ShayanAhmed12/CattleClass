"""Train, calibrate, and assess a controlled photo protocol on field references."""
import json
import re
import sqlite3
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, load_classifier, trait_release, traits
from .download import sha256

CAPTURES = {"ordinary", "side", "side_marker"}
PARTITIONS = ("development", "calibration", "test")


def check_partitions(*partitions):
    for first, second in combinations(partitions, 2):
        if set(first.animal_id) & set(second.animal_id):
            raise ValueError("A field animal appears in multiple partitions.")
        if (set(first.path) & set(second.path)) - {""}:
            raise ValueError("A field image path appears in multiple partitions.")
    if len(partitions) == 3:
        prior_sites = set(partitions[0].site_id) | set(partitions[1].site_id)
        if prior_sites & set(partitions[2].site_id):
            raise ValueError("Field test sites must be independent of development and calibration sites.")


def calibration_cutoff(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if not 0 < alpha < 1 or rank > len(scores) or rank < 1 or not np.isfinite(scores).all():
        raise ValueError("Insufficient finite calibration references for the requested coverage.")
    return float(scores[rank - 1])


def _bounds(predicted, cutoff, task):
    if task == "weight":
        values = np.asarray(predicted).reshape(-1)
        return np.maximum(0, np.floor(values - cutoff)), np.ceil(values + cutoff)
    included = 1 - np.asarray(predicted) <= cutoff
    return included


def field_report(frame, predictions, cutoff, task, policy, *, bands=None):
    """Count all attempts and apply the locked field thresholds after abstention."""
    true = frame.target.to_numpy(dtype=float)
    reference = np.isfinite(true)
    valid = frame.eligible.to_numpy(bool)
    if task == "weight":
        predictions = np.asarray(predictions).reshape(-1)
        low, high = _bounds(predictions, cutoff, task)
        displayed = (valid & np.isfinite(predictions) & (predictions > 0)
                     & (high - low <= policy["max_interval_width"]))
        covered = (true >= low) & (true <= high)
        guesses = predictions
    else:
        probabilities = np.asarray(predictions)
        included = _bounds(probabilities, cutoff, task)
        count = included.sum(axis=1)
        displayed = valid & np.isfinite(probabilities).all(axis=1) & (count > 0) & (count <= policy["max_age_bands"])
        covered = np.zeros(len(true), dtype=bool)
        covered[reference] = included[np.flatnonzero(reference), true[reference].astype(int)]
        guesses = probabilities.argmax(axis=1)
    shown = displayed & reference
    denominator = int(reference.sum())
    coverage = float(covered[shown].mean()) if shown.any() else None
    display_rate = float(displayed.mean())
    scores = {"attempts": len(frame), "eligible": int(valid.sum()), "references": denominator,
              "displayed": int(displayed.sum()), "displayed_with_reference": int(shown.sum()),
              "display_rate": display_rate, "reference_rate": float(reference.mean()),
              "coverage_displayed": coverage,
              "covered_per_reference": float((covered & shown).sum() / denominator) if denominator else None}
    if task == "weight":
        error = float(np.abs(true[shown] - guesses[shown]).mean()) if shown.any() else None
        quality = error is not None and error <= policy["max_mae_kg"]
        scores["mae_kg"] = error
    else:
        class_support = [int((shown & (true == band)).sum()) for band in range(bands)]
        recalls = [float(np.mean(guesses[shown & (true == band)] == band)) if support else None
                   for band, support in enumerate(class_support)]
        balanced = float(np.mean(recalls)) if all(class_support) else None
        quality = (balanced is not None and balanced >= policy["min_balanced_accuracy"]
                   and min(class_support) >= policy["min_per_band"])
        scores.update(balanced_accuracy=balanced, band_support=class_support)
    scores["approved"] = bool(quality and int(shown.sum()) >= policy["min_reference_count"]
                              and display_rate >= policy["min_display_rate"]
                              and scores["reference_rate"] >= policy.get("min_reference_rate", 0)
                              and coverage is not None
                              and coverage >= policy["min_coverage"])
    return scores


def _study(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    task = spec.get("task")
    cohort = spec.get("cohort", "")
    if (task not in {"age", "weight"} or spec.get("capture") not in CAPTURES
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", cohort) or not spec.get("intended_use")):
        raise ValueError("Specify a field task, capture mode, enrolled cohort and intended use.")
    if not all(re.fullmatch(r"[0-9a-f]{64}", spec.get(key, ""))
               for key in ("test_manifest_sha256", "test_media_sha256")):
        raise ValueError("Record the reserved test CSV and photo checksums before training.")
    strata = spec.get("required_strata", {})
    if not isinstance(strata, dict) or any(
        key not in {"sex", "breed"} or not isinstance(values, list) or not values
        or any(not isinstance(value, str) or not value for value in values)
        for key, values in strata.items()
    ):
        raise ValueError("Required field strata map sex or breed to expected labels.")
    policy = spec["policy"]
    required = ("max_interval_width", "max_mae_kg") if task == "weight" else ("max_age_bands", "min_balanced_accuracy", "min_per_band")
    for key in ("min_reference_count", "min_reference_rate", "min_coverage", "min_display_rate", *required):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Field policy requires a positive finite {key}.")
    for key in ("min_reference_rate", "min_coverage", "min_display_rate", "min_balanced_accuracy"):
        if policy.get(key, 0) > 1:
            raise ValueError(f"Field policy {key} must be at most 1.")
    if task == "age":
        edges = np.asarray(spec["age_edges_months"], dtype=float)
        if len(edges) < 3 or not np.isfinite(edges).all() or edges[0] < 0 or np.any(np.diff(edges) <= 0):
            raise ValueError("Field age needs ordered, finite band boundaries in months.")
    identity = traits._digest({"study": spec, "pipeline": sha256(Path(__file__))})
    root = config.MODELS_DIR / "traits" / "field" / identity
    return spec, identity, root


def _inside(base, name):
    if not name or Path(name).is_absolute():
        raise ValueError("Field files need relative paths inside the study folder.")
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Field files need relative paths inside the study folder.")
    return path


def _partition(path, spec, split):
    base = path.parent
    csv_path = _inside(base, spec[split])
    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    needed = {"animal_id", "site_id", "session_id", "path", "capture", "quality_ok", "photo_date"}
    if spec["task"] == "weight":
        needed |= {"weight_kg", "weight_source", "weight_equipment_id", "weighed_at"}
    else:
        needed |= {"birth_date", "birth_source"}
    if not needed <= set(frame):
        raise ValueError(f"{split} is missing fields: {', '.join(sorted(needed - set(frame)))}")
    frame = frame[frame.capture == spec["capture"]].reset_index(drop=True).copy()
    if frame.empty or (frame[["animal_id", "site_id", "session_id"]] == "").any().any():
        raise ValueError(f"{split} needs attempts with animal, site and session IDs.")
    if frame.animal_id.duplicated().any():
        raise ValueError(f"{split} permits one attempt per animal in the selected capture mode.")
    if not frame.quality_ok.isin(["yes", "no"]).all():
        raise ValueError(f"{split} quality_ok must be yes or no.")
    frame["path"] = frame.path.map(lambda name: str(_inside(base, name)) if name else "")
    frame["eligible"] = frame.quality_ok.eq("yes") & frame.path.ne("")
    if spec["capture"] == "side_marker":
        if not {"marker_cm", "marker_px"} <= set(frame):
            raise ValueError("Calibrated capture requires marker_cm and marker_px measurements.")
        frame["eligible"] &= traits._positive(frame.marker_cm).notna() & traits._positive(frame.marker_px).notna()
    captured = pd.to_datetime(frame.photo_date, utc=True, errors="coerce", format="ISO8601")
    if spec["task"] == "weight":
        weighed = pd.to_datetime(frame.weighed_at, utc=True, errors="coerce", format="ISO8601")
        verified = (frame.weight_source.eq("scale") & frame.weight_equipment_id.ne("")
                    & (captured - weighed).abs().le(pd.Timedelta(hours=2)))
        frame["target"] = traits._positive(frame.weight_kg).where(verified)
    else:
        birth = pd.to_datetime(frame.birth_date, utc=True, errors="coerce", format="ISO8601")
        age = (captured - birth).dt.total_seconds() / (86400 * 30.4375)
        edges = spec["age_edges_months"]
        bands = pd.cut(age, edges, right=False, labels=False)
        frame["target"] = bands.where(frame.birth_source.eq("record"))
    return frame


def _bind(frame):
    """Bind both labels and photo contents before opening the test partition."""
    images = sorted((path, sha256(path) if Path(path).is_file() else None)
                    for path in frame.path if path)
    photos = [checksum for _, checksum in images if checksum]
    if len(set(photos)) != len(photos):
        raise ValueError("Duplicate field photos must not count as independent animal attempts.")
    rows = traits._digest(frame.fillna("").to_dict("records"))
    return {"rows": rows, "linked_photos": traits._digest(images), "photos": sorted(photos)}


def _batches(frame, *, required=True):
    inputs, indices = [], []
    for idx, row in frame[frame.eligible].iterrows():
        try:
            image = data.letterbox(data.load_image(row.path))
        except data.ImageLoadError:
            frame.at[idx, "eligible"] = False
            continue
        inputs.append(image)
        indices.append(idx)
    if not inputs:
        if required:
            raise ValueError("The partition has no eligible readable photos.")
        return None, indices
    return data.center_crop_batch(np.stack(inputs)), indices


def _field_head(features, targets, task, bands):
    from tensorflow import keras

    keras.utils.set_random_seed(config.SEED)
    if task == "weight":
        return trait_release.fit_weight(features, targets, np.ones(len(targets)), alpha=100)
    if set(targets) != set(range(bands)):
        raise ValueError("Field age development lacks reference examples in at least one band.")
    network = keras.Sequential([keras.Input((features.shape[1],)),
                                keras.layers.Dense(bands, activation="softmax")])
    network.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    for _ in range(20):
        for index in range(0, len(features), config.BATCH_SIZE):
            section = slice(index, index + config.BATCH_SIZE)
            network.train_on_batch(features[section], targets[section])
    return network


def _predict(model, frame, task, bands):
    values = np.full((len(frame), bands), np.nan) if task == "age" else np.full(len(frame), np.nan)
    pixels, indices = _batches(frame, required=False)
    if pixels is None:
        return values
    raw = np.concatenate([np.asarray(model(pixels[i:i + config.BATCH_SIZE], training=False))
                          for i in range(0, len(pixels), config.BATCH_SIZE)])
    values[indices] = np.exp(raw.reshape(-1)) if task == "weight" else raw
    return values


def train_field(path):
    spec, identity, root = _study(path)
    if (root / "field-evaluation.json").exists():
        raise ValueError("This field study has an evaluated test cohort. Use a new locked study.")
    development, calibration = [_partition(path, spec, item) for item in PARTITIONS[:2]]
    check_partitions(development, calibration)
    bindings = [_bind(frame) for frame in (development, calibration)]
    if set(bindings[0]["photos"]) & set(bindings[1]["photos"]):
        raise ValueError("A duplicate field photo crosses development and calibration.")
    task = spec["task"]
    bands = len(spec.get("age_edges_months", [])) - 1
    pixels, indices = _batches(development)
    labeled = development.loc[indices, "target"].notna().to_numpy()
    if not labeled.any():
        raise ValueError("Field development needs eligible photos with verified references.")
    backbone = traits._backbone()
    pixels = pixels[labeled]
    features = np.concatenate([np.asarray(backbone(pixels[i:i + config.BATCH_SIZE], training=False))
                               for i in range(0, len(pixels), config.BATCH_SIZE)])
    head = _field_head(features, development.loc[indices, "target"].to_numpy()[labeled], task, bands)
    artifact, metadata = trait_release.save_model(root, "candidate", head, {
        "study_id": identity, "task": task, "stage": "field_candidate", "field_approved": False,
        "capture": spec["capture"], "cohort": spec["cohort"], "age_edges_months": spec.get("age_edges_months"),
        "preprocessing": "cattleclass-center-v1",
    }, backbone=backbone)
    predicted = _predict(load_classifier(artifact), calibration, task, bands)
    usable = calibration.eligible & calibration.target.notna()
    target = calibration.loc[usable, "target"].to_numpy()
    if task == "weight":
        scores = np.abs(predicted[usable] - target)
    else:
        if set(target) != set(range(bands)):
            raise ValueError("Field age calibration needs references for every band.")
        scores = 1 - predicted[usable][np.arange(len(target)), target.astype(int)]
    cutoff = calibration_cutoff(scores, .1)
    traits.write_summary(root / "locked.json", {**metadata, "bindings": bindings, "cutoff": cutoff,
                                               "calibration_references": int(usable.sum())})
    print(f"Locked {task} candidate using {usable.sum()} calibration references.", flush=True)


def subgroup_reports(frame, predicted, cutoff, task, policy, strata, *, bands):
    result = {}
    for field, labels in strata.items():
        present = frame[field] if field in frame else pd.Series("", index=frame.index)
        result[field] = {}
        for label in labels:
            subgroup = frame[present.eq(label)]
            result[field][label] = (
                field_report(subgroup, predicted[subgroup.index], cutoff, task, policy, bands=bands)
                if not subgroup.empty else {"attempts": 0, "approved": False}
            )
    return result


def _development_baseline(development, test, task, bands):
    targets = development.loc[development.eligible, "target"].dropna()
    true = test.target.dropna()
    if targets.empty or true.empty:
        return None
    if task == "weight":
        prediction = float(targets.median())
        return {"n": len(true), "constant_kg": prediction,
                "mae_kg": float(np.abs(true - prediction).mean())}
    band = int(targets.mode().iloc[0])
    support = [int(true.eq(i).sum()) for i in range(bands)]
    return {"n": len(true), "constant_band_index": band,
            "accuracy": float(true.eq(band).mean()),
            "balanced_accuracy": 1 / bands if all(support) else None}


def evaluate_field(path):
    spec, identity, root = _study(path)
    locked = json.loads((root / "locked.json").read_text())
    artifact = root / "candidate.keras"
    if locked.get("study_id") != identity or locked.get("model_sha256") != sha256(artifact):
        raise ValueError("Field evaluation requires a matching locked model and calibration record.")
    test_manifest = _inside(path.parent, spec["test"])
    if sha256(test_manifest) != spec["test_manifest_sha256"]:
        raise ValueError("The held-out test CSV does not match the registered checksum.")
    if test_media_checksum(test_manifest, spec["capture"], base=path.parent) != spec["test_media_sha256"]:
        raise ValueError("The held-out test images do not match the registered checksum.")
    partitions = [_partition(path, spec, split) for split in PARTITIONS]
    check_partitions(*partitions)
    bindings = [_bind(frame) for frame in partitions]
    if locked["bindings"] != bindings[:2]:
        raise ValueError("Field development or calibration data changed after locking.")
    for first, second in combinations(bindings, 2):
        if set(first["photos"]) & set(second["photos"]):
            raise ValueError("A duplicate photo crosses field partitions.")
    report_path = root / "field-evaluation.json"
    previous = json.loads(report_path.read_text()) if report_path.is_file() else None
    test_binding = traits._digest(bindings[-1])
    if previous and previous["test_binding"] != test_binding:
        raise ValueError("Use a newly specified locked study for a different test cohort.")
    if previous and previous["model_sha256"] != locked["model_sha256"]:
        raise ValueError("A test cohort already evaluated a different model in this study.")
    test = partitions[-1]
    task = spec["task"]
    bands = len(spec.get("age_edges_months", [])) - 1
    reserve_test_animals(test.animal_id, task, spec["capture"], traits._digest([identity, locked["model_sha256"]]))
    predicted = _predict(load_classifier(artifact), test, task, bands)
    report = field_report(test, predicted, locked["cutoff"], task, spec["policy"], bands=bands)
    site_reports = {
        site: field_report(part, predicted[part.index], locked["cutoff"], task, spec["policy"], bands=bands)
        for site, part in test.groupby("site_id")
    }
    strata = subgroup_reports(test, predicted, locked["cutoff"], task, spec["policy"],
                              spec.get("required_strata", {}), bands=bands)
    report["approved"] = (
        report["approved"] and all(site["approved"] for site in site_reports.values())
        and all(group["approved"] for labels in strata.values() for group in labels.values())
    )
    traits.write_summary(report_path, {
        "study_id": identity, "model_sha256": locked["model_sha256"], "test_binding": test_binding,
        "pooled": report, "sites": site_reports, "required_strata": strata,
        "development_baseline_on_all_references": _development_baseline(partitions[0], test, task, bands),
    })
    if report["approved"]:
        release = config.MODELS_DIR / "traits" / "approved" / f"{spec['cohort']}-{spec['capture']}-{task}.json"
        traits.write_summary(release, {**locked, "field_approved": True, "stage": "field",
                                     "policy": spec["policy"], "field_report": report})
    print(f"Field {task} report: {report}", flush=True)
    return report


def release_for(cohort, capture, task):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", cohort) or capture not in CAPTURES or task not in {"weight", "age"}:
        return None
    root = config.MODELS_DIR / "traits"
    path = root / "approved" / f"{cohort}-{capture}-{task}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = record["study_id"]
        if not (record["stage"] == "field" and record["field_approved"] is True
                and record["field_report"]["approved"] is True
                and record["preprocessing"] == "cattleclass-center-v1"
                and (record["cohort"], record["capture"], record["task"]) == (cohort, capture, task)
                and re.fullmatch(r"[0-9a-f]{64}", identity)
                and np.isfinite(record["cutoff"]) and record["cutoff"] >= 0):
            return None
        artifact = root / "field" / identity / "candidate.keras"
        if sha256(artifact) != record["model_sha256"]:
            return None
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return None
    return artifact, record


@lru_cache(maxsize=4)
def _released_model(path, checksum):
    return load_classifier(path)


def describe_prediction(release, batch, task):
    artifact, record = release
    raw = np.asarray(_released_model(artifact, record["model_sha256"])(batch, training=False))[0]
    cutoff = record["cutoff"]
    policy = record["policy"]
    if task == "weight":
        kg = float(np.exp(raw[0]))
        if not np.isfinite(kg) or kg <= 0:
            return None
        low, high = _bounds([kg], cutoff, "weight")
        low, high = float(low[0]), float(high[0])
        if high - low > policy["max_interval_width"]:
            return None
        return f"About {kg:.0f} kg · range {np.floor(low):.0f}–{np.ceil(high):.0f} kg"
    edges = record["age_edges_months"]
    if raw.shape != (len(edges) - 1,) or not np.isfinite(raw).all():
        return None
    included = np.flatnonzero(1 - raw <= cutoff)
    if not 0 < len(included) <= policy["max_age_bands"]:
        return None
    labels = [f"{edges[band]:g} to under {edges[band + 1]:g}" for band in included]
    return "Estimated age band: " + " or ".join(labels) + " months"


def test_media_checksum(manifest, capture, *, base=None):
    """Hash relative image names together with their contents, including missing files."""
    base = manifest.parent if base is None else base
    rows = pd.read_csv(manifest, dtype=str, usecols=["path", "capture"]).fillna("")
    linked = []
    for name in sorted(rows.loc[rows.capture.eq(capture), "path"]):
        path = _inside(base, name) if name else None
        linked.append((name, sha256(path) if path and path.is_file() else None))
    return traits._digest(linked)


def reserve_test_animals(animal_ids, task, capture, lock):
    """Protect a holdout from reuse after changes to the model or policy."""
    root = config.MODELS_DIR / "traits"
    root.mkdir(parents=True, exist_ok=True)
    keys = [traits._digest([task, capture, animal_id]) for animal_id in animal_ids]
    with sqlite3.connect(root / "field-test-ledger.sqlite") as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS used (animal TEXT PRIMARY KEY, lock_id TEXT NOT NULL)")
        for key in keys:
            previous = connection.execute("SELECT lock_id FROM used WHERE animal = ?", (key,)).fetchone()
            if previous and previous[0] != lock:
                raise ValueError("A field test animal was previously used for this trait and capture mode.")
        connection.executemany("INSERT OR IGNORE INTO used VALUES (?, ?)", [(key, lock) for key in keys])
