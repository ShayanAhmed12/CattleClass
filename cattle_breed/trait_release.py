"""Private CID candidates and field-only deployment checks."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from . import config, data, trait_models, traits
from .download import sha256


def fit_weight(features, target, sample_weight, *, alpha):
    """Pack the linear log-weight regression into a serializable Keras head."""
    from tensorflow import keras

    fit = Ridge(alpha=alpha).fit(features, np.log(target), sample_weight=sample_weight)
    output = keras.layers.Dense(1)
    model = keras.Sequential([keras.Input(shape=(features.shape[1],)), output])
    output.set_weights([fit.coef_.reshape(-1, 1), np.asarray([fit.intercept_])])
    return model


def save_model(root, name, head, metadata, *, backbone=None):
    from tensorflow import keras

    backbone = traits._backbone() if backbone is None else backbone
    inputs = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    model = keras.Model(inputs, head(backbone(inputs, training=False), training=False))
    model_path = root / f"{name}.keras"
    with data.atomic_output(model_path) as temporary:
        model.save(temporary)
    record = {**metadata, "model_sha256": sha256(model_path)}
    traits.write_summary(root / f"{name}.json", record)
    return model_path, record


def fit_final_cid():
    """Train a private weight candidate from the passing, verified CID study."""
    rows, pixels, breed_id, study_id = traits._study()
    summary_path = config.RESULTS_DIR / "traits_evaluation.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = summary["tasks"]["weight"]
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError("Run python evaluate.py --traits before fitting a candidate.") from error
    if summary["study_id"] != study_id or not result["passes_cid_gate"]:
        raise ValueError("A matching CID weight result must pass its research gate.")
    if any(choice["selected"] != "frozen_ridge" for choice in result["inner_selection"].values()):
        raise ValueError("The CID final Ridge candidate requires Ridge selection in every fold.")

    root = config.MODELS_DIR / "traits"
    features = traits.frozen_embeddings(pixels, breed_id, root / "cid-features.npz")
    del pixels
    eligible = rows[rows.weight_eligible]
    alpha = float(np.median([choice["settings"]["frozen_ridge"]
                             for choice in result["inner_selection"].values()]))
    head = fit_weight(features[eligible.index], eligible.weight_kg,
                      trait_models.animal_weights(eligible), alpha=alpha)
    path, _ = save_model(root / "candidates", "cid-weight", head, {
        "task": "weight", "stage": "research", "field_approved": False,
        "study_id": study_id, "ridge_alpha": alpha, "output": "log_kg",
        "preprocessing": "cattleclass-center-v1",
    })
    print(f"Saved private CID weight candidate to {path}", flush=True)
