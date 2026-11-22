"""Small Keras heads and inner model selection for the CID benchmark."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

from . import config, traits

TARGET = {"age": "age_months", "weight": "weight_kg"}
HEAD_EPOCHS = (12, 24)
RIDGE_ALPHA = (10.0, 100.0)


def inner_split(rows, *, task):
    splitter = GroupShuffleSplit(n_splits=16, test_size=0.2, random_state=config.SEED)
    for train, validation in splitter.split(rows, groups=rows.group_id):
        fit, check = rows.iloc[train], rows.iloc[validation]
        if task == "weight" or all(set(part.age_months) == set(traits.AGE_BANDS) for part in (fit, check)):
            return fit, check
    raise ValueError("No grouped inner validation split supports both CID age bands.")


def animal_weights(rows):
    weights = 1 / rows.groupby("sku").sku.transform("size").to_numpy()
    return weights / weights.mean()


def _weight_transform(target, sample_weight):
    logged = np.log(target)
    mean = float(np.average(logged, weights=sample_weight))
    std = float(np.sqrt(np.average((logged - mean) ** 2, weights=sample_weight)))
    return (mean, std or 1.0)


def _standardize(values, mean, scale):
    return (values - mean) / scale


def fit_head(features, target, sample_weight, *, task, epochs):
    from tensorflow import keras

    keras.utils.set_random_seed(config.SEED)
    target = np.asarray(target)
    mean = np.average(features, weights=sample_weight, axis=0)
    variance = np.average((features - mean) ** 2, weights=sample_weight, axis=0)
    transform = None
    if task == "weight":
        transform = _weight_transform(target, sample_weight)
        y = _standardize(np.log(target), *transform)
        outputs = 1
        loss = keras.losses.Huber()
    else:
        y = (target == traits.AGE_BANDS[1]).astype(int)
        outputs = 2
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    network = keras.Sequential([
        keras.Input(shape=(features.shape[1],)),
        keras.layers.Normalization(mean=mean, variance=np.maximum(variance, 1e-6)),
        keras.layers.Dense(32, activation="gelu", kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dense(outputs),
    ])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss)
    rng = np.random.default_rng(config.SEED)
    for _ in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(order), config.BATCH_SIZE):
            batch = order[start:start + config.BATCH_SIZE]
            network.train_on_batch(features[batch], y[batch], sample_weight=sample_weight[batch])
    return network, transform


def decode_predictions(raw, transform, *, task):
    raw = np.asarray(raw)
    if not np.isfinite(raw).all():
        raise ValueError("Keras trait outputs must be finite before decoding.")
    if task == "age":
        return np.asarray(traits.AGE_BANDS)[raw.argmax(axis=1)]
    mean, scale = transform
    return np.exp(raw.reshape(-1) * scale + mean)


def _fit_candidate(features, rows, method, setting, task):
    inputs = features[rows.index]
    target = rows[TARGET[task]].to_numpy()
    weights = animal_weights(rows)
    if method == "frozen_ridge":
        transform = _weight_transform(target, weights)
        fit = Ridge(alpha=setting).fit(inputs, _standardize(np.log(target), *transform), sample_weight=weights)
        return fit.predict, transform
    head, transform = fit_head(inputs, target, weights, task=task, epochs=setting)
    return lambda samples: head(samples, training=False), transform


def choose_settings(features, development, *, task):
    """Fix both candidate settings and family using only grouped inner data."""
    fit, validation = inner_split(development, task=task)
    cases = traits.representatives(validation)
    settings = {"frozen_keras": HEAD_EPOCHS}
    if task == "weight":
        settings["frozen_ridge"] = RIDGE_ALPHA
    chosen = {}
    for method, values in settings.items():
        trials = []
        for value in values:
            predict, transform = _fit_candidate(features, fit, method, value, task)
            guesses = decode_predictions(predict(features[cases.index]), transform, task=task)
            metrics = traits.score_traits(cases[TARGET[task]], guesses, task=task)
            loss = metrics["mae_kg"] if task == "weight" else -metrics["balanced_accuracy"]
            trials.append((loss, value))
        chosen[method] = min(trials)
    winner = min(chosen, key=lambda method: (chosen[method][0], method != "frozen_ridge"))
    return {method: value for method, (_, value) in chosen.items()}, winner


def candidate_predictions(features, development, test, *, task):
    settings, selected = choose_settings(features, development, task=task)
    predictions = {}
    for method, setting in settings.items():
        predict, transform = _fit_candidate(features, development, method, setting, task)
        predictions[method] = decode_predictions(predict(features[test.index]), transform, task=task)
    return {**predictions, "selected": predictions[selected]}, settings, selected
