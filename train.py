"""Score the majority baseline and train cattle breed classifiers."""
import argparse
import hashlib
import inspect
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cattle_breed import config, data
from cattle_breed.download import sha256
from cattle_breed.metrics import average_fold_scores, save_result, score


def run_baseline(index, *, path=None):
    """Record B0 by predicting Local for each CID test fold."""
    local = config.CLASS_TO_IDX['Local']
    folds = [score(fold['label_idx'], np.full(len(fold), local)) for _, fold in index.groupby('fold')]
    result = average_fold_scores(folds)
    save_result('B0', 'cid', '5-way', result, path=path)
    return result



def center_crop_batch(images):
    """Return a view of the centered model-sized region in every cached photo."""
    height, width = images.shape[1:3]
    size = config.IMG_SIZE
    top, left = (height - size) // 2, (width - size) // 2
    return images[:, top:top + size, left:left + size]


def write_json(path, result):
    with data.atomic_output(path) as temporary:
        temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def read_fold(root, fold, run_id, n_images):
    weights = root / f'fold_{fold}.keras'
    predictions = root / f'fold_{fold}.npz'
    if not weights.is_file() or not predictions.is_file():
        return None
    try:
        with np.load(predictions, allow_pickle=False) as archive:
            if (archive['run_id'].item() != run_id or archive['fold'].item() != fold
                    or archive['model_hash'].item() != sha256(weights)):
                return None
            result = {stage: archive[stage] for stage in ('e1', 'e2')}
            epochs = {stage: int(archive[stage]) for stage in ('head_epochs', 'ft_epochs')}
            if any(probs.shape != (n_images, len(config.CLASSES)) or not np.isfinite(probs).all()
                   for probs in result.values()) or min(epochs.values()) < 1:
                return None
            return {**result, **epochs}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _sample_training_animals(index, fraction):
    if fraction == 1:
        return index
    from sklearn.model_selection import train_test_split

    animals = index.groupby('animal')['label_idx'].agg(lambda labels: labels.mode().iloc[0])
    selected, _ = train_test_split(animals.index, train_size=fraction, stratify=animals,
                                    random_state=config.SEED)
    return index.loc[index['animal'].isin(selected)]


def run_folds(index, images, labels, fold_ids, root, run_id, learner, *, fraction=1):
    """Run the requested folds, preserving verified predictions on restart."""
    root = Path(root)
    for fold in fold_ids:
        test = index.index[index['fold'] == fold].to_numpy()
        if len(test) == 0:
            raise ValueError(f'CID test fold {fold} is empty')
        if read_fold(root, fold, run_id, len(test)) is not None:
            print(f'Fold {fold}: checkpoint verified, skipping', flush=True)
            continue
        train, val = data.split_validation(index[index['fold'] != fold])
        train = _sample_training_animals(train, fraction)
        test_images = center_crop_batch(images[test])
        head_prediction = []

        def save_head_prediction(network):
            head_prediction.append(network.predict(test_images, batch_size=config.BATCH_SIZE, verbose=0))

        print(f'Fold {fold}: training on {len(train)} photos, validating on {len(val)}, '
              f'testing on {len(test)}', flush=True)
        network = learner.build_model()
        chosen = learner.train(network, (images[train.index], labels[train.index]),
                               (images[val.index], labels[val.index]), on_head_trained=save_head_prediction)
        tuned = network.predict(test_images, batch_size=config.BATCH_SIZE, verbose=0)
        weights = root / f'fold_{fold}.keras'
        with data.atomic_output(weights) as temporary:
            network.save(temporary)
        with data.atomic_output(root / f'fold_{fold}.npz') as temporary:
            np.savez_compressed(temporary, run_id=run_id, fold=fold, model_hash=sha256(weights),
                                e1=head_prediction[0], e2=tuned, **chosen)
        del network

    all_folds = {}
    for fold, frame in index.groupby('fold'):
        result = read_fold(root, fold, run_id, len(frame))
        if result is not None:
            all_folds[int(fold)] = result
    write_json(root / 'folds.json', {'run_id': run_id, 'folds': {
        fold: {stage: result[stage] for stage in ('head_epochs', 'ft_epochs')}
        for fold, result in all_folds.items()
    }})
    return all_folds


def final_epochs(folds):
    """Choose the median selected epoch in each stage across all test folds."""
    if set(folds) != set(range(config.N_FOLDS)):
        raise ValueError('Finish all five current CID folds before training the final model.')
    return {stage: int(np.median([result[stage] for result in folds.values()]))
            for stage in ('head_epochs', 'ft_epochs')}


def record_cv(index, labels, folds, fraction):
    if set(folds) != set(range(config.N_FOLDS)):
        print(f'{len(folds)}/{config.N_FOLDS} current folds complete. Aggregate metrics require all five.')
        return
    details = {}
    experiments = {'E1': 'e1', 'E2': 'e2'} if fraction == 1 else {'E6': 'e2'}
    for experiment, stage in experiments.items():
        predictions = np.empty(len(index), dtype=np.int64)
        fold_scores = []
        for fold, frame in index.groupby('fold'):
            predictions[frame.index] = folds[fold][stage].argmax(axis=1)
            fold_scores.append(score(labels[frame.index], predictions[frame.index]))
        result = average_fold_scores(fold_scores)
        save_result(experiment, 'cid', '5-way', result)
        pooled = score(labels, predictions)
        details[experiment] = {'classes': list(config.CLASSES), 'per_class_f1': pooled['per_class_f1'],
                               'confusion_matrix': pooled['confusion_matrix'].tolist()}
        print(f"{experiment}: macro-F1 {result['macro_f1']:.3f} ± {result['macro_f1_std']:.3f}")
    name = 'cid_cv_details.json' if fraction == 1 else 'cid_e6_details.json'
    write_json(config.RESULTS_DIR / name, details)


def training_id(index, images, fraction):
    """Fingerprint training data, order, pixels and settings for safe resumption."""
    protocol = (2, fraction, config.CLASSES, config.SEED, config.VAL_FRAC, config.IMG_SIZE,
                config.BATCH_SIZE, config.DROPOUT, config.LABEL_SMOOTHING, config.HEAD_LR,
                config.HEAD_EPOCHS, config.HEAD_PATIENCE, config.FT_LR, config.FT_EPOCHS,
                config.FT_PATIENCE, config.AUG_ROTATION, config.AUG_BRIGHTNESS, config.AUG_CONTRAST)
    identity = hashlib.sha256(repr(protocol).encode('utf-8'))
    model_source = config.PROJECT_ROOT / 'cattle_breed' / 'model.py'
    identity.update(model_source.read_text(encoding='utf-8').encode('utf-8'))
    for operation in (run_folds, _sample_training_animals, center_crop_batch, data.split_validation):
        identity.update(inspect.getsource(operation).encode('utf-8'))
    identity.update(data.portable_index(index).to_csv(index=False).encode('utf-8'))
    identity.update(memoryview(images))
    return identity.hexdigest()


def train_final(index, images, labels, root, run_id, learner):
    folds = {int(fold): read_fold(root, fold, run_id, len(frame))
             for fold, frame in index.groupby('fold')}
    epochs = final_epochs({fold: result for fold, result in folds.items() if result is not None})
    checkpoint = root / 'final.keras'
    record = root / 'final.json'
    if checkpoint.is_file() and record.is_file():
        try:
            previous = json.loads(record.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            previous = {}
        if previous == {'run_id': run_id, 'model_hash': sha256(checkpoint), **epochs}:
            print('Final model: checkpoint verified, skipping')
            return
    network = learner.build_model()
    learner.train(network, (images, labels), **epochs)
    with data.atomic_output(checkpoint) as temporary:
        network.save(temporary)
    write_json(record, {'run_id': run_id, 'model_hash': sha256(checkpoint), **epochs})
    print(f'Wrote {checkpoint}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--baseline', action='store_true', help='score B0 without loading TensorFlow')
    mode.add_argument('--final', action='store_true', help='fit the all-CID model after cross-validation')
    mode.add_argument('--folds', nargs='+', type=int, choices=range(config.N_FOLDS))
    parser.add_argument('--fraction', type=float, choices=(0.5, 1.0), default=1.0)
    args = parser.parse_args()
    if (args.final or args.baseline) and args.fraction != 1:
        parser.error('--fraction 0.5 is for the optional E6 cross-validation run')
    if args.baseline:
        index = pd.read_csv(config.DATA_DIR / 'processed' / 'cid.csv')
        result = run_baseline(index)
        print(f"B0: accuracy {result['accuracy']:.3f}, balanced accuracy {result['balanced_accuracy']:.3f}, "
              f"macro-F1 {result['macro_f1']:.3f} ± {result['macro_f1_std']:.3f}")
        return

    from cattle_breed import model

    index, images, labels = data.load_cache('cid')
    run_id = training_id(index, images, args.fraction)
    root = config.MODELS_DIR if args.fraction == 1 else config.MODELS_DIR / 'fraction_0.5'
    if args.final:
        train_final(index, images, labels, root, run_id, model)
    else:
        if args.fraction == 1:
            run_baseline(index)
        fold_ids = args.folds if args.folds is not None else range(config.N_FOLDS)
        folds = run_folds(index, images, labels, fold_ids, root, run_id, model, fraction=args.fraction)
        record_cv(index, labels, folds, args.fraction)


if __name__ == '__main__':
    main()
