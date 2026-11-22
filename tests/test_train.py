"""Baseline and training orchestration tests without the real datasets."""
import csv

import numpy as np
import pandas as pd


def test_baseline_averages_folds_without_duplicate_result_rows(tmp_path):
    from train import run_baseline

    index = pd.DataFrame({'fold': [0, 0, 0, 1, 1], 'label_idx': [1, 1, 2, 1, 2]})
    path = tmp_path / 'metrics.csv'
    first = run_baseline(index, path=path)
    run_baseline(index, path=path)
    assert first['n_images'] == 5
    assert np.isclose(first['accuracy'], (2 / 3 + 1 / 2) / 2)
    assert np.isclose(first['balanced_accuracy'], 0.5)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1 and rows[0]['experiment'] == 'B0'


def test_fold_resume_rejects_stale_or_incomplete_artifacts(tmp_path, monkeypatch):
    import shutil

    from cattle_breed import data
    from train import read_fold, run_folds

    class Network:
        def predict(self, images, **kwargs):
            return np.tile([0, 1, 0, 0, 0], (len(images), 1))

        def save(self, path):
            path.write_bytes(b'valid model')

    class Learner:
        trained = 0

        def build_model(self):
            return Network()

        def train(self, network, train, val, *, on_head_trained):
            self.trained += 1
            on_head_trained(network)
            return {'head_epochs': 2, 'ft_epochs': 3}

    index = pd.DataFrame({'fold': [0, 1, 1], 'path': ['p0', 'p1', 'p2'], 'animal': ['a', 'b', 'c']})
    images = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    labels = np.asarray([1, 1, 2])
    learner = Learner()
    monkeypatch.setattr(data, 'split_validation', lambda frame: (frame.iloc[:1], frame.iloc[1:]))
    run_folds(index, images, labels, [0], tmp_path, 'current', learner)
    results = run_folds(index, images, labels, [0], tmp_path, 'current', learner)
    assert learner.trained == 1
    assert results[0]['head_epochs'] == 2
    assert results[0]['e1'].shape == results[0]['e2'].shape == (1, 5)
    for suffix in ('keras', 'npz'):
        shutil.copyfile(tmp_path / f'fold_0.{suffix}', tmp_path / f'fold_1.{suffix}')
    assert read_fold(tmp_path, 1, 'current', 1) is None
    run_folds(index, images, labels, [0], tmp_path, 'changed', learner)
    assert learner.trained == 2
    (tmp_path / 'fold_0.keras').unlink()
    run_folds(index, images, labels, [0], tmp_path, 'changed', learner)
    assert learner.trained == 3


def test_final_epoch_schedule_uses_all_fold_medians():
    import pytest
    from train import final_epochs

    folds = {i: {'head_epochs': head, 'ft_epochs': ft}
             for i, (head, ft) in enumerate([(1, 5), (2, 6), (9, 2), (4, 7), (5, 3)])}
    assert final_epochs(folds) == {'head_epochs': 4, 'ft_epochs': 5}
    del folds[4]
    with pytest.raises(ValueError, match='all five'):
        final_epochs(folds)


def test_breed_only_manifest_trains_private_keras_candidate(tmp_path, monkeypatch, make_image):
    import sys

    import pytest
    from tensorflow import keras

    import train
    from cattle_breed import config, load_classifier, model

    monkeypatch.setattr(config, 'MODELS_DIR', tmp_path / 'models')
    rows = []
    for split_index, split in enumerate(('development', 'validation')):
        for i, breed in enumerate(config.CLASSES):
            photo = make_image(f'{split}-{i}.png', color=(30 * i, 75 * split_index, 100))
            rows.append(dict(split=split, animal_id=photo.stem, path=photo.name, breed=breed))
    manifest = tmp_path / 'breed.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)

    def small_classifier():
        inputs = keras.Input((224, 224, 3))
        backbone = keras.Sequential([keras.Input((224, 224, 3)), keras.layers.Rescaling(1 / 255),
                                     keras.layers.GlobalAveragePooling2D()], name='backbone')
        outputs = keras.layers.Dense(len(config.CLASSES), activation='softmax')(backbone(inputs))
        return keras.Model(inputs, outputs)

    fit = model.train
    monkeypatch.setattr(model, 'build_model', small_classifier)
    monkeypatch.setattr(model, 'train', lambda *args: fit(*args, head_epochs=1, ft_epochs=1, verbose=0))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--breed-data', str(manifest)])

    train.main()

    candidate, = (config.MODELS_DIR / 'breed-research').glob('*.keras')
    assert load_classifier(candidate).output_shape == (None, len(config.CLASSES))
    assert not (config.MODELS_DIR / 'final.keras').exists()

    rows[-1]['animal_id'] = rows[0]['animal_id']
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match='animal.*multiple partitions'):
        train.main()
