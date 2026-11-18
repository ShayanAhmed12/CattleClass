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
