"""Score the majority baseline and train cattle breed classifiers."""
import argparse

import numpy as np
import pandas as pd

from cattle_breed import config


def average_fold_scores(folds):
    """Average rates over folds and sum the number of test photos."""
    averages = {key: float(np.mean([fold[key] for fold in folds]))
                for key in ('accuracy', 'balanced_accuracy', 'macro_f1')}
    return {**averages, 'n_images': sum(fold['n_images'] for fold in folds),
            'macro_f1_std': float(np.std([fold['macro_f1'] for fold in folds]))}


def run_baseline(index, *, path=None):
    """Record B0 by predicting Local for each CID test fold."""
    from cattle_breed.metrics import save_result, score

    local = config.CLASS_TO_IDX['Local']
    folds = [score(fold['label_idx'], np.full(len(fold), local)) for _, fold in index.groupby('fold')]
    result = average_fold_scores(folds)
    save_result('B0', 'cid', '5-way', result, path=path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    index = pd.read_csv(config.DATA_DIR / 'processed' / 'cid.csv')
    result = run_baseline(index)
    print(f"B0: accuracy {result['accuracy']:.3f}, balanced accuracy {result['balanced_accuracy']:.3f}, "
          f"macro-F1 {result['macro_f1']:.3f} ± {result['macro_f1_std']:.3f}")


if __name__ == '__main__':
    main()
