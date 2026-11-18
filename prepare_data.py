"""Download, index, clean and cache the cattle datasets."""
import argparse

from cattle_breed import config, data
from cattle_breed.download import EXPECTED, download_all


def prepare(datasets):
    raw_cid = data.build_index('cid')
    cid_hashes = data.hash_images(raw_cid['path'])
    for dataset in datasets:
        raw = raw_cid if dataset == 'cid' else data.build_index(dataset)
        if dataset == 'cid':
            index = data.clean_cid_index(raw, data.find_duplicates(raw['path'], hashes=cid_hashes))
        else:
            index = data.clean_external_index(raw, raw_cid['path'], cid_hashes=cid_hashes)
        images, labels, index = data.build_cache(index, return_index=True)
        if index.empty:
            raise ValueError(f'{dataset} has no usable images')
        if dataset == 'cid':
            index['fold'] = data.make_folds(index)
        data.save_cache(dataset, index, images, labels)
        if dataset == 'cid':
            with data.atomic_output(config.RESULTS_DIR / 'cid_folds.csv') as temporary:
                data.portable_index(index).to_csv(temporary, index=False)
        print(f'[{dataset}] {len(index)} images cached ({len(raw) - len(index)} removed after indexing)')
        del images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='only check the raw dataset downloads')
    mode.add_argument('--cid-samples', action='store_true', help='write the CID sample grid from existing data')
    parser.add_argument('--datasets', nargs='+', choices=EXPECTED, default=list(EXPECTED))
    args = parser.parse_args()
    if args.cid_samples:
        print(data.save_cid_samples(data.build_index('cid')))
        return

    datasets = list(dict.fromkeys(args.datasets))
    required = datasets if args.check or 'cid' in datasets else ['cid', *datasets]
    if not download_all(names=required, check_only=args.check):
        parser.exit(1, 'Some datasets are incomplete. Rerun prepare_data.py to resume downloads.\n')
    if not args.check:
        prepare(datasets)


if __name__ == '__main__':
    main()
