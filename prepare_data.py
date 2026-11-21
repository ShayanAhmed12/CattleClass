"""Download, index, clean and cache the CID cattle dataset."""
import argparse

from cattle_breed import config, data
from cattle_breed.download import download_cid


def prepare():
    raw = data.build_index("cid")
    index = data.clean_cid_index(raw, data.find_duplicates(raw["path"]))
    images, labels, index = data.build_cache(index, return_index=True)
    if index.empty:
        raise ValueError("CID has no usable images.")
    index["fold"] = data.make_folds(index)
    data.save_cache("cid", index, images, labels)
    with data.atomic_output(config.RESULTS_DIR / "cid_folds.csv") as temporary:
        data.portable_index(index).to_csv(temporary, index=False)
    print(f"[cid] {len(index)} images cached ({len(raw) - len(index)} removed after indexing)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="only check the raw CID files")
    mode.add_argument("--cid-samples", action="store_true", help="write the CID sample grid from existing data")
    parser.add_argument("--datasets", nargs="+", choices=("cid",),
                        help="CID is the supported dataset. The option remains for existing commands.")
    args = parser.parse_args()
    if args.cid_samples:
        print(data.save_cid_samples(data.build_index("cid")))
        return
    if not download_cid(check_only=args.check):
        parser.exit(1, "CID is incomplete. Rerun prepare_data.py to resume the download.\n")
    if not args.check:
        prepare()


if __name__ == "__main__":
    main()
