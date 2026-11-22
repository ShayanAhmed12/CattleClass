"""Download, index, clean and cache the CID cattle dataset."""
import argparse

import pandas as pd

from cattle_breed import config, data
from cattle_breed.download import download_cid, sha256

CID_LABELS_SHA256 = "1ddd01668d1ab20c616a104388d450ffa489fa5b5de9533d2164d62d592507f9"


def locked_index(index, manifest):
    """Match the prepared photos to the published fold assignment and order."""
    locked = pd.read_csv(manifest, dtype={"path": str, "animal": str})
    portable = data.portable_index(index)
    fields = list(portable)
    if (list(locked) != [*fields, "fold"] or locked.path.duplicated().any()
            or not locked.fold.isin(range(config.N_FOLDS)).all()
            or locked.groupby("animal").fold.nunique().gt(1).any()
            or portable.sort_values("path").to_dict("records")
            != locked[fields].sort_values("path").to_dict("records")):
        raise ValueError("Prepared CID photos do not match the locked CID manifest.")
    ordered = index.assign(_path=portable.path).set_index("_path").loc[locked.path].reset_index(drop=True)
    return ordered.assign(fold=locked.fold.to_numpy(dtype=int))


def prepare():
    if sha256(config.DATA_DIR / "raw/cid/dataset.csv") != CID_LABELS_SHA256:
        raise ValueError("CID labels do not match the locked benchmark checksum.")
    raw = data.build_index("cid")
    index = data.clean_cid_index(raw, data.find_duplicates(raw["path"]))
    index = locked_index(index, config.RESULTS_DIR / "cid_folds.csv")
    images, labels, usable = data.build_cache(index, return_index=True)
    if len(usable) != len(index) or set(index.fold) != set(range(config.N_FOLDS)):
        raise ValueError("The locked CID benchmark needs every readable photo and all five folds.")
    data.save_cache("cid", index, images, labels)
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
