"""Step 1: get the data ready.

    python prepare_data.py            download anything missing, then prepare it
    python prepare_data.py --check    only check the downloads

What it does (later steps are added as their tasks are done):
    1. download and check CID, Mendeley and Kaggle into data/raw/   (cattle_breed/download.py)
"""
import argparse
import sys

from cattle_breed.data import build_index, save_cid_samples
from cattle_breed.download import download_all

parser = argparse.ArgumentParser(description="Download and prepare the datasets.")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--cid-samples", action="store_true", help="write the CID sample grid from existing data")
mode.add_argument("--check", action="store_true", help="only check the downloads, change nothing")
args = parser.parse_args()

if args.cid_samples:
    print(save_cid_samples(build_index("cid")))
    sys.exit(0)

if not download_all(check_only=args.check):
    sys.exit("Some datasets are incomplete. Run  python prepare_data.py  again (downloads resume).")
print("All datasets OK.")
