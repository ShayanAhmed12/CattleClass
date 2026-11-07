"""Step 1: get the data ready.

    python prepare_data.py            download anything missing, then prepare it
    python prepare_data.py --check    only check the downloads

What it does (later steps are added as their tasks are done):
    1. download and check CID, Mendeley and Kaggle into data/raw/   (cattle_breed/download.py)
"""
import argparse
import sys

from cattle_breed.download import download_all

parser = argparse.ArgumentParser(description="Download and prepare the datasets.")
parser.add_argument("--check", action="store_true", help="only check the downloads, change nothing")
args = parser.parse_args()

if not download_all(check_only=args.check):
    sys.exit("Some datasets are incomplete. Run  python prepare_data.py  again (downloads resume).")
print("All datasets OK.")
