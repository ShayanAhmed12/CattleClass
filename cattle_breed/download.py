"""Download the three datasets into data/raw/ and check that nothing is missing.

No accounts are needed. Interrupted downloads resume, and a dataset that already passes
its check is skipped, so running this again is always safe.
"""
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import DATA_DIR, DOWNLOAD_DIR, IMAGE_SUFFIXES

CID_URL = "https://cid-21.s3.amazonaws.com/images.tar.gz"
CID_CSV_URL = "https://cid-21.s3.amazonaws.com/dataset.csv"
CID_TAR_BYTES = 833_046_375
MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/yd84s35d73/files/"
                "ca140aa2-171f-49b9-8811-cae5bc82fa68/file_downloaded")
KAGGLE_HANDLE = "lukex9442/indian-bovine-breeds"
KAGGLE_KEEP = ("Holstein_Friesian", "Red_Sindhi", "Sahiwal")  # the only Kaggle breeds we use

# What a complete download looks like (paths relative to DATA_DIR).
#   per_folder: exact image count in each class folder
#   folders / images: totals over the sub-folders of root (CID has one folder per animal)
EXPECTED = {
    # 514 animal folders x 4 photos. dataset.csv has 513 rows: folder BLF2334 has no row,
    # so it has no label and is dropped in Task 2.
    "cid": {"root": "raw/cid/images", "folders": 514, "images": 2056, "files": ["raw/cid/dataset.csv"]},
    "mendeley": {"root": "raw/mendeley/Cow Breed",
                 "per_folder": {"Holstain Friesian": 69, "Jersey": 73, "Local Deshi": 56, "Sahiwal": 65}},
    # Red_Sindhi also holds 4 GIFs, which are not counted (not in IMAGE_SUFFIXES).
    "kaggle": {"root": "raw/kaggle", "per_folder": {"Holstein_Friesian": 328, "Red_Sindhi": 162, "Sahiwal": 439}},
}


# --------------------------------------------------------------------------- checking
def count_images(root):
    """Number of image files (searched recursively) in each sub-folder of root."""
    def is_image(p):
        return p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")
    return {d.name: sum(is_image(p) for p in d.rglob("*")) for d in sorted(Path(root).iterdir()) if d.is_dir()}


def check_dataset(name, data_dir, spec):
    """Compare a dataset on disk with its EXPECTED entry. Returns a list of problems (empty = OK)."""
    root = Path(data_dir) / spec["root"]
    if not root.is_dir():
        return [f"{name}: missing folder {root}"]
    problems = [f"{name}: missing file {f}" for f in spec.get("files", []) if not (Path(data_dir) / f).is_file()]
    counts = count_images(root)
    for folder, want in spec.get("per_folder", {}).items():
        if counts.get(folder, 0) != want:
            problems.append(f"{name}/{folder}: expected {want} images, found {counts.get(folder, 0)}")
    if "folders" in spec and len(counts) != spec["folders"]:
        problems.append(f"{name}: expected {spec['folders']} folders, found {len(counts)}")
    if "images" in spec and sum(counts.values()) != spec["images"]:
        problems.append(f"{name}: expected {spec['images']} images, found {sum(counts.values())}")
    return problems


# --------------------------------------------------------------------------- downloading
def split_ranges(start, total, n):
    """Split bytes [start, total) into at most n back-to-back (first, last) ranges, last inclusive."""
    if total <= start:
        return []
    chunk = -(-(total - start) // min(n, total - start))  # ceiling division
    return [(s, min(s + chunk, total) - 1) for s in range(start, total, chunk)]


def fetch(url, dest, first=None, last=None, tries=5):
    """Download url, or only bytes first..last of it, to dest. Resumes a partial byte range."""
    want = None if first is None else last - first + 1
    for attempt in range(tries):
        have = dest.stat().st_size if dest.exists() else 0
        if want is not None and have == want:
            return
        headers = {"User-Agent": "cattle-breed-setup"}
        if want is not None:
            headers["Range"] = f"bytes={first + have}-{last}"
        elif have:
            dest.unlink()  # a whole-file download restarts from zero
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r, \
                    open(dest, "ab") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if want is None or dest.stat().st_size == want:
                return
        except OSError as e:
            print(f"  {dest.name}: attempt {attempt + 1} failed ({e}), retrying")
    raise RuntimeError(f"could not download {url}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_cid(downloads, workers=8):
    """CID: the 833 MB archive in parallel pieces (one connection is slow), then extract."""
    out = DATA_DIR / "raw" / "cid"
    out.mkdir(parents=True, exist_ok=True)
    fetch(CID_CSV_URL, out / "dataset.csv")
    tar_path = downloads / "cid_images.tar.gz"
    if not (tar_path.exists() and tar_path.stat().st_size == CID_TAR_BYTES):
        ranges = split_ranges(0, CID_TAR_BYTES, workers)
        parts = [downloads / f"cid_part_{i}" for i in range(len(ranges))]
        print(f"  downloading 833 MB in {len(parts)} parallel pieces")
        with ThreadPoolExecutor(len(parts)) as pool:
            list(pool.map(lambda job: fetch(CID_URL, job[0], *job[1]), zip(parts, ranges)))
        with open(tar_path, "wb") as f:
            for p in parts:
                f.write(p.read_bytes())
                p.unlink()
    print("  extracting (this also checks the archive is not corrupt)")
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if not Path(m.name).name.startswith("._")]  # macOS junk
        tf.extractall(out, members=members, filter="data")
    (out / "SHA256SUMS").write_text(f"{sha256(tar_path)}  images.tar.gz\n")
    return tar_path


def get_mendeley(downloads):
    out = DATA_DIR / "raw" / "mendeley"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = downloads / "mendeley_cow_breed.zip"
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        print("  downloading 105 MB")
        fetch(MENDELEY_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out, members=[m for m in z.namelist() if "__MACOSX" not in m and "/._" not in m])
    (out / "SHA256SUMS").write_text(f"{sha256(zip_path)}  mendeley_cow_breed.zip\n")
    return zip_path


def get_kaggle(downloads):
    """Kaggle: kagglehub downloads the whole 2.8 GB set; we copy 3 breed folders and delete the rest."""
    cache = downloads / "kagglehub"
    os.environ["KAGGLEHUB_CACHE"] = str(cache)  # must be set before kagglehub is imported
    import kagglehub

    print("  downloading 2.8 GB (only 331 MB is kept)")
    src = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    out = DATA_DIR / "raw" / "kaggle"
    for name in KAGGLE_KEEP:
        folder = next(d for d in src.rglob(name) if d.is_dir())
        shutil.copytree(folder, out / name, dirs_exist_ok=True)
    shutil.rmtree(cache, ignore_errors=True)
    return None


def download_all(names=tuple(EXPECTED), check_only=False):
    """Download whatever is missing, then check every dataset. Returns True if all are complete."""
    downloads = DOWNLOAD_DIR
    downloads.mkdir(parents=True, exist_ok=True)
    getters = {"cid": get_cid, "mendeley": get_mendeley, "kaggle": get_kaggle}
    all_ok = True
    for name in names:
        if not check_only and check_dataset(name, DATA_DIR, EXPECTED[name]):
            print(f"[{name}] downloading")
            archive = getters[name](downloads)
            if archive and not check_dataset(name, DATA_DIR, EXPECTED[name]):
                archive.unlink()  # keep the archive only while something is still wrong
        problems = check_dataset(name, DATA_DIR, EXPECTED[name])
        print(f"[{name}] {'OK' if not problems else 'INCOMPLETE'}", *[f"\n  {p}" for p in problems])
        all_ok &= not problems
    return all_ok
