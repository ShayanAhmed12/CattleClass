"""Download CID images and metadata, and check the local copy."""
import csv
import hashlib
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import DATA_DIR, DOWNLOAD_DIR, IMAGE_SUFFIXES

CID_URL = "https://cid-21.s3.amazonaws.com/images.tar.gz"
CID_CSV_URL = "https://cid-21.s3.amazonaws.com/dataset.csv"
CID_TAR_BYTES = 833_046_375
CID_SPEC = {
    "root": "raw/cid/images",
    "folders": 514,
    "images": 2056,
    "files": ["raw/cid/dataset.csv"],
}


def is_image_file(path: str | Path) -> bool:
    """True for .jpg/.jpeg/.png/.webp files (any case), excluding macOS '._' resource-fork files."""
    p = Path(path)
    return p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")


def count_images(root):
    """Number of image files (searched recursively) in each sub-folder of root."""
    return {
        directory.name: sum(path.is_file() and is_image_file(path) for path in directory.rglob("*"))
        for directory in sorted(Path(root).iterdir()) if directory.is_dir()
    }


def check_dataset(name, data_dir, spec):
    """Compare a dataset on disk with its expected layout. Returns a list of problems (empty = OK)."""
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


def cid_problems():
    problems = check_dataset("cid", DATA_DIR, CID_SPEC)
    source = DATA_DIR / "raw/cid/dataset.csv"
    if source.is_file():
        with source.open(encoding="utf-8-sig", newline="") as stream:
            columns = csv.DictReader(stream).fieldnames or ()
        missing = {"sku", "breed", "age_in_year", "weight_in_kg"} - set(columns)
        if missing:
            problems.append(f"cid: missing metadata columns {', '.join(sorted(missing))}")
    return problems


def download_cid(*, check_only=False):
    """Check CID, downloading it if needed."""
    problems = cid_problems()
    if problems and not check_only:
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print("[cid] downloading")
        archive = get_cid(DOWNLOAD_DIR)
        problems = cid_problems()
        if not problems:
            archive.unlink()
    print("[cid] INCOMPLETE" if problems else "[cid] OK", *[f"\n  {problem}" for problem in problems])
    return not problems
