"""Download the three datasets into data/raw/ and check that nothing is missing.

No accounts are needed. Interrupted downloads resume, and a dataset that already passes
its check is skipped, so running this again is always safe.
"""
import hashlib
import io
import shutil
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from .config import DATA_DIR, DOWNLOAD_DIR, IMAGE_SUFFIXES

CID_URL = "https://cid-21.s3.amazonaws.com/images.tar.gz"
CID_CSV_URL = "https://cid-21.s3.amazonaws.com/dataset.csv"
CID_TAR_BYTES = 833_046_375
MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/yd84s35d73/files/"
                "ca140aa2-171f-49b9-8811-cae5bc82fa68/file_downloaded")
KAGGLE_HANDLE = "lukex9442/indian-bovine-breeds/versions/5"
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


class ArchiveRanges(io.RawIOBase):
    """Seekable archive input backed by HTTP byte ranges."""

    def __init__(self, url, size):
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise ValueError("Cannot seek before the archive start")
        self.position = base + offset
        return self.position

    def readinto(self, buffer):
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        first, last = self.position, self.position + count - 1
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={first}-{last}", "Accept-Encoding": "identity",
        })
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if (response.status != 206
                            or response.headers.get("Content-Range") != f"bytes {first}-{last}/{self.size}"):
                        raise OSError("Unexpected response to the Kaggle archive range request")
                    chunk = response.read(count)
                if len(chunk) != count:
                    raise OSError("Incomplete response to the Kaggle archive range request")
            except OSError:
                continue
            buffer[:count] = chunk
            self.position += count
            return count
        raise OSError("Unable to read the requested Kaggle archive bytes after three attempts")


@contextmanager
def open_kaggle_zip():
    """Resolve the pinned ZIP once, then read it without downloading unrelated files."""
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    owner, dataset, _, version = KAGGLE_HANDLE.split("/")
    request = ApiDownloadDatasetRequest()
    request.owner_slug, request.dataset_slug = owner, dataset
    request.dataset_version_number = int(version)
    with build_kaggle_client() as client:
        with client.datasets.dataset_api_client.download_dataset(request) as response:
            url = response.url
            size = response.headers.get("Content-Length")
            if response.status_code != 200 or not size or response.headers.get("Accept-Ranges") != "bytes":
                raise OSError("Kaggle did not provide a seekable archive")
    with io.BufferedReader(ArchiveRanges(url, int(size)), buffer_size=4 << 20) as source:
        with zipfile.ZipFile(source) as archive:
            yield archive


def get_kaggle(_downloads):
    """Extract the matched image folders without downloading the whole Kaggle archive."""
    out = DATA_DIR / "raw" / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    existing = count_images(out)
    wanted = {name for name in KAGGLE_KEEP if existing.get(name) != EXPECTED["kaggle"]["per_folder"][name]}
    if not wanted:
        return None
    print("  extracting matched folders from Kaggle v5")
    with open_kaggle_zip() as archive:
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if (len(parts) != 4 or parts[:2] != ("Indian_bovine_breeds", "Indian_bovine_breeds")
                    or parts[2] not in wanted or entry.is_dir()):
                continue
            filename = parts[3]
            if (PurePosixPath(filename).suffix.lower() not in IMAGE_SUFFIXES
                    or filename.startswith("._") or any(char in filename for char in ("\\", ":"))):
                continue
            destination = out.joinpath(*parts[2:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size == entry.file_size:
                continue
            with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as target:
                temporary = Path(target.name)
                try:
                    with archive.open(entry) as picture:
                        shutil.copyfileobj(picture, target)
                    target.close()
                    temporary.replace(destination)
                finally:
                    target.close()
                    temporary.unlink(missing_ok=True)
    return None

def download_all(names=tuple(EXPECTED), check_only=False):
    """Download whatever is missing, then check every dataset. Returns True if all are complete."""
    downloads = DOWNLOAD_DIR
    if not check_only:
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
