"""Evaluate the frozen CID weight candidate on CowDatabase2 side images."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from cattle_breed import config, data, load_classifier, traits
from cattle_breed.download import sha256

REPO = "ruchaya/CowDatabase2"
REVISION = "3359f0a5a5e69d8416c2e939500b17d02da0ad81"
WORKBOOK = "Database2022v2.xlsx"
VIEWS = ("left", "right")
CID_MEDIAN_KG = 239.0
XLSX = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def fetch(url, limit):
    request = Request(url, headers={"User-Agent": "CattleClass-public-weight-study"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                contents = response.read(limit + 1)
            break
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
    if len(contents) > limit:
        raise ValueError(f"Source file exceeds its size limit: {url}")
    return contents


def git_blob(tree, path):
    entry = tree[path]
    source = fetch(f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}", entry["size"])
    digest = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
    if digest != entry["sha"]:
        raise ValueError(f"Source file checksum mismatch: {path}")
    return source


def read_workbook(source):
    """Read the pinned spreadsheet's frame, birthday and live weight fields."""
    with ZipFile(BytesIO(source)) as workbook:
        strings = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = ["".join(cell.itertext()) for cell in strings.findall("x:si", XLSX)]
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//x:row", XLSX):
        cells = {}
        for cell in row:
            value = cell.findtext("x:v", namespaces=XLSX)
            if value is not None:
                column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                cells[column] = strings[int(value)] if cell.get("t") == "s" else value
        rows.append(cells)
    fields = ("B", "J", "K")
    if tuple(rows[0][field].strip() for field in fields) != ("Frame number", "Birthday", "Live weigth"):
        raise ValueError("Unexpected CowDatabase2 worksheet columns.")
    records = {}
    for row in rows[1:]:
        if not row.get("A", "").strip().isdigit():
            continue
        frame, birthday, weight = (row[field].strip() for field in fields)
        datetime.strptime(birthday, "%d.%m.%Y")
        kg = float(weight)
        if not re.fullmatch(r"\d+(\.\d+)?", frame) or frame in records or not np.isfinite(kg) or kg <= 0:
            raise ValueError("Invalid or duplicate frame in CowDatabase2 measurements.")
        records[frame] = kg
    return records


def image_path(frame, view):
    return f"{frame}/raw_data/{view}/rgb.png"


def predict_views(model, tree, frames):
    def pixels(path):
        with Image.open(BytesIO(git_blob(tree, path))) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return data.letterbox(rgb)

    predictions = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for view in VIEWS:
            estimates = []
            for start in range(0, len(frames), 16):
                paths = [image_path(frame, view) for frame in frames[start:start + 16]]
                images = np.stack(list(pool.map(pixels, paths)))
                batch = data.center_crop_batch(images)
                estimates.extend(np.exp(model(batch, training=False).numpy().ravel()))
                print(f"{view}: scored {len(estimates)}/{len(frames)} images", flush=True)
            predictions[view] = np.asarray(estimates)
    return predictions


def main():
    model_path = config.MODELS_DIR / "traits/candidates/cid-weight.keras"
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    expected = {"task": "weight", "stage": "research", "field_approved": False,
                "output": "log_kg", "preprocessing": "cattleclass-center-v1",
                "model_sha256": sha256(model_path)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected the verified CID research weight candidate.")

    url = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    manifest = json.loads(fetch(url, 2_000_000))
    if manifest["truncated"]:
        raise ValueError("The CowDatabase2 source manifest is incomplete.")
    tree = {entry["path"]: entry for entry in manifest["tree"] if entry["type"] == "blob"}
    records = read_workbook(git_blob(tree, WORKBOOK))
    frames = [frame for frame in records if all(image_path(frame, view) in tree for view in VIEWS)]
    checksums = {tree[image_path(frame, view)]["sha"] for frame in frames for view in VIEWS}
    if len(records) != 96 or len(frames) != 95 or len(checksums) != len(frames) * len(VIEWS):
        raise ValueError("The paired cohort differs from the audited CowDatabase2 source.")

    true = np.array([records[frame] for frame in frames])
    cid = np.full_like(true, CID_MEDIAN_KG)
    prediction = predict_views(load_classifier(model_path), tree, frames)
    summary = {
        "source": f"https://github.com/{REPO}/tree/{REVISION}",
        "paper": "https://doi.org/10.3390/ani12162128",
        "model": metadata,
        "breed": "Aberdeen Angus",
        "sites": 1,
        "workbook_animals": len(records),
        "unmatched_animals": len(records) - len(frames),
        "paired_animals": len(frames),
        "primary_view": "left",
        "weight_range_kg": [float(true.min()), float(true.max())],
        "cid_median_kg": CID_MEDIAN_KG,
        "cid_median": traits.score_traits(true, cid, task="weight"),
    }
    for view, estimates in prediction.items():
        summary[view] = {**traits.score_traits(true, estimates, task="weight"),
                         "prediction_range_kg": [float(estimates.min()), float(estimates.max())],
                         "mae_improvement_over_cid_median_kg_95_ci":
                             traits.paired_improvement(true, cid, estimates, frames, task="weight")}
    summary["mae_improvement_right_over_left_kg_95_ci"] = traits.paired_improvement(
        true, prediction["left"], prediction["right"], frames, task="weight")
    summary["age_evaluated"] = False
    summary["field_approved"] = False
    traits.write_summary(config.RESULTS_DIR / "cowdatabase2_weight.json", summary)


if __name__ == "__main__":
    main()
