"""Download checks and byte-range splitting (no network needed)."""
from cattle_breed.download import check_dataset, count_images, split_ranges


def test_split_ranges_covers_every_byte_once():
    ranges = split_ranges(start=10, total=105, n=4)
    assert ranges[0][0] == 10 and ranges[-1][1] == 104
    assert all(b + 1 == c for (_, b), (c, _) in zip(ranges, ranges[1:]))  # contiguous, no overlap
    assert sum(e - s + 1 for s, e in ranges) == 95


def test_split_ranges_more_parts_than_bytes():
    assert split_ranges(start=0, total=3, n=8) == [(0, 0), (1, 1), (2, 2)]


def test_split_ranges_nothing_left():
    assert split_ranges(start=50, total=50, n=8) == []


def test_count_images_per_folder_ignores_junk(tmp_path, make_image):
    make_image("root/A/1.jpg"); make_image("root/A/2.png"); make_image("root/B/1.jpg")
    (tmp_path / "root/A/._1.jpg").write_bytes(b"junk")
    (tmp_path / "root/B/notes.txt").write_text("x")
    assert count_images(tmp_path / "root") == {"A": 2, "B": 1}


def test_check_dataset_ok_and_mismatch(tmp_path, make_image):
    make_image("raw/toy/A/1.jpg"); make_image("raw/toy/B/1.jpg")
    spec = {"root": "raw/toy", "per_folder": {"A": 1, "B": 1}}
    assert check_dataset("toy", tmp_path, spec) == []
    spec_bad = {"root": "raw/toy", "per_folder": {"A": 2, "C": 1}}
    problems = check_dataset("toy", tmp_path, spec_bad)
    assert any("A" in p and "2" in p for p in problems) and any("C" in p for p in problems)


def test_check_dataset_folder_totals(tmp_path, make_image):
    for sku in ("S1", "S 2"):
        make_image(f"raw/cid/images/{sku}/0.jpg")
    (tmp_path / "raw/cid/dataset.csv").write_text("sku\n")
    spec = {"root": "raw/cid/images", "folders": 2, "images": 2, "files": ["raw/cid/dataset.csv"]}
    assert check_dataset("cid", tmp_path, spec) == []
    assert check_dataset("cid", tmp_path, {**spec, "folders": 513}) != []


def test_check_dataset_missing_root(tmp_path):
    problems = check_dataset("toy", tmp_path, {"root": "raw/none", "folders": 1, "images": 1})
    assert problems and "missing" in problems[0]



def test_get_kaggle_extracts_only_incomplete_breed_folders(tmp_path, make_image, monkeypatch):
    from zipfile import ZipFile

    from cattle_breed import download

    root = tmp_path / 'data'
    monkeypatch.setattr(download, 'DATA_DIR', root)
    monkeypatch.setattr(download, 'EXPECTED', {
        'kaggle': {'per_folder': {name: 1 for name in download.KAGGLE_KEEP}}
    })
    completed = make_image('data/raw/kaggle/Holstein_Friesian/existing.jpg')
    archive = tmp_path / 'source.zip'
    with ZipFile(archive, 'w') as output:
        for breed in (*download.KAGGLE_KEEP, 'Jersey'):
            image = make_image(f'source/{breed}/new.jpg')
            output.write(image, f'Indian_bovine_breeds/Indian_bovine_breeds/{breed}/new.jpg')
    monkeypatch.setattr(download, 'open_kaggle_zip', lambda: ZipFile(archive), raising=False)

    download.get_kaggle(tmp_path / 'downloads')
    assert completed.is_file()
    assert not (completed.parent / 'new.jpg').exists()
    assert not (root / 'raw/kaggle/Jersey').exists()
    assert (root / 'raw/kaggle/Red_Sindhi/new.jpg').is_file()
    assert (root / 'raw/kaggle/Sahiwal/new.jpg').is_file()


def test_kaggle_archive_ranges_retry_and_clip_at_eof(monkeypatch):
    import io

    from cattle_breed import download

    archive = b'abcdefghij'
    requested = []

    def fetch_range(request, *, timeout):
        raw = request.get_header('Range')
        requested.append(raw)
        if len(requested) == 1:
            raise OSError('temporary failure')
        first, last = map(int, raw.removeprefix('bytes=').split('-'))
        response = io.BytesIO(archive[first:last + 1])
        response.status = 206
        response.headers = {'Content-Range': f'bytes {first}-{last}/{len(archive)}'}
        return response

    monkeypatch.setattr(download.urllib.request, 'urlopen', fetch_range)
    source = download.ArchiveRanges('https://example.com/archive.zip', len(archive))
    assert source.seek(-3, io.SEEK_END) == 7
    result = bytearray(7)
    assert source.readinto(result) == 3
    assert result[:3] == b'hij'
    assert source.readinto(result) == 0
    assert requested == ['bytes=7-9', 'bytes=7-9']
