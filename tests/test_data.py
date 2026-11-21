"""Task 1: image loading, letterboxing and centre cropping."""
import numpy as np
import pytest
from PIL import Image

from cattle_breed.data import ImageLoadError, center_crop_batch, is_image_file, letterbox, load_image


def test_load_image_converts_modes(make_image):
    # Greyscale, transparent PNG, palette PNG and normal JPEG must all come back as RGB uint8.
    for name, mode in [("g.png", "L"), ("a.png", "RGBA"), ("p.png", "P"), ("c.jpg", "RGB")]:
        img = load_image(make_image(name, mode=mode))
        assert img.dtype == np.uint8 and img.shape == (200, 320, 3), name


def test_load_image_applies_exif_rotation(make_image):
    # Phone photos store "rotate 90 degrees" in EXIF (orientation 6); height and width must swap.
    img = load_image(make_image("r.jpg", size=(300, 100), exif_orientation=6))
    assert img.shape[:2] == (300, 100)


def test_load_image_raises_on_corrupt(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        load_image(bad)


def test_letterbox_preserves_aspect():
    # A 100x200 (H x W) white image: the long side becomes 240, so content is 120 x 240, centred.
    out = letterbox(np.full((100, 200, 3), 255, np.uint8), 240)
    assert out.shape == (240, 240, 3) and out.dtype == np.uint8
    assert out[0, 120].sum() == 0  # black padding at the top
    assert out[120, 120].sum() == 765  # white content in the middle
    assert (out.sum(axis=2) > 0).sum() == pytest.approx(240 * 120, rel=0.02)


def test_letterbox_portrait():
    out = letterbox(np.full((300, 100, 3), 255, np.uint8), 240)
    assert out.shape == (240, 240, 3)
    assert out[120, 0].sum() == 0 and out[120, 120].sum() == 765  # padding on the sides


def test_center_crop_batch():
    images = np.arange(2 * 240 * 241 * 3, dtype=np.uint32).reshape(2, 240, 241, 3).astype(np.uint8)
    crops = center_crop_batch(images)
    assert crops.shape == (2, 224, 224, 3)
    np.testing.assert_array_equal(crops, images[:, 8:232, 8:232])


def test_is_image_file():
    assert is_image_file("a/B.JPG") and is_image_file("x.webp") and is_image_file("y.jpeg")
    assert not is_image_file("a/._B.jpg")  # macOS resource-fork junk in the CID archive
    assert not is_image_file("a/notes.txt")


def test_build_cid_index(tmp_path, make_image, monkeypatch):
    from cattle_breed import data

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    (tmp_path / "raw/cid").mkdir(parents=True)
    (tmp_path / "raw/cid/dataset.csv").write_text(
        "sku,breed\nBLF 2340,LOCAL\nBLF2341,HOSTINE_CROSS\nBLF2342,PABNA\n",
        encoding="utf-8",
    )
    local = make_image("raw/cid/images/BLF2340/0.jpg")
    holstein = make_image("raw/cid/images/BLF2341/0.jpeg")
    make_image("raw/cid/images/BLF2341/._0.jpeg")
    make_image("raw/cid/images/BLF2342/0.jpg")
    make_image("raw/cid/images/BLF2334/0.jpg")

    index = data.build_index("cid")
    assert list(index.columns) == ["path", "dataset", "animal", "label", "label_idx"]
    assert index.to_records(index=False).tolist() == [
        (str(local), "cid", "BLF2340", "Local", 1),
        (str(holstein), "cid", "BLF2341", "Holstein cross", 0),
    ]


def test_find_duplicates_of_resized_photo(make_image, tmp_path):
    from cattle_breed import data

    original = make_image('original.png', noise=True)
    distinct = make_image('distinct.png', noise=True, seed=72)
    resized = tmp_path / 'resized.png'
    with Image.open(original) as image:
        image.resize((160, 100), Image.Resampling.LANCZOS).save(resized)
    assert data.find_duplicates([original, distinct, resized]) == [(original, resized)]


def test_cid_duplicate_groups_are_transitive():
    import pandas as pd
    from cattle_breed import data

    index = pd.DataFrame({'path': ['a1', 'b1', 'b2', 'c1', 'c2'],
                          'animal': ['a', 'b', 'b', 'c', 'c']})
    clean = data.clean_cid_index(index, [('a1', 'b1'), ('b2', 'c1')])
    assert clean.path.tolist() == ['a1', 'b2', 'c2']
    assert clean.animal.tolist() == ['a', 'a', 'a']


def test_folds_and_validation_keep_animals_together():
    import pandas as pd
    from cattle_breed import data

    index = pd.DataFrame([(f'{breed}_{animal}', breed) for breed in range(5)
                          for animal in range(20) for _ in range(4)], columns=['animal', 'label_idx'])
    folds = data.make_folds(index)
    assert index.assign(fold=folds).groupby('animal').fold.nunique().eq(1).all()
    assert index.assign(fold=folds).groupby('fold').label_idx.nunique().eq(5).all()
    pd.testing.assert_series_equal(folds, data.make_folds(index))
    train, val = data.split_validation(index[folds != 0])
    assert set(train.animal).isdisjoint(val.animal)
    assert set(train.index).union(val.index) == set(index.index[folds != 0])
    assert val.animal.nunique() == 12
    assert val.label_idx.nunique() == train.label_idx.nunique() == 5


def test_cache_skips_unreadable_rows_and_preserves_labels(tmp_path, make_image, capsys):
    import pandas as pd
    from cattle_breed import data

    good = make_image('good.png', size=(30, 50))
    bad = tmp_path / 'bad.jpg'
    bad.write_bytes(b'invalid')
    index = pd.DataFrame({'path': [str(bad), str(good)], 'label_idx': [4, 2]})
    images, labels, valid = data.build_cache(index, return_index=True)
    assert images.shape == (1, 240, 240, 3) and images.dtype == np.uint8
    assert labels.tolist() == [2]
    assert valid.path.tolist() == [str(good)]
    assert 'bad.jpg' in capsys.readouterr().err
    assert data.build_cache(index.iloc[:1])[0].shape == (0, 240, 240, 3)


def test_cache_round_trip_with_relative_image_paths(tmp_path, make_image, monkeypatch):
    from cattle_breed import data

    monkeypatch.setattr(data, 'DATA_DIR', tmp_path)
    photo = make_image('raw/cid/images/BLF1/0.jpg')
    (tmp_path / 'raw/cid/dataset.csv').write_text('sku,breed,age_in_year,weight_in_kg\nBLF1,SINDHI,2,350\n')
    index = data.build_index('cid').assign(fold=0)
    images, labels = data.build_cache(index)
    data.save_cache('cid', index, images, labels)
    stored = (tmp_path / 'processed/cid.csv').read_text()
    assert 'raw/cid/images/BLF1/0.jpg' in stored and str(tmp_path) not in stored
    actual, restored, targets = data.load_cache('cid')
    assert actual.path.tolist() == [str(photo)] and targets.tolist() == [4]
    np.testing.assert_array_equal(images, restored)
    (tmp_path / 'processed/cid.csv').write_text(stored.replace('0.jpg', '1.jpg'))
    with pytest.raises(ValueError, match='does not match'):
        data.load_cache('cid')


def test_corruption_preserves_shape_and_is_seeded():
    from cattle_breed import data

    rng = np.random.default_rng(71)
    original = rng.integers(0, 256, size=(3, 30, 32, 3), dtype=np.uint8)
    copy = original.copy()
    for kind in ('blur', 'dark', 'occlude'):
        for severity in (1, 2):
            changed = data.corrupt(original, kind, severity)
            assert changed.shape == original.shape
            assert changed.dtype == np.uint8
            assert not np.array_equal(changed, original)
            np.testing.assert_array_equal(changed, data.corrupt(original, kind, severity))
    assert data.corrupt(original, 'dark', 2).mean() < data.corrupt(original, 'dark', 1).mean() < original.mean()
    np.testing.assert_array_equal(original, copy)
    with pytest.raises(ValueError, match='corruption'):
        data.corrupt(original, 'light', 1)


def test_load_image_propagates_oversize_as_load_error(make_image, monkeypatch):
    path = make_image('oversize.png', size=(20, 20))
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 100)
    with pytest.raises(ImageLoadError, match='cannot read image'):
        load_image(path)


def test_non_cid_caches_and_raw_indexes_are_rejected():
    from cattle_breed import data

    with pytest.raises(ValueError, match="uses CID"):
        data.build_index("breed_only")
    with pytest.raises(ValueError, match="uses CID"):
        data.load_cache("breed_only")
