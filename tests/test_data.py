"""Task 1: image loading, letterboxing and centre cropping."""
import numpy as np
import pytest

from cattle_breed.data import ImageLoadError, center_crop, is_image_file, letterbox, load_image


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


def test_center_crop():
    img = np.arange(240 * 240 * 3, dtype=np.uint32).reshape(240, 240, 3).astype(np.uint8)
    out = center_crop(img, 224)
    assert out.shape == (224, 224, 3) and (out == img[8:232, 8:232]).all()


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


def test_build_external_index(tmp_path, make_image, monkeypatch):
    from cattle_breed import data

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    holstein = make_image("raw/mendeley/Cow Breed/Holstain Friesian/nested/0.JPG")
    make_image("raw/mendeley/Cow Breed/Jersey/0.jpg")
    sindhi = make_image("raw/kaggle/Red_Sindhi/0.jpg")
    make_image("raw/kaggle/Local/0.jpg")
    make_image("raw/kaggle/Red_Sindhi/._0.jpg")

    assert data.build_index("mendeley").to_records(index=False).tolist() == [
        (str(holstein), "mendeley", str(holstein), "Holstein cross", 0),
    ]
    assert data.build_index("kaggle").to_records(index=False).tolist() == [
        (str(sindhi), "kaggle", str(sindhi), "Sindhi", 4),
    ]
