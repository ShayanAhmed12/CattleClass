import hashlib

import pytest

from cattle_breed import config, pretrained, research


@pytest.fixture
def release_files(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    payloads = {"cid-research.keras": b"model", "cid-research.json": b"metadata", "final.keras": b"breed"}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    monkeypatch.setattr(pretrained, "SHA256", hashes)
    return payloads


def test_install_and_reuse_keep_training_artifacts_intact(monkeypatch, release_files):
    training_path = config.MODELS_DIR / "traits/cid-research.keras"
    training_path.parent.mkdir(parents=True)
    training_path.write_bytes(b"local model")
    downloads = []

    def fetch(url, path):
        name = url.rsplit("/", 1)[-1]
        downloads.append(name)
        path.write_bytes(release_files[name])

    monkeypatch.setattr(pretrained, "fetch", fetch)
    pretrained.install()
    pretrained.install()
    pretrained.install(breed=True)

    assert downloads == ["cid-research.keras", "cid-research.json", "final.keras"]
    assert training_path.read_bytes() == b"local model"
    assert research.preview_path().parent == training_path.parent
    assert pretrained.installed_path("cid-research.json").read_bytes() == b"metadata"
    assert pretrained.installed_path("final.keras").read_bytes() == b"breed"


def test_checksum_failure_leaves_no_model_and_can_retry(monkeypatch, release_files):
    monkeypatch.setattr(pretrained, "fetch", lambda url, path: path.write_bytes(b"bad download"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        pretrained.install(breed=True)
    assert not list(pretrained.installed_path("final.keras").parent.iterdir())

    monkeypatch.setattr(pretrained, "fetch", lambda url, path: path.write_bytes(release_files["final.keras"]))
    pretrained.install(breed=True)
    assert pretrained.installed_path("final.keras").read_bytes() == b"breed"


def test_research_loads_installed_bundle_then_prefers_local(monkeypatch, release_files):
    import json

    installed = pretrained.installed_path("cid-research.keras")
    installed.parent.mkdir(parents=True)
    installed.write_bytes(release_files[installed.name])
    record = {"stage": "cid_research", "field_approved": False, "age_output": "months",
              "breeds": list(config.CLASSES), "sha256": pretrained.SHA256[installed.name]}
    installed.with_suffix(".json").write_text(json.dumps(record))
    monkeypatch.setattr(research, "load_classifier", lambda path: path)

    assert research.load_preview() == (installed, record)
    local = research.preview_path()
    local.parent.mkdir(parents=True)
    local.write_bytes(installed.read_bytes())
    local.with_suffix(".json").write_text(json.dumps(record))
    assert research.load_preview() == (local, record)
