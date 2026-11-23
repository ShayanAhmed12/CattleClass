"""Download and verify the published CID models for photo inference."""
import argparse

from . import config
from .data import atomic_output
from .download import fetch, sha256

MODEL_URL = "https://raw.githubusercontent.com/ShayanAhmed12/CattleClass/refs/tags/cid-models-v1"
SHA256 = {
    "cid-research.keras": "d5aab2625e30d65b63f025e56bdcbcc4f1a547d2d5fd85cdcedb5095c1adddf0",
    "cid-research.json": "e278c080ebc97fe84d83a1e836bd283f3d78ec131d92161485e182ab1bebd505",
    "final.keras": "738f02153e71ccfa61aed70f872d543facd8f243c155d1b65c9e7954262dcadf",
}


def installed_path(name):
    return config.MODELS_DIR / "pretrained" / name


def install(*, breed=False):
    names = ("final.keras",) if breed else ("cid-research.keras", "cid-research.json")
    for name in names:
        path = installed_path(name)
        if not path.is_file() or sha256(path) != SHA256[name]:
            print(f"Downloading {name}", flush=True)
            with atomic_output(path) as temporary:
                fetch(f"{MODEL_URL}/{name}", temporary)
                if sha256(temporary) != SHA256[name]:
                    raise ValueError(f"Checksum mismatch for {name}")
        print(f"Ready: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breed", action="store_true", help="Download the breed-only model.")
    args = parser.parse_args()
    try:
        install(breed=args.breed)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Model setup failed: {exc}. Rerun this command to retry.\n")


if __name__ == "__main__":
    main()
