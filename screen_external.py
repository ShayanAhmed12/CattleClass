"""Screen external images privately with a local, pinned CLIP checkpoint."""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cattle_breed import config, data
from cattle_breed.download import sha256

MODEL = "Xenova/clip-vit-base-patch32"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"
FILES = {
    "tokenizer.json": "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
    "onnx/text_model_quantized.onnx": "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
    "onnx/vision_model_quantized.onnx": "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
}
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
EXTERNAL_PROMPTS = {
    "valid": ("a real photograph of a domestic cow or bull", "a photograph of a calf or cattle in a farm or market",
              "a real photograph of cattle, perhaps far away or partially obscured"),
    "drawing": ("an illustration or painting of a cow, not a photograph", "a cartoon drawing of cattle"),
    "collage": ("a multi-panel collage of different cow photographs", "several distinct cattle photos combined into a poster"),
    "other species": ("a photograph of a goat, sheep, or horse with no cattle", "a toy or statue of an animal, not a living cow"),
    "text": ("a text-heavy poster or a screenshot of a document", "an image of large text covering most of a cow picture"),
    "not visible": ("an outdoor photograph with no discernible animal", "a landscape photograph with no visible cow"),
    "person": ("a close-up photograph of a recognizable human face", "a prominent license plate or personal identification document"),
}
VIEW_PROMPTS = {
    "side": ("a side profile photograph of a bull showing its flank",),
    "front": ("a head-on photograph of a bull taken from the front",),
    "rear": ("a straight rear view of a bull showing its tail and hindquarters",),
    "oblique": ("a three-quarter photograph of a bull taken at a diagonal angle",),
    "detail": ("a close-up photograph of part of a bull or cow",),
}
# Require the same reason to beat all alternatives on both image preparations.
# The values are conservative decision margins in cosine similarity, not probabilities.
MARGIN = 0.045
RUNNER_UP_MARGIN = 0.012


def decision(scores, labels):
    reasons = []
    for row in scores:
        first, second = np.argsort(row)[-2:][::-1]
        if (labels[first] == "valid" or row[first] - row[0] < MARGIN
                or row[first] - row[second] < RUNNER_UP_MARGIN):
            return ""
        reasons.append(labels[first])
    return reasons[0] if reasons[0] == reasons[1] else ""


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def pixels(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        variants = [ImageOps.fit(rgb, (224, 224), method=Image.Resampling.BICUBIC),
                    ImageOps.pad(rgb, (224, 224), method=Image.Resampling.BICUBIC, color=(123, 116, 104))]
        images = np.asarray(variants, dtype=np.float32) / 255
    return ((images - MEAN) / STD).transpose(0, 3, 1, 2)


def session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def text_vectors(weights, groups):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
    network = session(weights / "text_model_quantized.onnx")
    prompts = [prompt for phrases in groups.values() for prompt in phrases]
    tokens = tokenizer.encode_batch(prompts)
    feed = {"input_ids": np.asarray([token.ids for token in tokens], dtype=np.int64),
            "attention_mask": np.asarray([token.attention_mask for token in tokens], dtype=np.int64)}
    output = normalize(network.run(["text_embeds"], {key: feed[key] for key in (i.name for i in network.get_inputs())})[0])
    offsets = np.cumsum([0, *(len(phrases) for phrases in groups.values())])
    return normalize(np.stack([output[start:end].mean(axis=0) for start, end in zip(offsets, offsets[1:])]))


def score_photo(model, weights, path):
    embeddings = model.run(["image_embeds"], {"pixel_values": pixels(path)})[0]
    return normalize(embeddings) @ weights.T


def checkpoint(weights, *, download):
    files = []
    for name, checksum in FILES.items():
        destination = weights / Path(name).name
        if destination.is_file() and sha256(destination) != checksum:
            raise ValueError(f"Screening checkpoint checksum mismatch for {name}")
        if not destination.exists():
            if not download:
                raise FileNotFoundError(f"Missing {destination}. Pass --download to fetch the pinned weights.")
            with data.atomic_output(destination) as temporary:
                urllib.request.urlretrieve(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", temporary)
                if sha256(temporary) != checksum:
                    raise ValueError(f"Screening download checksum mismatch for {name}")
        files.append([name, checksum, destination.stat().st_size])
    return {"model": MODEL, "revision": REVISION, "files": files}


def run(weights, output, *, download=False):
    provenance = checkpoint(weights, download=download)
    groups = {**EXTERNAL_PROMPTS, **VIEW_PROMPTS}
    vectors = text_vectors(weights, groups)
    external = vectors[:len(EXTERNAL_PROMPTS)]
    views = vectors[len(EXTERNAL_PROMPTS):]
    model = session(weights / "vision_model_quantized.onnx")
    counts, exclusions, audit, viewpoints = {}, [], [], []
    for source in ("mendeley", "kaggle"):
        index = data.build_index(source, apply_review=False)
        counts[source] = {"screened": len(index), "excluded": {}}
        for i, filename in enumerate(index.path, start=1):
            path = Path(filename)
            relative = path.relative_to(config.DATA_DIR.resolve()).as_posix()
            try:
                scores = score_photo(model, external, path)
            except (OSError, Image.DecompressionBombError):
                reason = "corrupt"
            else:
                reason = decision(scores, tuple(EXTERNAL_PROMPTS))
                audit.append((source, relative, *[f"{term:.5f}" for term in scores.mean(axis=0)]))
            if reason:
                exclusions.append((source, relative, reason))
                bucket = counts[source]["excluded"]
                bucket[reason] = bucket.get(reason, 0) + 1
            if i % 100 == 0:
                print(f"{source}: {i}/{len(index)} screened", flush=True)
    cid = data.build_index("cid")
    for label in config.CLASSES:
        rows = cid[cid.label == label]
        animals = rows.animal.drop_duplicates().sample(n=4, random_state=config.SEED)
        for filename in rows[rows.animal.isin(animals)].path:
            path = Path(filename)
            scores = score_photo(model, views, path)
            guesses = scores.argmax(axis=1)
            suffix = path.stem.rsplit("_", 1)[-1]
            view = tuple(VIEW_PROMPTS)[guesses[0]] if len(set(guesses)) == 1 else "ambiguous"
            viewpoints.append((suffix, view))
    view_counts = {}
    for suffix, view in viewpoints:
        bucket = view_counts.setdefault(suffix, {})
        bucket[view] = bucket.get(view, 0) + 1
    tables = {
        "removed_images.csv": (("dataset", "path", "reason"), exclusions),
        "scores.csv": (("dataset", "path", *EXTERNAL_PROMPTS), audit),
    }
    for name, (header, records) in tables.items():
        with data.atomic_output(output / name) as temporary, temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(records)
    summary = {"checkpoint": provenance, "counts": counts, "cid_view_hypotheses": view_counts}
    with data.atomic_output(output / "summary.json") as temporary:
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--download", action="store_true", help="fetch and verify the pinned checkpoint files")
    args = parser.parse_args()
    run(args.weights, args.output, download=args.download)


if __name__ == "__main__":
    main()
