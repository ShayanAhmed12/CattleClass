# CattleClass datasets

CattleClass uses the Cow Images Dataset (CID), which links photos to recorded breed, age and weight. The images were collected in Bangladeshi cattle markets. Keep source photos and per-animal results private. [Upstream dataset and licence](https://github.com/bhuiyanmobasshir94/CID) · [Publication](https://doi.org/10.1145/3542954.3543018)

| Field | Use in CattleClass |
|---|---|
| Source SKU | Matches original photos with their labels. Whitespace is removed before joining. |
| Breed | Five supported labels are Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi. |
| Age | `age_in_year` is converted to months. The research demo fits a continuous target from all usable ages. The earlier two-label benchmark uses 24 and 30 months. |
| Weight | `weight_in_kg` is used as the recorded live-weight research label. |

## Preparation and splits

Run `python prepare_data.py` to download, check and cache CID. The image archive has 2,056 photos in 514 folders. Its SHA-256 is `77a5a7aadb079b4d0d64db8b69549131b95339a8e7023b6e4c812b5372e78448`. The metadata CSV has 513 rows.

Indexing retains 2,012 photos across 503 original animal IDs. Perceptual duplicate detection reduces these to 1,893 photos and links possible matches into 371 groups. Train and test partitions use those groups. Trait benchmarks score one representative photo per original animal ID. [Held-out folds](../results/cid_folds.csv) · [Trait benchmark and field data format](traits.md)

Preparation verifies the label-file checksum, matches the cleaned index against the published fold manifest, and caches images in the manifest order. The official benchmark uses the previously reproduced folds. The later Colab run used a different partition of the same photos and its results are available in Git history. [Alternate Colab run](https://github.com/ShayanAhmed12/CattleClass/commit/8b48bb7)

## Available viewpoints

A YOLOX-tiny detector and ViTPose-S model audited all indexed photos. Across 503 photos per suffix, the audit found 475, 497 and 434 image-left profiles for suffixes 0, 1 and 2, and 496 image-right profiles for suffix 3. These labels describe orientation within the frame. The camera geometry is uncalibrated. Run `python prepare_data.py --cid-samples` to make a private sample grid. [Pose audit](../results/viewpoint_and_detection_audit.json)

## Specialist datasets

The default commands reproduce the CID studies. A separate source can supply breed labels for breed research, birth records and photo dates for age development and calibration, or scale readings and photo times for weight development and calibration. Review source provenance, reuse terms, capture conditions and coverage for the intended task. [Trait workflow](traits.md)

To fit a private breed candidate, place a CSV beside its image folder with columns `split,animal_id,path,breed`. Use `development` or `validation` for `split`, paths relative to the CSV, and the supported breed names listed above. Use stable animal IDs across sources and assign all photos of an animal to one split. Review near-duplicate photos before training. The development split needs readable photos for all five supported breeds.

```bash
python train.py --breed-data private-study/breed.csv
```

This saves a Keras candidate under `models/breed-research/` in the work directory. The validation split is used for epoch selection. Reserve a shared final test with documented breed, age and scale weight for all three tasks. Keep its animals and sites independent of the data used by any model. [Field collection protocol](traits.md)
