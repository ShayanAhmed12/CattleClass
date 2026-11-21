# CID data

CattleClass uses the Cow Images Dataset (CID), which links photos to recorded breed, age and weight. The images were collected in Bangladeshi cattle markets. Keep source photos and per-animal results private. [Upstream dataset and licence](https://github.com/bhuiyanmobasshir94/CID) · [Publication](https://doi.org/10.1145/3542954.3543018)

| Field | Use in CattleClass |
|---|---|
| Source SKU | Matches original photos with their labels. Whitespace is removed before joining. |
| Breed | Five supported labels are Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi. |
| Age | `age_in_year` is converted to months. The research model uses the 24- and 30-month groups. |
| Weight | `weight_in_kg` is used as the recorded live-weight research label. |

## Preparation and splits

Run `python prepare_data.py` to download, check and cache CID. The image archive has 2,056 photos in 514 folders. Its SHA-256 is `77a5a7aadb079b4d0d64db8b69549131b95339a8e7023b6e4c812b5372e78448`. The metadata CSV has 513 rows.

Indexing retains 2,012 photos across 503 original animal IDs. Perceptual duplicate detection reduces these to 1,893 photos and links possible matches into 371 groups. Train and test partitions use those groups. Trait benchmarks score one representative photo per original animal ID. [Held-out folds](../results/cid_folds.csv) · [Trait benchmark and field data format](traits.md)

## Available viewpoints

A YOLOX-tiny detector and ViTPose-S model audited all indexed photos. Across 503 photos per suffix, the audit found 475, 497 and 434 image-left profiles for suffixes 0, 1 and 2, and 496 image-right profiles for suffix 3. These labels describe orientation within the frame. The camera geometry is uncalibrated. Run `python prepare_data.py --cid-samples` to make a private sample grid. [Pose audit](../results/viewpoint_and_detection_audit.json)

## Future field data

The CID weights and ages are source records of unconfirmed measurement provenance. The field workflow requires photos linked to a documented breed, a birth record and a scale weight, plus held-out test sites. [Field collection protocol](traits.md)
