# Data provenance

Fill this in during Task 0.3 and keep it current. It becomes the provenance table in the report. Images are never committed to git or redistributed.

| Field | CID (training) | Mendeley (external test 1) | Kaggle (external test 2) |
|---|---|---|---|
| Full name | Cow Images Dataset (CID) | A comprehensive image dataset of Bangladeshi cow breed identification | Indian Bovine Breeds |
| Authors / owner | Shagor, Alvi, Hasan (ICCA 2022) | Rupom, Tajuddin, Rahman, Tito, Anzum (Southeast University) | Kaggle user `lukex9442` |
| Citation / DOI | https://doi.org/10.1145/3542954.3543018 | https://doi.org/10.17632/yd84s35d73.1 | https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds |
| Download URL | https://cid-21.s3.amazonaws.com/images.tar.gz, `dataset.csv` | Mendeley Data page (browser download) | `kagglehub.dataset_download("lukex9442/indian-bovine-breeds")` |
| Licence | Apache-2.0 (repository) | CC BY 4.0 (attribution required) | CC0 as declared; images originally from Google Images, so test use only and no redistribution |
| How collected | Bangladeshi cattle market / farm photos, 4 per animal | Photos from different areas of Bangladesh | Google Images search by breed name, uncurated, non-expert labels |
| Labelled by | Dataset authors (metadata CSV) | Dataset authors | Uploader; curated by me (`docs/curation/kaggle_curation.csv`) |
| Used for | Training and CID cross-validation test (B0, E1, E2, E5, E6) | Test only (E3, E5) | Test only (E4, E5) |
| Expected counts | 513 animals / 2,052 images; 503 / 2,012 in 5 classes | 263 images; 190 in 3 matched classes | 5,949 images, 41 breeds; 933 in 3 matched classes |
| Download date | | | |
| SHA-256 | see `raw/cid/SHA256SUMS` on Drive: | | |
| Counts after curation and de-duplication | | | |
| Stored at | Drive `cattle-breed/data/raw/cid/` | Drive `cattle-breed/data/raw/mendeley/` | Drive `cattle-breed/data/raw/kaggle/` (3 class folders only) |

Pretrained weights: EfficientNet-B0 ImageNet weights via `keras.applications` (Tan & Le, 2019). Cite Keras and TensorFlow as well.
