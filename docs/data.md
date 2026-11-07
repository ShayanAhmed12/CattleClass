# Data

Sources, licences and counts for the three datasets, and the rules for reviewing the external images by hand. This becomes the data section and provenance table of the report. Images are never committed or redistributed.

## Provenance

| | CID (training) | Mendeley (test, E3) | Kaggle (test, E4) |
|---|---|---|---|
| Name | Cow Images Dataset | A comprehensive image dataset of Bangladeshi cow breed identification | Indian Bovine Breeds |
| Authors | Shagor, Alvi, Hasan (ICCA 2022) | Rupom, Tajuddin, Rahman, Tito, Anzum (Southeast University) | Kaggle user `lukex9442` |
| Reference | https://doi.org/10.1145/3542954.3543018 | https://doi.org/10.17632/yd84s35d73.1 | https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds (version 5) |
| Licence | Apache-2.0 | CC BY 4.0 (attribution required) | CC0 as declared. The images come from Google Images, so they are used for testing only, never redistributed, and only a few thumbnails appear in the report |
| How collected | Photos at Bangladeshi cattle markets, 4 per animal | Photos from several areas of Bangladesh | Google Images search by breed name; labels not checked by an expert |
| Downloaded | 2026-11-07 | 2026-11-07 | 2026-11-07 |
| On disk | 514 folders × 4 = 2,056 images; `dataset.csv` 513 rows (D-10) | 263 images: Holstain Friesian 69, Jersey 73, Local Deshi 56, Sahiwal 65 | 3 of 41 breeds kept: Holstein_Friesian 328, Red_Sindhi 162, Sahiwal 439 |
| Used | 503 animals / 2,012 images in 5 classes | 190 images in 3 classes (Jersey left out) | about 930 images in 3 classes, before review |
| SHA-256 | `77a5a7aadb079b4d0d64db8b69549131b95339a8e7023b6e4c812b5372e78448` (`images.tar.gz`) | `650e8cf58256988e1a743358a9cad15214bbed33a5c60b8c00e6ce2c378f0b43` (ZIP) | — |
| After review and de-duplication | *fill in T4* | *fill in T3/T4* | *fill in T3/T4* |

`python prepare_data.py` downloads all three and checks the counts. Pretrained weights: EfficientNet-B0 ImageNet weights from `keras.applications` (Tan & Le, 2019).

## Reviewing the external images (T3)

Look at every Mendeley and Kaggle image of the matched classes in a thumbnail view. Add each image you remove to `results/removed_images.csv` as `dataset,path,reason`, using one of these reasons:

| Reason | Remove when |
|---|---|
| `drawing` | Illustration, painting, render or cartoon, not a photo |
| `collage` | Several photos or animals combined into one image |
| `text` | Text or watermark covers much of the animal |
| `not visible` | The animal is too small, cut off or hidden to judge the breed |
| `other species` | Buffalo, goat, toy, statue |
| `wrong breed` | Clearly not the labelled breed, for example a black-and-white Holstein in the Sahiwal folder |
| `person` | Identifiable face, number plate or document is prominent (privacy) |
| `corrupt` | The file does not open |

**Keep hard images.** Poor light, odd poses, fences or people in the background, calves, cows and busy backgrounds are the realistic variation the test is meant to measure (D-09). When unsure, keep the image.

**Breed cues** (check them against a reliable source, such as the TNAU agritech or NDDB breed pages, and cite it):
- **Sahiwal:** reddish-brown to dun, loose skin, large dewlap, heavy hump in males, short horns.
- **Red Sindhi:** deep red, often darker on the face and shoulders; more compact than Sahiwal, and easily confused with it.
- **Holstein Friesian:** black-and-white (sometimes red-and-white) patches, large angular dairy frame, no hump.
- **Local / Deshi:** small, with varied colours; Bangladeshi indigenous cattle.
