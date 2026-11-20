# Data

CID supplies training and held-out animal groups. Mendeley and Kaggle supply external tests. Keep source photos private.

## Provenance

| | CID (training) | Mendeley (test, E3) | Kaggle (test, E4) |
|---|---|---|---|
| Name | Cow Images Dataset | A comprehensive image dataset of Bangladeshi cow breed identification | Indian Bovine Breeds |
| Authors | Shagor, Alvi, Hasan (ICCA 2022) | Rupom, Tajuddin, Rahman, Tito, Anzum (Southeast University) | Kaggle user `lukex9442` |
| Reference | https://doi.org/10.1145/3542954.3543018 | https://doi.org/10.17632/yd84s35d73.1 | https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds (version 5) |
| Licence | Apache-2.0 | CC BY 4.0 (attribution required) | CC0 as declared. The photos come from Google Images and are used only for testing. |
| How collected | Photos at Bangladeshi cattle markets, 4 per animal | Photos from several areas of Bangladesh | Google Images search by breed name; labels not checked by an expert |
| On disk | 514 folders × 4 = 2,056 images; `dataset.csv` 513 rows | 263 images: Holstain Friesian 69, Jersey 73, Local Deshi 56, Sahiwal 65 | 3 of 41 breeds kept: Holstein_Friesian 328, Red_Sindhi 162, Sahiwal 439 |
| Used | 503 animals / 2,012 images in 5 classes | 190 images in 3 classes (Jersey left out) | 929 images in 3 classes before review |
| SHA-256 | `77a5a7aadb079b4d0d64db8b69549131b95339a8e7023b6e4c812b5372e78448` (`images.tar.gz`) | `650e8cf58256988e1a743358a9cad15214bbed33a5c60b8c00e6ce2c378f0b43` (ZIP) | — |
| After deduplication and the automated screen | 1,893 images / 371 grouped IDs | 173 after pHash, 0 automated exclusions | 908 after pHash, 0 automated exclusions |

`python prepare_data.py` downloads and checks the raw datasets, then indexes, cleans and caches them. For CID alone use `--datasets cid`. External preparation always needs CID for the duplicate check. The Kaggle downloader reads the three matched folders from the version 5 ZIP using byte ranges and skips complete folders on rerun. `python prepare_data.py --cid-samples` builds the local `results/figures/cid_samples.png` grid from the downloaded CID files. Pretrained weights: EfficientNet-B0 ImageNet weights from `keras.applications` (Tan & Le, 2019).

## Reviewing the external images (T3)

Look at every Mendeley and Kaggle image of the matched classes in a thumbnail view. Add each image you remove to `results/removed_images.csv` as `dataset,path,reason`. Use dataset keys `mendeley` or `kaggle`, and paths relative to `data/` with forward slashes, for example `raw/kaggle/Sahiwal/photo.jpg`. Run preparation again after review. Use one of these reasons:

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

**Keep hard images.** Poor light, odd poses, fences or people in the background, calves, cows and busy backgrounds are the realistic variation the test is meant to measure. When unsure, keep the image.

**Breed cues** (check them against a reliable source, such as the TNAU agritech or NDDB breed pages, and cite it):
- **Sahiwal:** reddish-brown to dun, loose skin, large dewlap, heavy hump in males, short horns.
- **Red Sindhi:** deep red, often darker on the face and shoulders; more compact than Sahiwal, and easily confused with it.
- **Holstein Friesian:** black-and-white (sometimes red-and-white) patches, large angular dairy frame, no hump.
- **Local / Deshi:** small, with varied colours; Bangladeshi indigenous cattle.

## Automated screening

The published external results use a local CLIP screen with a pinned quantized checkpoint. It compared broad exclusion prompts with a valid-photo prompt on a center crop and an aspect-preserving view. In each view, an exclusion needed margins of 0.045 over the valid-photo prompt and 0.012 over the runner-up. All 1,119 matched external photos were screened, with zero threshold-qualified exclusions. The original hand review remains open. [Screening script and pinned inputs](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/screen_external.py) · [Aggregate screening results](../results/screening_summary.json)

## Independent object and viewpoint audit

A separate local audit used a COCO-trained YOLOX-tiny detector and ViTPose-S trained on APT-36K. It scanned all 2,012 indexed CID photos. A profile required nose and tail-root keypoint scores of at least 0.45 and horizontal separation of at least 45% of the detected cattle box width. Across 503 images per suffix, the pose audit found 475, 497 and 434 image-left profiles for suffixes 0, 1 and 2, and 496 image-right profiles for suffix 3. Suffixes 0–2 show predominantly image-left profiles. Suffix 3 shows predominantly image-right profiles. This describes the visible orientation in the image. The source metadata has no calibrated camera angles.

The detector also scanned the 1,119 matched external photos. It found no cow in 157 photos and detected at least one prominent person in 50 photos, using a person box of at least 15% of the frame. These counts can overlap. Detector misses can occur with valid cattle photos, and a person detection alone does not establish identifiability. The candidate list and images stay private. The benchmark retains the original automatically screened inputs, and the original hand-reviewed condition remains open. [Aggregate counts and checkpoint hashes](../results/viewpoint_and_detection_audit.json) · [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) · [ViTPose checkpoint](https://huggingface.co/JunkyByte/easy_ViTPose)

## Age and weight research labels

The CID trait study uses source SKU age and weight metadata as research labels, with duplicate-linked groups controlling the train and test split. See the verified cohort counts, age eligibility, exact-duplicate handling, and the private field manifest for scale and birth-record references. [Trait workflow](traits.md)
