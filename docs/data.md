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
| After deduplication and the automated screen | 1,893 images / 371 grouped IDs | 173 after pHash, 0 automated exclusions | 908 after pHash, 0 automated exclusions |

`python prepare_data.py` downloads and checks the raw datasets, then indexes, cleans and caches them. For CID alone use `--datasets cid`. External preparation always needs CID for the duplicate check. The Kaggle downloader reads the three matched folders from the version 5 ZIP using byte ranges and skips complete folders on rerun. The range reader was verified against the real archive after the individual-file endpoint rate-limited the setup. `python prepare_data.py --cid-samples` builds the local `results/figures/cid_samples.png` grid from the downloaded CID files. Pretrained weights: EfficientNet-B0 ImageNet weights from `keras.applications` (Tan & Le, 2019).

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

**Keep hard images.** Poor light, odd poses, fences or people in the background, calves, cows and busy backgrounds are the realistic variation the test is meant to measure (D-09). When unsure, keep the image.

**Breed cues** (check them against a reliable source, such as the TNAU agritech or NDDB breed pages, and cite it):
- **Sahiwal:** reddish-brown to dun, loose skin, large dewlap, heavy hump in males, short horns.
- **Red Sindhi:** deep red, often darker on the face and shoulders; more compact than Sahiwal, and easily confused with it.
- **Holstein Friesian:** black-and-white (sometimes red-and-white) patches, large angular dairy frame, no hump.
- **Local / Deshi:** small, with varied colours; Bangladeshi indigenous cattle.

## Autonomous screening

`screen_external.py` runs the quantized CLIP image and text encoders in ONNX Runtime on the same cloud workspace as the raw photos. The weight download contains model files only. Each image is compared with broad exclusion prompts on a center crop and an aspect-preserving view. An exclusion must win on both views by at least 0.045 cosine similarity over the valid-photo prompt and 0.012 over the runner-up. Uncertain photos remain in the set. The screen does not remove photos based on the study classifier's predictions or guess fine-grained breed errors.

```bash
python -m pip install onnxruntime==1.32.0 tokenizers==0.23.3
python screen_external.py data/curation/clip data/curation/run1 --download
```

The local run screened 190 matched Mendeley and 929 matched Kaggle images. It found zero threshold-qualified exclusions. This records the behavior of the screen and leaves the 173 and 908 previously deduplicated photos in the evaluation caches. It cannot establish that labels are correct or that all identifying details were detected. No source photos, review scores or galleries are distributed. The original hand-reviewed condition is reported as unverified. [Checkpoint metadata and aggregate counts](../results/screening_summary.json) · [Pinned checkpoint](https://huggingface.co/Xenova/clip-vit-base-patch32/tree/6b7c523703ae26e39be4b03774966f8571c7f523)

## Independent object and viewpoint audit

A separate local audit used a COCO-trained YOLOX-tiny detector and ViTPose-S trained on APT-36K. It scanned all 2,012 indexed CID photos. A profile required nose and tail-root keypoint scores of at least 0.45 and horizontal separation of at least 45% of the detected cattle box width. Across 503 images per suffix, the pose audit found 475, 497 and 434 image-left profiles for suffixes 0, 1 and 2, and 496 image-right profiles for suffix 3. Suffixes 0–2 show predominantly image-left profiles. Suffix 3 shows predominantly image-right profiles. This describes the visible orientation in the image. The source metadata has no calibrated camera angles.

The detector also scanned the 1,119 matched external photos. It found no cow in 157 photos and detected at least one prominent person in 50 photos, using a person box of at least 15% of the frame. These counts can overlap. Detector misses can occur with valid cattle photos, and a person detection alone does not establish identifiability. The candidate list and images stay private. The benchmark retains the original automatically screened inputs, and the original hand-reviewed condition remains open. [Aggregate counts and checkpoint hashes](../results/viewpoint_and_detection_audit.json) · [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) · [ViTPose checkpoint](https://huggingface.co/JunkyByte/easy_ViTPose)
