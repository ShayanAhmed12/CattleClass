# Curation procedure (Task 4)

The external test sets are reviewed by hand before testing. The Kaggle set in particular is uncurated, and its uploader states the labels are not expert-checked. The logs in this folder record every decision. They list file names only, never images.

## Files

- `mendeley_curation.csv` and `kaggle_curation.csv`: one row per image, with columns `path, label, decision, reason, note`. They are created by `python scripts/make_contact_sheets.py <dataset>` with every row set to `keep`. They are never overwritten once created.
- Contact sheets (numbered thumbnail grids) are written to `artifacts/curation/<dataset>/`, outside git.

## How to curate

1. Open a contact sheet. Each thumbnail shows its row number, which is its row in the CSV, counting from 0.
2. For each image that meets a **drop rule** below, set `decision` to `drop`, `reason` to the code, and optionally a short `note`.
3. Spend about 45 minutes per 300 images. When unsure, open the full image.
4. Run `apply_curation`. It refuses logs with missing rows or invalid codes. Then save `curation_summary` to `results/curation_summary.csv`.

## Drop rules (reason codes from `config.CURATION_REASONS`)

| Code | Drop when |
|---|---|
| `DRAWING` | Illustration, painting, 3D render, cartoon or stamp, not a photograph |
| `COLLAGE` | Several photos or several animals of different breeds combined into one image |
| `TEXT_WATERMARK` | Text, logos or watermarks cover a large part of the animal |
| `NOT_VISIBLE` | The animal is too small, heavily cropped or hidden to judge its breed at all |
| `OTHER_SPECIES` | Buffalo, goat, toy, statue or anything that is not a cow or bull |
| `WRONG_BREED` | Clearly not the labelled breed, for example a black-and-white Holstein in the Sahiwal folder |
| `UNCERTAIN_BREED` | You cannot confirm the label from the image, and its appearance contradicts the breed's typical look |
| `PERSON` | An identifiable face, number plate, ID card, screen or private document is prominent |
| `DUPLICATE` | The same photo appears earlier in the set. De-duplication catches most of these automatically |
| `CORRUPT` | The file will not open |

## Never drop an image because it is hard

Keep images with poor lighting, unusual pose, partial occlusion by fences or people in the background (faces not identifiable), calves, cows, dirty coats or busy backgrounds. These are the realistic variation the external test is meant to measure. Dropping them would inflate the results and is a form of cherry-picking (decision D-010).

## Breed reference cues (for `WRONG_BREED` / `UNCERTAIN_BREED`)

- **Sahiwal:** reddish-brown to dun; loose skin; large dewlap; heavy hump in males; short horns.
- **Red Sindhi:** deep red, often darker on the face and shoulders; compact body; smaller than Sahiwal. It is easily confused with Sahiwal, so mark `UNCERTAIN_BREED` only when it is clearly a different breed.
- **Holstein Friesian:** black-and-white (sometimes red-and-white) patches; large and angular dairy frame; no hump.
- **Local / Deshi (Mendeley):** small, varied colours; Bangladeshi indigenous cattle.

Check these cues against a reliable source, such as the TNAU agritech breed pages or NDDB, and cite that source in the report.
