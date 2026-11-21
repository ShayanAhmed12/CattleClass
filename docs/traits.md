# Age and weight research

CattleClass can train and score photo-based age and weight candidates using the existing CID folds. The public demo continues to show breed predictions. The controlled field interface shows an estimate only for a field-approved model and capture mode.

## Reproduce the CID benchmark

Use the same Python and TensorFlow environment as the breed workflow.

```bash
python prepare_data.py --datasets cid
python train.py
python train.py --traits
python evaluate.py --traits
```

Trait training uses the verified held-out breed predictions to make the predicted-breed weight baseline. It caches ImageNet features and saves private fold results so an interrupted run can resume. Trait evaluation writes aggregate results to `results/traits_metrics.csv` and `results/traits_evaluation.json`. It scores one photo per original SKU and keeps duplicate-linked groups intact. Images, feature caches, and per-animal predictions stay under the private work directory.

| Stage | Verified result |
|---|---|
| CID data | 1,893 photos, 503 original SKU records, and 371 duplicate-linked groups. Exact-byte duplicate review left the cohort unchanged. |
| Weight | 22.72 kg mean absolute error on 503 held-out records, compared with 39.33 kg for the predicted-breed median. The paired improvement was 16.61 kg with a group-bootstrap 95% interval of 13.15–20.26 kg. The CID research gate passed. |
| Age | 53.3% balanced accuracy for CID's 24- and 30-month labels on 498 records. The 95% paired interval for improvement over the constant baseline crossed zero. The 55% research gate failed. |
| Age fine-tuning check | On the fold 0 inner split only, the frozen age head reached 53.8% balanced accuracy. Unfreezing EfficientNet's top convolution for two epochs reached 51.7%. The held-out protocol uses the frozen model. |
| Background check | On 44 matched CID cases, Ridge reached 30.57 kg MAE on full photos, 28.66 kg on detected animal-box crops, and 23.98 kg on upper-left regions that excluded the animal. The small selected subset shows that background cues carry signal in CID. |
| Field validation | Weight and age fit, calibration, rejection, and demo withholding were tested on synthetic fixtures. Field metrics require independently collected reference measurements. |

CID results measure agreement with recorded labels. CID source weight and age measurement methods remain unverified. Before showing estimates, test with verified scale weights or documented birth dates on independent sites and report shortcut sensitivity across backgrounds and capture modes. [CID metrics](../results/traits_evaluation.json) · [Exploratory diagnostics](../results/traits_diagnostics.json) · [Implementation plan](age-and-weight-plan.md)

You can fit the private CID weight candidate for research.

```bash
python train.py --traits --final
```

The candidate is saved as `models/traits/candidates/cid-weight.keras` under the work directory. Its metadata excludes it from the field demo.

## Independent weight check on CowDatabase2

The public CowDatabase2 workbook contains 96 Aberdeen Angus records from one feedlot. Each has a birth date and reported live weight. The breed is documented for the cohort. We matched the workbook’s frame numbers to 95 unique left/right RGB pairs and excluded one record with no matching pair. The workbook’s live-weight range and mean match Table 1 of the study, which describes live weight among measurements obtained with 3D models. The release does not establish same-session scale weights or dated photos. [Public dataset](https://github.com/ruchaya/CowDatabase2/tree/3359f0a5a5e69d8416c2e939500b17d02da0ad81) · [Study and Table 1](https://doi.org/10.3390/ani12162128)

Run `python evaluate_public_weight.py` after fitting the CID candidate above. The script downloads the pinned workbook and side images in batches, verifies their checksums, and scores the frozen Keras model without fitting or calibrating on these animals. It uses the left view as the primary test and repeats the test on the right view for the same animals. Only aggregate results are saved. The public repository does not provide an explicit data licence, so the source files are linked here and excluded from the CattleClass repository. [Aggregate results](../results/cowdatabase2_weight.json)

| Prediction | Mean absolute error | Within 10% of recorded weight |
|---|---:|---:|
| CID weight model, left view | 222.41 kg | 0 of 95 |
| CID weight model, right view | 237.11 kg | 0 of 95 |
| Constant CID training median of 239 kg | 375.82 kg | 0 of 95 |

All predictions underestimated the reported weights, which ranged from 556 to 746 kg. Left-view error was 14.70 kg lower than right-view error on the same animals (paired bootstrap 95% interval of 5.98–23.47 kg). The error rules out using this candidate for this cohort. CowDatabase2 has no ordinary phone-photo or side-with-marker comparison. The study reports a cohort age of 16.5 months, and without photo dates individual ages at capture cannot be calculated. Aberdeen Angus also falls outside the five CID breed classes. The check qualifies neither trait for the demo.

## Collect and lock a field study

Make a separate private study folder for the planned capture comparison. Collect paired ordinary, side, and side-with-marker photos during the same session. Split by stable animal IDs, and reserve independent sites. Keep operator-confirmed marker placement and quality records. The current field model uses RGB pixels. Calibrated scale geometry, a second photo, and dental input are separate follow-up models if the paired field results warrant them.

Use separate development, calibration, and test CSV files, with one attempt per animal and capture mode. Use stable IDs across sites and studies so repeat animals can be recognized. The required common columns are `animal_id,site_id,session_id,path,capture,quality_ok,photo_date`. Use relative image paths, ISO 8601 dates, and `yes` or `no` for quality. All partitions must provide breed, weight and age references for the same animals. An attempt with an incomplete reference remains in the denominator. Leave an unavailable photo or label blank so the attempt can remain in the denominator.

- Set `capture` to `ordinary`, `side`, or `side_marker`. Marker captures also need positive `marker_cm` and `marker_px` measurements. Confirm placement and depth in the capture workflow.
- Record the documented breed in `breed`. For live weight, add `weight_kg,weight_source,weight_equipment_id,weighed_at`. The source must be `scale`, with timestamps for weighing and photography within two hours.
- For chronological age, add `birth_date,birth_source`. The source must be `record`. Define explicit month-range boundaries in the study JSON. The field model only assesses records within those bands.
- For subgroup gates, include `sex` when relevant and use `breed` and define the required labels in `required_strata`. A required group with insufficient evaluable cases prevents approval.

Create a private `study.json` with `task`, `capture`, `cohort`, `intended_use`, and the relative `development`, `calibration`, and `test` CSV paths. For age, add `age_edges_months`, such as `[0, 18, 30, 48]` for a protocol that covers those ranges. Add the SHA-256 of the reserved test CSV as `test_manifest_sha256`. Register the reserved photos for the selected mode as `test_media_sha256` using the same base directory as the study JSON. The custodian can compute both values before training.

```bash
python -c "from pathlib import Path; from cattle_breed.download import sha256; from cattle_breed.field_traits import test_media_checksum; base=Path('private-study'); p=base/'test.csv'; print(sha256(p)); print(test_media_checksum(p, 'ordinary', base=base))"
```

Set the `policy` object before training. Include `min_reference_count`, `min_reference_rate`, `min_display_rate`, and `min_coverage`. For weight, add `max_interval_width` and `max_mae_kg` in kilograms. For age, add `max_age_bands`, `min_balanced_accuracy`, and `min_per_band`. Set the thresholds for the actual use, population, and sample-size design.

```bash
python train.py --traits --final --traits-data private-study/study.json
python evaluate.py --traits --traits-data private-study/study.json
```

The first command fits and calibrates privately. The second checks the frozen model, data, policy and reserved animals, then evaluates the independent test. Test animals are tracked in a private ledger so a changed model or policy cannot simply reuse them for the same trait and capture mode. Results report the proportion of submissions receiving estimates, reference availability, and range coverage among displayed estimates. Approval depends on pooled, site, and declared subgroup gates. Keep the private ledger with the field data.

## Field data still needed

Field validation needs photographs and documented breeds, scale weights, and birth records linked by animal ID across independent sites. Record photo dates and weighing times so each reference can be checked against the capture.

## Start the controlled demo

Run the existing breed demo with `python demo.py`. For an enrolled field cohort and operator-verified capture, launch the local interface with its assigned cohort and mode.

```bash
python demo.py --traits --traits-cohort enrolled-program --traits-capture ordinary
```

The operator confirms the animal and capture protocol. Field mode can show an approved age band or a weight estimate and range. Missing, incompatible, or unapproved artifacts produce **Estimate unavailable** while breed predictions remain available. The controlled mode runs locally. [Implementation plan](age-and-weight-plan.md)
