# Age and weight research

## Three-output research prototype

Follow the [README](../README.md#run-the-research-demo) or [Colab notebook](../cattle-class.ipynb). To train instead of using the published model, follow the [training commands](../PROJECT.md#train-from-scratch). The final packaging command verifies held-out predictions, fits the age and weight heads, and saves `models/traits/cid-research.keras` with checksum metadata in the work directory.

The 503 CID references span 24–36 months and are concentrated at 24 and 30 months. Regression estimates on the held-out photos ranged from 23.3 to 27.8 months. Age error of 2.23 months exceeded the 1.59-month median baseline. Joint scoring uses one photo per animal. A correct prediction requires a matching breed, age within three months, and weight within 10%. The demo shows an average benchmark error for weight. [Joint metrics](../results/cid_joint.json)

## Reproduce the CID benchmark

Run `python train.py --traits` to estimate age and weight on held-out folds, then `python evaluate.py --traits` to publish the aggregate scores. This workflow includes the earlier 24- and 30-month age classification experiment. The current demo uses continuous age regression. `python train.py --traits --final` saves the separate CID weight candidate used in the external study. [Trait metrics](../results/traits_evaluation.json)

## Independent weight check on CowDatabase2

Run `python -m experiments.public_weight` after fitting the separate CID weight candidate. On 95 paired Angus animals, mean error was 222.41 kg for the left view and 237.11 kg for the right view. Every prediction underestimated the recorded weight. CowDatabase2 lacks verified same-session scale references and dated photos. [Aggregate results](../results/cowdatabase2_weight.json) · [Study](https://doi.org/10.3390/ani12162128)

## Capture and breed follow-up

To reproduce the crop and breed-rejection audit, install `rtmlib==0.0.16`, download [YOLOX-tiny weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx), and run `python -m experiments.capture /path/to/yolox_tiny.onnx`. A crop increased CID weight error from 22.20 to 25.76 kg on 478 matched photos. The tested rejection rule missed its target for all five simulated unknown breeds. [Aggregate results](../results/capture_diagnostics.json) · [Original study notes](https://github.com/ShayanAhmed12/CattleClass/blob/3090574/docs/traits.md#capture-and-breed-follow-up)

## Collect and lock a field study

Collect same-session photos and references, then reserve farms outside development and calibration. Use the paired capture pilot to choose a capture mode. [Field data plan](age-and-weight-plan.md#field-data-plan)

Use separate development, calibration, and test CSV files, with one attempt per animal and capture mode. Use stable IDs across sites and studies so repeat animals can be recognized. The required common columns are `animal_id,site_id,session_id,path,capture,quality_ok,photo_date`. Use relative image paths, ISO 8601 dates, and `yes` or `no` for quality.

Weight development and calibration require the weight fields below. Age development and calibration require the birth fields. These can come from different sources with verified labels for each task. The test CSV requires `breed` and both sets of reference fields. Leave an unavailable photo or label blank to keep that attempt in the denominator. [Breed-only training](data.md#specialist-datasets)

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

Use the same reserved animals and capture IDs across tasks for a future joint field test. Set age and weight tolerances and define how a correct breed is counted before opening the test. The current field commands report age and weight separately.

## Field data still needed

Field validation needs dated photos, documented breeds and birth dates, and same-session scale weights. The screened public releases did not provide all of those references for the same animals across independent farms.

### Additional public dataset screening

See the [source-by-source audit in Git history](https://github.com/ShayanAhmed12/CattleClass/blob/3090574/docs/traits.md#additional-public-dataset-screening).

## Start the controlled demo

Run the existing breed demo with `python demo.py`. For an enrolled field cohort and operator-verified capture, launch the local interface with its assigned cohort and mode.

```bash
python demo.py --traits --traits-cohort enrolled-program --traits-capture ordinary
```

The operator confirms the animal and capture protocol. Field mode can show an approved age band or a weight estimate and range. Missing, incompatible, or unapproved artifacts produce **Estimate unavailable** while breed predictions remain available. The controlled mode runs locally. [Implementation plan](age-and-weight-plan.md)
