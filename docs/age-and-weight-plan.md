# CattleClass age and weight implementation plan

**Date** — 21 November 2026

**Status** — Proposed. The metadata audit and constant baselines below are measured. Model training and field validation are planned.

Add estimated age and live weight alongside the existing five-breed classifier. Keep a single RGB photo as the first input to test. Use Python 3.12, TensorFlow's Keras API, EfficientNet-B0, and the existing Gradio app. Start with an isolated experiment on CID, then decide separately for age and weight which estimates can be shown to users.

## 1. What the existing data supports

I joined the original CID CSV to the checked-in fold manifest using the original SKU in each retained photo's path. The audit counts an original SKU once and keeps the existing duplicate-linked group as the split unit. The checked-in manifest contains the published breed splits. The current processed index contains breed and fold information. Age and weight are available in the raw metadata. [CID publication](https://doi.org/10.1145/3542954.3543018) · [Upstream dataset](https://github.com/bhuiyanmobasshir94/CID) · [CattleClass fold manifest](../results/cid_folds.csv)

| Evidence | Observed in the retained CID sample |
|---|---|
| Sample size | 1,893 photos, 503 original SKU records and 371 duplicate-linked groups |
| Recorded age | 375 at 24 months, 123 at 30 months, and 5 at 36 months |
| Recorded weight | 150 to 698 kg, with a median of 239 kg |
| Sex in the metadata | 499 bulls and 4 heifers |
| Label disagreement within duplicate-linked groups | 25 groups contain more than one age value. 47 contain more than one weight value. |
| Sparsity at the upper age label | The five held-out folds contain 0, 1, 2, 1, and 1 records at 36 months. |
| External data already in CattleClass | The existing Mendeley and Kaggle indexes provide breed labels. No verified age or live-weight labels are available in those indexes. |

In the 513 raw rows, the two-incisor label agrees with 24 months in all 381 records that report 24 months. Of the 126 records reporting 30 months, 125 list four incisors. All six records reporting 36 months list six incisors. This is strong evidence of association between the fields. The source materials reviewed do not establish how the individual ages or weights were measured. The upstream data-collection repository describes scraping cattle listings. Treat CID values as **recorded metadata** and seek confirmation of scale readings and how age was assigned before treating them as reference measurements. [Upstream collection repository](https://github.com/bhuiyanmobasshir94/Cow-weight-and-Breed-Prediction) · [Mississippi State Extension on estimating age using dentition](https://extension.msstate.edu/publications/estimating-cattle-age-using-dentition)

### Measured baselines to beat

Each row below uses the same five existing test folds. For each fold, I fitted the median or medians on the other four folds and scored one value per held-out original SKU. These are metadata benchmarks. Repeat them on the exact same eligible cases as each model.

| Baseline | Age MAE | Age exact-band accuracy | Weight MAE |
|---|---:|---:|---:|
| Median of the training fold data | 1.59 months | 74.6% | 44.10 kg |
| Training median for the true breed | 1.59 months | 74.6% | 34.52 kg |

The true-breed baseline uses the recorded breed as an input. For a fair image-only comparison, also build a **predicted-breed baseline** that classifies the photo and uses the corresponding training median. A weight model that beats 44.10 kg but trails the predicted-breed baseline may be learning a breed average. The 1.59-month age baseline reflects the narrow span and imbalance of CID. It says little about predicting the age of calves or mature cattle.

## 2. Approaches and recommended order

| Approach | What it tests | Decision |
|---|---|---|
| One RGB photo and small Keras heads | Learns cues from coat, body shape, and the rest of the visible animal. An ordinary photo can confound body size with camera distance and perspective. | Start here. Compare separate heads with a shared backbone before settling on the simpler model that meets the benchmarks. |
| Standardized side photo with a scale marker | Uses a full-body view taken with the animal standing side-on and a reference of known size in the same plane as the torso. Test geometric features such as height, body length, and silhouette area alongside the image model. | First fallback for **weight** if ordinary photos do not meet the usefulness threshold. Collect paired captures to measure the benefit. |
| Paired views or depth | A side and top or rear view can supply width. Depth can supply volume and physical scale under a defined capture protocol. | Evaluate later if a standardized side photo remains insufficient and collection conditions support a second view or sensor. |
| Lower-incisor photo and birth records | A specialist can use eruption and wear to estimate a broad age range. Reliable birth records and photo dates provide chronological age for validation. | First fallback for **age** if full-body performance is weak. Use images captured by trained handlers during routine checks and validate tooth-stage labels with an independent reviewer. |

Published work supports testing standardized views and depth. A dual-view RGB study used 3,210 paired images from 107 Simmental cattle under fixed acquisition conditions. A separate depth study used 196 Angus steers. Their reported errors apply to those datasets and setups. The public Girolando Figshare deposit contains figures and tables, so it is useful as a reference. Its listing does not show a complete set of animal-level photos paired with weights for training. [Dual-view RGB study](https://doi.org/10.3390/ani16162532) · [Depth study](https://doi.org/10.3389/fanim.2026.1752658) · [Girolando deposit](https://doi.org/10.6084/m9.figshare.12027582.v1)

**Recommended first version** — Run a CID research benchmark for weight in kilograms and a separate benchmark for the three recorded age bands. Assess each outcome independently. Weight is the stronger first candidate because CID has a much wider spread of recorded weights. Broader age estimates need reliable dates and better coverage across ages. A controlled photo, a second view, or a dental photo becomes a separate input mode only if the evidence supports it.

## 3. Data contract and split rules

- Derive `original_sku` from the photo's parent folder using the same whitespace normalization as CID indexing. Use the processed index's `animal` value as `group_id` and preserve its test `fold`. The raw SKU provides the label. The group controls the split. Keep original labels when duplicate-linked records disagree. Publish a sensitivity comparison with conflicting groups excluded.
- Join `age_in_year` as months and `weight_in_kg` as kilograms. Add `age_valid` and `weight_valid` independently. Missing, nonnumeric, nonfinite, or nonpositive targets fail validation for that target. Preserve valid labels for the other target. Stop on duplicate normalized SKU records with conflicting metadata. Report unusual but parseable values for review and document any exclusion rule before evaluation.
- Keep paths relative to the data directory and match the cached path order. Keep raw metadata, images, checkpoints, and per-animal predictions private. Track aggregate metrics and model provenance in Git.
- Use only image pixels as model input for the first experiment. Keep SKU, site, sex, breed labels, dentition, height, price, filename, and EXIF metadata out of the input tensor. Use metadata for analysis and splitting. Use original paths only to join records.
- Retain the five existing outer folds to compare against published breed performance. Fit preprocessing, target scaling, model settings, and early stopping using only training and grouped validation records inside each outer fold. Before a release-oriented test, lock the chosen settings and use a separate site or collection period. Keep all views and repeat visits of each animal in one partition and review possible cross-collection duplicates.
- The primary benchmark evaluates one eligible photo per original SKU. For the unfiltered benchmark, pick the first retained photo in sorted path order, before seeing labels or predictions. For a defined side-photo mode, select the first view passing a documented quality review. Report eligibility and coverage. Report all-photo sensitivity with each SKU weighted equally. In a separate multi-view experiment, combine scores from views of the same animal and report the number of photos required.
- Use all views during training with sample weights that give each original SKU equal total weight within each head's valid subset. Track image, SKU, and group counts. Compare a geometry-preserving crop with the current center crop for weight, since cropping or normalization can remove silhouette and scale cues. Keep the input format fixed for the initial benchmark before introducing additional comparisons.

## 4. Model and evaluation design

**Baselines.** Run the train-only median, true-breed median, and predicted-breed median. Compute the last one with a breed classifier fitted inside the corresponding training split. Existing out-of-fold breed predictions can be reused after verifying the run ID, row order, split, and checksum. If a model learns to combine predicted probabilities and images, use out-of-fold probabilities on its training data. Run a regularized linear weight model on frozen image embeddings before attempting full fine-tuning. If the standardized-photo experiment proceeds, compare a simple regression on measured silhouette features with the image model on the same animals.

**Keras models.** Start with an ImageNet-initialized EfficientNet-B0 backbone, global average pooling, and a small head per task. Train weight and age as separate experiments so each target can be assessed on its own. Then compare a shared three-output model if both heads show value. Retain the current breed loss and five-label order in that comparison. Use a three-output age softmax ordered as 24, 30, and 36 months. During CID model selection, prioritize the two well-supported labels and report 36 months descriptively across held-out predictions. Start with categorical cross-entropy, compare capped inverse-frequency weights on validation data, and use the most likely band for class metrics and the probability-weighted age in months for the primary research MAE. Sparse or absent training bands must be reported, with no claims about their accuracy. For weight, use a scalar output and Huber loss on standardized log kilograms. Fit the mean and scale on the training subset only, then invert the transform before computing metrics. Start with a frozen backbone. Fine-tune only if it improves held-out performance. For a shared model, use independent per-output sample weights and finite placeholders for missing targets, so one missing label never invalidates another head. Normalize each head's loss by the valid sample-weight sum for that head. Return zero for an all-missing head in a batch, and score task metrics on valid rows. Verify with a gradient or loss test.

| Question | Metrics and comparisons |
|---|---|
| Does age estimation add signal? | MAE in months, balanced accuracy, per-band recall and support, confusion matrix, and within-six-month accuracy. Pool out-of-fold predictions and compute primary balanced accuracy as the average recall of the supported 24- and 30-month classes. A 36-month prediction still counts as a miss for a 24- or 30-month record. Report 36-month recall and support descriptively, plus the overall three-band MAE for transparency. Compare primary MAE and balanced accuracy with the constant baseline on the same supported records. Report the three-month-or-smaller error fraction on new data if recording granularity supports it. |
| Does weight estimation add signal? | MAE and RMSE in kg, median absolute error, mean signed error, and the fraction within ±10% of reference weight. Compare all baselines on paired cases. Plot predicted versus recorded weight and residuals by weight range. |
| Who does the model work for? | Report support and errors by site, sex, breed, age or weight range, and view quality. Mark tiny or missing strata as insufficient. Show coverage and error together for any rule that withholds predictions. |
| Does sharing hurt breed prediction? | Re-run the existing CID breed metric on matched folds. Compare its macro-F1 and per-class recall with the current published breed model. Preserve the breed-only workflow if the shared model is weaker. |
| How uncertain is each result? | Report per-fold scores and paired bootstrap intervals for model-versus-baseline differences, resampling duplicate-linked groups. Report uncertainty in performance separately from any interval shown to an individual user. |

Lock the following proposed **pilot decision rules** before training. They are engineering targets for the experiment, not observed results.

1. **Weight** — Require at least 10% lower held-out MAE than the stronger of the global and predicted-breed baselines on paired cases. Require the 95% paired interval for the MAE improvement to exclude zero. With the preliminary global baseline alone, the 10% target is approximately 39.7 kg. If the model misses the criterion, advance to the standardized side-photo comparison.
2. **Age** — Require improvement in both held-out MAE and balanced accuracy over the median baseline. Treat the five 36-month records as descriptive only. Keep broad age display off until an external sample covers the intended age bands and demonstrates improvement. Birth dates and photo dates can support an age-in-months target there if the labels and image evidence justify it.
3. **Breed** — Use mean CID macro-F1 0.500 as a reference for the shared candidate, with a proposed tolerance of 0.02 absolute on the same evaluation protocol. Inspect per-breed recall before accepting the shared model. The existing breed model remains the fallback.
4. **Field display** — Before evaluating a field test set, agree on acceptable age and weight errors for the intended use, target populations, and capture mode. Separate training and tuning, interval calibration, and the reserved site test by animal. Calibrate separately for each capture mode. A simple starting option for weight is split conformal calibration on absolute errors, targeting 90% marginal coverage. Nominal conformal coverage assumes comparable, exchangeable calibration and test cases. Measure observed coverage and interval width on the reserved site, including among cases the demo would display after applying its withholding rule. For the discrete age head, report a calibrated set of supported bands if coverage permits. Disable a field estimate if its range is too wide to be useful, coverage is inadequate, the required capture is missing, or the population has insufficient evidence. Treat distribution shifts as a reason to validate again.

## 5. Field data plan

1. **Pilot the collection protocol on 30–50 independently identified cattle.** Use at least two participating farms or sites if feasible. Photograph each animal once with the ordinary phone workflow and once with a full-body side view and marker. Record how often the animal is fully visible, the marker is in the correct plane, and a reference measurement is available. The pilot estimates collection quality and variance. It is not an accuracy certification.
2. **Record reference labels at the time of photography.** Record live kilograms from a calibrated livestock scale, equipment ID, date/time and any failed weighing. Record an independently documented birth date and the photo date for chronological age. Use an approximate or unknown label flag where appropriate. A dental assessment can serve as a separate ordinal reference when birth records are unavailable. Keep those cases out of chronological-age validation.
3. **Capture protocol.** Use one standing animal, full torso and legs visible, minimal obstructions, and a straight side view with the camera approximately perpendicular to the animal. Place a known-size marker at approximately the torso plane. For a paired-view subset, add a consistent second view and keep both frames tied to the same animal and session. Record camera and site for analysis, with permission to retain the photos and anonymized records.
4. **Expand to the population the demo will serve.** Recruit across sexes, breeds, and age and weight ranges, including groups scarce in CID. Include different sites, phones, seasons, and reasonable lighting. Reserve at least one genuinely independent site or collection period for the final test and pool no repeated animals across splits. Estimate required sample size from pilot error variance, desired precision of interval coverage and MAE, site clustering, and support needed per subgroup. A provisional collection budget is 300–500 unique animals across multiple sites, to be revised after the pilot.
5. **External data screening.** For any public candidate source, verify license and access, stable animal IDs, capture setup, image-to-label pairing, units, and how reference weights or ages were established. Keep any source used for tuning separate from the external test. Mendeley and Kaggle can continue to evaluate breed as currently indexed.

## 6. Build sequence and repository changes

Use the existing four commands. The V1 extension adds one small traits module and a focused test module. Treat the training, evaluation, and demo flags below as proposed interfaces.

| File | Change |
|---|---|
| `cattle_breed/traits.py` | New module for the CID join, target preparation, Keras trait heads, and trait metrics. Keep the existing breed-only `model.py` and its checkpoint fingerprint compatible. |
| `train.py` | Add `--traits` outside the existing mutually exclusive mode group. Route it to the trait workflow before the breed-only path. Support `--traits --final` for fitting private field-validation candidates that pass the CID pilot criteria. Reject combinations of `--traits` with `--baseline`, `--folds`, or `--fraction` in V1. Store trait checkpoints under `models/traits/`. |
| `evaluate.py` | Add `--traits` to score saved trait fold predictions and write `results/traits_metrics.csv`. Use new field datasets via an explicit manifest after acquisition. Reject combinations with `--robustness` or `--figures` in V1. Existing Mendeley and Kaggle remain breed tests. |
| `demo.py` | Add `--traits` as an opt-in display. Keep the current breed upload. Read artifact metadata for each available head, supported input, and output range. Require explicit field-approved metadata for each head before enabling it in the demo. |
| `tests/test_traits.py` | New focused tests for target joins, units and masks, group isolation, metrics, and Keras serialization. Extend the existing demo tests for feature states. |
| `docs/data.md`, `docs/PROGRESS.md`, and `README.md` | Record label provenance, benchmark choices, results, and user instructions as each stage lands. |

### Milestone A — Reproduce the audit and start the benchmark

- [ ] Add `join_cid_targets(index, metadata_csv)` with a row-aligned table of normalized SKU, group, fold, age in months, weight in kg, and independent validity flags. Verify a raw SKU with embedded whitespace, two SKUs in one group, conflicting duplicate metadata, and one missing target.
- [ ] Add `score_traits(y_true, y_pred, *, task)` and baseline scoring. Verify metric values on small arrays in months and kg, equal SKU weighting, zero support, and absence of test groups from baseline fitting. Train and score baseline medians against the same test entries used by image models.
- [ ] Verify on CID that the unfiltered run matches the counts and preliminary baselines above. Export only aggregate results and leave the existing breed metrics intact.

### Milestone B — Train and compare Keras heads

- [ ] Implement single-task models first. Test output shapes and finite, positive inverse-transformed weights. Save and reload through Keras, and verify numerical parity on a fixed input. Run a synthetic end-to-end fit on a CPU before spending time on full folds.
- [ ] Run `python train.py --traits` using the frozen backbone, then compare staged fine-tuning with an inner validation protocol. Evaluate each head with `python evaluate.py --traits`. Add shared-head training only after independent results warrant it. Test incomplete-label rows and verify zero contribution to that head's loss.
- [ ] Record the data hash, model version, preprocessing, split, seed, target scaling, eligible views, and per-fold artifact checksums. Test safe resume and mismatched checkpoint rejection. Use a separate trait fingerprint that includes targets and capture policy. Preserve `model.py`, `run_folds`, `_sample_training_animals`, `center_crop_batch`, `split_validation` and the existing breed cache and index format, which determine breed checkpoint compatibility. Rerun the existing breed checkpoint identity check after adding the new modes. Check all five folds and reproduce aggregate rows from saved predictions.
- [ ] Decide independently whether to continue each head, restrict the population, or move to the relevant capture fallback using the decision rules above. Record which signals were still uncertain.

### Milestone C — Validate and connect the demo

- [ ] Acquire the paired pilot and collect verified reference labels, then expand under the field data plan. For weight, test ordinary, standardized side, and optional second-view capture on the same eligible animals. For age, compare full-body and dental modes only where their reference labels and capture quality support comparison.
- [ ] Lock the final model and interval policy before testing on the reserved site. Publish subgroup support, coverage, and error. Mark each approved head independently and bundle inference metadata with the approved Keras checkpoint. Reuse the existing upload and image preprocessing where appropriate, with an explicit full-body side-view instruction for a capture-dependent estimate.
- [ ] In `python demo.py --traits`, display the breed scores and show the age range or weight in kilograms with an interval only when the corresponding validated model is available. Otherwise show a clear state such as **Age estimate unavailable** or **Retake a full side-view photo**. Verify missing, incompatible, and partial trait checkpoints still allow breed predictions. Verify rejected photos, rounding, range labels, and single-photo results.
- [ ] Run `python -m pytest -q`, the existing breed baseline, the trait evaluation, and a real Gradio upload with representative inputs. Record commands, metrics, supported conditions, and known limits in the docs. Keep per-animal photos and outputs private.

**First deliverable** — A repeatable CID benchmark that compares the new image heads with strong baselines, reports what the data can and cannot establish, and identifies whether an age or weight estimate merits field validation. Documented performance on CID remains development evidence. The reserved field test determines which estimates are suitable to display.
