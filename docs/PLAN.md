# Project plan

**Question:** does a cattle breed classifier trained on one dataset still work on photos from other sources?

**Approach:**
1. Fine-tune EfficientNet-B0 (ImageNet weights) on CID, Bangladeshi market cattle in 5 breeds.
2. Test it on CID itself with 5-fold cross-validation split by animal.
3. Test it on two other datasets: Mendeley (same country, other photographers) and Kaggle (Indian web images).
4. Test it on blurred, darkened and occluded copies of all three.

The approved scope is in [proposal.docx](proposal.docx); the course rules are in [guidelines.pdf](guidelines.pdf). Progress is tracked in [PROGRESS.md](PROGRESS.md), reasons in [decisions.md](decisions.md).

## Experiments

| ID | What | Tested on |
|---|---|---|
| B0 | Always predict "Local" (the majority class) | CID, 5 folds |
| E1 | Frozen ImageNet backbone, only a new head trained | CID, 5 folds |
| E2 | Whole network fine-tuned | CID, 5 folds |
| E3 | Final E2 model | Mendeley (190 images, 3 classes) |
| E4 | Final E2 model | Kaggle (about 930 images, 3 classes, reviewed by hand) |
| E5 | Final E2 model (fold models for CID) | Corrupted copies: blur, dark, occlusion at 2 strengths |
| E6 | *Optional:* E2 trained on 50% of the animals | CID, 5 folds |

**Metrics:**
- Headline metrics are macro-F1 and balanced accuracy, because "Local" is 70% of CID and plain accuracy rewards always guessing it.
- Also reported: accuracy, per-class F1 and the confusion matrix.
- CID results are the mean ± std over 5 folds.
- External sets are scored two ways. **5-way** is the normal prediction. **Restricted** only chooses among the 3 classes that set contains.

## How the code is organised

```
prepare_data.py   step 1  download, index, clean, split, cache          -> data/, results/cid_folds.csv
train.py          step 2  B0, E1, E2 with 5-fold CV; then final model   -> models/, results/metrics.csv
evaluate.py       step 3  E3, E4, E5, confusion matrices, Grad-CAM      -> results/metrics.csv, results/figures/
demo.py           step 4  Gradio app: upload a photo, get the breed

cattle_breed/
  config.py       every setting (classes, label mapping, sizes, learning rates, corruptions)
  download.py     where the datasets come from; download and check them
  data.py         images -> index table -> duplicates removed -> folds -> .npz cache; corruptions
  model.py        EfficientNet-B0 + head, two-stage training, Grad-CAM
  metrics.py      scores, restricted scoring, writing results/metrics.csv
tests/            one test file per module; synthetic images only, no real data needed
```

Each step reads the files the previous step wrote, so any step can be rerun on its own.

## Rules that protect the results

1. **Split by animal, never by image.** All 4 photos of an animal stay in the same fold. Otherwise the model is tested on animals it has already seen.
2. **No duplicate photos across splits.** Perceptual hashes (distance ≤ 4) remove repeated photos within CID and any external image that also appears in CID.
3. **Never divide pixels by 255.** Keras EfficientNet rescales internally and expects values in 0–255.
4. **The backbone runs with `training=False`.** BatchNorm statistics stay fixed while fine-tuning on a small dataset.
5. **No hue or saturation augmentation.** Coat colour separates the breeds.
6. **Do not drop hard external images.** Remove only images that are invalid (drawings, other species, wrong breed, and so on). Hard but valid photos are exactly what the test is meant to measure.

## Tasks

Work in test-first cycles: write the test, see it fail, write the code, see it pass. Tick tasks off in [PROGRESS.md](PROGRESS.md).

**T0 Setup and downloads ✅**
- One-command setup: `setup.ps1` and `setup.sh`.
- `cattle_breed/download.py` downloads the three datasets; the counts are in `download.EXPECTED`.

**T1 Image loading ✅**
- `data.load_image(path)` loads an RGB array, upright according to EXIF.
- `data.letterbox(img, 240)` fits the image in a square without distortion.
- `data.center_crop(img, 224)` takes the centre square.

**T2 Index tables and CID check (day 2)**
- `data.build_index(dataset) -> DataFrame` with columns `path, dataset, animal, label, label_idx`.
- CID:
  - labels come from `dataset.csv`;
  - match folders to SKUs with spaces removed;
  - folder `BLF2334` has no row, so drop it;
  - drop Pabna, Brahma and Mir Kadim.
- External sets: the label is the folder name, mapped with `config.to_canonical`, and `animal` = path.
- Tests use a tiny fake CID (two animals, one unlabelled folder, one `._` file) and check columns, labels and the dropped rows.
- Real data must give 503 animals / 2,012 images: Local 353, Sahiwal 59, Sindhi 38, Holstein cross 32, Red Chittagong 21.
- Save a grid of sample photos per class as `results/figures/cid_samples.png` (report Figure 1). Note which views the photos show (decision D-07).

**T3 Review the external images by hand (days 6–9, about 3 hours, no code)**
- Look through every Mendeley and Kaggle image of the matched classes in a file browser with thumbnails.
- Add each removed image to `results/removed_images.csv` (`dataset, path, reason`), following the rules in [data.md](data.md).
- `build_index` skips the listed paths.

**T4 Duplicates and folds (day 3)**
- `data.find_duplicates(paths) -> list[(kept, duplicate)]` uses 64-bit pHash with Hamming distance ≤ 4.
  - Test: a resized copy is found, and two different images are not.
- If two CID animals share a photo, they are probably the same animal. Give them the same `animal` value so they land in the same fold, and drop the copy.
- External images that duplicate a CID image, or an earlier image in their own set, are dropped.
- `data.make_folds(cid_index) -> Series` uses `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by animal.
  - Test: no animal appears in two folds, and every class appears in every fold.
- `data.split_validation(train_index) -> (train, val)` holds out 15% of the training animals for early stopping, also by animal.

**T5 Image cache (day 4)**
- `data.build_cache(index) -> (images uint8 [N, 240, 240, 3], labels)` letterboxes every image once. Files that fail to load are reported and skipped.
- `prepare_data.py` now runs T2–T5 end to end. It writes `data/processed/{cid,mendeley,kaggle}.csv` and `.npz`, plus `results/cid_folds.csv`.

**T6 Metrics and baseline B0 (day 4)**
- `metrics.score(y_true, y_pred) -> dict` returns accuracy, balanced accuracy, macro-F1 over the classes present, per-class F1 and the confusion matrix.
- `metrics.restrict(probs, classes) -> predictions` takes the argmax over the allowed classes only.
- `metrics.save_result(experiment, dataset, scoring, scores)` appends one row to `results/metrics.csv`.
- Tests:
  - perfect predictions give 1.0;
  - a class absent from `y_true` does not lower macro-F1;
  - `restrict` never predicts a disallowed class.
- B0 on CID should give about 0.70 accuracy and 0.165 macro-F1.

**T7 Model and cross-validation (E1, E2) (days 5–13)**
- `model.build_model()`: augmentation (flip, ±7° rotation, brightness, contrast) → EfficientNet-B0 → pooling → dropout 0.3 → 5-way softmax.
- `model.train(model, train, val)`, in two stages:
  1. Freeze the backbone and train the head at lr 1e-3 for up to 10 epochs. Predict now: this is E1.
  2. Unfreeze the backbone and train at lr 1e-5 for up to 20 epochs. This is E2.
  - Both stages use Adam, balanced class weights, label smoothing 0.1 and early stopping on `val_loss`.
- `train.py`:
  - for each fold: split off validation, train, save `models/fold_k.keras` and the out-of-fold predictions;
  - a fold that is already done is skipped, so it can resume after a Colab disconnect;
  - at the end, write B0, E1 and E2 (mean ± std) to `results/metrics.csv`.
- Tests on 20 tiny random images with `weights=None`:
  - the model outputs 5 probabilities summing to 1;
  - one training epoch runs;
  - a finished fold is skipped on rerun.
- `colab.ipynb` already runs every step on Colab; nothing to add there.
- **Check:** E2 > E1 > B0 in macro-F1. If not, check in this order:
  1. the input range is 0–255;
  2. labels match `CLASS_TO_IDX`;
  3. class weights are used;
  4. the model can overfit 50 images.

  Change one thing at a time.
- *Optional E6:* `train.py --fraction 0.5`.

**T8 Final model (day 13)**
- `train.py --final` trains on all CID animals and saves `models/final.keras`.
- It uses the median number of epochs that early stopping chose in each stage across the 5 folds; `train.py` records those in `models/folds.json`.
- With no held-out data left, there is no early stopping in this run.

**T9 External test (E3, E4) (days 14–16)**
- `evaluate.py` scores Mendeley and Kaggle with the final model, 5-way and restricted.
- It saves the confusion matrices to `results/figures/`.
- **Must tier done here.**

**T10 Robustness (E5) (days 20–22)**
- `data.corrupt(images, kind, severity)` applies the blur, dark or occlusion settings from `config.CORRUPTIONS` with seed 42.
- Tests: the output shape is unchanged, "dark" lowers the mean, and the same seed gives the same result.
- `evaluate.py --robustness`:
  - CID: each fold model on its own corrupted test fold;
  - external sets: the final model on their corrupted copies.

**T11 Error analysis and Grad-CAM (days 20–22)**
- `model.gradcam(model, image, class_idx) -> heatmap [224, 224]` in 0–1.
  - Test: the output is finite, even for a blank image.
- `evaluate.py --figures` makes a gallery of correct and wrong predictions, and Grad-CAM for about 12 of them.
- Write down the likely cause of each error shown: similar breed, pose, lighting, background, calf or cow.

**T12 Demo (day 23)**
- `demo.py`: Gradio app that letterboxes and centre-crops the uploaded photo, then shows the top-3 breeds with probabilities. `--share` prints a public link, which is needed on Colab.
- Record a backup video of it working.

**T13 Report, slides and defence (days 24–30)**
- Follow [report.md](report.md).
- Do a fresh run from a clean clone and check it reproduces `results/metrics.csv`.
- Rehearse with [defense_questions.md](defense_questions.md).

## Schedule (2–3 hours a day)

| Days | Tasks | Checkpoint |
|---|---|---|
| 1 | T0, T1 | ✅ data downloaded, tests green |
| 2–4 | T2, T4, T5, T6 | B0 in `results/metrics.csv` |
| 5–13 | T7, T8 (GPU training runs in the background) | E2 > E1 > B0, `final.keras` saved |
| 6–9 | T3, while training runs | external sets reviewed |
| 14–16 | T9 | **Must tier done:** E3, E4 logged |
| 17–19 | Buffer | — |
| 20–22 | T10, T11, optional E6 | E5 and figures done |
| 23 | T12 | demo works, backup recorded |
| 24–30 | T13 | report, slides, defence |

**If time runs short, cut in this order:**
1. E6;
2. fewer Grad-CAM examples;
3. one corruption strength instead of two.

**Never cut:** B0–E4.

## Where to train

Native Windows TensorFlow has no GPU. Data steps, tests and evaluation run fine on CPU. For T7 and T8, use one of:
1. **WSL2 with the local NVIDIA GPU:**
   - clone the repo inside Ubuntu (not under `/mnt/d`, which is slow);
   - run `bash setup.sh --gpu --data`.

   WSL2 needs CPU virtualization and the Windows hypervisor turned on.
2. **Colab (free T4 GPU):** open `colab.ipynb` and run its cells. See "Moving to Colab" in the README. The code detects Colab by itself and keeps data and models in Google Drive.
3. **CPU overnight:** `python train.py --folds 0` one fold per night. It resumes.

Record the choice in [decisions.md](decisions.md).

## Risks

| Risk | What to do |
|---|---|
| Colab disconnects or runs out of GPU quota | Rerun `train.py` (it resumes). Fall back to MobileNetV2 by changing one line in `model.py` |
| E2 not better than E1 | Use the T7 checklist, one change at a time |
| Many noisy Kaggle labels | Report the removal counts; Mendeley is the cleaner check |
| Grad-CAM shows the model uses the background | Report it as a finding; it is a valid result |
| Busy week in another course | Use the buffer days, then cut in the order above |
