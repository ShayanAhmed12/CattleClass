# Cattle Breed Classification: Project Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Within 30 days, one student builds and evaluates an EfficientNet-B0 transfer-learning classifier. It is trained on the CID cow dataset (5 breeds) and tested on CID cross-validation folds, two independent online datasets and corrupted copies of all of them. The student delivers the notebook, README, an 8–10 page report, a 10-minute demo and an individual oral defence.

**Architecture:** All logic lives in a small tested Python package, `src/cattle_breed`. Thin command-line scripts in `scripts/` run one stage each, and `notebooks/run_all_colab.ipynb` runs those scripts on a Colab GPU, with data on Google Drive. Each stage writes one file that the next stage reads: index CSV → folds CSV → image cache (NPZ) → predictions CSV → metrics rows. Any stage can be rerun on its own, and each one can be explained on its own in the defence.

**Tech stack:** Python ≥ 3.10, TensorFlow / Keras 3 (`keras.applications.EfficientNetB0`, ImageNet weights), scikit-learn, pandas, Pillow, imagehash, matplotlib, Gradio, pytest, Git and GitHub (private repo), Google Colab and Google Drive.

**Spec:** `docs/proposal/CS4045_Project_Proposal_Cattle_Breed_Classification_OnlineOnly.docx` (the scope this plan implements) and `docs/reference/CS4045_Project_Guidelines.pdf` (the course rules and rubric). The farm-visit version of the proposal (`..._Revised.docx`) is kept only as an alternative. If it is the one approved, Task 3 and Task 4 change; see "If the farm version is approved" at the end.

---

## How to use this plan

1. **Work top to bottom.** Each task lists its files, its interfaces (exact names and types that other tasks rely on), checkbox steps and an exit check. Tick boxes in this file as you go and commit the file with your work.
2. **Daily routine (2–3 focused hours):**
   - Open this file and pick the first unticked step.
   - Work in test-first cycles: write the test, see it fail, write the code, see it pass, commit.
   - Spend the last 10 minutes on four things:
     - add any decision to `docs/decision_log.md`;
     - update `docs/PROGRESS.md` (task status, date, notes);
     - write 3–5 report-ready sentences about what you did into `docs/report/notes.md`;
     - push.
3. **Gates.** Do not start a phase until the previous gate passes (see the schedule). If a gate fails, use that task's "If it fails" box before moving on.
4. **Code written with AI help** must be read line by line, run through its tests, and explained in one sentence in the commit message. The oral defence can ask about any line (guidelines §5).
5. **Never** commit images, archives, caches or models. `.gitignore` already blocks them. Never upload dataset images to annotation platforms or AI tools.

### Where things run

| Work | Where | Why |
|---|---|---|
| Writing code, unit tests, docs | Your Windows PC (`.venv`, CPU) | Fast edit–test loop; tests use synthetic images |
| Downloads, caching, training, evaluation, demo | Google Colab with GPU (T4) | Native Windows TensorFlow has no GPU support |
| Datasets, caches, trained models | Google Drive `MyDrive/cattle-breed/` | Survives Colab disconnects; never in git |
| Code, small results, docs | Private GitHub repository | Colab clones it; your history is the audit trail |

Your RTX 5050 could be used through WSL2. That is not part of this plan, so don't spend time on it unless Colab becomes unusable.

---

## Schedule and gates

Hours assume 2–3 hours a day. The total is about 64 hours, including the Could-tier task E6.

| Day | Tasks | Hours | Gate at end of day |
|---|---|---|---|
| 1 | T0.1 approvals email, T0.2 environment, T0.3 downloads | 3 | **G0:** email sent; all three datasets on Drive; local `pytest` green |
| 2 | T1 image loading, T2 CID index and audit | 4 | — |
| 3 | T5a CID self-duplicates, T6 folds | 2.5 | — |
| 4 | T7 cache, T8 metrics and B0 | 3 | **G1:** `cid_240.npz` built; B0 row in `results/experiments.csv` |
| 5 | T9 pipeline, T10 model | 4 | — |
| 6–7 | T11 training and CV runner (code and smoke test); start the real CV run on Colab | 5 | **G2:** CV run started; fold 0 finished |
| 6–9 | T3 external indices, T4 manual curation, T5b external de-duplication (while the GPU runs) | 7.5 | **G3:** both external sets curated and de-duplicated |
| 8–13 | T11 CV results and debugging, T12 final model | 4 | **G4:** E2 macro-F1 > E1 > B0; `final_e2.keras` saved |
| 14–16 | T13 external evaluation (E3, E4) | 2.5 | **G5 (Must tier done):** E3 and E4 rows logged |
| 17–19 | **Buffer** for other courses or slipped tasks | 0–6 | — |
| 20–22 | T14 robustness (E5), T15 error analysis and Grad-CAM, T16 E6 if time allows | 7.5 | **G6:** Should-tier experiments done |
| 23 | T17 Gradio demo | 2.5 | **G7:** demo works on Colab, with a backup recording |
| 24–28 | T18 reproducibility run and README, T19 report | 14 | **G8:** report PDF complete; fresh-runtime rerun matches |
| 29–30 | T20 slides, rehearsal, defence preparation | 6 | Submission |

**Scope tiers.** Finish each tier before starting the next.
- **Must:** T0–T13. That covers B0, E1, E2, E3 and E4.
- **Should:** T14 (E5), T15 (error analysis), T17 (demo).
- **Could:** T16 (E6) and the YouTube archive.
- **Cut order if time runs short:** E6 → the YouTube archive → fewer Grad-CAM examples → one corruption severity in E5.

---

## Global Constraints

Every task implicitly includes these. The values live in `src/cattle_breed/config.py`. Changing one needs a decision-log entry.

- **Class order** (model output column = index): `0 Holstein cross, 1 Local, 2 Red Chittagong, 3 Sahiwal, 4 Sindhi` (`config.CLASSES`).
- **CID labels** (`dataset.csv` `breed` column):
  - `HOSTINE_CROSS` (misspelt in the source) → Holstein cross;
  - `LOCAL` → Local;
  - `RED_CHITTAGONG` → Red Chittagong;
  - `SAHIWAL` → Sahiwal;
  - `SINDHI` → Sindhi;
  - `PABNA_BREED`, `BRAHMA` and `MIR_KADIM` are dropped.
- **External labels:**
  - Mendeley: `Holstein Friesian` → Holstein cross, `Local Deshi` → Local, `Sahiwal` → Sahiwal; `Jersey` is dropped.
  - Kaggle: `Holstein_Friesian` → Holstein cross, `Red_Sindhi` → Sindhi, `Sahiwal` → Sahiwal; all other breeds are dropped.
- **Expected CID counts after filtering:** 503 animals and 2,012 images:

  | Class | Animals | Images |
  |---|---|---|
  | Local | 353 | 1,412 |
  | Sahiwal | 59 | 236 |
  | Sindhi | 38 | 152 |
  | Holstein cross | 32 | 128 |
  | Red Chittagong | 21 | 84 |

  `images_count` is 4 for every SKU. There are 507 `MALE_BULL` and 6 `FEMALE_HEIFER` animals.
- **CID image layout:** `images/<sku>/*.jpg`.
  - The folder name equals the `sku` value exactly. Some SKUs contain a space (`BLF 2340`) and some don't (`BLF2001`).
  - The archive contains macOS `._*` files, which must be ignored.
- **Training data:** CID main archive only (`images.tar.gz`, 833 MB). The YouTube archive is Could tier only.
- **Splits:**
  - CID is split by animal: `group` = SKU, merged where two SKUs share a duplicate photo.
  - `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
  - The inner validation split takes 15% of the training-fold animals, also by group.
- **Seed:** 42 everywhere (`keras.utils.set_random_seed`, `np.random.default_rng`, sklearn `random_state`).
- **Images:**
  - Loaded as RGB `uint8` with EXIF rotation applied.
  - Letterboxed (aspect kept, black padding) to 240×240 and cached.
  - Training takes a random 224×224 crop; evaluation and the demo take a centre 224×224 crop.
- **Model input** is `float32` in [0, 255]. Never divide by 255: Keras EfficientNet rescales internally.
- **Augmentation** (inside the model, so it is active only in training):
  - `RandomFlip("horizontal")`, `RandomRotation(0.02)`, `RandomBrightness(0.1, value_range=(0, 255))`, `RandomContrast(0.1)`.
  - No hue, saturation or colour-jitter layers.
- **Training recipe:**
  - Optimiser Adam, batch 32, dropout 0.3, `CategoricalCrossentropy(label_smoothing=0.1)`.
  - Class weights `"balanced"`, computed on the training-fold images.
  - Stage 1 (E1): backbone frozen, lr 1e-3, ≤ 10 epochs, early-stopping patience 3.
  - Stage 2 (E2): whole network, lr 1e-5, ≤ 20 epochs, patience 4.
  - Early stopping monitors `val_loss` with `restore_best_weights=True`.
  - The backbone is always called with `training=False`, so BatchNorm stays frozen.
- **Metrics:**
  - Headline metrics are macro-F1 and balanced accuracy, plus accuracy, per-class precision, recall and F1, and the confusion matrix (5×5, rows are true classes).
  - Macro-F1 averages over the classes present in `y_true`, with `zero_division=0`.
  - Cross-validation results are reported as mean ± sample std (ddof=1) over the 5 folds, plus a pooled out-of-fold confusion matrix.
- **External scoring:** two scores per set. "5-way" uses the plain argmax. "Restricted" takes the argmax only over `config.DATASET_CLASSES[dataset]`.
- **Corruptions (E5):**
  - Gaussian blur, radius 2 and 4 px.
  - Darkening, ×0.5 and ×0.3.
  - Occlusion: one grey (127) square covering 12.5% and 25% of the image.
  - Applied to 240 px cached images with seed 42.
- **Duplicates:** 64-bit pHash with Hamming distance ≤ 4.
- **Licences:**
  - CID is Apache-2.0.
  - Mendeley is CC BY 4.0: attribute it.
  - Kaggle is CC0 as declared, but the images come from Google Images: test only, not redistributed, and only a few thumbnails in the report.
- **Results registry:** every reported number is a row in `results/experiments.csv`, which carries the git commit hash.

## Review Focus

These are the five failure modes most likely to quietly ruin results. Each one is pinned by a test in the task that owns the code.

1. **Leakage.** A SKU appears in two folds, two SKUs share one photo, or a test image duplicates a training image. Expected: impossible by construction. Tests: `test_no_group_in_two_folds` (T6), `test_merge_groups_unions_duplicate_skus` (T5), `test_find_duplicates_detects_resized_copy` (T5).
2. **Unusual image files.** Covers `._*` macOS files, RGBA or palette PNGs, greyscale, EXIF-rotated phone photos and corrupt files from web scraping. Expected: junk is skipped, odd modes become RGB, and corrupt files are reported, never crashing a 2,000-image job. Tests: `test_load_image_converts_modes`, `test_load_image_applies_exif_rotation`, `test_is_image_file` (T1), `test_build_cid_index_filters_and_maps` (T2), `test_build_cache_skips_and_reports_corrupt` (T7).
3. **Double rescaling.** Images reach EfficientNet in [0, 1] instead of [0, 255], so the model trains but badly. Test: `test_dataset_outputs_0_255_range` (T9).
4. **Class order mismatch** between index labels, model columns, external mappings and demo labels. Tests: `label_idx == CLASS_TO_IDX[label]` checks in T2 and T3, and `test_predict_image_keys_are_config_classes` (T17).
5. **Classes missing from external sets.** These can produce zero-division warnings, a macro-F1 deflated by classes the set does not contain, and NaN Grad-CAM maps when gradients are zero. Tests: `test_macro_f1_ignores_absent_true_classes`, `test_restrict_predictions` (T8), `test_gradcam_finite_for_blank_image` (T15).

A sixth trap, Colab disconnecting mid-run, is handled by the resumable CV runner: `test_run_cv_resumes_completed_folds` (T11).

---

## Repository layout

```
Project/
├── README.md                      overview, setup, run order
├── pyproject.toml                 package + extras: [train] [demo] [dev]
├── src/cattle_breed/
│   ├── config.py                  ALL fixed values (done)
│   ├── data/
│   │   ├── images.py              load_image, letterbox, center_crop, is_image_file      (T1)
│   │   ├── indices.py             build_cid_index, build_external_index                  (T2, T3)
│   │   ├── curation.py            init_curation_log, apply_curation, make_contact_sheets (T3)
│   │   ├── dedup.py               compute_hashes, find_duplicates, dedupe_within, merge_groups (T5)
│   │   ├── splits.py              make_folds, inner_val_split, subsample_groups          (T6)
│   │   ├── cache.py               build_cache, load_cache                                (T7)
│   │   ├── pipeline.py            make_dataset, build_augmentation                       (T9)
│   │   └── corruptions.py         apply_corruption, corrupt_batch                        (T14)
│   ├── models/
│   │   ├── build.py               build_model, set_stage, load_model                     (T10)
│   │   ├── train.py               class_weights, train_stage, run_fold, run_cv, train_final (T11, T12)
│   │   └── gradcam.py             gradcam, overlay                                       (T15)
│   └── eval/
│       ├── metrics.py             compute_metrics, restrict_predictions, cv_summary      (T8)
│       ├── baselines.py           majority_predict                                       (T8)
│       ├── registry.py            append_result, git_commit                              (T8)
│       ├── predict.py             predict_probs, evaluate_external                       (T13)
│       └── plots.py               plot_confusion, plot_gallery                           (T15)
├── scripts/                       one CLI per stage (argparse), run from the repo root
├── notebooks/                     01_cid_audit, 06_error_analysis, run_all_colab
├── app/demo.py                    Gradio demo                                            (T17)
├── tests/                         mirrors src/ (data/, models/, eval/); TF tests skip without TensorFlow
├── results/                       experiments.csv, splits/, curation and dedup summaries, figures (small; committed)
├── data/                          raw/ interim/ processed/ (git-ignored; Drive on Colab)
├── artifacts/                     models, CV outputs, predictions, contact sheets (git-ignored; Drive on Colab)
└── docs/                          this plan, proposal, logs, curation logs, report, presentation
```

**File flow:**

| Stage | Writes | Read by |
|---|---|---|
| T2, T3 | `data/interim/{dataset}_index.csv` | T4, T5 |
| T4 | `docs/curation/{dataset}_curation.csv` | T5b |
| T5 | `data/interim/{dataset}_final.csv` | T6, T7 |
| T6 | `results/splits/cid_folds.csv` | T11 |
| T7 | `data/processed/{dataset}_240.npz` and `.csv` | T8, T11–T15 |
| T11 | `artifacts/cv/<run>/` | T8 `cv`, T14, T15 |
| T12 | `artifacts/models/final_e2.keras` | T13–T15, T17 |
| T8 / T13 / T14 | rows in `results/experiments.csv` | report |

**Index CSV columns** (`indices.INDEX_COLUMNS`), used by every data file:

| Column | Meaning |
|---|---|
| `path` | Relative to `DATA_ROOT`, forward slashes |
| `dataset` | `cid`, `mendeley` or `kaggle` |
| `group` | SKU for CID; the path itself for external sets |
| `raw_label` | Label exactly as in the source |
| `label` | Canonical class name |
| `label_idx` | `CLASS_TO_IDX[label]` |

---

## Phase 0: Administration and environment (Day 1)

### Task 0.1: Approval request

**Files:** `docs/admin/approval_request_email.md` (draft, ready), `docs/decision_log.md`

- [ ] **Step 1:** Fill in your name and ID in the draft. Send it to the instructor or TA together with the proposal (`docs/proposal/..._OnlineOnly.docx`, with placeholders filled).
- [ ] **Step 2:** Add decision-log entry D-011 with the date sent. When the reply arrives, record what was approved. If hand-collected data is still required, follow "If the farm version is approved" at the end of this plan.

### Task 0.2: Local and Colab environment

**Files:** `README.md`, `notebooks/run_all_colab.ipynb` (create from the cells in Appendix A)

- [ ] **Step 1: Local virtual environment.** The `.venv` was created at setup. Recreate it if needed:
  ```powershell
  py -3.14 -m venv .venv --system-site-packages
  .venv\Scripts\python -m pip install -e ".[dev]"
  ```
  `--system-site-packages` lets the venv reuse your already-installed TensorFlow, so the TF tests run locally on CPU.
- [ ] **Step 2: Run the tests.**
  ```powershell
  .venv\Scripts\python -m pytest
  ```
  Expected: `tests/test_config.py` passes, with 6 tests.
- [ ] **Step 3: Git and GitHub.**
  - The repository is https://github.com/ShayanAhmed12/CattleClass (public). It was initialised and first pushed at setup.
  - A public repo needs no token to clone in Colab.
  - In Colab, open the key icon (Secrets) and add the Kaggle API credentials (from kaggle.com → Settings → API) as the secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- [ ] **Step 4: Colab notebook.** Create `notebooks/run_all_colab.ipynb` with the Appendix A setup cell. Set Runtime → Change runtime type → T4 GPU, then run the cell. Expected:
  - `nvidia-smi` shows a T4;
  - `import cattle_breed` works;
  - `pytest -q` passes on Colab too.
  Record the Colab TensorFlow and Keras versions in `docs/decision_log.md` (D-012).
- [ ] **Step 5: Commit.**
  ```bash
  git add -A
  git commit -m "chore: project scaffold, config and Colab setup"
  ```

### Task 0.3: Download the datasets to Drive

**Files:** `docs/provenance.md`

- [ ] **Step 1: CID.** In Colab, run the Appendix A download cell. It fetches `dataset.csv` and `images.tar.gz` into `/content`, records their SHA-256, extracts to `$CATTLE_DATA_ROOT/raw/cid/`, and deletes `._*` files. Expected: 513 folders under `raw/cid/images/`.
- [ ] **Step 2: Mendeley.**
  - Open https://data.mendeley.com/datasets/yd84s35d73 in your browser and download the ZIP.
  - Upload it to `MyDrive/cattle-breed/data/raw/mendeley/` and unzip it there in Colab.
  - Expected: four class folders, 263 images.
- [ ] **Step 3: Kaggle.**
  - In Colab, run the Appendix A Kaggle cell. It downloads `lukex9442/indian-bovine-breeds` (about 3 GB) with `kagglehub` to local disk.
  - It then copies **only** `Sahiwal/`, `Red_Sindhi/` and `Holstein_Friesian/` to `raw/kaggle/`.
  - Expected: 439, 166 and 328 files.
- [ ] **Step 4: Provenance.** Fill in `docs/provenance.md` with the date, URL, licence, SHA-256 (CID) and file counts for each dataset. Commit.

**Exit (G0):** the approval email is sent, the three datasets are on Drive with counts recorded, and local and Colab `pytest` both pass.

---

## Phase 1: Data (Days 2–9)

### Task 1: Image loading and letterboxing

**Files:**
- Modify: `src/cattle_breed/data/images.py`
- Test: `tests/data/test_images.py`

**Interfaces:**
- Consumes: `config.CACHE_SIZE`, `config.IMG_SIZE`, `config.IMAGE_SUFFIXES`
- Produces:
  - `class ImageLoadError(Exception)`
  - `load_image(path: str | Path) -> np.ndarray`: (H, W, 3) `uint8` RGB with EXIF orientation applied. Raises `ImageLoadError` (wrapping the PIL error) on unreadable files.
  - `letterbox(img: np.ndarray, size: int = CACHE_SIZE) -> np.ndarray`: (size, size, 3) `uint8`. Scales the longer side to `size`, keeps the aspect ratio, centres the image and pads with 0.
  - `center_crop(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray`
  - `is_image_file(path: str | Path) -> bool`: true when the suffix (case-insensitive) is in `IMAGE_SUFFIXES` and the name does not start with `._`

- [x] **Step 1: Write the failing tests**

```python
import numpy as np, pytest
from cattle_breed.data.images import ImageLoadError, center_crop, is_image_file, letterbox, load_image

def test_load_image_converts_modes(make_image):
    for name, mode in [("g.png", "L"), ("a.png", "RGBA"), ("p.png", "P"), ("c.jpg", "RGB")]:
        img = load_image(make_image(name, mode=mode))
        assert img.dtype == np.uint8 and img.shape == (200, 320, 3)

def test_load_image_applies_exif_rotation(make_image):
    img = load_image(make_image("r.jpg", size=(300, 100), exif_orientation=6))
    assert img.shape[:2] == (300, 100)          # rotated 90 degrees: H and W swap

def test_load_image_raises_on_corrupt(tmp_path):
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        load_image(bad)

def test_letterbox_preserves_aspect():
    out = letterbox(np.full((100, 200, 3), 255, np.uint8), 240)
    assert out.shape == (240, 240, 3) and out.dtype == np.uint8
    assert out[0, 120].sum() == 0 and out[120, 120].sum() == 765   # padding on top, content in the middle
    assert (out.sum(axis=2) > 0).sum() == pytest.approx(240 * 120, rel=0.02)

def test_center_crop():
    img = np.arange(240 * 240 * 3, dtype=np.uint32).reshape(240, 240, 3).astype(np.uint8)
    out = center_crop(img, 224)
    assert out.shape == (224, 224, 3) and (out == img[8:232, 8:232]).all()

def test_is_image_file():
    assert is_image_file("a/B.JPG") and is_image_file("x.webp")
    assert not is_image_file("a/._B.jpg") and not is_image_file("a/notes.txt")
```

- [x] **Step 2: Run the tests and see them fail**

  Run `pytest tests/data/test_images.py -v`. Expected: FAIL with an `ImportError`.
- [x] **Step 3: Implement the four functions and the exception in `images.py`**

  Use `PIL.ImageOps.exif_transpose` then `.convert("RGB")`, and catch `(OSError, PIL.UnidentifiedImageError)`. Resize with `Image.Resampling.BILINEAR`.
- [x] **Step 4: Run the tests and see them pass**

  Run `pytest tests/data/test_images.py -v`. Expected: 7 passed (the 6 above plus `test_letterbox_portrait`).
- [ ] **Step 5: Commit**

  `git commit -am "feat: image loading, letterbox and centre crop"`

### Task 2: CID index and data audit

**Files:**
- Modify: `src/cattle_breed/data/indices.py`
- Create: `scripts/build_indices.py`, `notebooks/01_cid_audit.ipynb`
- Test: `tests/data/test_indices.py`

**Interfaces:**
- Consumes: `is_image_file`, `load_image` (T1); `config.to_canonical`, `CLASS_TO_IDX`, `DATA_ROOT`
- Produces:
  - `INDEX_COLUMNS = ["path", "dataset", "group", "raw_label", "label", "label_idx"]`
  - `build_cid_index(csv_path: Path, images_dir: Path, data_root: Path) -> tuple[pd.DataFrame, list[str]]`. Returns:
    - the index, filtered to the 5 classes and sorted by (group, path), with `path` relative to `data_root` using `/`;
    - the SKUs of kept classes that have no folder or no images.
  - CLI `python scripts/build_indices.py cid` writes `DATA_ROOT/interim/cid_index.csv` and prints a class × (animals, images) table plus the missing SKUs.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from cattle_breed.config import CLASS_TO_IDX
from cattle_breed.data.indices import INDEX_COLUMNS, build_cid_index

def _cid_fixture(tmp_path, make_image):
    csv = tmp_path / "raw/cid/dataset.csv"; csv.parent.mkdir(parents=True)
    csv.write_text("sku,sex,breed\nBLF 2340,MALE_BULL,LOCAL\nBLF2001,MALE_BULL,HOSTINE_CROSS\n"
                   "X9,MALE_BULL,PABNA_BREED\nMISSING1,MALE_BULL,SAHIWAL\n")
    for sku in ["BLF 2340", "BLF2001", "X9"]:
        for i in range(4):
            make_image(f"raw/cid/images/{sku}/{i}.jpg")
    (tmp_path / "raw/cid/images/BLF 2340/._0.jpg").write_bytes(b"junk")
    return csv, tmp_path / "raw/cid/images"

def test_build_cid_index_filters_and_maps(tmp_path, make_image):
    csv, images = _cid_fixture(tmp_path, make_image)
    df, missing = build_cid_index(csv, images, data_root=tmp_path)
    assert list(df.columns) == INDEX_COLUMNS
    assert len(df) == 8 and set(df.group) == {"BLF 2340", "BLF2001"}   # ._ file and PABNA ignored
    assert set(df.label) == {"Local", "Holstein cross"}
    assert (df.label_idx == df.label.map(CLASS_TO_IDX)).all()
    assert df.path.str.startswith("raw/cid/images/").all() and not df.path.str.contains("\\\\", regex=False).any()
    assert missing == ["MISSING1"]
    assert (df.dataset == "cid").all()
```

- [ ] **Step 2: Run it and see it fail**

  `pytest tests/data/test_indices.py -v`
- [ ] **Step 3: Implement `build_cid_index` and the `cid` subcommand of `scripts/build_indices.py`**

  - Read the CSV with `pd.read_csv(csv_path, dtype={"sku": str})`.
  - Use `(images_dir / sku)` exactly as written, and fall back to `sku.strip()` if that folder is missing.
  - Use `Path.as_posix()` for the relative paths.
- [ ] **Step 4: Run it and see it pass**

  Then commit: `git commit -am "feat: CID index builder"`
- [ ] **Step 5: Run on the real data in Colab**

  `!python scripts/build_indices.py cid`. Expected:
  - Local 353 animals / 1,412 images;
  - Sahiwal 59 / 236;
  - Sindhi 38 / 152;
  - Holstein cross 32 / 128;
  - Red Chittagong 21 / 84;
  - total 503 / 2,012;
  - no missing SKUs.

  Any difference goes in the decision log before you continue.
- [ ] **Step 6: Audit notebook `01_cid_audit.ipynb`**

  - Load every indexed image once with `load_image` and count failures. Expected: 0. Failures are listed in the decision log.
  - Plot a histogram of image width and height.
  - Draw a grid of the 4 photos for 3 random SKUs per class (seed 42). Save it as `results/figures/cid_samples.png`; this is report Figure 1.
  - Record which views the 4 photos show and whether the order is consistent.
- [ ] **Step 7: Decide on views (D-013)**

  - If the 4 photos are mixed views, keep all of them and describe the task as "any view". This is the default.
  - Filter to side views only if a consistent file name or order marks them; this choice is less likely.

  Commit the notebook with its outputs cleared, plus the figure.

### Task 3: External indices and curation tooling

**Files:**
- Modify: `src/cattle_breed/data/indices.py`, `scripts/build_indices.py`
- Modify: `src/cattle_breed/data/curation.py`
- Create: `scripts/make_contact_sheets.py`
- Test: `tests/data/test_indices.py`, `tests/data/test_curation.py`

**Interfaces:**
- Consumes: `INDEX_COLUMNS`, `is_image_file`, `config.to_canonical(raw, dataset)`, `CURATION_REASONS`, `CURATION_DECISIONS`, `CURATION_ROOT`
- Produces:
  - `build_external_index(root: Path, dataset: str, data_root: Path) -> pd.DataFrame`:
    - Uses `INDEX_COLUMNS`. The class is the name of each image's **parent folder**, found recursively under `root`.
    - Rows whose parent folder maps to `None` are skipped. `group` = `path`. Sorted by path.
  - `CURATION_COLUMNS = ["path", "label", "decision", "reason", "note"]`
  - `init_curation_log(index: pd.DataFrame, out_csv: Path) -> None`: writes every row with `decision="keep"`. Raises `FileExistsError` if `out_csv` exists, so an existing log is never overwritten.
  - `apply_curation(index: pd.DataFrame, log_csv: Path) -> pd.DataFrame`: returns the kept rows. Raises `ValueError` if:
    - an index path is missing from the log;
    - a decision is not in `CURATION_DECISIONS`;
    - a `drop` row has a reason outside `CURATION_REASONS`.
  - `curation_summary(log_csv: Path) -> pd.DataFrame`: counts per (label, decision, reason).
  - `make_contact_sheets(index: pd.DataFrame, data_root: Path, out_dir: Path, per_sheet: int = 48, thumb: int = 160) -> list[Path]`:
    - Writes PNG grids, 8 columns.
    - Each thumbnail is captioned with its row number in the index, so you can find it in the log CSV.
  - CLI:
    - `python scripts/build_indices.py mendeley|kaggle` writes `interim/{dataset}_index.csv`.
    - `python scripts/make_contact_sheets.py kaggle` writes `ARTIFACTS_ROOT/curation/kaggle/*.png` and creates the log if it is missing.

- [ ] **Step 1: Write the failing tests**

```python
def test_external_index_maps_and_drops(tmp_path, make_image):
    for folder in ["Holstein_Friesian", "Red_Sindhi", "Jersey", "Gir"]:
        make_image(f"raw/kaggle/Indian_bovine_breeds/{folder}/a.jpg")
    df = build_external_index(tmp_path / "raw/kaggle", "kaggle", data_root=tmp_path)
    assert set(df.label) == {"Holstein cross", "Sindhi"} and len(df) == 2
    assert (df.label_idx == df.label.map(CLASS_TO_IDX)).all() and (df.group == df.path).all()

def test_mendeley_local_deshi_maps_to_local(tmp_path, make_image):
    make_image("raw/mendeley/data/Local Deshi/1.jpg")
    assert build_external_index(tmp_path / "raw/mendeley", "mendeley", tmp_path).label.tolist() == ["Local"]

# tests/data/test_curation.py
def test_init_curation_log_refuses_overwrite(tmp_path, small_index):
    log = tmp_path / "log.csv"; init_curation_log(small_index, log)
    with pytest.raises(FileExistsError): init_curation_log(small_index, log)

def test_apply_curation_requires_every_path(tmp_path, small_index):
    log = tmp_path / "log.csv"; init_curation_log(small_index.iloc[:-1], log)
    with pytest.raises(ValueError, match="missing"): apply_curation(small_index, log)

def test_apply_curation_drops_and_validates_reason(tmp_path, small_index):
    log = tmp_path / "log.csv"; init_curation_log(small_index, log)
    df = pd.read_csv(log); df.loc[0, ["decision", "reason"]] = ["drop", "DRAWING"]; df.to_csv(log, index=False)
    assert len(apply_curation(small_index, log)) == len(small_index) - 1
    df.loc[0, "reason"] = "BAD"; df.to_csv(log, index=False)
    with pytest.raises(ValueError, match="reason"): apply_curation(small_index, log)

def test_contact_sheets_written(tmp_path, small_index):
    sheets = make_contact_sheets(small_index, tmp_path, tmp_path / "sheets", per_sheet=2)
    assert len(sheets) == -(-len(small_index) // 2) and all(p.exists() for p in sheets)
```

Add a `small_index` fixture to `tests/data/test_curation.py`: 5 images made with `make_image`, indexed with `build_external_index`.

- [ ] **Step 2: Run them and see them fail**
- [ ] **Step 3: Implement the functions and CLIs**

  - Read logs with `pd.read_csv(..., keep_default_na=False)`, so an empty reason is `""` rather than NaN.
  - Draw the grid with `PIL.Image.new` and `ImageDraw.text`.
- [ ] **Step 4: Run them and see them pass**

  Then commit: `git commit -am "feat: external indices and curation tooling"`
- [ ] **Step 5: Run in Colab**

  `build_indices.py mendeley` and `build_indices.py kaggle`. Expected:
  - Mendeley: Holstein cross 69, Local 56, Sahiwal 65 (190 in total).
  - Kaggle: Holstein cross 328, Sahiwal 439, Sindhi 166 (933 in total).

  If a folder name is spelt differently, add its normalised form to `config.LABEL_MAP`, record the decision (D-014), and add a test case to `tests/test_config.py`.

### Task 4: Manual curation (no code, about 3.5 hours)

**Files:** `docs/curation/mendeley_curation.csv`, `docs/curation/kaggle_curation.csv`, `results/curation_summary.csv`. The procedure is in `docs/curation/README.md`.

- [ ] **Step 1:** Run `python scripts/make_contact_sheets.py mendeley`, then the same for `kaggle`. This creates the logs and the sheets.
- [ ] **Step 2:** Open each sheet. For every image that meets a drop rule in `docs/curation/README.md`, set `decision=drop`, a `reason` code and an optional `note` in the CSV.
  - Time-box this to about 45 minutes per 300 images.
  - **Do not drop images because they are hard** (odd pose, poor light, calves, cows). Those are the domain shift being measured. Dropping them would inflate E4.
- [ ] **Step 3:** Run `apply_curation` for both sets. It must not raise. Write `curation_summary(...)` for both sets to `results/curation_summary.csv`.
  - Expected: at least 50 kept images per class in each set.
  - If Kaggle Sindhi falls below 50, record that in the decision log and report the class with its count. Do not relax the rules.
- [ ] **Step 4:** Commit the logs and the summary. The logs contain file names only, no images.

### Task 5: Near-duplicates and group merging

**Files:**
- Modify: `src/cattle_breed/data/dedup.py`
- Create: `scripts/dedup.py`
- Test: `tests/data/test_dedup.py`

**Interfaces:**
- Consumes: index CSVs (T2, T3), `apply_curation` (T3), `load_image` (T1), `config.DEDUP_MAX_DISTANCE`
- Produces:
  - `compute_hashes(index: pd.DataFrame, data_root: Path) -> np.ndarray`: `uint64` array of length N. Uses `int(str(imagehash.phash(pil_image)), 16)`.
  - `hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray`: (len(a), len(b)) int matrix of Hamming distances.
  - `find_duplicates(test_hashes, train_hashes, max_distance=DEDUP_MAX_DISTANCE) -> np.ndarray`: bool mask over test rows that are within `max_distance` of any train row.
  - `dedupe_within(hashes, max_distance=DEDUP_MAX_DISTANCE) -> np.ndarray`: bool mask of rows to drop, keeping the first occurrence.
  - `merge_groups(groups: Sequence[str], pairs: Iterable[tuple[int, int]]) -> list[str]`: union-find over row pairs. Every member of a merged set gets the lexicographically smallest group name.
  - CLI:
    - `python scripts/dedup.py cid` hashes CID, merges SKUs that share a duplicate photo, and writes `interim/cid_final.csv`.
    - `python scripts/dedup.py external mendeley|kaggle` applies curation, drops rows that duplicate CID or an earlier row in the same set, and writes `interim/{dataset}_final.csv`.
    - Both append counts to `results/dedup_summary.csv`.

- [ ] **Step 1: Write the failing tests**

```python
def test_find_duplicates_detects_resized_copy(tmp_path, make_image):
    a = make_image("a.png", size=(320, 200), noise=True, seed=1)
    Image.open(a).resize((160, 100)).save(tmp_path / "a_small.jpg", quality=85)
    make_image("b.png", size=(320, 200), noise=True, seed=2)
    idx = pd.DataFrame({"path": ["a.png", "a_small.jpg", "b.png"]})
    h = compute_hashes(idx, tmp_path)
    assert find_duplicates(h[1:], h[:1]).tolist() == [True, False]

def test_dedupe_within_keeps_first():
    h = np.array([0b1010, 0b1011, 0xFFFF_0000_FFFF_0000], dtype=np.uint64)
    assert dedupe_within(h, max_distance=1).tolist() == [False, True, False]

def test_hamming_matrix():
    assert hamming_matrix(np.array([0], np.uint64), np.array([0b111], np.uint64)).tolist() == [[3]]

def test_merge_groups_unions_duplicate_skus():
    assert merge_groups(["S1", "S2", "S3", "S4"], [(0, 1), (1, 2)]) == ["S1", "S1", "S1", "S4"]
```

Noise images give pHash something to work on; blank images all hash alike.

- [ ] **Step 2: Run them and see them fail**
- [ ] **Step 3: Implement**

  For `hamming_matrix`, XOR via broadcasting, `view(np.uint8)`, then `np.unpackbits(...).reshape(..., 64).sum(-1)`. Chunk over rows if `len(a) * len(b) > 5e6`.
- [ ] **Step 4: Run them and see them pass**

  Then commit: `git commit -am "feat: perceptual-hash de-duplication and group merging"`
- [ ] **Step 5a (Day 3): Run `dedup.py cid`**

  Record the number of merged SKU pairs in D-015. Two different SKUs sharing a photo means the same animal, or a labelling error, and must never be split across folds.
- [ ] **Step 5b (Day 9, after T4): Run `dedup.py external mendeley` and `dedup.py external kaggle`**

  Commit `results/dedup_summary.csv`.

### Task 6: Cross-validation folds

**Files:**
- Modify: `src/cattle_breed/data/splits.py`
- Create: `scripts/make_folds.py`
- Test: `tests/data/test_splits.py`

**Interfaces:**
- Consumes: `interim/cid_final.csv` (T5); `config.N_FOLDS`, `INNER_VAL_FRAC`, `SEED`
- Produces:
  - `make_folds(index: pd.DataFrame, n_splits=N_FOLDS, seed=SEED) -> np.ndarray`: int fold (0–4) per row, aligned with `index`. Uses `StratifiedGroupKFold` with `y=label_idx` and `groups=group`.
  - `inner_val_split(subset: pd.DataFrame, val_frac=INNER_VAL_FRAC, seed=SEED) -> tuple[np.ndarray, np.ndarray]`: positional (train, val) indices within `subset`. Grouped and stratified, using the first split of `StratifiedGroupKFold(n_splits=round(1 / val_frac))`.
  - `subsample_groups(subset: pd.DataFrame, fraction: float, seed=SEED) -> np.ndarray`: positions of rows whose groups form a random `fraction` of each class's groups, rounded, with at least 1 group per class.
  - CLI `python scripts/make_folds.py` writes `results/splits/cid_folds.csv` (`path, group, label, fold`) and prints an animals-per-class × fold table.

- [ ] **Step 1: Write the failing tests**

```python
def synthetic_index():
    counts = {"Local": 353, "Sahiwal": 59, "Sindhi": 38, "Holstein cross": 32, "Red Chittagong": 21}
    rows = [(f"{c}-{i}", c) for c, n in counts.items() for i in range(n) for _ in range(4)]
    df = pd.DataFrame(rows, columns=["group", "label"]); df["label_idx"] = df.label.map(CLASS_TO_IDX)
    return df

def test_no_group_in_two_folds():
    df = synthetic_index(); df["fold"] = make_folds(df)
    assert df.groupby("group").fold.nunique().max() == 1 and set(df.fold) == {0, 1, 2, 3, 4}

def test_every_class_in_every_fold():
    df = synthetic_index(); df["fold"] = make_folds(df)
    animals = df.drop_duplicates("group").groupby(["label", "fold"]).size().unstack()
    assert animals.loc["Red Chittagong"].min() >= 4 and animals.notna().all().all()

def test_folds_deterministic():
    assert (make_folds(synthetic_index()) == make_folds(synthetic_index())).all()

def test_inner_val_split_grouped_and_fraction():
    df = synthetic_index(); tr, va = inner_val_split(df)
    assert not set(df.group.iloc[tr]) & set(df.group.iloc[va])
    assert 0.12 <= df.group.iloc[va].nunique() / df.group.nunique() <= 0.18
    assert df.label.iloc[va].nunique() == 5

def test_subsample_groups_half():
    df = synthetic_index(); keep = df.iloc[subsample_groups(df, 0.5)]
    per_class = keep.drop_duplicates("group").groupby("label").size()
    assert per_class["Local"] == round(353 * 0.5) and per_class["Red Chittagong"] == round(21 * 0.5)
    assert (keep.groupby("group").size() == 4).all()          # whole animals only
```

- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and see them pass.**
- [ ] **Step 5: Run `make_folds.py` on the real data**

  Check the printed table: every class appears in every fold, and Red Chittagong has 4–5 animals per fold. Commit `results/splits/cid_folds.csv`.

### Task 7: Image cache

**Files:**
- Modify: `src/cattle_breed/data/cache.py`
- Create: `scripts/build_cache.py`
- Test: `tests/data/test_cache.py`

**Interfaces:**
- Consumes: `load_image`, `letterbox`, `ImageLoadError` (T1); final index CSVs (T5)
- Produces:
  - `build_cache(index: pd.DataFrame, data_root: Path, out_npz: Path, size=CACHE_SIZE) -> tuple[pd.DataFrame, list[str]]`:
    - Writes `out_npz` with `images` (uint8, shape (N, size, size, 3)) and `paths`, plus a sibling CSV `out_npz.with_suffix(".csv")` holding the written index rows in the same order.
    - Returns (written index, failed paths). Images that fail to load are skipped and reported, never written.
  - `load_cache(out_npz: Path) -> tuple[np.ndarray, pd.DataFrame]`: raises `ValueError` if `paths` in the NPZ ≠ `path` in the CSV.
  - CLI `python scripts/build_cache.py cid|mendeley|kaggle` reads `interim/{dataset}_final.csv` and writes `processed/{dataset}_240.npz`. On Colab it writes to `/content/cache` first and then copies to Drive.

- [ ] **Step 1: Write the failing tests**

  - `test_build_cache_preserves_order`: 3 images of different sizes. The output shape is (3, 240, 240, 3) and the CSV paths are in the same order.
  - `test_build_cache_skips_and_reports_corrupt`: one corrupt file. 2 images are written, and the failed list equals that file's path.
  - `test_load_cache_detects_mismatch`: edit the CSV, then `load_cache` raises `ValueError`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**

  Use `np.savez` without compression, since JPEG pixels compress poorly and loading is faster.
- [ ] **Step 4: Run them and see them pass.**

  Then commit: `git commit -am "feat: letterboxed image cache"`
- [ ] **Step 5: Build the CID cache on Colab**

  Expected: about 2,012 × 240 × 240 × 3 bytes ≈ 350 MB, with no failures. Build the external caches after T5b.

### Task 8: Metrics, results registry and majority baseline (B0)

**Files:**
- Modify: `src/cattle_breed/eval/metrics.py`, `eval/baselines.py`, `eval/registry.py`
- Create: `scripts/evaluate.py` (subcommands `b0` and `cv`; later tasks add `external` and `robustness`)
- Test: `tests/eval/test_metrics.py`, `tests/eval/test_registry.py`

**Interfaces:**
- Consumes: `config.CLASSES`, `RESULT_COLUMNS`, `RESULTS_ROOT`; `results/splits/cid_folds.csv`; `processed/cid_240.csv`
- Produces:
  - `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] | None = None) -> dict`. Keys:
    - `accuracy`, `balanced_accuracy`, `macro_f1` (floats);
    - `per_class`: `{class_name: {precision, recall, f1, support}}` for `labels_used`;
    - `confusion_matrix`: list of lists, 5×5 over all classes, rows are true classes;
    - `labels_used`: `sorted(unique(y_true))` when `labels` is None.

    Balanced accuracy is the mean recall over `labels_used`. Use `zero_division=0`.
  - `restrict_predictions(probs: np.ndarray, allowed: Sequence[int]) -> np.ndarray`: argmax over the `allowed` columns, returned as original class indices.
  - `cv_summary(per_fold: list[dict]) -> dict`: `{metric: {"mean", "std"}}` for the three headline metrics, std with ddof=1.
  - `majority_predict(train_labels: np.ndarray, n: int) -> np.ndarray`
  - `git_commit() -> str`: short hash, or `"unknown"`.
  - `append_result(row: dict, csv_path: Path = RESULTS_ROOT / "experiments.csv") -> None`: fills `date` and `commit`, writes `RESULT_COLUMNS` in order, and writes the header only if the file is new or empty.
  - CLI:
    - `python scripts/evaluate.py b0` uses the majority class of the training folds for each fold and appends `B0 / cid / 5-way` with the mean and std.
    - `python scripts/evaluate.py cv --run artifacts/cv/full --experiments E1 E2 [--as E6]` reads the predictions CSVs (T11), computes metrics per fold and pooled, appends one row per experiment, and saves `results/{exp}_cid_metrics.json` (per-fold and pooled metrics plus the confusion matrix). `--as` records a single experiment under another ID (used by T16).

- [ ] **Step 1: Write the failing tests**

```python
def test_perfect_predictions():
    y = np.arange(5); m = compute_metrics(y, y)
    assert m["macro_f1"] == 1 and m["balanced_accuracy"] == 1 and m["accuracy"] == 1

def test_macro_f1_ignores_absent_true_classes():
    y = np.array([0, 0, 3, 3, 4, 4]); p = np.array([0, 1, 3, 3, 4, 2])
    m = compute_metrics(y, p)
    assert m["labels_used"] == [0, 3, 4]
    assert m["macro_f1"] == pytest.approx(np.mean([2 / 3, 1.0, 2 / 3]))
    assert m["balanced_accuracy"] == pytest.approx(np.mean([0.5, 1.0, 0.5]))
    assert np.array(m["confusion_matrix"]).shape == (5, 5)

def test_restrict_predictions():
    probs = np.array([[0.1, 0.6, 0.1, 0.15, 0.05]])
    assert restrict_predictions(probs, [0, 3, 4]).tolist() == [3]

def test_majority_predict():
    assert majority_predict(np.array([1, 1, 0]), 4).tolist() == [1, 1, 1, 1]

def test_cv_summary():
    s = cv_summary([{"macro_f1": .5, "accuracy": .6, "balanced_accuracy": .4},
                    {"macro_f1": .7, "accuracy": .8, "balanced_accuracy": .6}])
    assert s["macro_f1"]["mean"] == pytest.approx(.6) and s["macro_f1"]["std"] == pytest.approx(0.14142, rel=1e-3)

def test_append_result_creates_header_once(tmp_path):
    p = tmp_path / "r.csv"
    for _ in range(2): append_result({"experiment": "B0", "dataset": "cid", "macro_f1": 0.16}, p)
    df = pd.read_csv(p)
    assert list(df.columns) == list(RESULT_COLUMNS) and len(df) == 2
```

- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**

  Use `sklearn.metrics` with explicit `labels=`. Compute balanced accuracy yourself as the mean of `recall_score(..., labels=labels_used, average=None)`, because sklearn's version ignores `labels`.
- [ ] **Step 4: Run them and see them pass.**

  Then commit: `git commit -am "feat: metrics, results registry and majority baseline"`
- [ ] **Step 5: Run `evaluate.py b0` in Colab**

  Expected: accuracy ≈ 0.70, macro-F1 ≈ 0.165, balanced accuracy = 0.20. This is the floor that E1 and E2 must beat. Commit `results/experiments.csv`.

**Exit (G1):** the CID cache exists and B0 is logged.

---

## Phase 2: Modelling (Days 5–13)

### Task 9: Data pipeline and augmentation

**Files:**
- Modify: `src/cattle_breed/data/pipeline.py`
- Test: `tests/data/test_pipeline.py`, starting with `tf = pytest.importorskip("tensorflow")`

**Interfaces:**
- Consumes: `config.IMG_SIZE`, `BATCH_SIZE`, `SEED`, `NUM_CLASSES`, `AUG_*`
- Produces:
  - `make_dataset(images: np.ndarray, labels: np.ndarray | None, training: bool, batch_size: int = 32, seed: int = SEED) -> tf.data.Dataset`:
    - Yields `(x, y)`, or `x` alone when `labels` is None.
    - `x` is float32 (B, 224, 224, 3) in [0, 255]; `y` is one-hot float32 (B, 5).
    - `training=True`: shuffle the full length each epoch, then a random 224 crop (`tf.image.random_crop`).
    - `training=False`: keep the order, then a centre crop. Never repeats. Uses `prefetch(AUTOTUNE)`.
  - `build_augmentation() -> keras.Sequential` named `"augmentation"`:
    - `RandomFlip("horizontal")`
    - `RandomRotation(AUG_ROTATION)`
    - `RandomBrightness(AUG_BRIGHTNESS, value_range=(0, 255))`
    - `RandomContrast(AUG_CONTRAST)`

- [ ] **Step 1: Write the failing tests**

```python
tf = pytest.importorskip("tensorflow")

def test_dataset_outputs_0_255_range():
    imgs = np.full((10, 240, 240, 3), 200, np.uint8); y = np.arange(10) % 5
    x, yb = next(iter(make_dataset(imgs, y, training=False, batch_size=4)))
    assert tuple(x.shape) == (4, 224, 224, 3) and x.dtype == tf.float32
    assert float(tf.reduce_max(x)) == pytest.approx(200.0)        # not rescaled to [0, 1]
    assert tuple(yb.shape) == (4, 5)

def test_eval_keeps_order():
    imgs = np.zeros((7, 240, 240, 3), np.uint8); y = np.array([0, 1, 2, 3, 4, 0, 1])
    got = np.concatenate([b.numpy().argmax(1) for _, b in make_dataset(imgs, y, training=False, batch_size=3)])
    assert got.tolist() == y.tolist()

def test_training_crops_vary():
    imgs = np.random.default_rng(0).integers(0, 256, (1, 240, 240, 3), dtype=np.uint8)
    ds = make_dataset(imgs, np.array([0]), training=True, batch_size=1)
    a, b = (next(iter(ds))[0].numpy() for _ in range(2))
    assert not np.array_equal(a, b)

def test_augmentation_has_no_colour_layers():
    names = [type(l).__name__ for l in build_augmentation().layers]
    assert names == ["RandomFlip", "RandomRotation", "RandomBrightness", "RandomContrast"]

def test_augmentation_inactive_at_inference():
    x = tf.random.uniform((2, 224, 224, 3), 0, 255)
    assert np.allclose(build_augmentation()(x, training=False), x)
```

- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and see them pass, locally on CPU.**
- [ ] **Step 5: Commit.**

### Task 10: Model builder

**Files:**
- Modify: `src/cattle_breed/models/build.py`
- Test: `tests/models/test_build.py`, starting with `pytest.importorskip("tensorflow")`

**Interfaces:**
- Consumes: `build_augmentation` (T9); `config.NUM_CLASSES`, `IMG_SIZE`, `TrainConfig`
- Produces:
  - `build_model(num_classes=NUM_CLASSES, dropout=0.3, weights: str | None = "imagenet") -> tuple[keras.Model, keras.Model]` returns `(model, backbone)`. The graph is:
    - `Input((224, 224, 3))`
    - `build_augmentation()`
    - `backbone(x, training=False)`, where `backbone = keras.applications.EfficientNetB0(include_top=False, weights=weights, pooling="avg")` is named `"backbone"`
    - `Dropout(dropout, name="dropout")`
    - `Dense(num_classes, activation="softmax", name="classifier")`
  - `set_stage(model, backbone, stage: int, lr: float, label_smoothing: float = 0.1) -> None`:
    - stage 1 sets `backbone.trainable = False`; stage 2 sets it to `True`.
    - Then compiles with `Adam(lr)`, `CategoricalCrossentropy(label_smoothing=...)` and `metrics=["accuracy"]`.
    - Raises `ValueError` for any other stage.
  - `load_model(path: Path) -> keras.Model`: thin wrapper around `keras.models.load_model`.

- [ ] **Step 1: Write the failing tests**

  All tests use `weights=None` for speed and offline use.
  - `test_output_shape_and_softmax`: `model(np.zeros((2, 224, 224, 3)))` has shape (2, 5) and each row sums to 1.
  - `test_stage1_trains_head_only`: after `set_stage(..., 1, 1e-3)`, `len(model.trainable_weights) == 2` and `backbone.trainable_weights == []`.
  - `test_stage2_unfreezes_backbone`: after stage 2, `len(backbone.trainable_weights) > 100`.
  - `test_backbone_batchnorm_stays_frozen`:
    - take the first `BatchNormalization` layer in `backbone`;
    - copy its `moving_mean`;
    - run `model.train_on_batch(random x, one-hot y)` in stage 2;
    - assert `moving_mean` is unchanged (`np.allclose`).
  - `test_save_load_round_trip(tmp_path)`: save as `.keras`, `load_model`, and predictions are equal (`atol=1e-5`).
  - `test_set_stage_rejects_bad_stage`: stage 3 raises `ValueError`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and see them pass.**

  Locally this takes about a minute on CPU.
- [ ] **Step 5: Commit.**

### Task 11: Training loop and cross-validation runner (E1 and E2)

**Files:**
- Modify: `src/cattle_breed/models/train.py`
- Create: `scripts/run_cv.py`
- Modify: `notebooks/run_all_colab.ipynb`
- Test: `tests/models/test_train.py` (TF)

**Interfaces:**
- Consumes: `make_dataset` (T9); `build_model`, `set_stage` (T10); `inner_val_split`, `subsample_groups` (T6); `load_cache` (T7); `results/splits/cid_folds.csv`; `TrainConfig`
- Produces:
  - `class_weights(labels: np.ndarray) -> dict[int, float]`: sklearn `compute_class_weight("balanced")` over the classes present.
  - `train_stage(model, train_ds, val_ds, epochs: int, patience: int, class_weight: dict | None) -> tuple[dict, int]`. Returns (history dict, best epoch, 1-based).
    - With `val_ds`: `EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)`, and best = argmin `val_loss` + 1.
    - With `val_ds=None`: exactly `epochs` epochs, and best = `epochs`.
  - `@dataclass FoldOutput`: `probs_e1: np.ndarray`, `probs_e2: np.ndarray`, `best_epochs: tuple[int, int]`, `history_e1: dict`, `history_e2: dict`, `model: keras.Model`
  - `run_fold(images, labels, train_idx, val_idx, test_idx, cfg: TrainConfig, weights="imagenet") -> FoldOutput`:
    - build the model;
    - stage 1, train, then predict the test fold → `probs_e1`;
    - stage 2, train, then predict → `probs_e2`.
  - `run_cv(images, index: pd.DataFrame, folds: pd.DataFrame, cfg: TrainConfig, out_dir: Path, fraction: float = 1.0, weights="imagenet", only_folds: Sequence[int] | None = None) -> None`:
    - Joins `index` with `folds` on `path`, and raises `ValueError` if a row has no fold.
    - For each fold k, writes into `out_dir`:
      - `fold_{k}.json` (best epochs, histories, test paths);
      - `fold_{k}_e2.keras`;
      - rows appended to `predictions_E1.csv` and `predictions_E2.csv` (`path, fold, y_true, p0..p4`).
    - **Skips any fold whose `fold_{k}.json` already exists**, so it can resume after a Colab disconnect.
    - `fraction < 1` applies `subsample_groups` to the training animals only, for E6.
  - `scripts/run_cv.py --out artifacts/cv/full [--fraction 0.5] [--folds 0 1]`. It sets `keras.utils.set_random_seed(42)` first.

- [ ] **Step 1: Write the failing tests**

  All tests use tiny synthetic data:
  - 20 random 240 px images, labels `np.arange(20) % 5`, groups `g0`–`g19`, 2 folds;
  - `weights=None`;
  - `TrainConfig(batch_size=4, head_epochs=1, ft_epochs=1)`.

  Tests:
  - `test_class_weights_balanced`: labels `[0, 0, 0, 1]` give `{0: 2/3, 1: 2.0}`.
  - `test_train_stage_without_val_runs_fixed_epochs`: returns best epoch == `epochs`.
  - `test_run_cv_writes_outputs`:
    - both prediction CSVs exist with 20 rows each;
    - `p0..p4` columns sum to 1 (`atol=1e-4`);
    - `fold_0.json` and `fold_1.json` exist.
  - `test_run_cv_resumes_completed_folds`:
    - run with `only_folds=[0]`, record the `fold_0.json` mtime;
    - run again with all folds;
    - the `fold_0.json` mtime is unchanged and the predictions CSVs have 20 rows with no duplicates.
  - `test_run_cv_rejects_rows_without_fold`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**

  When resuming, rebuild the prediction CSVs from the per-fold JSON plus probabilities so rows are never appended twice. Simplest approach:
  - save `probs_e1` and `probs_e2` inside `fold_{k}.json`;
  - rewrite both CSVs from all completed fold JSONs at the end of every fold.
- [ ] **Step 4: Run them and see them pass.**

  Then commit: `git commit -am "feat: two-stage training and resumable CV runner"`
- [ ] **Step 5: Run the real CV on Colab (T4 GPU)**

  Run `!python scripts/run_cv.py --out $CATTLE_ARTIFACTS_ROOT/cv/full`, copying the cache to `/content` first.
  - Expected: about 10–15 minutes per fold.
  - If Colab disconnects, rerun the same command and it resumes.
- [ ] **Step 6: Evaluate the CV run**

  Run `!python scripts/evaluate.py cv --run $CATTLE_ARTIFACTS_ROOT/cv/full --experiments E1 E2`. Commit `results/`.
- [ ] **Step 7: Gate G4 check**

  - E2 mean macro-F1 > E1 mean macro-F1 > 0.165 (B0).
  - Look at the pooled confusion matrix. If the Red Chittagong or Sindhi recall is 0, the model is collapsing to Local.

> **If it fails (E2 ≤ E1, or E1 barely beats B0), check in this order:**
> 1. Input range: print `x.min(), x.max()` of one batch. It must be about 0–255.
> 2. Label order: `label_idx` must match `CLASS_TO_IDX`, and the predictions CSV `y_true` must match the cache CSV.
> 3. Class weights: they must be passed in both stages.
> 4. Overfit test: train stage 2 on 50 images for 30 epochs. Training accuracy should approach 1.0. If it doesn't, the pipeline is broken, not the data.
> 5. Learning rate: try `ft_lr = 3e-5` on fold 0 only (`--folds 0`). Record the change in the decision log, then rerun all folds with the new value.
>
> Changing more than one thing at a time makes the defence answer "why?" impossible, so change one thing per run.

### Task 12: Final model

**Files:**
- Modify: `src/cattle_breed/models/train.py`
- Create: `scripts/train_final.py`
- Test: `tests/models/test_train.py`

**Interfaces:**
- Consumes: `run_cv` outputs (`fold_{k}.json` best epochs); `build_model`, `set_stage`, `train_stage`
- Produces:
  - `train_final(images, labels, cfg, epochs: tuple[int, int], weights="imagenet") -> keras.Model`: stage 1 for `epochs[0]` then stage 2 for `epochs[1]`, with no validation set.
  - `median_best_epochs(run_dir: Path) -> tuple[int, int]`: median across folds, rounded.
  - CLI `python scripts/train_final.py --run artifacts/cv/full` saves `ARTIFACTS_ROOT/models/final_e2.keras` and `final_e2.json` (epochs used, git commit).

- [ ] **Step 1: Write the failing tests**

  - `test_median_best_epochs`: two fold JSONs with (3, 7) and (5, 9) give (4, 8).
  - `test_train_final_smoke`: tiny data, `weights=None`, `epochs=(1, 1)`. Predictions have shape (n, 5).
- [ ] **Step 2: Run them, see them fail, implement, then see them pass.**
- [ ] **Step 3: Train on Colab**

  Sanity check: accuracy on the CID training images is above the E2 CV accuracy. It is seeing its own training data, so if it's lower, something is wrong. Then commit.

**Exit (G4):** E2 > E1 > B0 is logged and `final_e2.keras` is saved.

---

## Phase 3: Evaluation (Days 14–22)

### Task 13: External evaluation (E3 and E4)

**Files:**
- Modify: `src/cattle_breed/eval/predict.py`, `scripts/evaluate.py` (`external` subcommand)
- Test: `tests/eval/test_predict.py`

**Interfaces:**
- Consumes: `load_cache` (T7); `make_dataset(training=False)` (T9); `load_model` (T10); `compute_metrics`, `restrict_predictions`, `append_result` (T8); `config.DATASET_CLASSES`
- Produces:
  - `predict_probs(model, images: np.ndarray, batch_size: int = 64) -> np.ndarray`: (N, 5), using the eval pipeline (centre crop).
  - `evaluate_external(probs: np.ndarray, y_true: np.ndarray, dataset: str) -> dict[str, dict]`: `{"5-way": compute_metrics(y_true, argmax), "restricted": compute_metrics(y_true, restrict_predictions(probs, allowed))}`
  - CLI `python scripts/evaluate.py external --dataset mendeley --experiment E3` (and `kaggle` / `E4`):
    - writes `ARTIFACTS_ROOT/predictions/{exp}.csv` (`path, y_true, p0..p4`);
    - writes `results/{exp}_metrics.json`;
    - saves `results/figures/{exp}_confusion.png` (via T15's `plot_confusion` once it exists; until then save the JSON only);
    - appends two rows (`scoring = 5-way` and `restricted`).

- [ ] **Step 1: Write the failing tests**

  - `test_evaluate_external_restricted_not_lower`: probabilities put 0.6 on absent class 1 and 0.4 on the true class. The restricted macro-F1 is 1.0 and the 5-way macro-F1 is 0.0.
  - `test_predict_probs_shape`: tiny `weights=None` model, 3 images, shape (3, 5).
- [ ] **Step 2: Run them, see them fail, implement, then see them pass.** Commit.
- [ ] **Step 3: Run E3 and E4 on Colab.** Commit `results/`.
- [ ] **Step 4: Write the domain-shift paragraph in `docs/report/notes.md` the same day**

  Cover macro-F1 for CID (E2) → Mendeley (E3) → Kaggle (E4), the 5-way vs restricted gap, and the biggest confusion in each set.

**Exit (G5):** the Must tier is complete. Everything after this point improves the grade but is not required for a complete project.

### Task 14: Engineered test set and robustness (E5)

**Files:**
- Modify: `src/cattle_breed/data/corruptions.py`, `scripts/evaluate.py` (`robustness` subcommand)
- Test: `tests/data/test_corruptions.py`

**Interfaces:**
- Consumes: `config.CORRUPTIONS`, `OCCLUDE_VALUE`, `SEED`; `fold_{k}_e2.keras` and `fold_{k}.json` (T11); `final_e2.keras` (T12); `predict_probs`, `evaluate_external` (T13)
- Produces:
  - `apply_corruption(img: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray`: same shape, `uint8`.
    - `blur` uses `PIL.ImageFilter.GaussianBlur(radius)`.
    - `dark` multiplies by the factor and rounds.
    - `occlude` places one square of side `round(sqrt(frac * H * W))` filled with 127 at a uniformly random position fully inside the image.
    - Raises `ValueError` for an unknown kind or severity.
  - `corrupt_batch(images: np.ndarray, kind: str, severity: int, seed: int = SEED) -> np.ndarray`: one generator for the whole batch, so it is deterministic.
  - CLI `python scripts/evaluate.py robustness`. For each kind and severity:
    - **CID:** corrupt each fold's test images and predict with `fold_{k}_e2.keras`.
    - **External sets:** use `final_e2.keras`.
    - Appends to `results/robustness.csv` (`dataset, kind, severity, macro_f1, balanced_accuracy, accuracy`). Clean scores are severity 0, taken from E2/E3/E4.
    - Appends `E5` rows to `experiments.csv` with dataset `"{dataset}|{kind}|{severity}"`.

- [ ] **Step 1: Write the failing tests**

  - `test_shape_dtype_preserved`: all kinds and severities.
  - `test_dark_factor`: a uniform-200 image at severity 1 has mean ≈ 100 (± 1).
  - `test_occlude_area`: a zero image at severity 2. The fraction of pixels equal to 127 is 0.25 (± 0.01).
  - `test_blur_reduces_noise_std`: a noise image's std decreases.
  - `test_corrupt_batch_deterministic`: the same seed gives identical output.
  - `test_unknown_kind_raises`
- [ ] **Step 2: Run them, see them fail, implement, then see them pass.** Commit.
- [ ] **Step 3: Run `evaluate.py robustness` on Colab**

  Plot macro-F1 vs severity, one line per kind and one panel per dataset, to `results/figures/robustness.png`. Commit.

### Task 15: Error analysis and Grad-CAM

**Files:**
- Modify: `src/cattle_breed/models/gradcam.py`, `src/cattle_breed/eval/plots.py`
- Create: `notebooks/06_error_analysis.ipynb`, `docs/error_tags.csv`
- Test: `tests/models/test_gradcam.py`, `tests/eval/test_plots.py`

**Interfaces:**
- Consumes: `build_model`, `load_model` (T10); the predictions CSVs (T11, T13); `load_cache` (T7); `center_crop` (T1); `config.ERROR_CAUSES`
- Produces:
  - `gradcam(model, image: np.ndarray, class_idx: int | None = None, layer_name: str = "top_activation") -> np.ndarray`: (7, 7) float32 in [0, 1]. `image` is a 224×224×3 `uint8`. If `class_idx` is None, use the predicted class. Returns zeros, not NaN, when the map's max is 0.
  - `overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray`: upsamples with PIL bilinear, applies the matplotlib `"jet"` colormap, and returns `uint8` of the same size as `image`.
  - `plot_confusion(cm, class_names, out_path: Path, normalize: bool = True, title: str = "") -> Path`
  - `plot_gallery(images: list[np.ndarray], titles: list[str], out_path: Path, ncols: int = 4) -> Path`

  Grad-CAM approach (the nested backbone makes this non-obvious):

  ```python
  backbone = model.get_layer("backbone")
  feat = keras.Model(backbone.inputs, [backbone.get_layer(layer_name).output, backbone.output])
  with tf.GradientTape() as tape:
      conv, pooled = feat(x[None].astype("float32"), training=False)
      tape.watch(conv)
      probs = model.get_layer("classifier")(pooled)   # dropout is identity at inference
      score = probs[:, class_idx]
  grads = tape.gradient(score, conv)[0]               # (7, 7, 1280)
  cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(grads, axis=(0, 1)), axis=-1))
  ```

- [ ] **Step 1: Write the failing tests**

  - `test_gradcam_shape_and_range`: `weights=None` model, random image. Shape (7, 7), finite, values between 0 and 1.
  - `test_gradcam_finite_for_blank_image`: all-zero image gives finite output.
  - `test_overlay_shape`
  - `test_plot_confusion_writes_file(tmp_path)`
  - `test_plot_gallery_writes_file(tmp_path)`
- [ ] **Step 2: Run them, see them fail, implement, then see them pass.** Commit.
- [ ] **Step 3: Notebook `06_error_analysis.ipynb`**

  - Pooled CID confusion (E2) and the E3 and E4 confusions → `results/figures/*_confusion.png`.
  - For CID-E2 and Kaggle-E4, sample 16 errors and 8 correct predictions (seed 42) and draw galleries.
  - Tag each error with one `ERROR_CAUSES` code in `docs/error_tags.csv` (`path, dataset, true, pred, cause, note`).
  - Draw Grad-CAM for 12 images (6 correct, 6 wrong, across datasets) → `results/figures/gradcam.png`.
  - Count causes into a report table.
- [ ] **Step 4: Write the error-analysis section notes the same day.**

  Answer three questions:
  - Does the model look at the animal or the background?
  - Which breed pairs get confused?
  - What does the cows vs bulls shift do?

  Commit.

### Task 16 (Could tier): Data-size sensitivity (E6)

- [ ] **Step 1:** Run `!python scripts/run_cv.py --out $CATTLE_ARTIFACTS_ROOT/cv/half --fraction 0.5`.
- [ ] **Step 2:** Run `!python scripts/evaluate.py cv --run $CATTLE_ARTIFACTS_ROOT/cv/half --experiments E2 --as E6`. This records the result as experiment `E6` with `notes="50% training animals"`. Commit.

**Exit (G6):** E5 and the error analysis are done; E6 too if there was time.

---

## Phase 4: Demo and delivery (Days 23–30)

### Task 17: Gradio demo

**Files:**
- Create: `app/demo.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: `letterbox`, `center_crop` (T1); `load_model` (T10); `config.CLASSES`
- Produces:
  - `predict_image(model, img: np.ndarray) -> dict[str, float]`: `letterbox(240)` → `center_crop(224)` → `model.predict(batch, verbose=0)`. Keys are exactly `config.CLASSES` and the values sum to 1.
  - `build_interface(model) -> gradio.Blocks`:
    - image upload → `gr.Label(num_top_classes=3)`;
    - a fixed note: "Local = Bangladeshi indigenous cattle (CID label). Decision-support demo, not breed verification."
  - `main()` uses argparse `--model` and `--share` and calls `launch`.

- [ ] **Step 1: Write the failing tests**

  These use a stub model whose `predict` returns `[[.1, .2, .3, .2, .2]]` and do not need TensorFlow.
  - `test_predict_image_keys_are_config_classes`
  - `test_predict_image_any_size`: inputs of 50×400 and 1000×300 both work.
- [ ] **Step 2: Run them, see them fail, implement, then see them pass.** Commit.
- [ ] **Step 3: Run on Colab**

  `!python app/demo.py --model $CATTLE_ARTIFACTS_ROOT/models/final_e2.keras --share`. Try one image per class from the external sets.
- [ ] **Step 4: Record a backup**

  Record a 60-second screen capture of the demo as a backup for the presentation, in case the network fails. Store it on Drive, not in git.

### Task 18: Reproducibility run and README

- [ ] **Step 1:** Finish `notebooks/run_all_colab.ipynb` so it runs every stage in order with a `RUN_TRAINING = False` switch. When the switch is off, it reuses the saved CV runs and the final model.
- [ ] **Step 2: Fresh-runtime test.**
  - Runtime → Disconnect and delete runtime, then Run all with `RUN_TRAINING = False`.
  - Expected: every metric in `experiments.csv` is reproduced exactly from the saved predictions.
  - Then run `run_cv.py --folds 0` into a new folder. Expected: fold-0 E2 macro-F1 within ± 0.03 of the logged value. GPU nondeterminism allows small drift; note it in the report.
- [ ] **Step 3:** Run `!pip freeze > results/environment.txt` and record the Python, TensorFlow and Keras versions in the README.
- [ ] **Step 4:** Complete the README: run order, expected runtimes, data download steps, and where outputs go. Then:
  ```bash
  git tag v1.0-submission
  ```
  Commit and push.

### Task 19: Report (8–10 pages)

**Files:** `docs/report/outline.md` (structure, page budget, figure list, rubric map; ready), `docs/report/notes.md` (daily notes), the final report as `docs/report/report.docx` or PDF.

- [ ] **Step 1 (Day 24):** Assemble a draft from `notes.md` into the outline structure. Every number must be copied from `results/experiments.csv` or a `results/*.json`, never retyped from memory.
- [ ] **Step 2 (Day 25):** Insert the figures listed in the outline, each with a caption that states the takeaway.
- [ ] **Step 3 (Day 26):** Write the limitations, ethics and licensing sections, and the AI-use disclosure that guidelines §5 requires.
- [ ] **Step 4 (Day 27):** Self-review against the outline's rubric checklist. Read it aloud once to catch unclear sentences.
- [ ] **Step 5 (Day 28):** Export to PDF, check the page count is 8–10, and commit.

**Exit (G8):** report PDF and fresh-runtime rerun.

### Task 20: Presentation and oral defence

**Files:** `docs/presentation/defense_questions.md` (question bank; ready), slides in `docs/presentation/`

- [ ] **Step 1 (Day 29): Build the slides.** Ten minutes, about one slide per minute:

  | # | Slide | Time |
  |---|---|---|
  | 1 | Title and question | 0:30 |
  | 2 | Problem and use | 0:45 |
  | 3 | Data and the two-step domain-shift design | 1:00 |
  | 4 | Curation and leakage controls | 0:45 |
  | 5 | Model and training recipe | 1:00 |
  | 6 | CID results (B0 → E1 → E2) | 1:00 |
  | 7 | Domain shift (E2 → E3 → E4) | 1:00 |
  | 8 | Robustness (E5) | 0:45 |
  | 9 | Errors and Grad-CAM | 1:00 |
  | 10 | Live demo (backup video ready) | 1:30 |
  | 11 | Limitations and takeaways | 0:45 |
- [ ] **Step 2: Rehearse three times with a timer.** Each run must be under 10:00. Cut words, not slides.
- [ ] **Step 3: Defence preparation.**
  - Answer every question in `defense_questions.md` aloud.
  - For each file in `src/cattle_breed/`, write a 3-line explanation without looking at it, then check it against the code.
  - Re-read the decision log: every "why" answer is there.

---

## Risk register

| Risk | Trigger | Response |
|---|---|---|
| Online-only test data are refused | Instructor reply | Switch to the farm-version plan for T3–T4 (see below), using the Day 17–19 buffer |
| Colab GPU quota or disconnects | `run_cv` interrupted | Rerun the same command (it resumes). Train at off-peak hours. Fall back to `MobileNetV2` (change only the backbone line, log as D-0xx) |
| E2 not better than E1 | G4 check | Use the T11 "If it fails" checklist, one change at a time |
| External labels noisy | Many `WRONG_BREED` drops | Report the counts. Mendeley is the cleaner check. Never drop "hard" images |
| Kaggle Sindhi under 50 images after curation | T4 Step 3 | Report it with its count and wider uncertainty; do not relax the rules |
| Background shortcut found by Grad-CAM | T15 | Report it as a finding. Optionally re-evaluate with a tighter centre crop (Could tier) |
| Heavy week in another course | Behind the gate by more than 1 day | Use the buffer first, then cut in tier order (E6 → YouTube → Grad-CAM count → E5 severities) |
| Drive full (15 GB free) | Copy fails | Kaggle: keep only the 3 class folders. Delete the `.tar.gz` after extraction. Caches total about 550 MB |

## If the farm version is approved instead

Keep T0–T2 and T5–T20 unchanged. Replace T3–T4 with the following:

1. Visit the farm on Days 6–10. Photograph 25–35 animals, about 3 photos each.
2. Store them as `raw/farm/<animal_id>/*.jpg` plus `raw/farm/farm_metadata.csv` (`animal_id, breed, confirmed_by, sex, age, date, photographer`).
3. Write `build_farm_index`, which uses `group = animal_id` and maps to Sahiwal, Sindhi and Holstein cross.
4. E3 becomes the farm set, scored per image and per animal (majority vote over each animal's photos).
5. The permission and privacy steps from the farm proposal apply.

---

## Appendix A: Colab cells

**Setup (top of every Colab session):**

```python
from google.colab import drive, userdata
drive.mount("/content/drive")
import os
os.environ["CATTLE_DATA_ROOT"] = "/content/drive/MyDrive/cattle-breed/data"
os.environ["CATTLE_ARTIFACTS_ROOT"] = "/content/drive/MyDrive/cattle-breed/artifacts"
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
!test -d /content/cattle-breed || git clone -q https://github.com/ShayanAhmed12/CattleClass.git /content/cattle-breed
%cd /content/cattle-breed
!git pull -q && pip install -q -e ".[demo,dev]"
!nvidia-smi --query-gpu=name,memory.total --format=csv
import tensorflow as tf, keras; print(tf.__version__, keras.__version__, tf.config.list_physical_devices("GPU"))
```

**CID download (once):**

```bash
%%bash
set -e
mkdir -p /content/dl "$CATTLE_DATA_ROOT/raw/cid" && cd /content/dl
wget -q https://cid-21.s3.amazonaws.com/dataset.csv
wget -q https://cid-21.s3.amazonaws.com/images.tar.gz
sha256sum dataset.csv images.tar.gz | tee "$CATTLE_DATA_ROOT/raw/cid/SHA256SUMS"
tar -xzf images.tar.gz && find images -name '._*' -delete
cp dataset.csv "$CATTLE_DATA_ROOT/raw/cid/" && cp -r images "$CATTLE_DATA_ROOT/raw/cid/"
ls "$CATTLE_DATA_ROOT/raw/cid/images" | wc -l      # expect 513
```

**Kaggle download (once):**

```python
import kagglehub, shutil, pathlib, os
src = pathlib.Path(kagglehub.dataset_download("lukex9442/indian-bovine-breeds"))
dst = pathlib.Path(os.environ["CATTLE_DATA_ROOT"]) / "raw/kaggle"
for breed in ["Sahiwal", "Red_Sindhi", "Holstein_Friesian"]:
    folder = next(src.rglob(breed))
    shutil.copytree(folder, dst / breed, dirs_exist_ok=True)
    print(breed, len(list((dst / breed).iterdir())))   # expect 439 / 166 / 328
```

Install `kagglehub` first with `pip install -q kagglehub` if it's missing.

**Fast cache access (before training):**

```python
!mkdir -p /content/cache && cp $CATTLE_DATA_ROOT/processed/*_240.* /content/cache/
os.environ["CATTLE_CACHE_DIR"] = "/content/cache"   # scripts read caches from here when set
```

Scripts read caches from `config.CACHE_DIR`. It defaults to `DATA_ROOT/processed` and is overridden by `CATTLE_CACHE_DIR`. `build_cache.py` writes to `DATA_ROOT/processed`.

## Appendix B: Experiment registry

| ID | What | Data | Scoring | Tier |
|---|---|---|---|---|
| B0 | Always predict Local | CID, 5-fold | 5-way | Must |
| E1 | Frozen backbone, head only | CID, 5-fold | 5-way | Must |
| E2 | Two-stage fine-tuning | CID, 5-fold | 5-way | Must |
| E3 | Final model | Mendeley (curated, de-duplicated) | 5-way and restricted | Must |
| E4 | Final model | Kaggle (curated, de-duplicated) | 5-way and restricted | Must |
| E5 | Fold models / final model | Corrupted CID / Mendeley / Kaggle | 5-way (CID), restricted (external) | Should |
| E6 | Two-stage on 50% of animals | CID, 5-fold | 5-way | Could |

Every row in `results/experiments.csv` = one (experiment, dataset, scoring). The report's results tables are generated from this file.
