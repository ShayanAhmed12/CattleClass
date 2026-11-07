# Decisions

Each design choice and its reason. The defence will ask "why did you do X?", so the answer should be here. Add an entry whenever a setting in `cattle_breed/config.py` or the method changes.

**D-01 · Individual project with online datasets only (approved 2026-11-07).**
- CID is the training set. Mendeley and Kaggle are test-only, standing in for the hand-collected test set the guidelines ask for.
- The two external sets give two steps of domain shift: the same country with new photographers, then a new country with web images.

**D-02 · Five classes and label mapping.**
- The classes are Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi.
- Dropped: Pabna (6 animals), Brahma (2) and Mir Kadim (2), which are too few to train and test on; and Jersey, which is not a CID class.
- Assumptions stated in the report:
  - CID's Sindhi is Red Sindhi;
  - purebred Holstein Friesian counts as Holstein cross;
  - Mendeley's Local Deshi is Local.
- Source spellings handled in `config.LABEL_MAP`: CID `HOSTINE_CROSS` and Mendeley `Holstain Friesian`.

**D-03 · EfficientNet-B0 with ImageNet weights.**
- Good accuracy per parameter at 224 px, built-in input rescaling, and it fits a free Colab GPU.
- MobileNetV2 is used only if B0 does not train or fit.

**D-04 · 5-fold cross-validation, split by animal.**
- Why: with only 21 Red Chittagong animals, a single 70/15/15 split leaves about 3 of them for testing, which gives a noisy score.
- With 5 folds every animal is tested once, and results are mean ± std plus one pooled confusion matrix.
- Splitting by animal keeps photos of one animal from appearing in both training and test.

**D-05 · One fixed two-stage training recipe, no hyperparameter search.**
- Stage 1 trains the head only at lr 1e-3. Stage 2 fine-tunes everything at lr 1e-5.
- Both use early stopping, balanced class weights and label smoothing 0.1.
- Why: there is no time for a search, and tuning on the test folds would inflate the CV score.
- E1 and E2 come from the same run, after stage 1 and after stage 2.
- No hue or saturation augmentation, because coat colour is a breed cue.

**D-06 · Letterbox to 240 × 240, then crop to 224.**
- Cattle side-view photos are wide. Stretching distorts body shape, and a square crop cuts off heads or tails.
- Images are cached once as uint8, so every training run starts fast.

**D-07 · CID photo views.** *Decide in T2* after looking at the sample grid. The default is to keep all 4 photos per animal and call the task "any view".

**D-08 · Metrics: macro-F1 and balanced accuracy; external sets scored two ways.**
- "Local" is 70% of CID, so always guessing it scores 0.70 accuracy but only 0.165 macro-F1 (B0).
- Each external set holds only 3 of the 5 classes:
  - **5-way** counts a prediction of an absent class as an error;
  - **restricted** only chooses among the 3 classes present.
- The gap between the two shows how often the model falls back on an absent class.

**D-09 · Review rule: remove only invalid images, never hard ones.**
- Drawings, collages, other species, clearly wrong breeds, heavy watermarks and identifiable people are removed.
- Poor light, odd poses and busy backgrounds stay. Removing them would hide the domain shift being measured.

**D-10 · Data facts found while downloading (2026-11-07).**
- CID has 514 animal folders of 4 photos, but `dataset.csv` has 513 rows.
  - Folder `BLF2334` has no row and so no label; it is dropped.
  - The CSV writes some SKUs with a space (`BLF 2340`), so folders are matched with spaces removed.
- Kaggle `Red_Sindhi` holds 166 files, 4 of them GIFs. GIFs are not used, which leaves 162 images.

**D-11 · Python 3.12 and one-command setup.**
- TensorFlow's stable release has no wheels for Python 3.14, so the project uses 3.12 (3.13 also works).
- `setup.ps1` / `setup.sh` create `.venv`, install `requirements.txt` and run the tests.
- `prepare_data.py` downloads and checks the data, so every machine has identical raw data.

**D-12 · Keep the code and documents small (2026-11-07).**
- The project must be explainable in a 10-minute presentation and an oral defence. So the code is 5 modules and 4 run scripts, and the documents are 6 files.
- Every experiment in the approved proposal is kept. Only the machinery around them is simpler:
  - no results registry with git hashes;
  - no curation reason codes or contact-sheet tool.

**D-13 · Where training runs.** *Decide in T7:* WSL2 with the local GPU, Colab, or CPU overnight. Record the TensorFlow and Keras versions used.

**D-14 · Ready to move to Colab at any time (2026-11-07).**
- `config.py` detects Colab: datasets and models go to *My Drive/cattle-breed*, and archives go to Colab's temporary disk.
- `colab.ipynb` runs the same four steps. The code is the same on both machines.
- What moves between machines:
  - code and `results/` through GitHub;
  - trained models through Drive;
  - datasets are re-downloaded by `prepare_data.py`.
