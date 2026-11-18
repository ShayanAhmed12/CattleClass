# Report (8–10 pages)

Write a few sentences under the matching heading at the end of each working day, so the report is mostly drafted by day 24. Every number must come from `results/metrics.csv`.

## Outline

| § | Section | Pages | Content | Figures and tables | Rubric (points) |
|---|---|---|---|---|---|
| 1 | Introduction | 0.75 | Problem, intended use (decision support), research question | — | Problem definition (5) |
| 2 | Related work and novelty | 0.5 | CID paper and other breed CNNs test on their own dataset; this project tests across datasets | — | Novelty and real-life use (5) |
| 3 | Data | 1.5 | Three datasets, provenance, label mapping and assumptions, review rules and counts, duplicates, class imbalance, CID bias (young bulls) | Sample grid; provenance table; class counts | Dataset quality and provenance (5) |
| 4 | Method | 1.5 | Letterbox and crop; EfficientNet-B0; two stages; augmentation (and why no hue); class weights; 5-fold CV by animal; final model; 5-way vs restricted scoring; corruptions | Pipeline diagram | Implementation and pretrained model (10) |
| 5 | Results | 2.0 | B0 → E1 → E2 on CID (mean ± std); per-class F1; E3, E4 domain shift; E5 robustness | Main results table; CID confusion matrix; macro-F1 bars CID → Mendeley → Kaggle; robustness plot | Evaluation and error analysis (10) |
| 6 | Error analysis | 1.25 | Which breeds are confused and why; Grad-CAM (animal or background?); cows vs bulls | Error gallery; Grad-CAM grid | (same) |
| 7 | Limitations | 0.5 | CID is mostly young bulls; label assumptions; noisy web labels; no animal IDs in external sets; small classes | — | Report (5) |
| 8 | Ethics and licences | 0.5 | Licences and attribution; Kaggle copyright; no redistribution; privacy in review; intended-use caveat | — | Report (5) |
| 9 | AI-use disclosure | 0.25 | Tools used, what for, how the output was checked (guidelines §5) | — | Report (5) |
| 10 | Reproducibility | 0.25 | Repository, commit, run order, seed, run times, versions | — | Report (5) |
| 11 | Contributions | 0.25 | Who did which tasks (from `PROGRESS.md` Owner column) | — | Required by guidelines §3 |
| — | References | — | CID; Mendeley DOI; Kaggle URL; EfficientNet (Tan & Le 2019); Grad-CAM (Selvaraju et al. 2017); label smoothing (Szegedy et al. 2016); Keras; TensorFlow; scikit-learn; breed reference | — | — |

## Final check

- [ ] Every number matches `results/metrics.csv`.
- [ ] B0 is shown next to E1 and E2.
- [ ] CID results are mean ± std.
- [ ] Both external scorings are reported, with the gap explained.
- [ ] Figure captions state the takeaway.
- [ ] Label assumptions and removal counts are stated.
- [ ] The report covers limitations, ethics and AI use.
- [ ] Each member's contribution is stated.
- [ ] It is 8–10 pages.

## Notes

### 3 Data

Before deduplication, the CID index contained 2,012 photos of 503 animals, with four photos per animal. The class counts by animal were Local 353, Sahiwal 59, Sindhi 38, Holstein cross 32 and Red Chittagong 21. The index maps space-normalized SKUs to metadata labels and excludes the unlabelled folder and three small classes. The local Figure 1 grid shows all four photos of one example animal per class.

### 4 Method

A 64-bit perceptual hash with Hamming distance at most four identified 135 matching CID pairs, all across different source SKUs. Counting each later photo once removed 119 photos. Linked SKUs share an ID before splitting, leaving 1,893 photos in 371 groups. Stratified group cross-validation held out each group once. Within each training fold a second stratified holdout selected 15% of remaining groups for validation. All five classes occurred in every training, validation and test split.

EfficientNet-B0 receives RGB inputs on the 0–255 scale. Training randomly crops the letterboxed images and applies modest geometric and brightness/contrast augmentation. The frozen-head stage and the full fine-tuning stage both use class weighting and label smoothing. The backbone keeps BatchNorm moving statistics fixed throughout. Validation loss selects the epoch reported for each stage. The final-model fit uses all CID images and the median selected epoch count for each stage. E5 applies two levels each of Gaussian blur, brightness reduction and a seeded gray occluder to the centered model inputs. CID corruptions are evaluated by each photo’s held-out fold model. External corruptions use the all-CID model.

### 5 Results

The B0 predictor assigns every CID photo to Local. On 1,893 held-out predictions across the five folds, mean accuracy was 0.703, balanced accuracy was 0.200 and macro-F1 was 0.165 (fold standard deviation 0.001). This is the reference for E1 and E2.

### 6 Error analysis

The evaluation command selects correct and incorrect external examples by confidence and computes Grad-CAM for the predicted class. The candidate gallery is kept locally. The team records a likely cause and marks ambiguous cases before reporting examples.

### 7 Limitations

### 9 AI-use disclosure

OpenAI Codex was used to implement and debug the Python pipeline and update the documentation. Verification used synthetic tests and local dataset counts and split checks. Manual image review and interpretation of the project results are the project team's responsibility.
