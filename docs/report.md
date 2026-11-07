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
| — | References | — | CID; Mendeley DOI; Kaggle URL; EfficientNet (Tan & Le 2019); Grad-CAM (Selvaraju et al. 2017); label smoothing (Szegedy et al. 2016); Keras; TensorFlow; scikit-learn; breed reference | — | — |

## Final check

- [ ] Every number matches `results/metrics.csv`.
- [ ] B0 is shown next to E1 and E2.
- [ ] CID results are mean ± std.
- [ ] Both external scorings are reported, with the gap explained.
- [ ] Figure captions state the takeaway.
- [ ] Label assumptions and removal counts are stated.
- [ ] The report covers limitations, ethics and AI use.
- [ ] It is 8–10 pages.

## Notes

### 3 Data

### 4 Method

### 5 Results

### 6 Error analysis

### 7 Limitations
