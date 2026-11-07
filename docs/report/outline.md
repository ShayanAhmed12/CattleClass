# Report outline (8–10 pages)

Write each section's notes in `notes.md` on the day its work is done. Every number comes from `results/experiments.csv` or `results/*.json`.

| § | Section | Pages | Content | Figures and tables | Rubric criterion (points) |
|---|---|---|---|---|---|
| 1 | Introduction | 0.75 | Problem, intended use (decision support), research question, contributions | — | Problem definition (5) |
| 2 | Related work and novelty | 0.5 | Breed classification work (CID paper, Cowbree, zebu breed CNNs); gap: same-dataset testing only; this project's two-step cross-dataset test | — | Novelty and real-life use (5) |
| 3 | Data | 1.5 | Three datasets; provenance table; label mapping and assumptions; curation rules and counts; de-duplication and group merging; class imbalance; CID bias (young bulls) | **F1** CID sample grid; **T1** provenance; **T2** class counts per dataset (before and after curation) | Dataset quality, provenance, labels (5) |
| 4 | Method | 1.5 | Letterbox/crop preprocessing; EfficientNet-B0; two-stage recipe; augmentation and why no hue; class weights; label smoothing; grouped 5-fold CV; inner validation; final model; scoring (5-way vs restricted); corruptions | **F2** pipeline diagram (data → cache → CV → final → external) | Implementation and pretrained model use (10) |
| 5 | Results | 2.0 | B0 → E1 → E2 on CID (mean ± std); per-class F1; E3 and E4 domain shift; E5 robustness; E6 if done | **T3** main results table; **F3** pooled CID confusion; **F4** macro-F1 across CID → Mendeley → Kaggle (bars, 5-way and restricted); **F5** robustness lines | Performance evaluation and error analysis (10) |
| 6 | Error analysis | 1.25 | Error causes (counts from `error_tags.csv`); breed confusions; Grad-CAM findings (animal vs background); cows vs bulls | **F6** error gallery; **F7** Grad-CAM grid; **T4** error-cause counts | (same as above) |
| 7 | Limitations | 0.5 | CID mostly young bulls; label mappings; noisy web labels; no animal IDs in external sets; small minority classes; single backbone; GPU nondeterminism | — | Report (5) |
| 8 | Ethics and licensing | 0.5 | Licences and attribution; Kaggle image copyright; no redistribution; privacy rules in curation; no upload to AI or annotation tools; intended-use caveat | — | Report (5) |
| 9 | AI-use disclosure | 0.25 | Required by guidelines §5: tools used, purposes, how output was verified | — | Report (5) |
| 10 | Reproducibility | 0.25 | Repo, commit tag, run order, seeds, runtimes, environment versions | — | Report (5) |
| — | References | — | CID; Mendeley DOI; Kaggle URL; EfficientNet (Tan & Le 2019); Grad-CAM (Selvaraju et al. 2017); label smoothing (Szegedy et al. 2016); Keras; TensorFlow; scikit-learn; breed reference source | — | — |

The technical understanding and defence criterion (10 points) is covered by the presentation and oral defence, not the report.

## Self-review checklist (Day 27)

- [ ] Every number in the text matches `results/experiments.csv`.
- [ ] B0 is shown next to E1 and E2, so readers see the 70%-accuracy floor.
- [ ] CID results are reported as mean ± std across folds, not one number.
- [ ] Both external scorings are reported, and the gap is explained.
- [ ] Each figure caption states the takeaway, not only what is plotted.
- [ ] Label-mapping assumptions and curation drop counts are stated.
- [ ] Limitations include the young-bull bias and noisy web labels.
- [ ] The ethics section covers licences, attribution, privacy and redistribution.
- [ ] The AI-use disclosure section is present (guidelines §5).
- [ ] The report is 8–10 pages and every figure is referenced in the text.
