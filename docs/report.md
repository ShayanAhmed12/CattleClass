---
title: Cattle Breed Classification from Images Using Transfer Learning
subtitle: CS4045 Deep Learning for Perception · Working report
geometry: margin=0.85in
fontsize: 11pt
---

## 1 Introduction

This project asks whether a cattle breed classifier trained on photos from one source can generalize to independently collected photos. Farmers, livestock traders and dairy record keepers may encounter cattle in unfamiliar settings. A photo classifier could supply a candidate breed for inspection. The intended use is decision support alongside an experienced handler or verified records. Breed identity can depend on ancestry, and the model observes only what is visible in a photograph.

The task is five-class image classification. Each input is one RGB photo and each model output is a probability over Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi. The CID label Local groups indigenous cattle under a common dataset label. The model uses CID for development and tests generalization on photos from Mendeley and Kaggle. The evaluation design also tests controlled changes to lighting, sharpness and visibility.

**Status as of 19 November 2026.** Data preparation, all five grouped splits and the majority-class baseline have been verified on real CID data. The EfficientNet training pipeline and downstream workflows have passed synthetic tests. Real-model results, manual external curation and interpretation are pending. The results table on page 6 gives the current evidence.

### Related work and contribution

The CID publication and dataset provide images and metadata for Bangladeshi market cattle. EfficientNet-B0 supplies an ImageNet-pretrained visual representation for transfer learning. The contribution here is the evaluation design. It holds out animals within CID, then uses the same final model across two independent external sources. Mendeley changes photographers and collection context within Bangladesh. Kaggle changes both geography and acquisition, using web images of Indian bovine breeds. These are two different external domains, and disagreement can reflect differences in breed composition, annotation or capture conditions as well as geography. [CID publication](https://doi.org/10.1145/3542954.3543018) · [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html)

\newpage

## 2 Data sources and label mapping

| Source | Role | Images in matched labels before deduplication | Licence |
|---|---|---:|---|
| CID | Development and grouped cross-validation | 2,012 | Apache-2.0 |
| Mendeley | External evaluation E3 | 190 | CC BY 4.0 |
| Kaggle v5 | External evaluation E4 | 929 | Declared CC0 |

CID contains 514 four-photo folders and 513 metadata records. The index removes spaces when matching SKUs, excludes the folder without a label and keeps the five classes with enough animals for grouped evaluation. Pabna, Brahma and Mir Kadim are excluded. Local makes up roughly 70% of the retained animals, so an accuracy figure alone can hide poor results on the other classes. The local CID sample grid displays all four indexed views of one example per class. All four photos enter the any-view protocol, and the specific viewpoint names remain to be confirmed privately. [Data provenance and checksums](data.md)

| CID class | Source animal IDs | Indexed photos |
|---|---:|---:|
| Holstein cross | 32 | 128 |
| Local | 353 | 1,412 |
| Red Chittagong | 21 | 84 |
| Sahiwal | 59 | 236 |
| Sindhi | 38 | 152 |
| **Total** | **503** | **2,012** |

Mendeley supplies Holstein Friesian, Local Deshi and Sahiwal after excluding Jersey. Kaggle supplies Holstein Friesian, Red Sindhi and Sahiwal. The protocol maps Holstein Friesian to CID Holstein cross, Local Deshi to Local and Red Sindhi to Sindhi. The mapped labels reflect assumptions needed to compare these datasets. They also constrain the interpretation of test errors. Neither external source provides the same animal-ID grouping as CID, so its scores are per photo.

\newpage

## 3 Curation, duplicate detection and grouped splits

Before external scoring, a reviewer inspects every photo in the matched Mendeley and Kaggle classes. Invalid examples include clearly wrong breeds, drawings, non-cattle species, corrupt files and images where an identifiable person or document is prominent. Difficult but valid examples, including unfamiliar poses, fences and poor lighting, remain part of the test. Each exclusion goes into a path-and-reason CSV. Before manual curation, perceptual hashing and decoding left 173 Mendeley photos and 908 Kaggle photos in the provisional caches. [Review criteria](data.md#reviewing-the-external-images-t3)

Every decodable image is compared using a 64-bit perceptual hash computed after EXIF orientation correction. Hamming distance at most four marks a pair as near-duplicate. CID produced 135 matching pairs, all between distinct source SKUs. Because a later photo can match multiple earlier photos, counting each removed photo once gives 119 removals. Linked SKUs are assigned the same group transitively, leaving 1,893 images in 371 groups. External photos are compared against all indexed CID photos and earlier photos in their own source. This prevents obvious cross-dataset copies from inflating performance.

Some linked groups contain multiple source labels. Each retained photo keeps its metadata label. Grouping keeps potentially related images in the same partition despite label differences. Seventeen linked groups contain more than one label. The hash detects similarity in the pixels, and these groups may contain genuinely different animals. The grouped data therefore represent an evaluation safeguard rather than independent verification of animal identity.

The five test folds come from stratified group cross-validation with seed 42. A photo is tested exactly once. Within each training split, a second group-level split holds out 15% of its animal groups for validation. That holdout is stratified by each group's most frequent label. Every real training, validation and test split contains all five labels. The grouped split was checked for overlap at both levels. [Fold manifest](../results/cid_folds.csv)

\newpage

## 4 Model and training procedure

The pipeline decodes images upright in RGB, resizes with aspect ratio preserved and pads them to 240 × 240 pixels before caching as `uint8`. Training takes a random 224 × 224 crop. Validation, evaluation and the demo use the center crop. Inputs stay in the 0–255 range because EfficientNet includes an input rescaling layer. Large or unreadable images are reported and excluded from the cache. The prepared index and cache record row order and fold identity so downstream commands can detect mismatches.

The classifier uses ImageNet-pretrained EfficientNet-B0, global average pooling, dropout with probability 0.3 and a five-output softmax. Training augments the input using horizontal flips, rotations of about seven degrees, and mild brightness and contrast adjustments. Hue and saturation are preserved because coat color helps distinguish some classes. The backbone runs in inference mode throughout training so its BatchNorm moving statistics stay fixed. The fine-tuning stage can still update its trainable weights.

| Stage | Trainable layers | Adam learning rate | Maximum epochs |
|---|---|---:|---:|
| E1 | New head | 0.001 | 10 |
| E2 | Head and backbone | 0.00001 | 20 |

Both stages use balanced class weights and categorical cross-entropy with label smoothing of 0.1. Early stopping monitors validation loss and restores the best weights. E1 test predictions are captured at the end of the frozen stage before E2 starts. After all five folds, the final model fits all CID photos using the median number of selected epochs in each stage. The final fit has no validation holdout. Each complete fold records its predictions, epoch choices, fold ID and model checksum so a rerun can reuse verified outputs. [Keras EfficientNet reference](https://keras.io/api/applications/efficientnet/) · [Label smoothing](https://arxiv.org/abs/1512.00567)

\newpage

## 5 Evaluation protocol

B0 always predicts Local. E1 and E2 use exactly the same five held-out CID folds as B0. The summary reports fold means and the standard deviation of macro-F1. The pooled out-of-fold predictions also produce the per-class F1 values and confusion matrix. A confusion matrix shows the number of photos from each true class assigned to each predicted class.

The headline metrics are macro-F1 and balanced accuracy. Macro-F1 averages the F1 scores of the classes present in the ground truth and gives a rare class the same weight as a common class. Balanced accuracy is the mean recall across the same set of classes. Plain accuracy remains useful to show the fraction of correctly classified photos, especially when presented with these class-balanced metrics.

E3 and E4 evaluate the final all-CID model on the reviewed Mendeley and Kaggle datasets. Five-way scoring uses the highest probability across all classes. Restricted scoring uses the highest probability among the three class labels in that source. A five-way prediction of an absent class is counted as an error. Restricted evaluation describes performance when the label set is known in advance. Interpret the gap alongside the confusion matrix and class coverage.

E5 tests the same label sets after controlled corruption. Gaussian blur uses radii of two and four pixels. Pixel brightness is multiplied by 0.5 and 0.3. Occlusion adds a seeded gray square covering approximately one-eighth or one-quarter of the cropped input. CID uses each photo's held-out fold model. External tests use the final all-CID model. Comparing each corrupted condition with its clean counterpart controls for the underlying image set. [Experiment plan](PLAN.md)

\newpage

## 6 Results and interpretation

| Experiment | Dataset | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| B0 | CID, 1,893 photos | 0.703 | 0.200 | 0.165 ± 0.001 |
| E1 | CID | Pending | Pending | Pending |
| E2 | CID | Pending | Pending | Pending |
| E3 | Mendeley | Pending | Pending | Pending |
| E4 | Kaggle | Pending | Pending | Pending |
| E5 | CID, Mendeley and Kaggle | Pending | Pending | Pending |

The B0 accuracy closely follows the fraction of Local photos in the folds. Its balanced accuracy is exactly one fifth because Local is the only class it recalls and all five classes occur in every fold. Its macro-F1 is lower than 0.20 because every non-Local photo is a false positive for Local and the other classes have zero F1. The small variation across folds is consistent with the grouped stratification. These results establish a reference before judging the CNN.

The model comparison will test the prespecified expectation that fine-tuning improves on frozen transfer learning and that both improve over B0 in macro-F1. Report the measured outcome even if ordering differs. If the ordering differs, the implementation checks concern input range, class mapping, class weights and the ability to fit a small training subset. Use held-out fold performance for reporting and preserve the distinction between checks and model tuning.

The external comparisons will report both scoring protocols together with class coverage and curation counts. The robustness comparison will state the difference from the clean macro-F1 for every source and severity. Numeric evidence belongs in the metrics CSV. Each final chart should state which subset and model produced it. [Recorded metrics](../results/metrics.csv)

\newpage

## 7 Error analysis, limitations and ethics

The evaluation command selects up to three correct and three incorrect predictions per external source by confidence. It stores the gallery, annotations and Grad-CAM overlays in a local folder. Grad-CAM uses the gradient of the predicted-class logit with respect to the last spatial features. It can help identify whether strongly weighted regions coincide with the animal, its coat or the background. Interpret the overlays with the predictions and image context. Actual examples and probable causes are pending real external scoring and private review. [Grad-CAM](https://arxiv.org/abs/1610.02391)

Potential error explanations include similar pigmentation between Sindhi and Sahiwal, unusual poses, partial visibility, calves versus mature animals and background effects. These are hypotheses until individual predictions are reviewed. The final report should name the confused labels, the visual cue considered and any ambiguity in the source label. High-confidence mistakes deserve particular attention because the interface displays the largest probabilities to users.

CID is a market dataset dominated by Local cattle and containing many young bulls. Some breeds are represented by few groups. The external label mapping joins labels that are related but may have different definitions. The web dataset may have noisy labels and unclear underlying photo rights despite the dataset's declared licence. External photos lack reliable animal IDs, and a perceptual hash cannot establish that visibly different photos depict different animals. The reported results apply to these collections and evaluation assumptions.

The dataset and model files remain outside the repository. Private review excludes images with prominent identifying details. Photo-containing analysis figures stay local and reports or presentations can include examples only after the team checks their provenance and suitability. Kaggle's web photos are used for testing. The demo illustrates candidate breeds to support human judgement and is unsuitable as a standalone basis for decisions about ancestry, purchase or breeding.

\newpage

## 8 Reproducibility, contributions and AI use

The Python 3.12 project uses NumPy, Pillow, pandas, scikit-learn, TensorFlow/Keras, Matplotlib and Gradio. The initial cloud run uses TensorFlow 2.21.0 and Keras 3.15.1. Record the runtime and commit that produced the report's final checkpoints, including the accelerator and software versions. Image provenance, checksums, label mappings and review rules are recorded in the repository. The CID archive was checked against its recorded SHA-256. [Data provenance](data.md) · [Decisions](decisions.md)

```bash
python -m pytest -q
python prepare_data.py --datasets cid
python train.py --baseline
python train.py
python train.py --final
# After private review of both external sources:
python prepare_data.py --datasets mendeley kaggle
python evaluate.py --robustness --figures
python demo.py
```

The B0 output can be reproduced independently of TensorFlow training. The full reproduction check requires training all five folds, fitting the final model, applying the completed review manifest and comparing results from a clean checkout. Keep the trained checkpoints as private artifacts. Export aggregate metrics and figures for the report. The recording for the demonstration should come from the trained final model.

Task ownership is recorded in the progress tracker. Shayan is recorded as the owner of the completed setup, image preparation, grouping and B0 work, plus the model and reporting tasks in progress. The team should review the final contributor details and rehearse the individual oral defence before submission. [Progress tracker](PROGRESS.md) · [Defence questions](defense_questions.md)

OpenAI Codex assisted with implementation, debugging, automated tests and documentation. Synthetic tests, local dataset checks and B0 reproduction validate the current work. The project team is responsible for private image review, interpreting experiment results and explaining the implementation during the defence.
