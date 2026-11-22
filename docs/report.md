---
title: Cattle Breed Classification from Images Using Transfer Learning
subtitle: CS4045 Deep Learning for Perception · Working report
geometry: margin=0.85in
fontsize: 11pt
---

## 1 Introduction

This project measures how well image models predict cattle breed and recorded traits on independently held-out animals from CID. Farmers, livestock traders and dairy record keepers may encounter cattle in unfamiliar settings. A photo classifier could supply a candidate breed for inspection. The intended use is decision support alongside an experienced handler or verified records. Breed identity can depend on ancestry, and the model observes only what is visible in a photograph.

The task is five-class image classification. Each input is one RGB photo and each model output is a probability over Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi. The CID label Local groups indigenous cattle under a common dataset label. The study uses CID because it links breed, age and weight to the photographed animals. The breed evaluation tests controlled changes to lighting, sharpness and visibility.

**Status as of 21 November 2026.** The grouped breed evaluation, robustness tests and age and weight research benchmarks use CID. The final breed model and demo recording are stored privately. Trait estimates need independent field validation.

### Related work and contribution

The CID publication and dataset provide images and metadata for Bangladeshi market cattle. EfficientNet-B0 supplies an ImageNet-pretrained representation for transfer learning. The evaluation holds out duplicate-linked animal groups and compares breed predictions under clean and corrupted conditions. The same groups support the trait benchmarks. [CID publication](https://doi.org/10.1145/3542954.3543018) · [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html)

\newpage

## 2 Data sources and label mapping

CID contains 514 four-photo folders and 513 metadata records. The index removes spaces when matching SKUs, excludes the folder without a label and keeps the five classes with enough animals for grouped evaluation. Pabna, Brahma and Mir Kadim are excluded. Local makes up roughly 70% of the retained animals, so an accuracy figure alone can hide poor results on the other classes. The local CID sample grid displays all four indexed views of one example per class. All four photos enter the any-view protocol. Across 503 images per suffix, the pose audit found 475, 497 and 434 image-left profiles for suffixes 0, 1 and 2, and 496 image-right profiles for suffix 3. The labels describe direction within the frame. [Data provenance and checksums](data.md)

| CID class | Source animal IDs | Indexed photos |
|---|---:|---:|
| Holstein cross | 32 | 128 |
| Local | 353 | 1,412 |
| Red Chittagong | 21 | 84 |
| Sahiwal | 59 | 236 |
| Sindhi | 38 | 152 |
| **Total** | **503** | **2,012** |

CID records indicate 375 animals at 24 months, 123 at 30 months and 5 at 36 months after the breed and image filtering. The weight data run from 150 to 698 kg for the 503 source animal IDs. The joint benchmark scores one representative image per source animal across all recorded ages. [Trait audit](age-and-weight-plan.md)

\newpage

## 3 Curation, duplicate detection and grouped splits

Every decodable image is compared using a 64-bit perceptual hash computed after EXIF orientation correction. Hamming distance at most four marks a pair as near-duplicate. CID produced 135 matching pairs, all between distinct source SKUs. Because a later photo can match multiple earlier photos, counting each removed photo once gives 119 removals. Linked SKUs are assigned the same group transitively, leaving 1,893 images in 371 groups.

Some linked groups contain multiple source labels. Each retained photo keeps its metadata label. Grouping keeps potentially related images in the same partition despite label differences. Seventeen linked groups contain more than one label. The hash detects similarity in the pixels, and these groups may contain genuinely different animals. The grouped data therefore represent an evaluation safeguard rather than independent verification of animal identity.

The five test folds come from stratified group cross-validation with seed 42. A photo is tested exactly once. Within each training split, a second group-level split holds out 15% of its animal groups for validation. That holdout is stratified by each group's most frequent label. Every real training, validation and test split contains all five labels. The grouped split was checked for overlap at both levels. [Fold manifest](../results/cid_folds.csv)

\newpage

## 4 Model and training procedure

The pipeline decodes images upright in RGB, resizes with aspect ratio preserved and pads them to 240 × 240 pixels before caching as `uint8`. Training takes a random 224 × 224 crop. Validation, evaluation and the demo use the center crop. Inputs stay in the 0–255 range because EfficientNet includes an input rescaling layer. Large or unreadable images are reported and excluded from the cache. The prepared index and cache record row order and fold identity so downstream commands can detect mismatches.

The TensorFlow Keras classifier uses ImageNet-pretrained EfficientNet-B0, global average pooling, dropout with probability 0.3 and a five-output softmax. Training augments the input using horizontal flips, rotations of about seven degrees, and mild brightness and contrast adjustments. Hue and saturation are preserved because coat color helps distinguish some classes. The backbone runs in inference mode throughout training so its BatchNorm moving statistics stay fixed. The fine-tuning stage can still update its trainable weights.

| Stage | Trainable layers | Adam learning rate | Maximum epochs |
|---|---|---:|---:|
| E1 | New head | 0.001 | 10 |
| E2 | Head and backbone | 0.00001 | 20 |

Both stages use balanced class weights and categorical cross-entropy with label smoothing of 0.1. Early stopping monitors validation loss and restores the best weights. E1 test predictions are captured at the end of the frozen stage before E2 starts. After all five folds, the final model fits all CID photos using the median number of selected epochs in each stage. The final fit has no validation holdout. Each complete fold records its predictions, epoch choices, fold ID and model checksum so a rerun can reuse verified outputs. [Keras EfficientNet reference](https://keras.io/api/applications/efficientnet/) · [Label smoothing](https://arxiv.org/abs/1512.00567)

\newpage

## 5 Evaluation protocol

B0 always predicts Local. E1 and E2 use exactly the same five held-out CID folds as B0. The summary reports fold means and the standard deviation of macro-F1. The pooled out-of-fold predictions also produce the per-class F1 values and confusion matrix. A confusion matrix shows the number of photos from each true class assigned to each predicted class.

The headline metrics are macro-F1 and balanced accuracy. Macro-F1 averages the F1 scores of the classes present in the ground truth and gives a rare class the same weight as a common class. Balanced accuracy is the mean recall across the same set of classes. Plain accuracy remains useful to show the fraction of correctly classified photos, especially when presented with these class-balanced metrics.

E5 tests the CID breed labels after controlled corruption. Gaussian blur uses radii of two and four pixels. Pixel brightness is multiplied by 0.5 and 0.3. Occlusion adds a seeded gray square covering approximately one-eighth or one-quarter of the cropped input. CID uses each photo's held-out fold model. Comparing each corrupted condition with its clean counterpart controls for the underlying image set.

\newpage

## 6 Results and interpretation

| Experiment | Dataset | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| B0 | CID, 1,893 photos | 0.703 | 0.200 | 0.165 ± 0.001 |
| E1 | CID, 1,893 photos | 0.611 | 0.533 | 0.453 ± 0.053 |
| E2 | CID, 1,893 photos | 0.656 | 0.580 | 0.500 ± 0.046 |
| E6 | CID, half training groups | 0.619 | 0.559 | 0.467 ± 0.034 |
| E5 | Six CID corruption conditions | See CSV | See CSV | See below |

Strong blur had the lowest CID macro-F1, at 0.189 compared with 0.500 on clean photos.

The B0 accuracy closely follows the fraction of Local photos in the folds. Its balanced accuracy is exactly one fifth because Local is the only class it recalls and all five classes occur in every fold. Its macro-F1 is lower than 0.20 because every non-Local photo is a false positive for Local and the other classes have zero F1. The small variation across folds is consistent with the grouped stratification. These results establish a reference before judging the CNN.

The five-fold results followed the expected macro-F1 ordering: E2 exceeded E1, and both exceeded B0. For the optional E6 run, each fold trained on a stratified sample of half its training animal groups, with the same validation and test splits. E6 macro-F1 was 0.033 lower relative to E2. Both were evaluated on all 1,893 held-out CID photos across the five folds.

The CID weight benchmark reached 22.72 kg mean absolute error against 39.33 kg for the predicted-breed median. The age benchmark reached 53.3% balanced accuracy and missed its 55% research gate. The weight background-control result motivates independent field testing. [Breed metrics](../results/metrics.csv) · [Trait results](../results/traits_evaluation.json)

The three-output research demo packages the breed classifier and the CID age and weight regressions in a Keras checkpoint. On 503 held-out animals, 179 (35.6%) had a correct breed, age within three months, and weight within 10% of the recorded values. The numeric age model averaged 2.23 months of error versus 1.59 for the training-median baseline. The held-out estimates clustered between 23.3 and 27.8 months, so the benchmark gives limited evidence about older cattle. [Joint metrics](../results/cid_joint.json) · [Demo instructions](traits.md#three-output-research-prototype)

\newpage

## 7 Error analysis, limitations and ethics

The evaluation command selects up to three correct and three incorrect held-out predictions by confidence. Each Grad-CAM overlay comes from the fold model that predicted the corresponding photo. These overlays can help identify whether influential regions coincide with the animal, coat or background. Interpret them with the predictions and image context. Galleries and annotations stay private. [Grad-CAM](https://arxiv.org/abs/1610.02391)

Potential error explanations include similar pigmentation between Sindhi and Sahiwal, unusual poses, partial visibility, calves versus mature animals and background effects. These are hypotheses until individual predictions are reviewed. High-confidence mistakes deserve particular attention because the interface displays the largest probabilities to users.

CID is a cattle-market dataset dominated by Local cattle and young bulls. Some breeds have few groups. Its weight and age labels come from source records of unconfirmed measurement provenance. A background-only weight control performed well on a small subset, so the weight results require validation on new sites. Future data collection requires all three reference labels linked to photographs of the same animal. [Field protocol](traits.md)

Raw photo files, trained models and photo-containing galleries stay private. The demo provides candidate breeds for a human to check.

\newpage

## 8 Reproducibility, contributions and AI use

The initial cloud run used Python 3.12, TensorFlow 2.21.0 and Keras 3.15.1 on CPU. The grouped folds, breed metrics and trait scores are available with run instructions. [Run instructions](../README.md#run-the-study) · [CID provenance](data.md) · [Grouped folds](../results/cid_folds.csv)

Shayan Ahmed owns the project tasks. The live individual defence remains open. [Progress tracker](PROGRESS.md) · [Defence questions](defense_questions.md)

OpenAI Codex assisted with implementation, debugging, automated tests and documentation. Synthetic tests and real CID runs verify the current workflow.
