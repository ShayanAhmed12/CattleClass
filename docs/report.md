---
geometry: margin=0.85in
fontsize: 11pt
documentclass: article
mainfont: DejaVu Serif
sansfont: DejaVu Sans
colorlinks: true
linkcolor: blue
urlcolor: blue
papersize: a4
---

```{=latex}
\begin{titlepage}
\centering
\vspace*{2.2cm}
{\large\sffamily CS4045 · Deep Learning for Perception\par}
\vspace{1.8cm}
{\huge\sffamily\bfseries Cattle Breed Classification\\[4pt]from Images Using Transfer Learning\par}
\vspace{0.55cm}
{\Large\sffamily CattleClass · Final project report\par}
\vspace{1.8cm}
{\large\sffamily Group members\par}
\vspace{0.3cm}
\begin{tabular}{ll}
Muneeb Ul Haq & 23K-0578\\[6pt]
Shaikh Hassan Nafees & 23K-0769\\[6pt]
Shayan Ahmed & 23K-0551
\end{tabular}
\vspace{1.4cm}

\textbf{GitHub repository}\\[3pt]
\url{https://github.com/ShayanAhmed12/CattleClass}
\vfill
Fall 2026 · Submission deadline 29 November 2026
\end{titlepage}
```

## 1 Introduction

CattleClass estimates breed from one RGB photograph. It gives farmers, livestock traders and dairy record keepers a breed suggestion they can check against records or with an experienced handler. A photograph shows appearance, so the prediction cannot establish ancestry.

The classifier assigns probabilities to five labels from the Cow Images Dataset (CID), which documents cattle photographed at markets in Bangladesh. The labels are Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi. CID uses Local as a collective label for indigenous cattle. Its records also include age and weight, which we used in a separate exploratory study. We used EfficientNet-B0 with ImageNet weights as the starting point for the classifier. [CID publication](https://doi.org/10.1145/3542954.3543018) · [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html)

We tested on held-out CID animals and checked how changes in sharpness, lighting and visibility affected breed predictions. The group omitted the farm test proposed as E3. Performance on independent farms is therefore unknown. Both inference models are available from GitHub, and the demo recording is stored privately. [Model release](https://github.com/ShayanAhmed12/CattleClass/releases/tag/cid-models-v1)

## 2 Data sources and label mapping

The main CID archive contains 514 folders with four photos each and 513 metadata records. We matched folders to records by removing spaces from the stock keeping unit (SKU), CID's animal identifier. We skipped the folder with no label and excluded Pabna, Brahma and Mir Kadim because they had too few animals for grouped evaluation. The study used the main archive. CID's separate YouTube archive was outside the study. [Data provenance and checksums](data.md)

| CID class | Source animal IDs | Indexed photos |
|---|---:|---:|
| Holstein cross | 32 | 128 |
| Local | 353 | 1,412 |
| Red Chittagong | 21 | 84 |
| Sahiwal | 59 | 236 |
| Sindhi | 38 | 152 |
| **Total** | **503** | **2,012** |

Local accounts for roughly 70% of the animals, so accuracy alone can hide poor results for smaller classes. The private sample grid shows all four views of one animal per class. The evaluation considered all four views before deduplication. In the pose audit, each filename suffix had 503 images. Suffixes 0, 1 and 2 had 475, 497 and 434 profiles facing left in the image, respectively. Suffix 3 had 496 profiles facing right.

The dataset records 375 animals at 24 months, 123 at 30 months and 5 at 36 months. Recorded weights range from 150 to 698 kg across the 503 source animal IDs. The joint benchmark uses one representative image per source animal and includes all three recorded ages. [Trait audit](age-and-weight-plan.md)

\newpage

## 3 Curation, duplicate detection and grouped splits

We corrected image orientation using EXIF metadata, then computed a 64-bit perceptual hash for each readable image. Two photos counted as near-duplicates when their hashes differed in at most four bits. We found 135 matching pairs across different SKUs. Some photos matched more than one other photo, so the 135 pairs led to 119 removals. We grouped linked SKUs together, including chains of matches. This left 1,893 photos in 371 groups.

Seventeen groups have more than one breed label. Each photo kept its original label. A hash match reflects visual similarity, so a linked group can contain different animals. Keeping those photos together reduces the risk that similar images appear in both training and testing.

We used five stratified group folds with seed 42. Each photo appears in one test fold. Within each fold's training set, we held out 15% of the animal groups for validation, stratified by each group's most common breed label. Every training, validation and test split contains all five labels. We checked for group overlap in both the outer test split and the inner validation split. [Fold manifest](../results/cid_folds.csv)

\newpage

## 4 Model and training procedure

We load photos upright in RGB, resize without stretching and pad them to 240 × 240 pixels before caching them as `uint8`. Training uses a random 224 × 224 crop. Validation, evaluation and the demo use a center crop. Inputs stay in the 0 to 255 range because EfficientNet rescales them internally. The cache builder reports large or unreadable images and skips them. For the locked CID benchmark, preparation stops if any expected photo is missing from the cache. The index records row order and fold identity so later commands can check for mismatches.

The TensorFlow/Keras model uses ImageNet-pretrained EfficientNet-B0, global average pooling, dropout of 0.3 and a five-output softmax. Training uses horizontal flips, rotations of about seven degrees and mild changes in brightness and contrast. We preserve hue and saturation because coat color can help distinguish breeds. We run the backbone in inference mode to keep its BatchNorm statistics fixed. Its trainable weights can still change during fine-tuning.

| Stage | Trainable layers | Adam learning rate | Maximum epochs |
|---|---|---:|---:|
| E1 | New head | 0.001 | 10 |
| E2 | Head and backbone | 0.00001 | 20 |

Both stages use balanced class weights and categorical cross-entropy with label smoothing of 0.1. Early stopping tracks validation loss and restores the best weights. We save E1 test predictions before fine-tuning. After cross-validation, we train the final model on all CID photos using the median number of selected epochs for each stage. This fit uses the entire dataset without a separate validation split. Each fold saves its predictions, selected epochs, fold ID and model checksum so later runs can reuse verified outputs. [Keras EfficientNet reference](https://keras.io/api/applications/efficientnet/) · [Label smoothing](https://arxiv.org/abs/1512.00567)

\newpage

## 5 Evaluation protocol

B0 predicts Local for every photo. We scored B0, E1 and E2 on the same five test folds. The overall table reports fold means and the standard deviation of macro-F1. The per-breed table and confusion matrices use the pooled predictions. The matrices put true labels in rows and predicted labels in columns.

Accuracy is the share of correct predictions. For one breed, precision is the share of predictions for that breed that were correct, and recall is the share of its photos the model found. Balanced accuracy averages recall across the breeds present in the test data. Macro-F1 averages the F1 scores for those breeds, giving each breed equal weight.

For E5, we scored the same test photos with and without deliberate image changes. We applied Gaussian blur with a radius of two or four pixels, multiplied brightness by 0.5 or 0.3, or covered about one-eighth or one-quarter of the crop with a gray square placed using a fixed random seed. Each photo used its own held-out fold model in every condition.

![Clean CID and the strongest severity in each corruption family.](submission/figures/cid-robustness.png){width=94%}

\newpage

## 6 Results and interpretation

| Experiment | Dataset | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| B0 | CID, 1,893 photos | 0.703 | 0.200 | 0.165 ± 0.001 |
| E1 | CID, 1,893 photos | 0.611 | 0.533 | 0.453 ± 0.053 |
| E2 | CID, 1,893 photos | 0.656 | 0.580 | 0.500 ± 0.046 |
| E6 | CID, half training groups | 0.619 | 0.559 | 0.467 ± 0.034 |
| E5 | Six CID corruption conditions | See CSV | See CSV | See Figure 1 |

Local is common enough that B0 was right on about 70% of the test photos. All five classes appear in every fold, and B0 finds only Local, giving it 0.200 balanced accuracy. Its macro-F1 is lower because four breeds score zero and every other breed predicted as Local counts against Local's precision. Grouped stratification kept the proportion of Local photos similar across folds.

Fine-tuning raised mean macro-F1 from 0.453 to 0.500. E6, the repository label for proposal E4, used a stratified sample of half the training groups in each fold. It kept the same validation and test splits and scored 0.033 lower than E2 across the same 1,893 test photos. Strong blur caused the largest drop in the corruption study, from 0.500 on clean photos to 0.189.

\newpage

### 6.1 Per-breed results and research extension

For E2, support is the number of held-out photos with each recorded breed.

| Breed | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Holstein cross | 122 | 52.6% | 82.8% | 64.3% |
| Local | 1,330 | 91.6% | 70.8% | 79.9% |
| Red Chittagong | 79 | 28.4% | 55.7% | 37.6% |
| Sahiwal | 225 | 29.9% | 49.8% | 37.3% |
| Sindhi | 137 | 29.4% | 30.7% | 30.0% |

Sindhi had the lowest recall. The model classified 64 of its 137 photos as Sahiwal. Local had the highest F1, and the model still classified 179 Local photos as Sahiwal. [Per-class E1 and E2 metrics](../results/cid_per_class.csv) · [Confusion matrices](../results/cid_cv_details.json)

We studied age and weight as optional extensions. The weight model averaged 22.72 kg of absolute error on CID. Using the median weight for the predicted breed gave 39.33 kg. The earlier age classification experiment reached 53.3% balanced accuracy, below its 55% threshold. [Breed metrics](../results/metrics.csv) · [Trait results](../results/traits_evaluation.json)

The research demo combines the breed classifier with models that estimate age in months and weight in kilograms in one Keras checkpoint. Of 503 held-out animals, 179 (35.6%) met all three criteria on the same photo. The breed matched, age was within three months and weight was within 10% of the recorded value. The age estimates averaged 2.23 months of absolute error, compared with 1.59 months for the training-median baseline. They ranged from 23.3 to 27.8 months, so the benchmark offers little evidence about older cattle. [Joint metrics](../results/cid_joint.json) · [Demo instructions](traits.md#three-output-research-prototype)

\newpage

## 7 Error analysis, limitations and ethics

The evaluation script selects up to three correct and three incorrect predictions by confidence. It uses the fold model that scored each photo to create a Grad-CAM overlay. The overlay helps locate image regions that influenced the prediction. We interpret it alongside the photo and the predicted breed. [Grad-CAM](https://arxiv.org/abs/1610.02391)

In the private gallery, the model called two dark brown Sahiwal cattle Local with 95% and 93% confidence. It called one dark-coated Local animal Holstein cross with 97% confidence. The three high-confidence correct examples were all Local. The photos show similar coat and background details, and the gallery alone cannot establish what caused the errors. The controlled tests give firmer evidence for image quality effects. Strong blur cut macro-F1 by 0.311, and strong occlusion cut it by 0.261. [Anonymized error examples](error-analysis.md)

![Pooled E2 confusion matrix. Labels show counts. Colour uses a logarithmic scale to distinguish small classes.](submission/figures/cid-confusion.png){width=72%}

CID mostly represents Local cattle and young bulls photographed at markets. Some other breeds have few groups. We could not verify how the original age and weight records were measured. A background-only control performed well on a small weight subset, which raises the possibility that the weight model uses clues from the setting. A future field study would need verified breed, age and weight for the same photographed animals. The raw photos, training checkpoints, galleries and annotations stay private. [Field protocol](traits.md)

\newpage

## 8 Reproducibility, contributions and AI use

The initial cloud run used Python 3.12, TensorFlow 2.21.0 and Keras 3.15.1 on CPU. The repository provides instructions for running the demo, reproducing the grouped splits and checking the published scores. We checked the workflow with synthetic tests and real CID runs. [Run instructions](../README.md#run-the-research-demo) · [CID provenance](data.md) · [Grouped folds](../results/cid_folds.csv)

The original proposal names the group members. Shayan Ahmed maintained the repository and recorded the technical work in the progress log. The presentation and individual defence are scheduled course activities. OpenAI Codex assisted with implementation, debugging, automated tests and documentation. [Progress tracker](PROGRESS.md) · [Defence questions](defense_questions.md)

### References and submission sources

- Shagor, M. B., Alvi, M. Z. H., and Hasan, K. T. (2022). *CID: Cow Images Dataset for Regression and Classification.* Proceedings of ICCA. [DOI](https://doi.org/10.1145/3542954.3543018)
- Tan, M., and Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* PMLR 97, 6105–6114. [Publisher record](https://proceedings.mlr.press/v97/tan19a.html)
- [Original group proposal and final submission files](submission/README.md)
