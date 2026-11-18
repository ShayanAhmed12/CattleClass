---
title: Cattle Breed Classification
subtitle: CS4045 · Working presentation
aspectratio: 169
---

## Research question · 45 seconds

- Can a breed classifier trained on market photos recognize the same labels in photos from other sources?
- One photo yields five probabilities.
- The intended use is to suggest a breed for a human to check.

## Three sources · 60 seconds

- **CID:** 503 source animal IDs and 2,012 photos before deduplication.
- **Mendeley:** 190 matched photos from other photographers in Bangladesh.
- **Kaggle:** 929 matched web photos of Indian bovine breeds.
- External labels cover three classes per source. Review and label mapping matter.

## Preventing leakage · 75 seconds

- Match CID photos to metadata using normalized SKUs.
- Compare perceptual hashes and group linked SKUs before splitting.
- Keep 1,893 CID photos in 371 groups after deduplication.
- Hold out animals in five test folds. Use a separate grouped validation split.
- Verify every train, validation and test split has all five classes.

## Transfer learning · 75 seconds

- Cache 240-pixel RGB letterboxes. Crop to 224 pixels for the model.
- E1 trains the new head on frozen ImageNet EfficientNet-B0 features.
- E2 fine-tunes the backbone while keeping BatchNorm statistics fixed.
- Use class weights, label smoothing and validation-based early stopping.
- Train the final model using all CID photos and median selected epochs.

## Measured baseline · 60 seconds

| B0 on CID | Value |
|---|---:|
| Mean accuracy | 70.3% |
| Mean balanced accuracy | 20.0% |
| Mean macro-F1 | 0.165 ± 0.001 |

Always predicting Local achieves high accuracy because Local is common. Class-balanced metrics expose the missing recall on four classes.

## Generalization and robustness · 75 seconds

- E1/E2 compare frozen transfer and fine-tuning on held-out CID groups.
- E3/E4 compare five-way and restricted predictions on reviewed external data.
- E5 measures changes under blur, darkness and occlusion.
- Add measured E1–E5 values and confusion matrices after the real runs.

## Errors and limitations · 60 seconds

- Check rare breeds, ambiguous source labels and differences in photography.
- Use local Grad-CAM overlays to discuss candidate visual cues.
- Add examples from measured outputs after private review.
- CID is market-heavy. External photos have no reliable animal IDs.

## Demo and conclusion · 90 seconds

- Upload a reviewed cattle photo and show the top three predicted breeds.
- The demo uses the same crop and pixel range as evaluation.
- Demonstrate the trained final model and keep a private backup recording.
- The current verified result is B0. The trained-model comparison will answer the research question.

## Discussion · 60 seconds

- Why split by animal and compare perceptual hashes?
- Why report macro-F1 and restrict labels only in a secondary score?
- What does the model learn from coat, pose and background?
- Codex assisted with implementation, tests and documentation. The team reviews images and interprets the results.
