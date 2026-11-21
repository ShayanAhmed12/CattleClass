---
title: Cattle Breed Classification
subtitle: CS4045 · Working presentation
aspectratio: 169
---

## Research question · 45 seconds

- How accurately can a model predict cattle breed and recorded traits for held-out CID animals?
- One photo yields five probabilities.
- The intended use is to suggest a breed for a human to check.

## CID labels · 60 seconds

- **CID:** 503 source animal IDs and 2,012 photos before deduplication.
- CID links breed, age and live weight to the photographed animal.
- The breed model predicts five classes. The trait models run on the same grouped folds.

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

## Held-out results and robustness · 75 seconds

- E1/E2 compare frozen transfer and fine-tuning on held-out CID groups.
- E5 measures changes under blur, darkness and occlusion.
- On the five real CID folds, E1 macro-F1 was 0.453 and E2 macro-F1 was 0.500.

## Errors and limitations · 60 seconds

- Check rare breeds, ambiguous source labels and differences in photography.
- Use local Grad-CAM overlays to discuss candidate visual cues.
- CID is market-heavy. The weight and age records need independent field confirmation.
- A background-only model performed well on a small CID weight subset.

## Demo and conclusion · 90 seconds

- Upload a reviewed cattle photo and show the top three predicted breeds.
- The demo uses the same crop and pixel range as evaluation.
- Demonstrate the trained final model and keep a private backup recording.
- E6 used half the CID training groups and reached 0.467 macro-F1 on the same test folds.
- CID trait research reached 22.72 kg weight MAE and 53.3% age balanced accuracy.

## Discussion · 60 seconds

- Why split by animal and compare perceptual hashes?
- Why report macro-F1 alongside accuracy?
- What does the model learn from coat, pose and background?
- Codex assisted with implementation, tests and documentation. The individual defence covers the methods, results and limits.
