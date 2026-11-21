# Oral defence

Trace one photo from disk to the demo, then use the results to explain why each step exists. [Report](report.md) · [CID data and provenance](data.md) · [Published metrics](../results/metrics.csv)

| Question | Points to explain |
|---|---|
| What problem does the model solve? | It suggests cattle breeds that a person can check. CID's Local label groups indigenous cattle. An image shows visible traits, while breed identity can depend on ancestry. |
| Why hold out animals? | Four photos can show the same animal. Perceptual hashing links potentially related source IDs before splitting, so their photos stay together even if source labels differ. |
| How does an image become a prediction? | Apply EXIF orientation, convert to RGB, pad to 240 × 240 without stretching, then crop to 224 × 224. EfficientNet rescales the 0–255 input. The demo ranks the model's five probabilities. |
| Why train in two stages? | Train the new head on fixed ImageNet features first, then fine-tune at a lower learning rate. Keeping BatchNorm in inference mode preserves pretrained running statistics. Class weights and early stopping address imbalance and overfitting. |
| Why report macro-F1? | B0 gets 70.3% accuracy by always predicting Local, with zero recall on four breeds. It gets 20% balanced accuracy and 0.165 macro-F1. |
| What changed with fine-tuning and less data? | E1 reached 0.453 macro-F1, E2 reached 0.500, and E6 reached 0.467 using half the training groups. The held-out photos were the same. |
| What do the held-out results show? | E2 scored 0.500 breed macro-F1 on held-out CID groups. An independent field study would measure performance beyond the CID setting. |
| What do the trait results show? | CID weight MAE was 22.72 kg. A background-only control performed well on a small subset. Age balanced accuracy was 53.3% for the two supported bands. |
| What did robustness and Grad-CAM show? | Strong blur produced the lowest CID breed score. Grad-CAM highlights regions that support a prediction. Individual errors require more evidence to explain. |
| How do training and evaluation resume? | A fold is reused when the data and protocol fingerprint, fold ID and model checksum match. The final fit uses the median selected epoch counts. |
| What would you improve, and how did you validate the work? | Explain the use of AI assistance, the synthetic tests and the independent rerun. Priorities for further validation are photos from new sites with breed, birth and scale records linked to the same animals. |
