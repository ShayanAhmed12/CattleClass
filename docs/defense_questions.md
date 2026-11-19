# Oral defence

Trace one photo from disk to the demo, then use the results to explain why each step exists. [Report](report.md) · [Data sources and review rules](data.md) · [Published metrics](../results/metrics.csv)

| Question | Points to explain |
|---|---|
| What problem does the model solve? | It suggests cattle breeds that a person can check. CID's Local label groups indigenous cattle. An image shows visible traits, while breed identity can depend on ancestry. |
| Why hold out animals? | Four photos can show the same animal. Perceptual hashing links potentially related source IDs before splitting, so their photos stay together even if source labels differ. |
| How does an image become a prediction? | Apply EXIF orientation, convert to RGB, pad to 240 × 240 without stretching, then crop to 224 × 224. EfficientNet rescales the 0–255 input. The demo ranks the model's five probabilities. |
| Why train in two stages? | Train the new head on fixed ImageNet features first, then fine-tune at a lower learning rate. Keeping BatchNorm in inference mode preserves pretrained running statistics. Class weights and early stopping address imbalance and overfitting. |
| Why report macro-F1? | B0 gets 70.3% accuracy by always predicting Local, with zero recall on four breeds. It gets 20% balanced accuracy and 0.165 macro-F1. |
| What changed with fine-tuning and less data? | E1 reached 0.453 macro-F1, E2 reached 0.500, and E6 reached 0.467 using half the training groups. The held-out photos were the same. |
| How did the model generalize? | Five-way macro-F1 was 0.561 on Mendeley and 0.458 on Kaggle. Restricting predictions to known source classes gave 0.728 and 0.482. Class coverage, label provenance and collection conditions differ across sources. |
| What did screening establish? | A local image-text model checked each photo in two views. Zero photos met its exclusion thresholds. Breed and privacy checks by hand remain open. |
| What did robustness and Grad-CAM show? | Strong blur produced the lowest five-way score on CID and Mendeley. Strong occlusion was lowest on Kaggle. Grad-CAM highlights regions that support a prediction. Individual errors require more evidence to explain. |
| How do training and evaluation resume? | A fold is reused when the data and protocol fingerprint, fold ID and model checksum match. The final fit uses the median selected epoch counts. A fresh checkout independently reproduced all 38 published metric rows. |
| What would you improve, and how did you validate the work? | Explain the use of AI assistance, the synthetic tests and the independent rerun. Priorities for further validation are hand review and photos from new sites with independently verified labels and animal IDs. |
