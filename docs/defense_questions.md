# Oral defence questions

Practise answering each one aloud without notes. The answers are in [decisions.md](decisions.md) (D-xx), [PLAN.md](PLAN.md) or the code.

**Problem and data**
1. Why breed classification, and who would use it?
2. Why these five classes? What is "Local", and why is it a problem? (D-02, D-08)
3. Why online test sets instead of your own photos? (D-01)
4. Why split by animal and not by image? What would go wrong otherwise? (D-04)
5. How do you know no test image was seen in training? (PLAN rules 1–2, `data.find_duplicates`)
6. How did you review the Kaggle images without cherry-picking? (D-09, data.md)
7. What biases does CID have, and how do they show up in the results?

**Model and training**

8. What is transfer learning here, and what do the ImageNet weights give you?
9. Why EfficientNet-B0? (D-03)
10. Why two stages, and why is the second learning rate 100× smaller? (D-05)
11. Why keep BatchNorm frozen while fine-tuning? (PLAN rule 4)
12. Why don't you divide pixels by 255? (PLAN rule 3)
13. Which augmentations, and why no hue changes? (D-05)
14. What do class weights and label smoothing do?
15. Why letterbox instead of stretching or cropping? (D-06)

**Evaluation**

16. Why macro-F1 instead of accuracy? What does B0 show? (D-08)
17. Why 5-fold cross-validation? What does the standard deviation tell you? (D-04)
18. What is the difference between 5-way and restricted scoring? (D-08)
19. Which external source has the larger measured change from CID, if either, and which differences could account for it?
20. Which breeds get confused, and why?
21. How does Grad-CAM work, and what did it show?
22. Which corruption hurt most, and why?

**Code** (be ready to open any file)

23. Follow one image from disk to prediction. (`data.py` → `model.py`)
24. How does `make_folds` keep each animal in one fold?
25. How does `train.py` resume after a disconnect?
26. Which parts did AI tools help with, and how did you check them?
27. Which tasks did you do yourself? Explain a part a teammate wrote.
28. With two more weeks, what would you change first?

## Current rehearsal anchors

- B0 scores 70.3% accuracy and 0.165 macro-F1 because it predicts Local for every photo. Explain how the four zero-recall classes lead to 20.0% balanced accuracy.
- The real fold manifest has 1,893 photos in 371 groups. Explain why related source SKUs can share a group despite different label spellings or labels, and how this affects the interpretation of grouped counts.
- A fold checkpoint is reusable only if its model checksum, fold ID and protocol fingerprint match. The final epoch schedule uses the median chosen count across all five matching folds.
- Use the real E1–E5 values and privately reviewed examples before answering the result-dependent questions.

## Short answers for a mock defence

| Question | Answer to explain aloud |
|---|---|
| Why put groups before folds? | Near-duplicate photos and photos of the same source animal can share visual cues. Joining their source IDs before splitting helps prevent the test fold from containing copies of the training examples. |
| What does B0 establish? | Always predicting Local gives about 70% accuracy because Local is common. It recalls only one of five breeds and gets 0.165 macro-F1. |
| Why train the head before unfreezing? | The head first learns a mapping from pretrained features to the five project labels. Fine-tuning then adjusts those features with a smaller learning rate. |
| Why keep BatchNorm in inference mode? | Small, imbalanced batches can alter pretrained running statistics. Inference mode keeps those statistics fixed while gradients still reach trainable backbone weights. |
| What is the external protocol difference? | The five-way score lets the classifier select any training class. The restricted score selects only from the labels present in the external set and depends on knowing its class coverage. |
| How did autonomous screening work? | A pinned local image-text model compared broad exclusion prompts on a center crop and an aspect-preserving view. It applied a conservative agreement rule and kept uncertain cases. It did not use the trained breed classifier to choose the test set. |
| How does screening affect interpretation? | The screen can miss issues, especially close breed errors or identifying details. Scoring is based on an automatically screened test set. The original hand-reviewed condition remains a separate protocol. |
| What can Grad-CAM establish? | The map shows spatial features whose gradients support a selected prediction. Context may contribute, and a map alone cannot establish a cause for an individual error. |
| How does a run resume? | It checks the input and code fingerprint, fold ID and model checksum before reusing a fold. The all-CID fit uses the median selected epoch counts from all five matching folds. |
| What would improve the project? | Improve label and privacy verification, collect cattle from new sites with independent IDs, and validate the demo with an external set whose breed labels are verified. |

## Questions about the measured results

| Question | Answer to explain aloud |
|---|---|
| Why is accuracy lower for E2 than for B0? | B0 accurately predicts the majority class and has zero recall on four breeds. On CID, E2 reaches 65.6% accuracy, 58.0% balanced accuracy, and 0.500 macro-F1. The latter two scores capture the gains across breeds. |
| What did fine-tuning change? | Mean macro-F1 rose from 0.453 for the frozen backbone to 0.500 for fine-tuning across the same five held-out folds. Fold standard deviations were 0.053 and 0.046. |
| What did the half-data test show? | Using half the training animal groups in each fold gave 0.467 macro-F1, about 0.033 below the full-training result on the same held-out photos. This comparison comes from the recorded stratified samples. |
| How did the model generalize? | Five-way macro-F1 was 0.561 on Mendeley and 0.458 on Kaggle. Restricted scores were 0.728 and 0.482. The sources differ in collection context, breed coverage, and label provenance, so the score gap alone cannot identify a cause. |
| What was the weakest corruption condition? | Strong blur had the lowest five-way macro-F1 on CID and Mendeley, at 0.189 and 0.428. Strong occlusion was lowest on Kaggle, at 0.351. Compare each value with the clean score for the same source. |
| What can you show about reproducibility? | The clean checkout regenerated the CID fold manifest. Scoring the saved predictions against the rebuilt cache reproduced the published E1, E2, and E6 fold means. Independent retraining is a separate reproduction check. |
