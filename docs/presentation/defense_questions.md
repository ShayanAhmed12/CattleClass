# Oral defence question bank

Answer each one aloud without notes. The "where" column points to the file or decision that holds the answer.

## Problem and data

| # | Question | Where |
|---|---|---|
| 1 | Why breed classification, and who would use it? | Proposal §2 |
| 2 | Why these five classes and not all eight CID breeds? | D-003 |
| 3 | What is the "Local" class, and why is it a problem? | D-003, D-009 |
| 4 | Why not test on a hand-collected set? | D-002, D-011 |
| 5 | How did you make sure no test image was seen in training? | T5, T6, D-005, D-015 |
| 6 | Why split by animal instead of by image? What would happen if you split by image? | D-005 |
| 7 | How did you curate the Kaggle images, and how do you know you didn't cherry-pick? | `docs/curation/README.md`, D-010 |
| 8 | What biases does CID have, and how do they show in your results? | Report §7 |

## Model and training

| # | Question | Where |
|---|---|---|
| 9 | Why EfficientNet-B0? What is a compound-scaled network? | D-004 |
| 10 | What does "transfer learning" mean here? What do the ImageNet weights give you? | T10 |
| 11 | Why two stages? Why is the second learning rate 100× smaller? | D-007 |
| 12 | Why keep BatchNorm in inference mode while fine-tuning? | T10 `test_backbone_batchnorm_stays_frozen` |
| 13 | Why don't you divide pixels by 255? | Global Constraints; T9 |
| 14 | Why no hue augmentation? Which augmentations did you use and why? | D-006 |
| 15 | What do class weights do mathematically? And label smoothing? | T11, D-007 |
| 16 | How does early stopping choose the epoch? How was the final model's epoch count chosen? | T11, T12 |
| 17 | Why letterbox instead of resizing or centre-cropping? | D-008 |

## Evaluation

| # | Question | Where |
|---|---|---|
| 18 | Why macro-F1 and balanced accuracy instead of accuracy? What does B0 show? | D-009, B0 row |
| 19 | Why 5-fold cross-validation? What does the std tell you? | D-005 |
| 20 | What is the difference between 5-way and restricted scoring? | D-009, T13 |
| 21 | Why does performance drop on Kaggle compared with Mendeley? | E3, E4 results; error tags |
| 22 | Which breeds are confused, and why? | Confusion matrices, Grad-CAM |
| 23 | How does Grad-CAM work? What did it show? | T15 |
| 24 | Which corruption hurt most, and why might that be? | E5 |
| 25 | If you had two more weeks, what would you change first? | Limitations |

## Code (be ready to open any file)

| # | Question | Where |
|---|---|---|
| 26 | Walk through what happens to one image from disk to prediction. | T1 → T7 → T9 → T10 |
| 27 | How does `make_folds` guarantee one animal is in one fold? | `splits.py` |
| 28 | How does the CV runner resume after a disconnect? | `train.py` `run_cv` |
| 29 | How is the pHash Hamming distance computed? | `dedup.py` |
| 30 | Which parts did AI tools help with, and how did you verify them? | Report §9 |
