# Progress

Owner throughout: Shayan Ahmed. Updated 20 November 2026.

| Tasks | Status |
|---|---|
| T0–T2 · Setup, image loading and indexing | Complete. Downloaded and checked all three sources. Indexed 503 CID animal IDs and 2,012 photos. |
| T3 · External image screening | Automated screen and independent detector audit complete. The original hand review of Mendeley and Kaggle remains open. |
| T4–T5 · Deduplication, folds and cache | Complete. 1,893 CID photos in 371 groups, with 173 Mendeley and 908 Kaggle photos in the provisional evaluation caches. |
| T6–T8 · Baseline, training and final model | Complete. Verified five held-out folds and the final all-CID checkpoint. |
| T9–T11 · External evaluation, robustness and error analysis | Complete for the automatically screened caches. Both scoring modes and all corruption conditions have published results. |
| T12 · Demo | Complete. Verified the real-model upload flow and saved the backup recording privately. |
| T13 · Report, slides and individual defence | Report and slides published. The live individual defence remains open. |
| Optional E6 · Half the training groups | Complete. Scored on the same five held-out CID folds. |

A fresh cloud checkout independently rebuilt the CID folds and reproduced all 38 published metric rows. The cleanup consolidated the run instructions and duplicate code, and reduced the active tree from 57 to 40 tracked files. All 54 active tests passed. Reusing the verified checkpoints reproduced all 38 metric rows and regenerated the six published plots byte for byte. [Reproduction record](../results/reproduction.json)

The follow-up put center cropping in the data module and made evaluation and the demo share one Keras checkpoint loader. All 54 tests passed on TensorFlow 2.21.0 with Keras 3.15.1. The tests cover both training stages and model reload. The Gradio demo returned three scores for a real photo using the final checkpoint. Re-evaluation with the verified checkpoints reproduced all 38 published metric rows and all six aggregate plots.

[Report](report.md) · [Slides](slides.md) · [Defence guide](defense_questions.md) · [Project overview and run guide (PDF)](cattleclass-run-guide.pdf)

## Age and weight extension

- CID preparation reproduced 503 source SKUs and 371 groups. The age cohort contains 498 records in the supported bands.
- The weight candidate reached 22.72 kg MAE against 39.33 kg for the predicted-breed median. The background control also predicted weight on a matched subset, so independent field testing is necessary.
- The age benchmark reached 53.3% balanced accuracy and missed its research gate. A limited fine-tuning check did not improve inner validation.
- Private Keras candidate reload, field fit/calibration/evaluation, and demo withholding were exercised. Field approval remains open pending paired photos with independent scale weights or documented birth dates. [Workflow](traits.md) · [Results](../results/traits_evaluation.json) · [Plan](age-and-weight-plan.md)

## Project archive

[Original plan](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/PLAN.md) · [Decision log](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/decisions.md) · [Proposal](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/proposal.docx) · [Course guidelines](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/guidelines.pdf) · [Previous progress log](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/PROGRESS.md)
