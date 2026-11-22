# Progress

Owner throughout: Shayan Ahmed. Updated 22 November 2026.

| Tasks | Status |
|---|---|
| T0–T2 · Setup, image loading and indexing | Complete. Indexed 503 CID animal IDs and 2,012 photos. |
| T4–T5 · Deduplication, folds and cache | Complete. 1,893 CID photos in 371 groups. |
| T6–T8 · Baseline, training and final model | Complete. Verified five held-out folds and the final all-CID checkpoint. |
| T9–T11 · Evaluation and error analysis | CID evaluation uses the matching fold model for clean predictions, robustness and Grad-CAM. |
| T12 · Demo | Verified a real-photo upload and saved the backup recording privately. |
| T13 · Report, slides and individual defence | Report and slides updated for the CID scope. The individual defence remains open. |
| Optional E6 · Half the training groups | Complete. Scored on the same five held-out CID folds. |

Dataset scope changed to CID because it links breed, age and weight to the photographed animal. A CID-only workspace rebuilt the five-fold manifest, and the held-out run reproduced all ten CID breed rows and the existing trait results. The test suite passed all 71 tests. [Earlier study and reproduction record](https://github.com/ShayanAhmed12/CattleClass/blob/a52241ad0df375bee3945196d59cba5578cd2c32/results/reproduction.json)

[Report](report.md) · [Slides](slides.md) · [Defence guide](defense_questions.md) · [Project overview and run guide (PDF)](cattleclass-run-guide.pdf)

## Age and weight extension

- CID preparation reproduced 503 source SKUs and 371 groups. The age cohort contains 498 records in the supported bands.
- The weight candidate reached 22.72 kg MAE against 39.33 kg for the predicted-breed median. The background control also predicted weight on a matched subset, so independent field testing is necessary.
- The age benchmark reached 53.3% balanced accuracy and missed its research gate. A limited fine-tuning check did not improve inner validation.
- The public CowDatabase2 check scored paired left and right views for 95 Aberdeen Angus animals. Mean absolute weight error was 222.41 kg and 237.11 kg, respectively. Every prediction fell below its recorded weight and outside 10% error. The public files lack photo dates and verified scale references, so the independent age and field tests remain open. [Public-data check](traits.md#independent-weight-check-on-cowdatabase2) · [Results](../results/cowdatabase2_weight.json)
- A matched crop study scored 478 CID cases. Cropping increased CID error from 22.20 to 25.76 kg. On CowDatabase2 it reduced error for both views and still left every estimate outside 10%. Only five CID animals have an age label outside 24 and 30 months. [Capture and age audit](traits.md#capture-and-breed-follow-up)
- An experimental breed rejection rule retained 90.7% of CID known cases and rejected all 95 Angus images. It missed the 80% rejection target for all five CID breeds used as simulated unknowns. Both demo modes now explain the five-breed scope. [Breed audit](../results/capture_diagnostics.json)
- Additional public-data screening checked the Hanwoo archive, Hereford workbooks and calf metadata, plus other published releases. No additional candidate passed the linked-photo, breed, birth-record and scale-weight requirements. [Dataset screening](traits.md#additional-public-dataset-screening)
- The specialist-data policy now allows a breed-only research CSV and task-specific age or weight development and calibration. The reserved final test still requires all three linked references. [Data formats](data.md#specialist-datasets)
- The CID research prototype now estimates continuous age in months. On 503 locked held-out animals, age error was 2.23 months, versus 1.59 for the training-median baseline. Breed, age and weight met their joint criteria on 179 animals (35.6%). [Research mode and results](traits.md#three-output-research-prototype)
- Private Keras candidate reload, field fit/calibration/evaluation, and demo withholding were exercised. Field approval remains open pending an independent cohort with paired photos, breed labels, scale weights and documented birth dates. [Workflow](traits.md) · [Results](../results/traits_evaluation.json) · [Plan](age-and-weight-plan.md)

## Project archive

[Original plan](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/PLAN.md) · [Decision log](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/decisions.md) · [Proposal](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/proposal.docx) · [Course guidelines](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/guidelines.pdf) · [Previous progress log](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/docs/PROGRESS.md)
