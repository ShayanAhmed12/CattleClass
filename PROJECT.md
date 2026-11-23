# CattleClass project reference

Owner: Shayan Ahmed · Course: CS4045 Deep Learning for Perception · Updated 23 November 2026

Start here for project context and development. Read the linked source files for exact behavior and the published result files for exact measurements. Update this reference and the [progress log](docs/PROGRESS.md) when a change affects behavior, data, results, or project status.

## Purpose and current status

CattleClass takes one RGB cattle photo. It estimates breed, age in months, and live weight in kilograms. The course project measures how much these traits can be inferred from photos of previously unseen animals in the Cow Images Dataset (CID). The app is a research prototype for decision support alongside documented records or experienced handlers. The final course study uses CID. The group omitted the planned independent farm test, so field generalization remains unmeasured. The individual course defence is pending. [Course submission](docs/submission/README.md)

| Mode | Command | Input and output |
| --- | --- | --- |
| Research | `python demo.py --research` | Upload one clear side photo. See the three highest breed scores, a continuous age estimate in months, and a weight estimate in kilograms. The displayed weight error is a CID benchmark average. |
| Breed only | `python demo.py` | Upload one photo. See the three highest breed scores from the same five-class label set. |
| Controlled field | `python demo.py --traits --traits-cohort enrolled-program --traits-capture ordinary` | An operator uploads a photo of an enrolled animal and confirms the capture protocol. Breed scores are available. The app withholds age or weight when its release checks fail. A released age result uses bands and a released weight result includes an interval. |

The breed classifier scores Holstein cross, Local, Red Chittagong, Sahiwal, and Sindhi, in that output order. **Local** is a CID label for indigenous cattle. A photo of an unsupported animal or breed can still receive a high score. Age is numeric in research mode, but the available labels are concentrated at 24 and 30 months and predictions on held-out images clustered around 23–28 months. Reliability across a wide age range remains unverified. [UI and CLI](demo.py) · [Label order](cattle_breed/config.py)

## Run from a clean checkout

Use Python 3.12 from the repository root. TensorFlow, Keras, Gradio, and the other dependencies are pinned. [Dependencies](requirements.txt)

```bash
python -m pip install -r requirements.txt
python -m cattle_breed.pretrained
python demo.py --research
```

The installer downloads and verifies the published Keras bundle and its metadata once, then reuses them. For the breed-only app, run `python -m cattle_breed.pretrained --breed` once before `python demo.py`. The installer stores artifacts in `models/pretrained/` inside `CATTLE_WORK_DIR` (the checkout by default). The app prefers locally trained artifacts. An installation needs internet access. Inference can run offline after setup. Open the Gradio URL and select **Estimate traits** after uploading. Use **Reset** to try another photo. In [Colab](cattle-class.ipynb), `--share` creates a public Gradio URL. [Published assets](https://github.com/ShayanAhmed12/CattleClass/releases/tag/cid-models-v1)

### Train from scratch

Use a fresh `CATTLE_WORK_DIR` if you want to train without reusing any local checkpoints. A GPU helps with breed training. In Colab, mount Drive to retain the dataset and training outputs across runtime resets.

```bash
python prepare_data.py
python train.py
python train.py --final
python train.py --traits
python -m cattle_breed.research
python demo.py --research
```

If preparation reports an incomplete download, rerun it to resume. If it reports a manifest or checksum mismatch, check the source files and work directory before rebuilding. [Data provenance](docs/data.md)

| Task | Command |
| --- | --- |
| Check downloaded CID files | `python prepare_data.py --check` |
| Run software tests | `python -m pytest -q` |
| Check the majority-breed baseline or train selected folds | `python train.py --baseline` or `python train.py --folds 0 1` |
| Evaluate held-out breed results and optional robustness or galleries | `python evaluate.py` or `python evaluate.py --robustness --figures` |
| Reproduce the half-training-data breed experiment | `python train.py --fraction 0.5` |
| Score saved weight and earlier two-band age benchmarks | `python evaluate.py --traits` after `python train.py --traits` |
| Fit the separate weight candidate and run the external study | `python train.py --traits --final`, then `python -m experiments.public_weight` |
| Reproduce the crop and breed-rejection study | `python -m experiments.capture /path/to/yolox_tiny.onnx` with the pinned optional setup. [Capture setup](docs/traits.md#capture-and-breed-follow-up) |

## Data, models, and evaluation

1. **Prepare CID.** CID contains Bangladeshi cattle-market photos and source metadata. The SKU joins photos to labels. The code converts `age_in_year` to months and reads `weight_in_kg` as recorded weight. Indexing retains 2,012 usable photos from 503 source animals across the five supported labels. Preparation verifies the source and then matches the cleaned index to the published fold manifest. [Data details](docs/data.md) · [Dataset and licence](https://github.com/bhuiyanmobasshir94/CID)
2. **Prevent train/test overlap.** Perceptual hashes link suspected duplicate photos and their source animal IDs. Deduplication leaves 1,893 photos in 371 groups. Five stratified grouped folds keep linked photos together. Each breed model has a separate group-level validation split within its training fold. Breed evaluation uses all retained photos. Trait and joint headline metrics use one representative photo per original animal ID. [Locked folds](results/cid_folds.csv) · [Methodology](docs/report.md)
3. **Process pixels consistently.** Decode upright RGB, keep the aspect ratio, and pad to 240 × 240. Training takes a random 224 × 224 crop. Evaluation and inference take the center crop and preserve the 0–255 input range for EfficientNet's internal rescaling. [Image pipeline](cattle_breed/data.py)
4. **Train breed models.** ImageNet-pretrained EfficientNet-B0 feeds global average pooling, dropout, and a five-output softmax. E1 trains the head with the backbone frozen. E2 fine-tunes the network while keeping BatchNorm moving statistics fixed. Balanced class weights and validation early stopping are used. The demo breed model is refit on all prepared CID photos after cross-validation. [Model](cattle_breed/model.py) · [Training](train.py)
5. **Estimate age and weight.** The research bundle combines the final breed classifier with a separate frozen ImageNet EfficientNet-B0 feature extractor and log-target Ridge heads. The age output is decoded in the Keras graph as `age_months`. The app exponentiates `log_kg` to get kilograms. The earlier age benchmark evaluates only the two common CID ages and remains a separate research result. [Research export and scoring](cattle_breed/research.py) · [Trait benchmarks](cattle_breed/traits.py)
6. **Evaluate before interpreting.** The joint score requires a correct breed, age within three months, and weight within 10% of the recorded value, all for the same photo. Breed macro-F1 weights each represented class equally. MAE means mean absolute error in the displayed unit. Robustness uses the proper held-out model for each photo under blur, darkening, and occlusion. Grad-CAM galleries are private qualitative diagnostics. [Evaluation code](evaluate.py)

| Published finding | Result |
| --- | --- |
| Breed macro-F1 on 1,893 held-out photos | 0.165 for the Local-only baseline, 0.453 for the frozen head, 0.500 after fine-tuning. Strong blur reduced it to 0.189. [Metrics](results/metrics.csv) |
| Joint research benchmark on 503 held-out animals | 179 met all three criteria (35.6%). Weight MAE was 22.72 kg. Age MAE was 2.23 months versus 1.59 for the training-median baseline. [Joint metrics](results/cid_joint.json) |
| Age support in CID | 375 records at 24 months, 123 at 30, and five at 36. Held-out estimates ranged from 23.3 to 27.8 months. The earlier two-band classifier scored 53.3% balanced accuracy and missed its 55% research gate. [Joint metrics](results/cid_joint.json) · [Older trait benchmark](results/traits_evaluation.json) |
| External weight and capture checks | The CID weight candidate underestimated all 95 paired Angus animals in each view in CowDatabase2. Left and right MAE were 222.41 and 237.11 kg. Cropping worsened CID weight error on the matched subset. The trial breed rejection rule missed its target on all five simulated unknown breeds. [External weight results](results/cowdatabase2_weight.json) · [Capture audit](results/capture_diagnostics.json) |

CID is imbalanced and drawn from a limited collection setting. Its age and weight records have unconfirmed measurement provenance. A background-only weight control also performed well on a subset, which makes independent testing especially important. CowDatabase2 is a transfer check with missing photo dates and verified scale provenance. [Limitations and research history](docs/traits.md)

## Independent field validation

Separate specialist datasets can supply documented breed, birth dates, or live weights for development and calibration. A future joint test needs all three references linked to the same photographed animals, with independent farms and animals reserved from every model. Start by comparing ordinary, side, and side-with-marker photos taken in the same session. Record stable animal and capture IDs, actual photo and birth dates, and same-session scale weights. Keep failed attempts in the test denominator. [Collection plan](docs/age-and-weight-plan.md) · [Specialist data formats](docs/data.md#specialist-datasets)

The implemented field workflow runs per trait. It fits on development data, calibrates on a separate partition, and evaluates against locked references and media with a prespecified policy. Its private ledger prevents a changed candidate from reusing the same test animals for the same trait and capture mode. Report coverage and withholding along with error. A field release requires compatible model metadata and passing pooled, site and declared subgroup gates. Development and software tests exist for this workflow. An independently approved release and joint field result still require real field data. [Field files and policy](docs/traits.md#collect-and-lock-a-field-study) · [Implementation](cattle_breed/field_traits.py)

## Development map

| Area | Source and responsibility |
| --- | --- |
| Configuration and preparation | [config.py](cattle_breed/config.py) defines label order, paths and training settings. [download.py](cattle_breed/download.py) downloads and checks CID. [prepare_data.py](prepare_data.py) builds the locked cache with [data.py](cattle_breed/data.py). |
| Breed | [train.py](train.py) orchestrates folds and the final fit. [model.py](cattle_breed/model.py) defines the classifier. [evaluate.py](evaluate.py) runs held-out evaluation and [metrics.py](cattle_breed/metrics.py) calculates scores. |
| Traits | [traits.py](cattle_breed/traits.py) aligns labels and scores folds. [trait_models.py](cattle_breed/trait_models.py) fits and selects candidates. [trait_release.py](cattle_breed/trait_release.py) fits and packages trait heads. [research.py](cattle_breed/research.py) exports the three-output research bundle. [field_traits.py](cattle_breed/field_traits.py) owns the controlled field workflow. |
| App and experiments | [demo.py](demo.py) is the shared Gradio app. [experiments/](experiments/) contains optional CowDatabase2 transfer, capture, and breed-rejection studies. |
| Verification and records | [tests/](tests/) and [GitHub Actions](.github/workflows/tests.yml) cover core software flows with fixture data. [results/](results/) holds aggregate evidence and the locked manifest. [docs/](docs/) holds protocols, course materials and the progress history. |

Raw inputs and processed caches live under `CATTLE_WORK_DIR/data/`. Models and per-fold predictions live under `CATTLE_WORK_DIR/models/`. The breed app loads `models/final.keras`, and the research app loads `models/traits/cid-research.keras` with adjacent JSON metadata. Those paths are relative to the work directory. Aggregate results stay in the repository's `results/`, even when the work directory is elsewhere. The versioned GitHub model release links to the two inference models. Keep original images, training checkpoints, per-animal records and predictions, private field ledgers, and photo galleries outside Git. [Ignore rules](.gitignore)

When changing input preparation, keep training, evaluation, and inference transformations aligned. Preserve the published folds and label order for direct comparisons. If a change alters an experiment, document the protocol and regenerate its aggregate metrics. Run pytest for code changes and verify a real upload when changing a demo or model artifact. Update this reference with any resulting change in supported behavior or validation status.
