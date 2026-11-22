# CattleClass

Upload one cattle photo to estimate breed, age in months, and weight in kilograms. The TensorFlow/Keras models use CID, a dataset of Bangladeshi cattle photos and recorded traits. [Open the Colab notebook](cattle-class.ipynb)

[Project reference for developers and assistants](PROJECT.md)

## Run the research demo

Use Python 3.12 from the repository root. A GPU helps with breed training. In Colab, mount Drive to keep the dataset and checkpoints across sessions.

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train.py
python train.py --final
python train.py --traits
python -m cattle_breed.research
python demo.py --research
```

Open the URL shown in the terminal. Upload a clear side photo, select **Estimate traits**, then use **Reset** to try another animal. Add `--share` in Colab. Set `CATTLE_WORK_DIR` to choose where data and models are stored. Training reuses matching checkpoints.

## Understand the results

EfficientNet extracts features from the photo. The breed model scores five supported breeds. Separate regressions estimate age and weight from image features. Test folds keep photos of the same animal and near-duplicates together. [Data and supported breeds](docs/data.md) · [Model and evaluation methods](docs/report.md)

The [project reference](PROJECT.md#data-models-and-evaluation) explains the benchmarks and their limits. Accuracy on independent farms remains unverified.

## Explore the project

| Where | What it contains |
|---|---|
| `prepare_data.py`, `train.py`, `evaluate.py`, `demo.py` | The main commands for data, models, evaluation, and the app. |
| `cattle_breed/` and `tests/` | The implementation and automated tests. Run `python -m pytest -q`. |
| `results/` | The locked folds and published benchmark results. |
| `experiments/` | Optional external and camera studies. [Commands and results](docs/traits.md) |

For the breed-only app, run `python demo.py`. To reproduce the breed robustness plots, run `python evaluate.py --robustness --figures`. [Field validation](docs/traits.md#collect-and-lock-a-field-study) · [Project progress](docs/PROGRESS.md) · [Archived run guide PDF](docs/cattleclass-run-guide.pdf)
