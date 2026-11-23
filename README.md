# CattleClass

Upload one cattle photo to estimate breed, age in months, and weight in kilograms. The TensorFlow/Keras models use CID, a dataset of Bangladeshi cattle photos and recorded traits. [Open the Colab notebook](cattle-class.ipynb)

[Project reference for developers and assistants](PROJECT.md)

## Run the research demo

Use Python 3.12 and run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m cattle_breed.pretrained
python demo.py --research
```

The one-time download verifies and caches the trained model. Open the URL from the terminal, upload a clear side photo, and select **Estimate traits**. Use **Reset** to try another photo. For the breed-only demo, run `python -m cattle_breed.pretrained --breed` once, then `python demo.py`. [Download files on GitHub](https://github.com/ShayanAhmed12/CattleClass/releases/tag/cid-models-v1) · [Optional training from scratch](PROJECT.md#train-from-scratch)

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
