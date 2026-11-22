# CattleClass

CattleClass uses TensorFlow and Keras to study five breeds of Bangladeshi cattle from CID photos. The research demo shows breed, an age estimate in months, and an estimated weight from one photo. [Colab notebook](cattle-class.ipynb)

[Download the project overview and step-by-step run guide (PDF)](docs/cattleclass-run-guide.pdf) · [Age and weight research workflow](docs/traits.md)

## Set up

From the repository root, use Python 3.12.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux or macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

On Linux or WSL2 with a compatible NVIDIA GPU, install `tensorflow[and-cuda]` for GPU support. A mounted Google Drive in Colab can hold the data and checkpoints across sessions.

## Run the study

| Command | What it does |
|---|---|
| `python prepare_data.py` | Downloads CID photos and labels, removes duplicates, checks the locked animal-grouped folds and caches decoded images. |
| `python train.py` | Measures the majority-class baseline and trains the two-stage classifier on five CID folds. |
| `python train.py --final` | Fits the model on all CID photos using epoch counts selected during cross-validation. |
| `python evaluate.py --robustness --figures` | Evaluates held-out CID images and generates robustness plots and local Grad-CAM galleries. |
| `python demo.py` | Starts a local Gradio app that uses the final model. |

Use `python prepare_data.py --check` to verify downloads and `python train.py --fraction 0.5` to reproduce E6. Training reuses verified fold checkpoints. Each command documents its options with `--help`. Set `CATTLE_WORK_DIR` to store downloaded data and checkpoints elsewhere.

You can train separate research models on specialist datasets. Breed training accepts a private labeled CSV. Age and weight accept task-specific development and calibration data, and the reserved final test requires all three references. [Specialist data format](docs/data.md#specialist-datasets) · [Field workflow](docs/traits.md#collect-and-lock-a-field-study)

## How the model works

1. Turn each photo into an upright RGB image, pad it to a square without stretching it, then crop to 224 × 224 pixels. Training uses random crops. Evaluation and the demo use the center.
2. `keras.applications.EfficientNetB0` starts from ImageNet weights. A small classification head produces five probabilities. Keras trains the head first, then fine-tunes the backbone at a lower learning rate. [Model code](cattle_breed/model.py)
3. Keras saves each trained model as a `.keras` checkpoint. The demo loads the final checkpoint. Evaluation uses the corresponding held-out fold checkpoint for each CID photo.

## Read the results

Breed macro-F1 was **0.500** on the locked CID folds. Mean absolute error was **22.72 kg** for weight and **2.23 months** for numeric age. The simple training-median age baseline averaged **1.59 months** of error. On the 503 shared CID animals, **179 (35.6%)** had a correct breed, age within three months, and weight within 10% of the recorded values. Field accuracy remains unverified. [Report](docs/report.md) · [Joint results](results/cid_joint.json) · [Research demo steps](docs/traits.md#three-output-research-prototype)

- `cattle_breed/` contains settings, downloads, image preparation, model training and scoring. `tests/` exercises the workflow on small fixtures.
- `results/` contains the published tables, fold manifest and audit summaries. Evaluation regenerates plots under `results/figures/`. [CID confusion matrix](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/results/figures/cid_5-way_confusion.png)
- Keep data, checkpoints and photo galleries local. [CID provenance and licence](docs/data.md)
- The course deliverables and work still open are listed in [Progress](docs/PROGRESS.md).
