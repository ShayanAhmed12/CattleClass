# CattleClass

Can a model trained on Bangladeshi cattle-market photos recognize the same breed labels in photos from other sources? CattleClass uses TensorFlow's Keras API (`tf.keras`) to fine-tune EfficientNet-B0 on CID to classify **Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi**. It tests on animals held out from CID, on Mendeley and Kaggle photos, and on blurred, darkened and partly blocked copies. The demo shows the three highest model scores for an uploaded photo.

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

On Linux or WSL2 with a compatible NVIDIA GPU, install `tensorflow[and-cuda]` for GPU support. A mounted Google Drive in Colab can hold the data and checkpoints across sessions. [Colab example](https://github.com/ShayanAhmed12/CattleClass/blob/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/colab.ipynb)

## Run the study

| Command | What it does |
|---|---|
| `python prepare_data.py` | Downloads matched datasets, removes duplicates, splits CID by animal group and caches decoded images. |
| `python train.py` | Measures the majority-class baseline and trains the two-stage classifier on five CID folds. |
| `python train.py --final` | Fits the model on all CID photos using epoch counts selected during cross-validation. |
| `python evaluate.py --robustness --figures` | Scores both external sources and generates robustness plots and local Grad-CAM and error galleries. |
| `python demo.py` | Starts a local Gradio app that uses the final model. |

Use `python prepare_data.py --check` to verify downloads and `python train.py --fraction 0.5` to reproduce E6. Training reuses verified fold checkpoints. Each command documents its options with `--help`. Set `CATTLE_WORK_DIR` to store downloaded data and checkpoints elsewhere.

## How the model works

1. Turn each photo into an upright RGB image, pad it to a square without stretching it, then crop to 224 × 224 pixels. Training uses random crops. Evaluation and the demo use the center.
2. `keras.applications.EfficientNetB0` starts from ImageNet weights. A small classification head produces five probabilities. Keras trains the head first, then fine-tunes the backbone at a lower learning rate. [Model code](cattle_breed/model.py)
3. Keras saves each trained model as a `.keras` checkpoint. Evaluation and the demo load the same final checkpoint for external scoring and individual predictions. CID robustness uses the corresponding held-out fold checkpoint for each photo.

## Read the results

Five-way macro-F1 was **0.500** on held-out CID groups, **0.561** on Mendeley and **0.458** on Kaggle. The external results use automated screening. The original hand review remains open. [Report](docs/report.md) · [Published metrics](results/metrics.csv)

- `cattle_breed/` contains settings, downloads, image preparation, model training and scoring. `tests/` exercises the workflow on small fixtures.
- `results/` contains the published tables, fold manifest and audit summaries. Evaluation regenerates plots under `results/figures/`. [Published plots](https://github.com/ShayanAhmed12/CattleClass/tree/82b6f78e15bed8dcb36bed79c1df79f7775a9bed/results/figures)
- Keep data, checkpoints and photo galleries local. [Data sources, licensing and screening](docs/data.md)
- The course deliverables and work still open are listed in [Progress](docs/PROGRESS.md).
