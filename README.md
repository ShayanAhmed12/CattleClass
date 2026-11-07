# Cattle Breed Classification (CS4045)

Does a breed classifier trained on one dataset still work on cattle photos from somewhere else?

The project fine-tunes **EfficientNet-B0** (ImageNet weights, Keras) on the **CID** dataset of Bangladeshi market cattle in 5 breeds: Holstein cross, Local, Red Chittagong, Sahiwal and Sindhi. It then tests the model on:
- CID itself, with 5-fold cross-validation split by animal;
- **Mendeley**, photos from the same country taken by other photographers;
- **Kaggle**, Indian web images;
- blurred, darkened and occluded copies of all three.

The plan is in [docs/PLAN.md](docs/PLAN.md), progress in [docs/PROGRESS.md](docs/PROGRESS.md).

## Setup

You need Git and **Python 3.12**. Version 3.13 also works; TensorFlow does not support 3.14 yet.

```powershell
git clone https://github.com/ShayanAhmed12/CattleClass.git
cd CattleClass
powershell -ExecutionPolicy Bypass -File setup.ps1 -Data     # Windows
```
```bash
bash setup.sh --data                                          # Linux / macOS / WSL2 (add --gpu for NVIDIA)
```

The setup script:
1. creates `.venv` and installs `requirements.txt`;
2. runs the tests;
3. with `-Data` / `--data`, downloads the three datasets into `data/` (about 30–60 minutes, 1 GB kept, no accounts needed).

It is safe to run again. Afterwards, activate the environment in each new terminal: `.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` elsewhere.

## Running it

| Step | Command | Produces |
|---|---|---|
| 1 | `python prepare_data.py` | datasets downloaded, indexed, de-duplicated, split and cached in `data/` |
| 2 | `python train.py` | B0, E1, E2 with 5-fold CV, then the final model in `models/` |
| 3 | `python evaluate.py` | external tests (E3, E4), robustness (E5), figures |
| 4 | `python demo.py` | Gradio app: upload a photo, get the breed |

Run the tests with `python -m pytest`. Steps 2–4 are added as the tasks in the plan are completed. Training needs a GPU to be quick: use WSL2 or Colab (see the plan).

## Moving to Colab (any time)

Nothing in the code changes. On Colab, data and models are kept automatically in **Google Drive → My Drive → cattle-breed**.

1. **On your PC:** commit and push. Colab gets the code from GitHub.
2. **Only if you have already trained models:** upload your local `models/` folder to *My Drive/cattle-breed/models*. Training continues from the finished folds.
3. **Open [colab.ipynb in Colab](https://colab.research.google.com/github/ShayanAhmed12/CattleClass/blob/main/colab.ipynb).** Choose *Runtime → Change runtime type → T4 GPU*, then run the cells from top to bottom. The first run downloads the datasets into your Drive, which takes about 10 minutes.
4. **Before closing the tab:** run the "Save the results" cell. Then bring `results/` back to your PC, either by downloading it from Drive or with `git pull`.

To move back to your PC, download *My Drive/cattle-breed/models* into `models/` and carry on locally.

## Files

```
prepare_data.py, train.py, evaluate.py, demo.py   the four steps
colab.ipynb        the same four steps on Google Colab
cattle_breed/      config (all settings), download, data, model, metrics
tests/             unit tests on small synthetic images
results/           metrics.csv, folds, figures (committed)
data/, models/     datasets and trained models (never committed)
docs/              plan, progress, decisions, data sources, report outline, defence questions,
                   approved proposal, course guidelines
```

## Data and licences

The datasets are not included and must never be committed. Sources and checksums are in [docs/data.md](docs/data.md).
- CID: Shagor et al., ICCA 2022, Apache-2.0.
- Bangladeshi cow breed dataset: Rupom et al., Mendeley Data 2025, CC BY 4.0.
- Indian Bovine Breeds (Kaggle): CC0 as declared. The images come from Google Images, so they are used for testing only.
