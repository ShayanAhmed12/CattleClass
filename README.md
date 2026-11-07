# Cattle Breed Classification (CS4045)

Transfer-learning breed classifier (EfficientNet-B0, Keras 3) trained on the CID cow dataset and tested on CID cross-validation folds, two independent online datasets (Mendeley, Kaggle) and corrupted copies of all of them. The project asks how well a breed classifier generalises beyond its own dataset.

**Status:** scaffold and Task 1 (image loading) done. Follow [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) task by task; progress is tracked in [docs/PROGRESS.md](docs/PROGRESS.md).

## Layout

```
src/cattle_breed/   package: config, data/, models/, eval/   (all logic, unit-tested)
scripts/            one command-line script per pipeline stage
notebooks/          CID audit, error analysis, run_all_colab (end-to-end runner)
app/                Gradio demo
tests/              pytest, synthetic images only (TensorFlow tests skip without TF)
results/            experiments.csv, splits, summaries, figures (committed)
data/, artifacts/   datasets, caches, models (git-ignored; on Google Drive in Colab)
docs/               plan, progress log, proposal, decision log, curation, report, presentation
```

## Local setup (Windows, for coding and tests)

```powershell
py -3.14 -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Colab setup (for data, training, evaluation, demo)

See Appendix A of [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). Mount Drive, set `CATTLE_DATA_ROOT` and `CATTLE_ARTIFACTS_ROOT`, clone this repository, then run `pip install -e ".[demo,dev]"`.

## Run order (filled in as tasks are completed)

| Stage | Command | Output |
|---|---|---|
| Index | `python scripts/build_indices.py cid\|mendeley\|kaggle` | `data/interim/*_index.csv` |
| Curate | `python scripts/make_contact_sheets.py mendeley\|kaggle`, then edit logs | `docs/curation/*_curation.csv` |
| De-duplicate | `python scripts/dedup.py cid`, `python scripts/dedup.py external <dataset>` | `data/interim/*_final.csv` |
| Folds | `python scripts/make_folds.py` | `results/splits/cid_folds.csv` |
| Cache | `python scripts/build_cache.py <dataset>` | `data/processed/*_240.npz` |
| Baseline | `python scripts/evaluate.py b0` | `results/experiments.csv` |
| CV (E1, E2) | `python scripts/run_cv.py --out artifacts/cv/full` | `artifacts/cv/full/` |
| Evaluate CV | `python scripts/evaluate.py cv --run artifacts/cv/full --experiments E1 E2` | `results/` |
| Final model | `python scripts/train_final.py --run artifacts/cv/full` | `artifacts/models/final_e2.keras` |
| External (E3, E4) | `python scripts/evaluate.py external --dataset mendeley --experiment E3` (and `kaggle`, `E4`) | `results/` |
| Robustness (E5) | `python scripts/evaluate.py robustness` | `results/robustness.csv` |
| Demo | `python app/demo.py --model artifacts/models/final_e2.keras --share` | Gradio link |

## Data

The datasets are not included and must never be committed. See [docs/provenance.md](docs/provenance.md) for sources and licences:
- CID (Apache-2.0);
- Bangladeshi cow breed dataset, Mendeley (CC BY 4.0);
- Indian Bovine Breeds, Kaggle (CC0 as declared; images from Google Images, so test use only).

## Progress

See [docs/PROGRESS.md](docs/PROGRESS.md).
