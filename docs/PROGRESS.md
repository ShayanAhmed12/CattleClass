# Progress log

Update this at the end of every working day: set each task's status, add the completion date, and note anything left over. Task details are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

**Status key:** ✅ Done · 🟡 In progress · ⬜ Not started · ⏭️ Skipped (with reason)

**Last updated:** 2026-11-07 · **Tasks done:** 1 of 23 (plus the scaffold) · **Next up:** T0.1 approval email, then finish T0.2 and T0.3

## Tasks

| Task | Title | Phase | Plan day | Tier | Status | Completed | Notes |
|---|---|---|---|---|---|---|---|
| — | Project scaffold (folders, `config.py`, packaging, fixtures, docs) | Setup | 0 | Must | ✅ | 2026-11-07 | `tests/test_config.py` 6/6 passing |
| T0.1 | Approval request (solo work, CID, online-only testing) | 0 Admin | 1 | Must | ⬜ | | Draft ready in `docs/admin/approval_request_email.md`; fill in name, ID and email first |
| T0.2 | Local and Colab environment | 0 Admin | 1 | Must | 🟡 | | Done: `.venv`, local pytest, git repo and first push. Remaining: `notebooks/run_all_colab.ipynb`, Kaggle secrets in Colab, record Colab TF/Keras versions (D-012) |
| T0.3 | Download datasets to Drive | 0 Admin | 1 | Must | ⬜ | | CID, Mendeley, Kaggle (3 class folders); fill in `docs/provenance.md` |
| T1 | Image loading and letterboxing | 1 Data | 2 | Must | ✅ | 2026-11-07 | `data/images.py`; 7 tests in `tests/data/test_images.py` |
| T2 | CID index and data audit | 1 Data | 2 | Must | ⬜ | | Needs T0.3 for the real-data steps; views decision D-013 |
| T3 | External indices and curation tooling | 1 Data | 6–9 | Must | ⬜ | | |
| T4 | Manual curation (no code) | 1 Data | 6–9 | Must | ⬜ | | About 3.5 hours; follow `docs/curation/README.md` |
| T5 | Near-duplicates and group merging | 1 Data | 3 and 9 | Must | ⬜ | | Step 5a (CID) on Day 3; step 5b (external) after T4 |
| T6 | Cross-validation folds | 1 Data | 3 | Must | ⬜ | | |
| T7 | Image cache | 1 Data | 4 | Must | ⬜ | | |
| T8 | Metrics, results registry, B0 | 1 Data | 4 | Must | ⬜ | | Expect B0 ≈ 0.70 accuracy, ≈ 0.165 macro-F1 |
| T9 | Data pipeline and augmentation | 2 Modelling | 5 | Must | ⬜ | | |
| T10 | Model builder | 2 Modelling | 5 | Must | ⬜ | | |
| T11 | Training loop and CV runner (E1, E2) | 2 Modelling | 6–13 | Must | ⬜ | | |
| T12 | Final model | 2 Modelling | 8–13 | Must | ⬜ | | |
| T13 | External evaluation (E3, E4) | 3 Evaluation | 14–16 | Must | ⬜ | | |
| T14 | Engineered set and robustness (E5) | 3 Evaluation | 20–22 | Should | ⬜ | | |
| T15 | Error analysis and Grad-CAM | 3 Evaluation | 20–22 | Should | ⬜ | | |
| T16 | Data-size sensitivity (E6) | 3 Evaluation | 20–22 | Could | ⬜ | | Cut first if time runs short |
| T17 | Gradio demo | 4 Delivery | 23 | Should | ⬜ | | Record a backup video |
| T18 | Reproducibility run and README | 4 Delivery | 24–28 | Must | ⬜ | | |
| T19 | Report (8–10 pages) | 4 Delivery | 24–28 | Must | ⬜ | | Outline in `docs/report/outline.md` |
| T20 | Presentation and oral defence | 4 Delivery | 29–30 | Must | ⬜ | | Questions in `docs/presentation/defense_questions.md` |

## Gates

| Gate | Day | Condition | Status |
|---|---|---|---|
| G0 | 1 | Approval email sent; all three datasets on Drive; local and Colab `pytest` green | ⬜ |
| G1 | 4 | `cid_240.npz` built; B0 row in `results/experiments.csv` | ⬜ |
| G2 | 7 | CV run started; fold 0 finished | ⬜ |
| G3 | 9 | Both external sets curated and de-duplicated | ⬜ |
| G4 | 13 | E2 macro-F1 > E1 > B0; `final_e2.keras` saved | ⬜ |
| G5 | 16 | E3 and E4 logged (Must tier complete) | ⬜ |
| G6 | 22 | E5 and error analysis done | ⬜ |
| G7 | 23 | Demo works on Colab; backup recording saved | ⬜ |
| G8 | 28 | Report PDF complete; fresh-runtime rerun matches | ⬜ |

## Daily entries

Add one short line per working day, newest first: date, hours, what was done, what is blocked.

| Date | Hours | Done | Blocked or next |
|---|---|---|---|
| 2026-11-07 | — | Scaffold, plan, Task 1 (image loading) | Send approval email; set up Colab; download data |
