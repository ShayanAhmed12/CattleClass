# Progress

**Updated:** 2026-11-19 · **Done:** T0, T1, T4 (CID and Mendeley verified), T6 · **Next:** T7/T8 (model training), plus CID view inspection and T3 manual review

Tasks are described in [PLAN.md](PLAN.md). Status: ✅ done · 🟡 in progress · ⬜ not started.
**Claim a task before starting it:** put your name under *Owner* and push. This table is also the record of who did what, which the proposal and report must state.

| Task | What | Tier | Owner | Status | Notes |
|---|---|---|---|---|---|
| T0 | Setup and downloads | Must | Shayan | ✅ | Approval received (D-01); one-command setup (D-11); all three datasets downloaded and checked |
| T1 | Image loading | Must | Shayan | ✅ | `data.load_image`, `letterbox`, `center_crop` |
| T2 | Index tables and CID check | Must | Shayan | 🟡 | Index code complete. Verified 503 CID animals / 2,012 photos and 190 matched Mendeley photos. View inspection pending (D-07). |
| T3 | Review external images by hand | Must |  | ⬜ | About 3 hours; rules in `data.md` |
| T4 | Duplicates and folds | Must | Shayan | ✅ | 1,893 CID photos in 371 grouped IDs. Every train, validation and test split has all five classes. |
| T5 | Image cache | Must | Shayan | 🟡 | End-to-end CID and Mendeley cache runs verified. Kaggle verification pending. |
| T6 | Metrics and B0 | Must | Shayan | ✅ | B0 on real CID folds: 0.703 accuracy, 0.200 balanced accuracy and 0.165 macro-F1. |
| T7 | Model and CV (E1, E2) | Must | Shayan | 🟡 | Two-stage training and fold resume verified on synthetic inputs. Real CID run started on the cloud CPU. |
| T8 | Final model | Must | Shayan | 🟡 | Final training uses the median epoch in each stage. Waiting for all five real fold checkpoints. |
| T9 | External test (E3, E4) | Must | Shayan | 🟡 | Both scoring methods tested on synthetic caches. Real scoring awaits T3 and the final CNN. |
| T10 | Robustness (E5) | Should | Shayan | 🟡 | Both strengths of each corruption pass deterministic synthetic tests. Waiting for real models and reviewed external caches. |
| T11 | Error analysis and Grad-CAM | Should | Shayan | 🟡 | Local galleries and annotation sheet verified on synthetic data. Real errors still need review. |
| T12 | Demo | Should | Shayan | 🟡 | Gradio upload and top-three predictions verified with a synthetic Keras checkpoint. Real demo and backup video pending. |
| T13 | Report, slides, defence | Must |  | ⬜ | |
| — | E6 (50% of animals) | Could |  | ⬜ | Cut first if short of time |

## Log

Newest first: date, who, what was done, what is next or blocked.

| Date | Who | Done | Next / blocked |
|---|---|---|---|
| 2026-11-19 | Shayan | Verified the Gradio upload and prediction flow with a synthetic checkpoint | Use the trained final model and record the backup |
| 2026-11-19 | Shayan | Tested both external scoring protocols, robustness workflow and local Grad-CAM galleries | Run official external experiments after private curation and training |
| 2026-11-19 | Shayan | Implemented deterministic blur, darkness and occlusion and checked oversized-image handling | Run E5 after the trained checkpoints and T3 |
| 2026-11-19 | Shayan | Added the all-CID fit and verified epoch selection and checkpoint identity | Finish the five real folds before the final fit |
| 2026-11-19 | Shayan | Implemented and tested EfficientNet training and fold resume. Started real CID training on CPU | Collect five-fold E1/E2 metrics |
| 2026-11-19 | Shayan | Scored the majority-class baseline on all five CID folds | Begin E1/E2 training code and verification |
| 2026-11-19 | Shayan | Cached 1,893 CID and 173 provisional Mendeley photos and checked read-back | Verify Kaggle after download and rerun external preparation after hand review |
| 2026-11-19 | Shayan | Verified CID pHash cleaning and grouped folds on the real data | T5 cache and Kaggle verification after download |
| 2026-11-19 | Shayan | Built and verified the CID and external index code; generated the local CID sample grid | Review the CID views and external thumbnails locally |
| 2026-11-07 | Shayan | Team-ready: branch per task, tests on GitHub, Colab notebook for any member and branch (D-15) | Add members as GitHub collaborators; team approved |
| 2026-11-07 | Shayan | Colab-ready: `colab.ipynb`, automatic Drive storage, README "Moving to Colab" (D-14) | Push before first Colab run |
| 2026-11-07 | Shayan | Simplified the project: 5 modules, 4 run scripts, 6 docs (D-12) | T2 |
| 2026-11-07 | Shayan | Setup scripts; all three datasets downloaded and checked (D-10, D-11) | — |
| 2026-11-07 | Shayan | Scaffold, plan, T1 image loading; approval received | — |
