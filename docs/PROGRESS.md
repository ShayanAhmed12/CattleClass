# Progress

**Updated:** 2026-11-07 · **Done:** T0, T1 · **Next:** T2 (index tables and CID check)

Tasks are described in [PLAN.md](PLAN.md). Status: ✅ done · 🟡 in progress · ⬜ not started.
**Claim a task before starting it:** put your name under *Owner* and push. This table is also the record of who did what, which the proposal and report must state.

| Task | What | Tier | Owner | Status | Notes |
|---|---|---|---|---|---|
| T0 | Setup and downloads | Must | Shayan | ✅ | Approval received (D-01); one-command setup (D-11); all three datasets downloaded and checked |
| T1 | Image loading | Must | Shayan | ✅ | `data.load_image`, `letterbox`, `center_crop` |
| T2 | Index tables and CID check | Must |  | ⬜ | Drop unlabelled folder `BLF2334`; match SKUs without spaces (D-10); decide views (D-07) |
| T3 | Review external images by hand | Must |  | ⬜ | About 3 hours; rules in `data.md` |
| T4 | Duplicates and folds | Must |  | ⬜ | |
| T5 | Image cache | Must |  | ⬜ | |
| T6 | Metrics and B0 | Must |  | ⬜ | Expect B0 ≈ 0.70 accuracy, ≈ 0.165 macro-F1 |
| T7 | Model and CV (E1, E2) | Must |  | ⬜ | Decide where to train (D-13) |
| T8 | Final model | Must |  | ⬜ | |
| T9 | External test (E3, E4) | Must |  | ⬜ | |
| T10 | Robustness (E5) | Should |  | ⬜ | |
| T11 | Error analysis and Grad-CAM | Should |  | ⬜ | |
| T12 | Demo | Should |  | ⬜ | Record a backup video |
| T13 | Report, slides, defence | Must |  | ⬜ | |
| — | E6 (50% of animals) | Could |  | ⬜ | Cut first if short of time |

## Log

Newest first: date, who, what was done, what is next or blocked.

| Date | Who | Done | Next / blocked |
|---|---|---|---|
| 2026-11-07 | Shayan | Team-ready: branch per task, tests on GitHub, Colab notebook for any member and branch (D-15) | Add members as GitHub collaborators; team approved |
| 2026-11-07 | Shayan | Colab-ready: `colab.ipynb`, automatic Drive storage, README "Moving to Colab" (D-14) | Push before first Colab run |
| 2026-11-07 | Shayan | Simplified the project: 5 modules, 4 run scripts, 6 docs (D-12) | T2 |
| 2026-11-07 | Shayan | Setup scripts; all three datasets downloaded and checked (D-10, D-11) | — |
| 2026-11-07 | Shayan | Scaffold, plan, T1 image loading; approval received | — |
