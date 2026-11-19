# Progress

**Updated:** 2026-11-19 · **Done:** T0, T1, T4–T12 · **Open:** CID viewpoint verification, original hand review and individual defence. See the table for E6.

Tasks are described in [PLAN.md](PLAN.md). Status: ✅ done · 🟡 in progress · ⬜ not started.
**Claim a task before starting it:** put your name under *Owner* and push. This table is also the record of who did what, which the proposal and report must state.

| Task | What | Tier | Owner | Status | Notes |
|---|---|---|---|---|---|
| T0 | Setup and downloads | Must | Shayan | ✅ | Approval received (D-01); one-command setup (D-11); all three datasets downloaded and checked |
| T1 | Image loading | Must | Shayan | ✅ | `data.load_image`, `letterbox`, `center_crop` |
| T2 | Index tables and CID check | Must | Shayan | 🟡 | Index code complete. Verified 503 CID animals / 2,012 photos and 190 matched Mendeley photos. An 80-photo local automated viewpoint audit was inconclusive. All four views retained (D-07). |
| T3 | External image screening | Must | Shayan | 🟡 | Local automated screen examined all 1,119 matched photos. No exclusions met its threshold. The hand-reviewed protocol remains unverified. |
| T4 | Duplicates and folds | Must | Shayan | ✅ | 1,893 CID photos in 371 grouped IDs. Every train, validation and test split has all five classes. |
| T5 | Image cache | Must | Shayan | ✅ | Verified caches for 1,893 CID, 173 provisional Mendeley, and 908 provisional Kaggle photos. |
| T6 | Metrics and B0 | Must | Shayan | ✅ | B0 on real CID folds: 0.703 accuracy, 0.200 balanced accuracy and 0.165 macro-F1. |
| T7 | Model and CV (E1, E2) | Must | Shayan | ✅ | Five real CID folds. E1 macro-F1 0.453, E2 0.500. |
| T8 | Final model | Must | Shayan | ✅ | Verified all-CID checkpoint stored in the private cloud workspace. |
| T9 | External test (E3, E4) | Must | Shayan | ✅ | Both scoring modes recorded for the automatically screened caches. |
| T10 | Robustness (E5) | Should | Shayan | ✅ | Measured both strengths of all three corruptions in each source. |
| T11 | Error analysis and Grad-CAM | Should | Shayan | ✅ | Quantified per-class errors and tested crop, brightness and edge sensitivity on local examples. |
| T12 | Demo | Should | Shayan | ✅ | Verified UI upload and saved the real-model backup recording privately. |
| T13 | Report, slides, defence | Must | Shayan | 🟡 | Real CID, external and robustness results are recorded in the eight-page report and ten-minute presentation. A clean checkout reproduced the CID folds and B0 byte for byte. Individual defence remains open. |
| — | E6 (50% of animals) | Could |  | ⬜ | Cut first if short of time |

## Log

Newest first: date, who, what was done, what is next or blocked.

| Date | Who | Done | Next / blocked |
|---|---|---|---|
| 2026-11-19 | Shayan | Published real CID, external and robustness results and updated the report and slides. Verified the private demo recording. | Monitor optional E6 and retain the recorded protocol limitations |
| 2026-11-19 | Shayan | Verified and recorded real CID E1 and E2 | Check the automatically screened external results and their limitations |
| 2026-11-19 | Shayan | Verified the first real EfficientNet checkpoint and held-out predictions | Continue the remaining CID folds before reporting aggregate metrics |
| 2026-11-19 | Shayan | Finished local automated external screening and audited CID viewpoints. Added a mock defence guide, aligned Colab instructions and verified the private recording workflow | Finish real training, then score the automatically screened external caches |
| 2026-11-19 | Shayan | Verified the Kaggle v5 byte-range reader against the real archive and reproduced the CID fold manifest and B0 in a clean checkout | GitHub checks for the draft and real CNN training |
| 2026-11-19 | Shayan | Finished Kaggle cache verification and drafted the report and slides | Add real model results, reviewed figures and rehearse the defence |
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
