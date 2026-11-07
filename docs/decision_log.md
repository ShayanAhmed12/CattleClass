# Decision log

One entry per decision that changes data, model, evaluation or scope. The oral defence will ask "why did you do X?". The answer should be here.

Format: **ID · date · decision.** Then the why, the alternatives considered, and what it affects.

---

**D-001 · 2026-11-07 · Individual project.**
One student does all the work, so the scope is split into Must / Should / Could tiers (see `PROJECT_PLAN.md`). Approval to work alone is requested with the proposal.

**D-002 · 2026-11-07 · Online datasets only: CID for training; Mendeley and Kaggle for external testing.**
- Why: no farm visit. Two independent test sets give a two-step domain shift: same country with a new source, then a new country with web images.
- Alternatives: a farm visit, kept as the `_Revised.docx` proposal; a single external set, which gives a weaker novelty claim.
- Affects T3–T5 and T13. The guidelines ask for a hand-collected test set, so this substitution needs instructor approval (D-011).

**D-003 · 2026-11-07 · Five classes and label mappings.**
- Classes: Holstein cross, Local, Red Chittagong, Sahiwal, Sindhi (`config.CLASSES`, fixed order).
- CID's `HOSTINE_CROSS` is a misspelling of Holstein cross.
- Mapped as the same class: Mendeley/Kaggle Holstein Friesian → Holstein cross; Kaggle Red Sindhi → Sindhi; Mendeley Local Deshi → Local.
- Dropped: PABNA_BREED (6 animals), BRAHMA (2), MIR_KADIM (2), which are too few to train and test on; and Jersey (not a CID class).
- Assumptions stated in the report: CID "Sindhi" is Red Sindhi, and purebred Holstein Friesian is close enough to "Holstein cross".

**D-004 · 2026-11-07 · EfficientNet-B0 (ImageNet) in Keras 3; MobileNetV2 only as a fallback.**
- Why: strong accuracy per parameter at 224 px, its own input rescaling, and it fits a free Colab T4 comfortably.
- The fallback is used only if B0 fails to train or does not fit in memory.

**D-005 · 2026-11-07 · Stratified group 5-fold cross-validation by animal (SKU).**
- Why: with 21 Red Chittagong animals, a single 70/15/15 split leaves about 3 test animals, so per-class scores would be noise.
- Grouping by SKU stops photos of one animal from leaking between train and test.
- Results are mean ± std plus a pooled out-of-fold confusion matrix.

**D-006 · 2026-11-07 · No hue, saturation or colour-jitter augmentation.**
- Why: coat colour separates breeds (red Sindhi and Red Chittagong vs black-and-white Holstein crosses). Hue shifts would teach the model to ignore a true cue.
- Kept: horizontal flip, ±7° rotation, brightness 0.1, contrast 0.1, random 224 crop.

**D-007 · 2026-11-07 · Fixed two-stage recipe with no hyperparameter search.**
- Stage 1: head only, lr 1e-3, ≤ 10 epochs. Stage 2: full network, lr 1e-5, ≤ 20 epochs.
- Both use early stopping on `val_loss`, BatchNorm frozen, label smoothing 0.1 and balanced class weights.
- Why: solo time budget, and a search on the CV folds would bias the CV estimate.
- E1 is read after stage 1 and E2 after stage 2 of the same run.

**D-008 · 2026-11-07 · Letterbox to 240×240 (aspect kept, black padding), cache as uint8, 224 crop.**
- Why: side-view cattle photos are wide. Centre-cropping to a square would cut off heads or tails, and stretching distorts body shape.
- Caching once makes every Colab run fast.

**D-009 · 2026-11-07 · Headline metrics are macro-F1 and balanced accuracy; external sets are scored two ways.**
- Why: Local is 70% of CID, so accuracy alone rewards always guessing Local (B0 ≈ 0.70 accuracy, ≈ 0.165 macro-F1).
- External macro-F1 averages over the classes present in that set.
- "Restricted" scoring (argmax over the classes in the set) shows how often the model picks an absent class.

**D-010 · 2026-11-07 · Curation rules: drop only images that are invalid or unverifiable, never images that are merely hard.**
- Why: dropping difficult photos (poor light, odd pose, calves) would hide the very domain shift being measured and inflate E3 and E4.
- Every drop is logged with a reason code.

<!-- Next entries come from the plan: D-011 approval outcome, D-012 Colab versions, D-013 CID views,
     D-014 external folder names, D-015 CID duplicate-SKU merges, then anything else you change. -->
