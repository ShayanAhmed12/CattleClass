# CID error analysis

The [pooled CID confusion matrices](../results/cid_cv_details.json) combine held-out predictions from all five folds. Rows are true breeds and columns are predicted breeds. The [per-class table](../results/cid_per_class.csv) reports precision, recall, F1 and the number of true photographs per class. The [overall experiment table](../results/metrics.csv) reports averages across folds.

## Breed-label confusion

| Example from the private held-out gallery | Observed in the image | Model output |
| --- | --- | --- |
| Correct examples | Three Local cattle with pale or tan coats. The photographs differ in angle and coat shading. | Local, about 99% on each example |
| Sahiwal mistaken for Local | Two dark brown side views. Both use similar railings and a yellow scale board. | Local, 95% and 93% |
| Local mistaken for Holstein cross | Dark-coated cattle with the whole body visible at a slight angle. | Holstein cross, 97% |

In the full E2 confusion matrix, **179 Local photos were predicted as Sahiwal**, **64 Sindhi photos as Sahiwal**, and **47 Sahiwal photos as Sindhi**. Sindhi had the lowest E2 recall, at 30.7%. The selected gallery cases are examples of high-confidence predictions. Coat and background are visible in them, and their causal influence is unverified. The private gallery can be rebuilt with `python evaluate.py --figures` using matching fold checkpoints. [Evaluation code](../evaluate.py)

## Controlled image changes

Each corruption is applied to the same held-out CID photos. The corresponding fold model scores both the clean and altered images.

| Condition | Clean macro-F1 | Altered macro-F1 | Difference |
| --- | ---: | ---: | ---: |
| Strong blur | 0.500 | 0.189 | −0.311 |
| Strong occlusion | 0.500 | 0.239 | −0.261 |
| Strong darkening | 0.500 | 0.477 | −0.023 |

These tests show how the classifier responds to controlled changes in sharpness, visibility, and lighting. The gallery does not establish a cause for each natural error. Pose, dirt, age, and similar coat patterns remain plausible explanations where no controlled test was run.
