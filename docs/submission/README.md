# CS4045 submission files

The deadline supplied for this submission is Sunday, 29 November 2026. Attach these files to the assignment in Google Classroom.

1. [Final project report](CattleClass-Project-Report.pdf)
2. [Original project proposal](CattleClass-Project-Proposal.pdf)
3. [GitHub link as plain text](CattleClass-GitHub-Link.txt)

Use the [presentation PDF](CattleClass-Presentation.pdf) or [editable PowerPoint](CattleClass-Presentation.pptx) for the scheduled demonstration. The PowerPoint includes speaker notes. [Presenter guide](../slides.md) The report and original proposal use the group details in the proposal. The report explains the decision to omit the planned farm test and states the limits of the CID evidence.

## Repository contents

| Requirement | Location |
| --- | --- |
| Dataset and training reproducibility | [CID provenance and download instructions](../data.md), [locked data manifest](../../results/cid_folds.csv), and [download and preparation script](../../prepare_data.py). The script retrieves the original image archive and metadata from the authors. |
| Notebook and app | [Colab notebook](../../cattle-class.ipynb), [Gradio app](../../demo.py), and [published models](https://github.com/ShayanAhmed12/CattleClass/releases/tag/cid-models-v1). |
| Papers | [Research references](references/README.md), including the publisher PDF for EfficientNet and the official CID paper link. |
| Reproducibility and results | [Project reference](../../PROJECT.md), [per-class metrics](../../results/cid_per_class.csv), and [error analysis](../error-analysis.md). |
