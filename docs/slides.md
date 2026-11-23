# CattleClass presentation

The nine-slide deck is available as an [editable PowerPoint with speaker notes](submission/CattleClass-Presentation.pptx) and a [presentation PDF](submission/CattleClass-Presentation.pdf).

| Slide | Time | What to say |
| --- | ---: | --- |
| CattleClass | 0:30 | Introduce the team and the question. Can one photo suggest a cattle breed? |
| What one photo can tell us | 0:55 | Explain the five labels and how a handler can check the suggestion. |
| How we prepared the data | 1:10 | Explain why photos of the same animal and near-duplicates stay in the same test split. |
| How we trained the model | 1:05 | Describe the pretrained model, the new breed layer, and fine-tuning in Keras. |
| Fine-tuning improved breed predictions | 1:20 | Explain macro-F1 as an average that gives every breed equal weight. Read the chart from the majority guess to the fine-tuned model. |
| Where the classifier struggled | 1:00 | Show the effect of blur and the Sindhi/Sahiwal confusion. Say that no independent farm test was run. |
| Age and weight are experimental | 0:55 | Explain the three joint targets. The median-age baseline had less error than the age model. |
| Live demo | 1:50 | Upload a clear side photo, select **Estimate traits**, and point out the results. Keep a private backup recording available. |
| What we learned | 0:55 | Restate the scope of the evidence and leave time for questions. |

The suggested timing is 9 minutes 40 seconds. Open the app before presenting. Follow the [pretrained setup](../README.md#run-the-research-demo) and launch it with `python demo.py --research`.

## Sources

- [CID data and folds](data.md) · [Breed metrics](../results/metrics.csv) · [Error analysis](error-analysis.md) · [Joint results](../results/cid_joint.json)
- The cover uses an illustrative public-domain photo by Keith Weller / USDA. The demo slide shows the actual Gradio interface before a photo is uploaded. [Cover photo source](https://commons.wikimedia.org/wiki/File:Cow_female_black_white.jpg)
