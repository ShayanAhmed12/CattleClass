# Age and weight roadmap

The CID research demo returns a breed score, age in months, and estimated weight from one photo. [Run the prototype](traits.md#three-output-research-prototype) · [Joint benchmark](../results/cid_joint.json)

<a id="5-field-data-plan"></a>

## Field data plan

1. **Pilot paired photos.** Start with 30–50 animals across at least two sites if feasible. Photograph each animal in the same session with an ordinary photo, a side view, and a side view with a scale marker. Keep failed attempts in the log.
2. **Record independent references.** Link each photo to a stable animal ID, documented breed and birth date, capture time, scale weight, weighing time, and scale ID.
3. **Select a capture method.** Compare the paired modes and choose the simplest one that meets the intended accuracy. Use the pilot to size the final evaluation.
4. **Lock the field test.** Use independent farms and animals. Freeze models, the capture protocol, and pass criteria before testing. Measure error, coverage, and how often each estimate is withheld. [Field manifest and commands](traits.md#collect-and-lock-a-field-study)

Separate specialist sources can support training. The final cohort needs all three references for each scored animal. [Public dataset audit](traits.md#additional-public-dataset-screening) · [Original implementation plan](https://github.com/ShayanAhmed12/CattleClass/blob/3090574/docs/age-and-weight-plan.md)
