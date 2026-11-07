# Oral defence questions

Practise answering each one aloud without notes. The answers are in [decisions.md](decisions.md) (D-xx), [PLAN.md](PLAN.md) or the code.

**Problem and data**
1. Why breed classification, and who would use it?
2. Why these five classes? What is "Local", and why is it a problem? (D-02, D-08)
3. Why online test sets instead of your own photos? (D-01)
4. Why split by animal and not by image? What would go wrong otherwise? (D-04)
5. How do you know no test image was seen in training? (PLAN rules 1–2, `data.find_duplicates`)
6. How did you review the Kaggle images without cherry-picking? (D-09, data.md)
7. What biases does CID have, and how do they show up in the results?

**Model and training**

8. What is transfer learning here, and what do the ImageNet weights give you?
9. Why EfficientNet-B0? (D-03)
10. Why two stages, and why is the second learning rate 100× smaller? (D-05)
11. Why keep BatchNorm frozen while fine-tuning? (PLAN rule 4)
12. Why don't you divide pixels by 255? (PLAN rule 3)
13. Which augmentations, and why no hue changes? (D-05)
14. What do class weights and label smoothing do?
15. Why letterbox instead of stretching or cropping? (D-06)

**Evaluation**

16. Why macro-F1 instead of accuracy? What does B0 show? (D-08)
17. Why 5-fold cross-validation? What does the standard deviation tell you? (D-04)
18. What is the difference between 5-way and restricted scoring? (D-08)
19. Why does performance drop more on Kaggle than on Mendeley?
20. Which breeds get confused, and why?
21. How does Grad-CAM work, and what did it show?
22. Which corruption hurt most, and why?

**Code** (be ready to open any file)

23. Follow one image from disk to prediction. (`data.py` → `model.py`)
24. How does `make_folds` keep each animal in one fold?
25. How does `train.py` resume after a disconnect?
26. Which parts did AI tools help with, and how did you check them?
27. With two more weeks, what would you change first?
