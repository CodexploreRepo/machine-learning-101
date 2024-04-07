# Classification

## Evaluation Metrics

### Confusion Matrix

- Example: confusion matrix for a binary classifier to predict if the patient has a disease or not.

<img src="../../assets/img/confusion_matrix_simple2.png">

- **True Positives (TP):** These are cases in which we predicted yes (they have the disease), and they do have the disease, which are 100.
- **True Negatives (TN):** We predicted no, and they don't have the disease, which are 50.
- **False Positives (FP):** We predicted yes, but they don't actually have the disease. (Also known as a "Type I error."), which are 10.
- **False Negatives (FN):** We predicted no, but they actually do have the disease. (Also known as a "Type II error."), which are 5.

### General Metrics

- **Accuracy:** Overall, how often is the classifier correct?
  - $Accuracy=\frac{TP+TN}{\text{All records}}=\frac{TP+TN}{TP+TN+FP+FN}$ = (100+50)/(50+100+10+5) = 0.91
- **Precision**: How accurate the classifier predicts the positive class
  - $Precision=\frac{TP}{TP+FP}$
- **Recall (True Positive Rate or Sensitivity):** When it's actually yes, how often does it predict yes?
  - $Recall=\frac{TP}{\text{Actual Yes}}=\frac{TP}{TP+FN}$ = 100/105 = 0.95 also known as "Sensitivity" or "Recall"
- **False Positive Rate (FPR):** When it's actually no, how often does it predict yes?
  - $FPR=\frac{FP}{\text{Actual No}}=\frac{FP}{FP+TN}$ = 10/60 = 0.17
- $F_1$ score: harmonic mean of precision and recall
  - $F_1 = 2*\frac{Precision*Recall}{Precision + Recall}$

### ROC Curve & ROC AUC

- Reference: [A Gentle Introduction to Threshold-Moving for Imbalanced Classification](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)
- **ROC curve**: to understand the trade-off in the true-positive rate (TPR) and false-positive rate (FPR) for different thresholds.
- **ROC AUC** (Area Under the ROC Curve): provides a single number to summarize the performance of a model in terms of its ROC Curve with a value between
  - 0.5 (no-skill)
  - 1.0 (perfect skill).

#### Optimal Threshold for ROC Curve

- There are many ways we could locate the threshold with the optimal balance between false positive and true positive rates.
- **Geometric Mean** or G-Mean is a metric for imbalanced classification that, if optimized, will seek a balance between the sensitivity and the specificity.
  - $G_{mean} = sqrt(Sensitivity * Specificity)$
  - Where:
    - $Sensitivity = TPR$
    - $Specificity = 1 - FPR$

### Mathew Correlation coefficient (MCC)

The Matthews correlation coefficient (MCC), instead, is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset.

- Worst value: `-1`
- Best Value: `+1`

$$MCC=\frac{TP*TN - FP*FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
