# Machine Learning 101

## Day 1
### Dimensionality Reduction Techniques
- PCA
- t-SNE
- MCA
### Data Pre-processing
#### Scaler
- Min Max Scaler
- PowerTransformer

- Note: NO need to scale those are One-Hot encoded
#### Categorical Feature
##### Rare Category

- A **rare category** is a category which is not seen very often, or a new category that is not present in train
- Define our criteria for calling a value **RARE**. Let’s say the requirement for a value being rare in this column is a count of less than 2000
- Wherever the value count for a certain category is less than 2000, replace it with rare. So, now, when it comes to test data, all the new, unseen categories will be mapped to **RARE**, and all missing values will be mapped to **NONE**.

### Model Training

#### Binary Classification

- If the target is **skewed** (class 0 dominates class 1) &#8594; the best metric for this binary classification problem would be `Area Under the ROC Curve (AUC)``.
  - We can use precision and recall too, but AUC combines these two metrics.
