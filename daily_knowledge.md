# Machine Learning 101

## Day 2
### Pandas
- `pd.cut` to divide the data into multiple range
```Python
# If right == True (the default), then the bins [1, 2, 10**6, float('inf')] 
#                              indicate (1,2], (2,3], (10**6, float('inf')]. 
# This argument is ignored when bins is an IntervalIndex.
mapping = pd.cut(make_df['mean'], [0, 10**4, 2.5*(10**4), 3.6*(10**4), 5.5*(10**4), 10**5, 10**6, float('inf')], 
        labels=['low', ' mid-1', 'mid-2', 'high-class', 'luxury' ,'super luxury', 'exotic'], right=True)

# make_category = {
#     'exotic': (10**6, float('inf')],
#     'super luxury': (10**5, 10**6],
#     'luxury': (5.5*(10**4), 10**5],
#     'high-class': (3.6*(10**4), 5.5*(10**4)],
#     'mid-2': (2.5*(10**4), 3.6*(10**4)],
#     'mid-1': (10**4, 2.5*(10**4)],
#     'low'  : (0, 10**4]
# }
```
### Scikit-learn
- `.fit()`, `.transform()`, `.predict()` needs to provide as `[[1], [2], [1], ..]` format
  - Solution: 
    - **Method 1** using reshape: `[1,2,1].reshape(-1, 1)` or `ohe.fit_transform(df['col'].values.reshape(-1, 1))`
    - **Method 2** using `[[]]`: `model.fit(df[['col']].values)`
- `UserWarning: X does not have valid feature names, but IsolationForest was fitted with feature name`
  - Root Cause: this happens when we `.fit()`the model/encoder/sklearn object with DataFrame, but when we `.predict()` with Numpy array
  - Solution: `.fit(df.values)` so that we will provide to the sklearn object the Numpy array
### Numpy
- Convert sparse matrix (only store position of where has value != 0) to dense: `sparse_matrix.to_array()`
  - FYI: the matrix returns after one-hot encoding is a spare matrix
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

#### Regression
- Target variable: as the y has a wide range, so better to convert to log scale before training
```Python
y_train = np.log1p(df_train.price.values)
y_val   = np.log1p(df_val.price.values)
y_test  = np.log1p(df_test.price.values)
```
#### Binary Classification

- If the target is **skewed** (class 0 dominates class 1) &#8594; the best metric for this binary classification problem would be `Area Under the ROC Curve (AUC)``.
  - We can use precision and recall too, but AUC combines these two metrics.
