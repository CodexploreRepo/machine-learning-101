# Machine Learning 101

## Day 3

### Sklearn's Pipeline

- `SelectFromModel` transformer based on the feature importance of `RandomForestRegressor` before the final regressor:

```Python
from sklearn.feature_selection import SelectFromModel

selector_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),  # min feature importance score
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])

```

### Hyperparameter Tuning

#### Grid Search

- Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

```Python
# params: 'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000.0]
# best hyper-params
grid_search.best_params_ # {'svr__C': 10000.0, 'svr__kernel': 'linear'}
```

### Vocab

- Stochastics: randomness
  - For example: k-means is a stochastic algorithm, meaning that it relies on randomness to locate the clusters

### EDA

- The basic metric for location is the `mean`, but it can be **sensitive to extreme values (outlier)**.
- Other metrics (`median`, `trimmed mean`) are **less sensitive to outliers and unusual distributions** and hence are more robust.
- The **median** is the same thing as the **50th percentile**.

#### Plots

- **Violin plot** is an enhancement to the boxplot, show nuances (sac thai) in the distribution
- **Boxplot** is to show the outliers in the data

#### Correlation

- Like the mean and standard deviation, the **correlation coefficient** is **sensitive** to **outliers** in the data.
- Pearson’s correlation coefficient always lies between

  - $+1$ perfect positive correlation
  - $–1$ perfect negative correlation
  - $0$ indicates no correlation.

- NOTE: Variables can have an association that is **not linear**, in which case the correlation coefficient may not be a useful metric.
  - For example: . The relationship between tax rates and revenue raised is an example: as tax rates increase from zero, the revenue raised also increases. However, once tax rates reach a high level and approach 100%, tax avoidance increases and tax revenue actually declines.

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

- There are two common ways to get all attributes to have the same scale:

  - Min-max Scaling (a.k.a Normalization): &#8594; Range (0 & 1)

    - This is performed by subtracting the min value and dividing by the difference between the min and the max.
    - It has a feature_range hyperparameter that lets you change the range.

    ```Python
    from sklearn.preprocessing import MinMaxScaler

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    ```

  - Standardization: it subtracts the mean value (so standardized values have a zero mean), then it divides the result by the standard deviation (so standardized values have a standard deviation equal to 1)
    - Unlike min-max scaling, standardization does not restrict values to a specific range. However, standardization is much less affected by outliers.

- Note 1: NO need to scale those are One-Hot encoded
- Note 2: While the training set values will always be scaled to the specified range, if new data contains outliers, these may end up scaled outside the range.
  - If you want to avoid this, just set the `clip` hyperparameter to `True`
- Note 3: Neural networks work best with zero-mean inputs, so the scaled range of –1 to 1 is preferable

#### Categorical Feature

##### Encoding

- If a categorical attribute has a large number of possible categories (e.g., country code, profession, species), then one-hot encoding will result in a large number of input features.
- Solution:
  - For example, a country code could be replaced with the country’s population and GDP per capita).
  - Alternatively, you can use one of the encoders provided by the category_encoders package on GitHub.
  - When dealing with neural networks, you can replace each category with a learnable, low-dimensional vector called an embedding. This is an example of representation learning (see Chapters [13](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch13.html#data_chapter) and [17](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch17.html#autoencoders_chapter) for more details).

##### Rare Category

- A **rare category** is a category which is not seen very often, or a new category that is not present in train
- Define our criteria for calling a value **RARE**. Let’s say the requirement for a value being rare in this column is a count of less than 2000
- Wherever the value count for a certain category is less than 2000, replace it with rare. So, now, when it comes to test data, all the new, unseen categories will be mapped to **RARE**, and all missing values will be mapped to **NONE**.

### Model Training

- Split Train, Val, Test with 60-20-20

```Python
# Train, Test, Val Splits
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, shuffle=True, test_size=0.25) # 0.25 x 0.8 = 0.2
```

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
