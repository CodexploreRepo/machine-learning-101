# XGBoost

## Introduction

- eXtreme Gradient Boosting (XGBoost) is a scalable and improved version of the gradient boosting algorithm (terminology alert) designed for efficacy, computational speed and model performance.
- Some of the advantages of using XGBoost over Gradient Boosting are:
  - **Regularization:** - XGBoost Incorporates both L1 (LASSO) and L2 (Ridge) regularization terms in the objective function, providing better control over model complexity.
  - **Parallelization:** XGBoost is optimized for parallel computing, making it more efficient and scalable. This is achieved through parallel tree construction, which is particularly beneficial for large datasets.
  - **Handling Missing Values:** XGBoost can handle missing values internally, reducing the need for explicit imputation.
  - **Tree Pruning:** XGBoost utilizes "max_depth" and "min_child_weight" parameters during tree construction to control the depth and size of trees, enabling more effective pruning.
  - **Cross-validation:** XGBoost has built-in cross-validation capabilities, simplifying the model selection process while cross-validation in gradient boosting needs to be implemented separately.

## Code Usage

- **Objective**: The two most popular classification objectives are:
  - `binary:logistic` binary classification (the target contains only two classes, i.e., cat or dog)
  - `multi:softprob` multi-class classification (more than two classes in the target, i.e., apple/orange/banana)
- **Early Stopping Round**: when given an unnecessary number of boosting rounds, XGBoost starts to overfit and memorize the dataset. This, in turn, leads to validation performance drop because the model is memorizing instead of generalizing. early_stopping_rounds helps to prevent that.
  If value of `early_stopping_rounds` is set to 10 then model will stop the training process if there is no major improvement in the evaluation parameters.
- **Evaluation Metric**: The performance measure.
  - For example, `r2` for regression models, precision for classification models. `auc` (Area under curve) because it performs well with the imbalanced data.
- **Evaluation set**: `X_val` and `y_val` both are used for the evaluation purpose.

```Python
clf_xgb = XGBClassifier(seed=42)
clf_xgb.fit(X_train,
            y_train,
            verbose=False,
            early_stopping_rounds=10,
            eval_metric='auc',
            eval_set=[(X_val, y_val)])
```
