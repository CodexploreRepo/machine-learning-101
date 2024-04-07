# Light Gradient-Boosting Machine (LightGBM)

## Introduction

- Light GBM is a high-performance gradient boosting framework for efficient tree-based machine learning, employing a leaf-wise growth strategy and histogram-based learning, ideal for large datasets and tasks where speed is paramount.
- Key Characteristics:
  - Gradient Boosting: It is an ensemble learning method that builds a series of weak learners (usually decision trees) to create a strong learner.
  - Leaf-Wise Growth: Light GBM grows trees leaf-wise instead of level-wise, leading to faster training times.
  - Histogram-Based Learning: Utilizes histograms to find the best splits during tree growth, reducing memory usage and improving computational efficiency.

## Code Usage

```Python
lgb_model = LGBMClassifier()
lgb_model.fit(X_train_pre,
            y_train,
            eval_metric='auc',
            eval_set=[(X_val_pre, y_val)])
```

## Hyper-parameter Space

- `num_iterations` number of trees
  - The more trees you have, the more stable your predictions will be
  - How many trees should you choose:
    - If your model needs to deliver results with low latency, you might want to limit the number of trees to around 200.
    - If your model runs once a week (e.g.: sales forecasting) and has more time to make the predictions, you could consider using up to 5,000 trees
- `learning_rate`
  - Rule of thumb: start by fixing the number of trees and then focus on tuning the learning_rate
  - The more trees you have, the smaller the learning rate should be.
  - Range: 0.001 and 0.1 (`trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)`)
- `num_leaves` maximum number of terminal nodes (leaves) that can be present in each tree.
  - num_leaves is equivalent to `max_depth` parameter in other tree-based models
  - In a decision tree, a leaf represents a decision or an outcome.
  - Range: powers of 2, starting from 2 and going up to 1024
  - Pros: By increasing the num_leaves, you allow the tree to grow more complex, creating a higher number of distinct decision paths.
  - Cons: increasing the number of leaves may also cause the model to **overfit** the training data, as it will have a lower amount of data per leaf.
- `subsample` control the amount of data used for building each tree in your model.
  - Range: a fraction that ranges from 0 to 1, representing the proportion of the dataset to be randomly selected for training each tree (Recommend: 0.05 and 1)
  - By using only a subset of the data for each tree, the model can benefit from the diversity and reduce the correlation between the trees, which may help combat overfitting.
- `bagging_freq` is the frequency at which the data is sampled.
  - Rule of thumb: to set bagging_freq to a **positive** value or LightGBM will ignore `subsample`.
- `colsample_bytree` determines the proportion of features to be used for each tree.
  - Range: from 0 to 1, where a value of 1 means that all features will be considered for every tree
- `min_data_in_leaf` sets the minimum number of data points that must be present in a leaf node in each tree.
  - This parameter helps control the complexity of the model and prevents overfitting.
  - Range: 1 to 100
  - If you have a leaf node with only 1 data point, your prediction will be the value of that single data point.
  - If you have a leaf node with 30 data points, your prediction will be the average of those 30 data points.

```Python
def objective(trial):
    param = {
        "objective": "regression",
        "metric": "auc",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }


    # Fit the model
    model = LGBMClassifier(**param)
    model.fit(X_train_pre,
                y_train,
                eval_set=[(X_val_pre, y_val)])

    # Make predictions
    y_pred_proba = model.predict_proba(X_val_pre)[:, 1]
    # Evaluate predictions
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)

print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

```
