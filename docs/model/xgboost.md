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

## Hyper-parameter Space

- `n_estimators` number of trees
  - The more trees you have, the more reliable your predictions will be.
  - How many trees should you pick?
    - Quick result: limit the number of trees to around 200.
    - Model onlr runs once a week: up to 5,000 trees.
- `learning_rate` regulates how much each tree contributes to the final prediction. The more trees you have, the smaller the learning rate should be.
  - Range: between 0.001 and 0.1.
- `max_depth` decides the complexity of each tree in your model & refers to the maximum depth that a tree can grow to.
  - Range: 1 to 10
- `subsample` controls the amount of data used for building each tree in your model.
  - Range: fraction that ranges from 0 to 1 (recommended: 0.05 and 1)
    - representing the proportion of the dataset to be randomly selected for training each tree.
  - By using only a portion of the data for each tree, the model can benefit from diversity and reduce the correlation between the trees, which may help combat overfitting.
- `colsample_bytree` proportion of features to be considered for each tree.
  - Range: from 0 to 1, where
    - a value of 1 means that all features will be considered for every tree
    - a lower value indicates that only a subset of features will be randomly chosen before building each tree.
- `min_child_weight` sets the minimum sum of instance weights that must be present in a child node in each tree.
  - Range: 1 to 20

```Python
def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        #"tree_method": "exact",
        'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        # defines booster, gblinear for linear functions.
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight within a logarithmic scale (log=True)
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Fit the model
    """
    clf_xgb = XGBClassifier(**param, seed=42)
    clf_xgb.fit(X_train_pre,
                y_train,
                verbose=False,
                early_stopping_rounds=10,
                eval_metric='auc',
                eval_set=[(X_val_pre, y_val)])

    # Make predictions
    y_pred_proba = clf_xgb.predict_proba(X_val_pre)[:, 1]
    # Evaluate predictions
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    # follow the cross validation in optuna:
    # https://www.kaggle.com/code/iqbalsyahakbar/ps4e1-3rd-place-solution#CatBoost
    model = XGBClassifier(**param, seed=42)
    roc_auc = np.round(
                    np.mean(
                        cross_val_score(model, X_train_pre, y_train,
                        scoring="roc_auc", cv=kfold)
                    ), 3
                )

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
