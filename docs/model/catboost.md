# Cat Boost

## Introduction

- CatBoost is a machine learning algorithm developed by Yandex, designed for categorical feature support and gradient boosting on decision trees. It is particularly effective in handling categorical variables without the need for extensive preprocessing.
- Key Characteristics:
  - Categorical Feature Support: CatBoost efficiently handles categorical features without the need for manual encoding.
  - Gradient Boosting: Utilizes the gradient boosting framework for ensemble learning.
  - Robustness to Overfitting: Implements techniques to reduce overfitting, making it robust in various scenarios.

## Code Usage

- Catboost can handle the categorical data without the need of encoding

```Python
from catboost import CatBoostClassifier
# Initialize data
cat_features = [0, 1]
train_data = [["a", "b", 1, 4, 5, 6],
              ["a", "b", 4, 5, 6, 7],
              ["c", "d", 30, 40, 50, 60]]
train_labels = [1, 1, -1]
eval_data = [["a", "b", 2, 4, 6, 8],
             ["a", "d", 1, 4, 50, 60]]

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2,
                           learning_rate=1,
                           depth=2)
# Fit model
model.fit(train_data, train_labels, cat_features)
# Get predicted classes
preds_class = model.predict(eval_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_data, prediction_type='RawFormulaVal')


```

## Hyper-param Space

- `iterations` represents the steps (or rounds of refinement) the algorithm takes to create a more accurate model that learns from the data.
  - For real-time inference: 100-200 iterations for faster prediction
  - For batch inference: 1000-2000 iterations
- `learning_rate`
  - Range: 0.001 to 0.1 is a good starting point.
  - A smaller learning rate signifies that each tree offers a smaller “voice,” or a smaller update to the model &#8594; This can lead to higher accuracy but increases the risk of underfitting and longer training times.
  - A larger learning rate, on the other hand, means each tree has a more significant impact on the model, speeding up the learning process &#8594; high learning rate can result in overfitting or model instability.
- `depth` the height of decision trees in your CatBoost model
  - Range: 1 and 10
  - A higher depth can capture more intricate patterns in your data, leading to better performance.
- `subsample` to randomly choose a fraction of the dataset when constructing each tree.
  - Range: 0.05 to 1
  - Lower values increase diversity but may result in underfitting.
- `colsample_bylevel` fraction of features to choose when determining the best split for each node at a specific level during the tree building process.
  - Range: 0.05 and 1
- `min_data_in_leaf` specifies the minimum number of samples required to create a leaf, effectively controlling the split creation process.
  - Range: 1 and 100
  - Higher values generate less complex trees, reducing overfitting risks, but might result in underfitting.
  - Lower values lead to more complex trees that might overfit.

```Python
def objective(trial):
    param = {
        #"task_type": "GPU",
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        #"used_ram_limit": "3gb",
        "eval_metric": "AUC", # TODO: need to update
    }

#     if param["bootstrap_type"] == "Bayesian":
#         param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
#     elif param["bootstrap_type"] == "Bernoulli":
#         param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


    kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=2024)
    # follow the cross validation in optuna:
    # https://www.kaggle.com/code/iqbalsyahakbar/ps4e1-3rd-place-solution#CatBoost
    model = CatBoostClassifier(**param, verbose=False)
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

## Tips

- Tip 1: Tuning catboost model will take a lot of time (6-8 hours), so usually the default parameters in the catboost model will work better than the catboost with the tunned params with limited tuning times (1 hour)
  - Hence, sometime, we only tune catboost model only with 3 different bootstrap type each: no bootstrap, Bayesian, and Bernoulli.
  - Example: the default CatBoost model (with `iteration=1000`) works better on this dataset (`AUC=0.88434` on private score) in compare with the fine-tuned hype-parameters using Optuna (`AUC=0.88323` on private score)
- Tip 2: Use the fixed `iterations` like 100-200 for real-time and 1000-2000 for batch and turning the learning rate.
- Tip 3: For the imbalance dataset, we can use - `class_weight` to ask model to pay attention on the minority class
  - For example: `{"0": 1, "1": 5}` since the `class=1` is the minority class, so we can assign a stronger penalty (in this case is 5) to the model when it fails to detect a true positive (Exiter).

```Python
cat_model = CatBoostClassifier(
                eval_metric='AUC',
                class_weights={"0": 1, "1": 5},
                iterations=1000
)

cat_model.fit(X_train_pre,
              y_train,
              eval_set=(X_val_pre,y_val),
              verbose=False
             )
```
