# Hyperparamter Tuning

- Grid/Randomize Search + Pipeline
- Bayesian Optimization with Gaussian Process
- Hyperopt
- Optuna

## Grid & Randomize Search + Pipeline

- Create a pipeline

```Python
from sklearn.svm import SVR
svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])
```

### Grid Search

```Python
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'svr__kernel': ['linear'], 'svr__C': [10., 30., 100., 300., 1000.,
                                               3000., 10000., 30000.0]
        }, # option 1
        {'svr__kernel': ['rbf'], 'svr__C': [1.0, 3.0, 10., 30., 100., 300.,
                                            1000.0],
         'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        }, # option 2
    ]
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
```

- The best model achieves the following score (evaluated using 3-fold cross validation):

```Python
svr_grid_search_rmse = -grid_search.best_score_
svr_grid_search_rmse # 69814.13889867254

# best hyper-params
grid_search.best_params_ # {'svr__C': 10000.0, 'svr__kernel': 'linear'}
```

- Note: Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

### Randomize Search

- `expon()` distribution for `gamma`, with a scale of 1, so RandomSearch mostly searched for values roughly of that scale: about 80% of the samples were between 0.1 and 2.3 (roughly 10% were smaller and 10% were larger)
- `loguniform()` distribution for `C`, meaning we did not have a clue what the optimal scale of C was before running the random search. It explored the range from 20 to 200 just as much as the range from 2,000 to 20,000 or from 20,000 to 200,000.

```Python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, loguniform

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `loguniform()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': loguniform(20, 200_000),
        'svr__gamma': expon(scale=1.0),
    }

rnd_search = RandomizedSearchCV(svr_pipeline,
                                param_distributions=param_distribs,
                                n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=42)
rnd_search.fit(X_train, y_train)

```

- We also can create the new pipeline based on the `rnd_search.best_params_`
  - In this pipeline, we also use a `SelectFromModel` transformer based on the feature importance of `RandomForestRegressor` before the final regressor:

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

### How to choose the sampling distribution for a hyperparameter

- [Notebook](../notebooks/tutorial/hyperparameter_sampling_distribution.ipynb)

* `scipy.stats.randint(a, b+1)`: for hyperparameters with _discrete_ values that range from a to b, and all values in that range seem equally likely.
* `scipy.stats.uniform(a, b)`: this is very similar, but for _continuous_ hyperparameters.
* `scipy.stats.geom(1 / scale)`: for discrete values, when you want to sample roughly in a given scale. E.g., with scale=1000 most samples will be in this ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
* `scipy.stats.expon(scale)`: this is the continuous equivalent of `geom`. Just set `scale` to the most likely value.
* `scipy.stats.loguniform(a, b)`: when you have almost no idea what the optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then you're just as likely to sample a value between 0.01 and 0.1 as a value between 10 and 100.

## Optuna

- `Optuna` uses a smart technique called Bayesian optimization to find the best hyperparameters for your model.
  - Bayesian optimization is like a treasure hunter using an advanced metal detector to find hidden gold, instead of just digging random holes (random search) or going through the entire area with a shovel (grid search).

```Python
import optuna
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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


    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    # follow the cross validation in optuna:
    # https://www.kaggle.com/code/iqbalsyahakbar/ps4e1-3rd-place-solution#CatBoost
    model = XGBClassifier(**param, seed=42)
    roc_auc = np.round(
                    np.mean(
                        cross_val_score(model, X_train, y_train,
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

#Visualize parameter importances.
optuna.visualization.plot_param_importances(study)
```

- Once the best model has been identified, we can init the model's instance with the best params

```Python
# re-fit with the best set of params
clf_xgb = XGBClassifier(**study.best_trial.params,
                        tree_method='gpu_hist',
                        seed=42)
clf_xgb.fit(X_train,
            y_train,
            verbose=False,
            early_stopping_rounds=10,
            eval_metric='auc',
            eval_set=[(X_val_pre, y_val)])
```
