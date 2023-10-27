from functools import partial

import numpy as np
import pandas as pd
from sklearn import ensemble, metrics, model_selection
from skopt import gp_minimize, space

# Bayesian Optimization with Gaussian Process


def optimize(params, param_names, X, y):
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for train_idx, test_idx in kf.split(X=X, y=y):
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        fold_acc = metrics.accuracy_score(y_test, y_preds)
        accuracies.append(fold_acc)
    # Since optimization =  minimize
    # we want to maximize [accuracy] = minimize [(-1)*accuracy]
    return -1.0 * np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv("../input/dataset/mobile_price_train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features"),
    ]
    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

    optimization_function = partial(optimize, param_names=param_names, X=X, y=y)
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10,
    )
    print(dict(zip(param_names, result.x)))
