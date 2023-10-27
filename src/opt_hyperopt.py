from functools import partial

import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn import ensemble, metrics, model_selection

# Hyperopt


def optimize(params, X, y):
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

    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1),
    }

    optimization_function = partial(optimize, X=X, y=y)
    trials = Trials()
    result = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials,
    )
    print(result)
