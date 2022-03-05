import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from functools import partial
import optuna
#Optuna

def optimize(trial, X, y):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_depth",3,15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion
    )
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for (train_idx, test_idx) in kf.split(X=X, y = y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        
        X_test = X[test_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        fold_acc = metrics.accuracy_score(y_test, y_preds)
        accuracies.append(fold_acc)
    #Since optimization =  minimize
    #we want to maximize [accuracy] = minimize [(-1)*accuracy]
    return -1.0*np.mean(accuracies)
        

if __name__ == "__main__":
    df = pd.read_csv("../input/dataset/mobile_price_train.csv")
    X = df.drop("price_range", axis = 1).values 
    y = df.price_range.values
    optimization_function = partial(optimize, X=X, y=y)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=15)


