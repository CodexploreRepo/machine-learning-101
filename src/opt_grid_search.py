import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


from sklearn import decomposition, preprocessing, pipeline

if __name__ == "__main__":
    df = pd.read_csv("../input/dataset/mobile_price_train.csv")
    X = df.drop("price_range", axis = 1).values 
    y = df.price_range.values
    
    
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()   
    rf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=2022) #n_jobs = -1 use all the core
    
    classifier = pipeline.Pipeline([
        ("scaling", scl),
        ("pca", pca),
        ("rf",rf)
    ])
    
    param_grid = {
        "pca__n_components": np.arange(5,10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max_depth": np.arange(1,20),
        "rf__criterion": ["gini", "entropy"],
    }
    kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    model = model_selection.RandomizedSearchCV(
        classifier,
        param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1, #use only 1 core
        cv = kfold
    )
    model.fit(X,y)
    print(model.best_score_)
    print(model.best_params_)
    