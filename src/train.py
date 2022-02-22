import joblib
import os 
import argparse

from . import config, model_dispatcher

import pandas as pd
from sklearn import metrics


def run(fold, model):
    #read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    x_train = df_train.drop("target", axis = 1).values
    y_train = df_train["target"].values
    
    x_valid = df_valid.drop("target", axis = 1).values
    y_valid = df_valid["target"].values
    
    #Fetch the model from model_dispatcher
    clf =  model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    
    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    
    #Save model
    # joblib.dump(clf, 
    #             os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    #         )
    
if __name__ == "__main__":
    # initialize Argument Parser Class
    parser = argparse.ArgumentParser()
    #Add (arg, type)
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    #read the arguments from the command line
    args = parser.parse_args()
    
    run(fold=args.fold,
        model=args.model)
  