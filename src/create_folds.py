import pandas as pd
import numpy as np
from sklearn import model_selection

def create_folds(df, target_col, regression=False):
    #create a new kfold column & filled with -1
    df["kfold"] = -1
    
    #Step 1: Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True) #Drop the old column index
    #fetch targets
    if regression:
        #Calculate number of bins by Sturge's Rule
        num_bins = int(np.floor(1+np.log2(len(df))))
        df.loc[:, "bins"] = pd.cut(
            df[target_col], bins=num_bins, labels=False
        )
        y = df.bins.values
    else:
        y = df[target_col].values

    #initialize the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    if regression: df = df.drop("bins", axis = 1) #for regression, drop "bins" column after split
    
    df.to_csv("../input/train_folds.csv", index=False)

if __name__ == "__main__":
    df = pd.read_csv("../input/dataset/winequality-red.csv")
    
    create_folds(df, "quality")