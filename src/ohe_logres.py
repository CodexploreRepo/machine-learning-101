import pandas as pd

from sklearn import linear_model, metrics, preprocessing

def run(fold):
    #load the full training data with folds
    df = pd.read_csv("../input/train_folds.csv")
    
    #all columns are features, except id, target, and kfold
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    
    #fill all NaN values with NONE
    #since all columns are categories, can convert to STRING
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
        
    #get the training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    ohe = preprocessing.OneHotEncoder()
    #fit ohe on traing + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data)
    
    #transform training data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    
    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(x_train, df_train.target.values)
    
    valid_preds = model.predict_proba(x_valid)[:,1]
    
    #get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == "__main__":
    for i in range(5):
        run(i)