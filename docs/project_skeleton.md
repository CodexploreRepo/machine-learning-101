# Project Skeleton

## Data Loading & Inspection

### Missing Columns

```Python
df.isna().sum()*100 / df.shape[0]

#Showing missing data at which index of the data
fig, ax = plt.subplots(figsize=(15,4))
sns.heatmap(df.isna().transpose(), ax=ax, cmap="crest")
plt.show()
```

### Duplicates Values

```Python
# Count duplicate rows in train_data
train_duplicates = df.duplicated().sum()
test_duplicates = df.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in df_train: {train_duplicates}")
print(f"Number of duplicate rows in df_test: {test_duplicates}")
```

### Numercial & Catergorical Feature Identification

- We can identify the catergorical columns by the number of unique values

```Python
def find_categorical(df, cutoff=12):
    """
        Function to find categorical columns in the dataframe.
        cutoff (int): is determinied when plotting the histogram distribution for numerical cols
    """
    cat_cols = []
    for col in df.columns:
        if len(df[col].unique()) <= cutoff:
            cat_cols.append(col)
    return cat_cols

def to_categorical(cat_cols, df):
    """
        Converts the columns passed in `columns` to categorical datatype for keras model
    """
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df
# Step 1: identify catergorical columns in the df
cat_cols = find_categorical(df)
# Step 2: convert atergorical columns astype "category"
# This step can be performed later as categorical column with numerical value like (1,2,3) can be used in the EDA
# df = to_categorical(cat_cols, df)
# Step 3: identify the numerical columns
num_cols = list(set(df.columns) - set(cat_cols))

# Extra: You also can get the summary on the catergorical columns
def summarize_categoricals(df, show_levels=False):
    """
        Display uniqueness in each column
        df: dataframe contains only catergorical features
        show_levels: how many unique values in a catergoical column
    """
    data = [[df[c].unique().to_list(), len(df[c].unique()), df[c].isnull().sum()] for c in df.columns]
    df_temp = pd.DataFrame(data, index=df.columns,
                           columns=['Levels', 'No. of Levels', 'No. of Missing Values'])
    return df_temp.iloc[:, 0 if show_levels else 1:]

summarize_categoricals(df[cat_cols], show_levels=True)
```

## EDA

### Univariate Analysis

#### Numerical Columns

- For the numerical, histogram plot will help to identify the distribution and data skew (log transform if needed) & box plot to identify the outliers

```Python
import matplotlib.pyplot as plt # plot
import seaborn as sns

palette = [ '#0077b6' , '#00b4d8' , '#90e0ef' , '#caf0f8']
# color_palette = sns.color_palette(palette)
sns.set_palette(palette)
sns.set_style('whitegrid')
# histogram plot & box plot
for col in num_cols:
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle(f"Histogram Distribuiton for {col}")
    sns.histplot(data=df, x=col, bins=50, kde=True, color=palette[1], ax=ax[0])
    sns.boxplot(x=df[col], ax=ax[1], palette=palette[1:])
```

#### Catergorical Columns

```Python
for column in cat_cols:
    fig, ax=plt.subplots(1,2,figsize=(20,6))

    df[column].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1],  colors=palette)
    sns.countplot(x=column, data=df, ax=ax[0], palette=palette)

    fig.suptitle(f'{column}')
    plt.show()
```

### Bi-Variate Analysis

#### Correlation between Quantitative Variables

- Quantitative Variables include catergorical variables which are numeric, so we can use `.select_dtypes(include=np.number)` to identify the list of Quantitative Variables.

```Python
quantitative_variables = df.select_dtypes(include=np.number).columns.to_list()
```

- Perform the correlation & the visualisation by heatmap

```Python
corr_df = df[quantitative_variables].corr()
plt.figure(figsize= (14, 8))
sns.heatmap(corr_df, annot = True, fmt = '.2f', linewidths= 0.8  , cmap= palette[::-1])
plt.show()
```

##### Collinear Feature Removal

- Removing collinear within the input features can help a model to generalize and improves the interpretability of the model.
- The function below will help to remove collinear features in a dataframe with a correlation coefficient greater than the threshold.

```Python
def remove_collinear_features(df,
                              quantitative_variables,
                              target_column,
                              threshold = 0.99):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        df: original dataframe
        quantitative_variables: list of quantitative variables
        target_column: this to ignore the target_column as this function is to remove col-linear among features
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''
    x = df[quantitative_variables]
    x = x.drop(target_column, axis=1)
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(f"({col.values[0]:15s} | {row.values[0]:15s} | {round(val[0][0], 2)}) -> Remove '{col.values[0]}'")
                drop_cols.append(col.values[0])


    drops = set(drop_cols)
    # Drop one of each pair of correlated columns in the original df
    df = df.drop(columns=drops, axis=1)

    return df

remove_collinear_features(
    df,
    quantitative_variables,
    target_column="Exited",
    threshold=0.8
)
```

- Once finished, we might need to re-update the list of numerical & categorical columns

#### Numberical Features vs Target Variables

- In the case of classification, we can specify the `hue=target_columns` to observe the different distribution with respect to the different class

```Python
# numerical feature vs target
for column in num_cols:
    fig, ax = plt.subplots(figsize=(18, 4))
    fig = sns.histplot(data=df, x=column, hue="Exited", bins=50, kde=True, palette=palette)
    plt.show()

fig = plt.figure(figsize=(14, len(cat_cols)*3))
for i, col in enumerate(cat_cols):
    plt.subplot(len(cat_cols)//2 + len(cat_cols) % 2, 2, i+1)
    sns.countplot(x=col, hue='Exited', data=df, palette=palette[0:2], color='#26090b', edgecolor='#26090b')
    plt.title(f"{col} countplot by target", fontweight = 'bold')
    plt.ylim(0, df[col].value_counts().max() + 10)

plt.tight_layout()
plt.show()
```

##### Binning

```Python
age_bins = [0, 18, 25, 30, 40, 65, np.inf]  # Adjust the bin edges as needed
age_labels = ['0-18', '18-25', '25-30', '30-40', '40-65', '> 65']

# Bin the Age column
age_bin_df = pd.cut(df[['Age']].iloc[:, 0], bins=age_bins, labels=age_labels, right=False)

fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.histplot(data=age_bin_df, kde=True, palette=palette)
plt.show()
```

## Pipeline

- Create custome transformers for data cleaning or feature engineering.

```Python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self,factor=1.5):
        self.factor = factor

    def _outlier_detector(self,X,y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X: np.ndarray,y=None):
        # if fit in another dataset, this two arrays must be reset
        self.lower_bound = []
        self.upper_bound = []

        X.apply(self._outlier_detector)
        self.feature_names_in_ = X.shape[1] # this is required for get_feature_names_out
        return self

    def transform(self,X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy() # convert X into Pandas dataframe to use .iloc[:, i]
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        self.columns = X.columns
        return X
    def get_feature_names_out(self, feature_names):
        return [col for col in self.columns]
```

- Depends on different set of columns, we can define different pipelines to process & perform feature engineering
- Default numerical pipepline

```Python
num_pipeline = make_pipeline(
                OutlierRemover(), # from the custom transformer
                SimpleImputer(strategy="median"),
                MinMaxScaler()
)
```

- For the categorical feature, we might need to use different encoders such as OneHot, Ordinal, Hash encoders

```Python
def make_cat_pipeline(encoder):
    # the function to generate the cat_pipeline base on different encoder
    return make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                encoder
    )

# One-hot encoder pipeline
oh_cat_pipeline = make_cat_pipeline(OneHotEncoder(handle_unknown='ignore'))
# Ordinal encoder pipeline
ord_cat_pipeline = make_cat_pipeline(
                OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                )
            )
```

- For feature engineering, we can use the functional transformers to combine multiple columns

```Python
# this example is to use FunctionTransformer to combine (Geography + Gender) column into Geo-Gender column such as SpainMale, GemarnyFemale
def geo_gender_name(function_transformer, feature_names_in):
    # '__GeoGender to be appended
    return ["GeoGender"]  # feature names out

geo_gender_pipeline = make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                # X = df[['Geography','Gender']]
                FunctionTransformer(lambda X: X[:, [0]] + X[:, [1]], feature_names_out=geo_gender_name), # concat__GeoGender
                OneHotEncoder(handle_unknown='ignore')
)
```

- For binning, we can use the built-in Sklearn's `KBinsDiscretizer`

```Python
# we also can define
def binning_pipeline(n_bins, encode, bin_strategy):
    """
    KBinsDiscretizer: Bin continuous data into intervals.
        n_bins: number of bins
        encode: {‘onehot’, ‘onehot-dense’, ‘ordinal’}, default=’onehot’
        strategy: {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
    """
    return make_pipeline(
        OutlierRemover(), # custom transformer
        SimpleImputer(strategy="median"),
        KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=bin_strategy)
    )
```

### Full Pipeline

- We can ensemble all seperate pipelines for each set of feature columns

```Python
preprocessing_pipeline = Pipeline([
    ("preprocessing", ColumnTransformer([
            ("bin_oh",binning_pipeline(n_bins=5, encode='onehot', bin_strategy='quantile'), ['Age']),
            ("bin_ord", binning_pipeline(n_bins=7, encode='ordinal', bin_strategy='kmeans'), ['CreditScore']),
            #(oh_cat_pipeline, make_column_selector(dtype_include='category')),
            ("cat_oh", oh_cat_pipeline, ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']),
            ("cat_ord",ord_cat_pipeline, ['Tenure', 'NumOfProducts']),
            ("concat", geo_gender_pipeline, ['Geography', 'Gender'])  # feature engineering
            ],
            remainder=num_pipeline
        ),
    )
])

X_train_pre = preprocessing_pipeline.fit_transform(X_train, y_train)
X_val_pre = preprocessing_pipeline.transform(X_val)

column_names = preprocessing_pipeline.get_feature_names_out()

X_train_pre=pd.DataFrame(X_train_pre, columns=column_names)
X_val_pre  =pd.DataFrame(X_val_pre, columns = column_names)
```

## Model Training

### Baseline Model

- First, we can define the list of models with default parameters

```Python
random_state = 2024
model_list = [
    BernoulliNB(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(random_state=random_state, max_depth=10, max_features='sqrt', n_estimators=300),
    AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.5),
    GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, max_depth=10, n_estimators = 200),
    CatBoostClassifier(verbose=False),
    XGBClassifier(verbose=False),
    LGBMClassifier(verbose=False)
]
```

- Define the trainer class to perform the cross validation

```Python
class Trainer:
    def __init__(self,
                 model_list) -> None:
        self.model_list = model_list

    def fit_and_evaluate(self, X_train, y_train, X_val, y_val, metrics: str, cv: int=5) -> pd.DataFrame:
        baseline_results = pd.DataFrame(columns=['model_name', f'{metrics}_train_cv', f'{metrics}_val'])
        for idx in tqdm.tqdm(range(len(model_list))):
            clf = model_list[idx]
            # cross_val_score uses the KFold strategy with default parameters for making the train-test splits,
            # which means splits into consecutive chunks rather than shuffling. -> shuffle=True
            # Stratified is to ensure the class distribution equal in each fold
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=2024)
            # using cross_val_score
            # list of "scoring": https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            metrics_train = np.round(
                            np.mean(
                                cross_val_score(clf, X_train, y_train,
                                scoring=metrics, cv=kfold)
                            ), 3
                        )
            # test on val_set
            clf.fit(X_train, y_train)
            y_pred_val = clf.predict_proba(X_val)[:, 1]
            metrics_val = self.cal_metrics(y_val, y_pred_val)
            baseline_results.loc[len(baseline_results)] = [clf.__class__.__name__, metrics_train, metrics_val]
        return baseline_results \
                    .sort_values(by=f'{metrics}_val', ascending=False) \
                    .set_index('model_name')

    def cal_metrics(self, y, y_pred) -> float:
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        return auc(fpr, tpr)
```

- You can initialise the Trainer class and perform the evaluation

```Python
base_trainer = Trainer(model_list)
baseline_results = base_trainer.fit_and_evaluate(X_train_pre, y_train, X_val_pre, y_val, 'roc_auc')

# under-sampling
X_train_rus, y_train_rus = RandomUnderSampler().fit_resample(X_train_pre, y_train)
rus_baseline_results = base_trainer.fit_and_evaluate(X_train_rus, y_train_rus, X_val_pre, y_val, 'roc_auc')

# over-sampling
X_train_smote, y_train_smote = SMOTE().fit_resample(X_train_pre, y_train)
smote_baseline_results = base_trainer.fit_and_evaluate(X_train_smote, y_train_smote, X_val_pre, y_val, 'roc_auc')
```

### Evaluation Function

#### Classification

- For classification, we will need to examine the confusion matrix, classification report
  - Threshold fine-tuning is also needed for the classification problems

```Python
def evaluate_model(model, X_test, y_test, display_labels: List[str]=['Not exited', 'Exited']):

    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.show()

    # Precision and recall
    print(classification_report(y_test, y_pred))

def finetune_threshold(model, X_test, y_test, display_labels: List[str]=['Not exited', 'Exited']):
    """
    This function is to fine-tune the threshold based on the G-means
    """
     # ROC curve and AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold: calculate the G-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest G-mean
    idx = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[idx]}, G-Mean={gmeans[idx]:.3f}')

    fig, ax = plt.subplots()
    ax.scatter(fpr[idx], tpr[idx], marker="*", s=100, label=f'Optimal Threshold (G-mean={gmeans[idx]:.3f})')
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])

    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.show()
```
