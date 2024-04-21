# Drift in Machine Learning (ML)

## Introduction

- Drift is one of the top priorities and concerns of data scientists and machine learning engineers.
- From day 1, the data that our models utilize to infer the predictions is already different from the data on which they trained.

### What causes drift in ML?

- Errors in data collection
- Changes in the way people behave
- Time gaps that alter what is considered a good prediction and what is not.

### Type of ML drifts

- **Concept drift** (**model drift**) is formaly defined as “a change in the joint probability distribution, i.e., $P_t(X,y)  \neq  P_{t+n}(X,y).$”
- We can decompose the joint probability $P(X,y)$ into smaller components to better understand what changes in these components can trigger concept drift:
  - **Covariate shift** $P(X)$ (also known as **input drift**, **data drift**, or **population drift**): occurs when there are changes in the distribution of the input variables (i.e., features).
    - Detection:
      - Covariate shift can be detected on a univariate level, also referred to as **feature drift**
      - It may also be analyzed on a **multi-variate** (e.g: train the classifier to differentiate training features and inference features) level across the entire feature space distribution.
  - **Prior probability shift** $P(y)$ (**label drift**, **unconditional class shift**, or **prior probability shift**): occurs when there are changes in the distribution of the class variable $y$.
  - **Posterior class shift** $P(y | X)$ (concept shift, or "**real concept drift**") refers to changes in the relationship between the input variables and the target variables.

### Drift Patterns

- **Gradual**: a gradual transition will happen over time when new concepts come into play. For example, in a movie recommendation task, movies, genres, and user preferences all change gradually over time.
- **Sudden**: a drift can happen suddenly, for example, when a sensor is replaced by another one with a different calibration.
- **Incremental**: drift can also happen in a sequence of small steps, such that it is only noticed after a long period of time. An example may be when a sensor wears down and becomes less accurate over time.
- **Blip**: spikes or blips are basically outliers or one-off occurrences that influence the model. This may be a war or some exceptional event. Of course, there may be recurring anomalies that take place—with no clue as to when they may happen again.

<p align="center"><img src="../../assets/img/patterns-of-drift.png" width=400></p>

## Data Drift

### Univariate Drift Detection

- To measure the [**statistics distance**](#statistical-distance-metrics-for-drift-detection) between the tested distribution and the reference distribution of a given feature.

### Multivariate Drift Detection

- Many machine learning models leverage dozens, hundreds, or even thousands of different features. In scenarios of high dimensionality, looking for drift only at the feature level will quickly lead to an overload of alerts due to the granularity of the measurement method.
- Solution:
  - Model-based detection: train a classifier to differetiate between the $X_{reference}$ assigning with class $0$, and $X_{current}$ assigning with class $1$. If the classifier `AUC_ROC` > 0.5, meaning that the classifier is able to detect the difference between $X_{reference}$ and $X_{current}$, hence there is a multivariate drift.
    - Note: $X_{reference}$ and $X_{current}$ have the same set of feature columns

## Concept Drift Detection

- **Method 1 - [Statistics Distance](#statistical-distance-metrics-for-drift-detection)**: to ensure the model prediction distribution doesn’t drastically change from a baseline. The baseline can be a training production window of data or a training or validation dataset.
  - This is useful when there is the delayed ground truth to compare against production model decisions.
- **Method 2 - Change in feature importance**

## Statistical Distance Metrics for Drift Detection

- Statistical distance measures are defined between two distributions: **reference** distribution and **current** distribution.

  | Metrics                                              | Usage                                                                                                                                                                | Symmetry  | Range               | Threshold                                                                                                                                 | Note                                                                                                 |
  | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
  | Population Stability Index (PSI)                     | #1: monitor **top features** for **feature drift**; monitor **prediction output** and **ground truths** for concept drift<br><br>#2: Numeric and Categorical         | Symmetry  |                     | `PSI < 0.1` no significant population change<br>`0.1 < PSI < 0.2` moderate population change<br>`PSI > 0.2` significant population change | **PSI bins** and **PSI quantiles** (calculating the same metric with 2 different binning strategies) |
  | Pearson’s Chi-Square Distance                        | To be updated                                                                                                                                                        |           |                     |                                                                                                                                           |                                                                                                      |
  | Kullback–Leibler (KL) Divergence                     | #1: One distribution has a high variance relative (training data) to **small sample size** (inference data)<br><br> #2: Numerical (binning required) and Categorical | Asymmetry | $KL \in [0,\infty)$ | When $KL=0$, it means that the two distributions are identical.                                                                           |                                                                                                      |
  | Jensen Shannon                                       | Numerical (binning required) and Categorical                                                                                                                         | Symmetry  | Between 0 and 1     |                                                                                                                                           | a bounded variant of KL divergence                                                                   |
  | Wasserstein Distance (a.k.a. Earth Mover’s distance) | Numerical                                                                                                                                                            |           |                     |                                                                                                                                           |                                                                                                      |
  | Kolmogorov–Smirnov                                   |                                                                                                                                                                      |           |                     |                                                                                                                                           |                                                                                                      |

## Reference

- [Using Statistical Distances for Machine Learning Observability](https://medium.com/towards-data-science/using-statistical-distance-metrics-for-machine-learning-observability-4c874cded78)
