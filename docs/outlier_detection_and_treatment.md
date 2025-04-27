# Outlier Detection & Treatment

- [Tutorial Notebook](../notebooks/tutorial/outlier_detection_and_treatment.ipynb)
- Reference:
  - [Outlier Detection and Treatment: Z-score, IQR, and Robust Methods](https://medium.com/@aakash013/outlier-detection-treatment-z-score-iqr-and-robust-methods-398c99450ff3)

## Outlier Impact

- **Skewed Statistics**: Means and standard deviations can be distorted.
- **Misleading Visualizations**: Outliers affect the readability of charts and graphs.
- **Model Performance Issues**: In machine learning, outliers can cause models to misfit data, leading to poor generalization.

## Outlier Detection Techniques

- There are several ways to detect outliers, each with its own strengths and limitations.
  - Z-score method
  - IQR (Interquartile Range) method
  - Robust methods

| Method  | Best For                  | Limitations                          | Suitable Distributions |
| ------- | ------------------------- | ------------------------------------ | ---------------------- |
| Z-score | Normally distributed data | Sensitive to extreme values          | Normal                 |
| IQR     | Skewed distributions      | May not work well for small datasets | Skewed or normal       |
| MAD     | Skewed or non-normal data | Interpretation can be complex        | Highly skewed          |
