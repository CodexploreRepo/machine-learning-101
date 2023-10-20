# Exploratory Data Analysis (EDA)

- The first step in any data science project: exploring the data.
- In statistical theory, 
    - **Location** and **Variability** are referred to as the first and second moments of a distribution. 
    - **Skewness** and **Kurtosis** are called as the third and fourth moments 
        - Skewness refers to whether the data is skewed to larger or smaller values
        - Kurtosis indicates the propensity of the data to have extreme values. 
## 1. Elements of Structured Data
- There are two basic types of structured data: numeric and categorical.
- **Numeric**: Data that are expressed on a numeric scale.
    - **Continuous**: Data that can take on any value in an interval. (Synonyms: interval, float, numeric)
    - **Discrete**: Data that can take on only integer values, such as counts. (Synonyms: integer, count)
- **Categorical**: Data that can take on only a specific set of values representing a set of possible categories. (Synonyms: enums, enumerated, factors, nominal)
    - **Binary**: A special case of categorical data with just two categories of values, e.g., 0/1, true/false. (Synonyms: dichotomous, logical, indicator, boolean)
    - **Ordinal**: Categorical data that has an explicit ordering. (Synonym: ordered factor)

## 2. Estimates of Location
- A basic step in exploring your data is getting an estimate of where most of the data is located (i.e., its **central tendency**).
### 2.1. Mean (Average)
- **Mean** is the sum of all values divided by the number of values.
$$\text{Mean} = \bar x = \frac{\sum_{i=1}^n x_i}{n}$$
#### 2.1.1. Trimmed mean
- A variation of the mean is a trimmed mean
$$\text{Trimmed Mean}= \bar x = \frac{\sum_{i=p+1}^{n-p} x_i}{n-2p}$$ 

#### 2.1.2. Weighted mean
- Weighted mean, which you calculate by multiplying each data value $x_i$ by a user-specified weight $w_i$ and dividing their sum by the sum of the weights. 
$$\text{Weighted Mean}= \bar x_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$$ 

- For example: If we want to compute the average murder rate for the country, we need to use a weighted mean or median to account for different populations in the states. 
```Python
np.average(state['Murder.Rate'], weights=state['Population']) # weight used here is population of each state
```
### 2.2. Median 
- The median is the middle number on a sorted list of the data.
- The median is referred to as a **robust estimate of location** since it is not influenced by outliers (extreme cases) that could skew the results 
- When outliers are the result of bad data, the mean will result in a poor estimate of location, while the median will still be valid.
## 3. Estimates of Variability
- Location is just one dimension in summarizing a feature. 
A second dimension, **variability**, also referred to as `dispersion`, measures whether the data values are tightly clustered or spread out. 
### 3.1. Deviation
-  Deviations tell us how dispersed the data is around the central value.
#### 3.1.1. Mean Absolute Deviation
- Mean absolute deviation and is computed with the formula
$$\text{Mean Absolute Deviation}= \frac{\sum_{i=1}^{n} |x_i - \bar x |}{n}$$ where $\bar x$ is the sample mean
- For example: a set of data {1, 4, 4}, the mean is 3 and the median is 4. The deviations from the mean are the differences: 1 – 3 = –2, 4 – 3 = 1, 4 – 3 = 1. 
    - The absolute value of the deviations is {2 1 1}, and their average is (2 + 1 + 1) / 3 = 1.33.
#### 3.1.2. Variance & Standard Deviation (MAD)
- The best-known estimates of variability are the variance and the standard deviation, which are based on *squared deviations*. 
- The **variance** is an average of the squared deviations
$$\text{Variance} = s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar x )^2}{n-1}$$
- The **standard deviation** is the square root of the variance
    - The standard deviation is much easier to interpret than the variance since it is on the same scale as the original data
$$\text{Standard Deviation} = s = \sqrt{\text{Variance}}$$

### 3.2. Estimates Based on Percentiles
- A different approach to *estimating dispersion* is based on the spread of the sorted data. 
- The most basic measure is the **range**: the difference between the largest and smallest numbers. 
    - The *minimum* and *maximum* values themselves are useful to know and are helpful in identifying outliers, but the range is extremely sensitive to outliers and **not very useful as a general measure of dispersion (range) in the data**.
- To avoid the sensitivity to outliers, we can look at the range of the data after dropping values from each end.
- `Pth percentile` is a value such that at least $P$ percent of the values take on this value or less and at least $(100 – P)$ percent of the values take on this value or more.
    - For example: to find the 80th percentile, sort the data. 
        - Then, starting with the smallest value, proceed 80 percent of the way to the largest value.
        - Note: the median is the same thing as the 50th percentile. 
    - The percentile is essentially the same as a quantile, with quantiles indexed by fractions (so the .8 quantile is the same as the 80th percentile).
- `Interquartile Range` (or `IQR`) is a common measurement of variability is the difference between the **25th percentile** and the **75th percentile**
    - For example: we have an array {3,1,5,3,6,7,2,9}. 
        - We sort these to get {1,2,3,3,5,6,7,9}. 
        - The 25th percentile is at 2.5, and the 75th percentile is at 6.5, so the interquartile range is 6.5 – 2.5 = 4. 

```Python
state['Population'].quantile(0.75) - state['Population'].quantile(0.25)
```
## 4. Exploring the Data Distribution
- Each of the estimates sums up the data in a single number to describe the location or variability of the data. 
- It is also useful to explore how the data is distributed overall.
### 4.1. Percentiles and Boxplots
- Percentiles are also valuable for summarizing the entire distribution

<p align="center"><img src="../assets/img/box_plot_explain.png" witdh=300></p>

```Python
ax = (state['Population'] / 1_000_000).plot.box(figsize=(4,6))
ax.set_ylabel('Population (millions)')
plt.show()
```
<p align="center"><img src="../assets/img/box_plot_ex1.png" height=350></p>

- Understand the box plot:
    - The top and bottom of the box are the 75th (7M people) and 25th (2M people) percentiles
    - The median (5M people) is shown by the horizontal line in the box.
    - The dashed lines, referred to as **whiskers**, extend from the top and bottom of the box to indicate the range for the bulk of the data
    - Any data outside of the whiskers is plotted as single points or circles (often considered **outliers**).

### 4.2. Frequency Tables and Histograms
#### Frequency Tables
- A frequency table of a variable divides up the variable range into equally spaced segments and tells us how many values fall within each segment
- For example: The function `pandas.cut` creates a series that maps the values into the segments.
    - The least populous state is Wyoming, with 563,626 people, and the most populous is California, with 37,253,956 people. 
    - This gives us a range of 37,253,956 – 563,626 = 36,690,330, which we must divide up into equal size bins—let’s say 10 bins. 
    - With 10 equal size bins, each bin will have a width of 3,669,033, so the first bin will span from $(0.527M, 4.233M]$

```Python
binnedPopulation = pd.cut(state['Population'], 10)
```

| binnedPopulation   |   count | States                                                                                                                                           |
|:-------------------|--------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| (0.527, 4.233]     |      24 | ['AK', 'AR', 'CT', 'DE', 'HI', 'ID', 'IA', 'KS', 'ME', 'MS', 'MT', 'NE', 'NV', 'NH', 'NM', 'ND', 'OK', 'OR', 'RI', 'SD', 'UT', 'VT', 'WV', 'WY'] |
| (4.233, 7.902]     |      14 | ['AL', 'AZ', 'CO', 'IN', 'KY', 'LA', 'MD', 'MA', 'MN', 'MO', 'SC', 'TN', 'WA', 'WI']                                                             |
| (7.902, 11.571]    |       6 | ['GA', 'MI', 'NJ', 'NC', 'OH', 'VA']                                                                                                             |
| (11.571, 15.24]    |       2 | ['IL', 'PA']                                                                                                                                     |
| (15.24, 18.909]    |       1 | ['FL']                                                                                                                                           |
| (18.909, 22.578]   |       1 | ['NY']                                                                                                                                           |
| (22.578, 26.247]   |       1 | ['TX']                                                                                                                                           |
| (26.247, 29.916]   |       0 | []                                                                                                                                               |
| (29.916, 33.585]   |       0 | []                                                                                                                                               |
| (33.585, 37.254]   |       1 | ['CA'] 

#### Histogram
- A histogram is a way to visualize a frequency table, with bins on the x-axis and the data count on the y-axis.

```Python
ax = (state['Population'] / 1_000_000).plot.hist(figsize=(4, 4))
ax.set_xlabel('Population (millions)')
plt.show()
```
<p align="center"><img src="../assets/img/histogram_ex.png" witdh=300></p>

- Understand the histogram:
    - Empty bins are included in the graph.
    - Bins are of equal width.
    - The number of bins (or, equivalently, bin size) is up to the user.
    - Bars are contiguous — no empty space shows between bars, unless there is an empty bin.
## 5. Outliers
-  An **outlier** is any value that is very distant from the other values in a data set.