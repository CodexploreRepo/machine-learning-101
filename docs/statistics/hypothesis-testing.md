# Hypothesis Testing

## What is Hypothesis testing ?

- Hypothesis testing is a formal procedure for investigating our ideas about the world using statistics.
- There are 5 main steps in hypothesis testing:
  1. State your research hypothesis as a null hypothesis and alternate hypothesis ($H_0$) and ($H_a$ or $H_1$).
  2. Collect data in a way designed to test the hypothesis.
  3. Perform an appropriate [statistical test](#statistical-test).
  4. Decide whether to reject or fail to reject your null hypothesis.
  5. Present the findings in your results and discussion section.

### Null hypothesis

- All statistical tests have a null hypothesis. For most tests, the null hypothesis is that there is no relationship between your variables of interest or that there is no difference among groups.

### Alternate hypothesis

- Alternate hypothesisis usually your initial hypothesis that predicts a relationship between variables.

### Example: Null and alternative hypothesis

- Example 1: You want to know whether there is a difference in longevity between two groups of mice fed on different diets, diet A and diet B. You can statistically test the difference between these two diets using a two-tailed t test.
  - **Null hypothesis ($H_0$)**: there is no difference in longevity between the two groups.
  - **Alternative hypothesis ($H_A$ or $H_1$)**: there is a difference in longevity between the two groups.

<p align="center"><img src="../../assets/img/statistical-significance-p-value.png" width=500><br>Statistical Significance & P-Value</p>

### p-value

- P-value is evidence against the null hypothesis. The smaller the p-value stronger the chances to reject the $H_0$
  - **p-value** is the area

### Statistical Significance

- In a hypothesis test, the p value is compared to the **significance level** $\alpha$ to decide whether to reject the null hypothesis.
  - If the p value is **higher** than the significance level, the null hypothesis is not refuted (prove a statement or theory to be wrong), and the results are **not statistically significant**.
  - If the p value is **lower** than the significance level $\alpha$, the results are interpreted as refuting (prove a statement or theory to be wrong) the null hypothesis and reported as **statistically significant**
- Example: Through your hypothesis test, you obtain a p value of 0.0029. Since this p value is lower than your significance level of 0.05 (5%), you consider your results statistically significant and reject the null hypothesis.

### Confidence interval

- The confidence interval is an observed range in which a given percentage of test outcomes fall. We manually select our desired confidence level at the beginning of our test. Generally, we take a 95% confidence interval

## Statistical Tests

- Reference: [Choosing the Right Statistical Test | Types & Examples](https://www.scribbr.com/statistics/statistical-tests/)

### t-test

#### What is t-test ?

- Reference: [Scribbr](https://www.scribbr.com/statistics/t-test/)
- A **t-test** is a statistical test that is used to compare the means of two groups. It is often used in hypothesis testing to determine whether a process or treatment actually has an effect on the population of interest, or whether two groups are different from one another.
- For example: consider a telecom company that has two service centers in the city. The company wants to find out whether the average time required to service a customer is the same in both stores.
  - The company measures the average time taken by 50 random customers in each store. Store A takes 22 minutes, while Store B averages 25 minutes.
  - Can we say that Store A is more efficient than Store B in terms of customer service?
  - Simply looking at the average sample time might not be representative of all the customers who visit both stores.
  - This is where the t-test comes into play. It helps us understand if the difference between two sample means is actually real or simply due to chance.

#### When to use a t-test

- When the sample size is small (say < 30) and the population mean is unknown.
- A t-test can only be used when comparing the **means of two groups** (a.k.a. pairwise comparison).
  - If you want to compare more than two groups, or if you want to do multiple pairwise comparisons, use an ANOVA test or a post-hoc test

#### Assumptions for Performing a T-test

- Normal distribution
- Similar variance
- Sample sizes:
  - Same number (if 2-sample t-test)
  - 20-30+ (more than that should use z-test)

#### Results of a t-test

- `t value`
- `p value` related to your sample size, and shows how many ‘free’ data points are available in your test for making comparisons. The greater the degrees of freedom, the better your statistical test will work.
- `degrees of freedom`

#### Types of t-test

##### One-sample t-test

- In a **one-sample t-test**, we compare the average (or mean) parameter of one group (sample) against the _set average_ (or mean).
  - This _set average_ can be any **theoretical value** (or it can be the **population mean**).
- Formula:
  $$t-statistic = \frac{\bar{x}-\mu}{s/\sqrt{n}}$$

  - $\bar{x}$ sample mean
  - $\mu$ population mean or set average
  - $s$ sample standard deviation
  - $n$ number of examples in the sample

- Example 1: A research scholar wants to determine if the average eating time for a standard-size burger differs from a set value (Let’s say this set value is 10 minutes).
  - How do you think the research scholar can go about determining this?
- Example 2: A company claims to produce ball bearings of 10 cm diameter (null hypothesis), and you decide to audit if that is true. You collect 21 ball bearings from multiple shops around the city and measure their diameter with sample mean is 11 cm, and the standard deviation is 1 cm.
  - You must determine if the company’s claim is false (alternate hypothesis) based on the measurements with the confidence of 95%.

```Python
import numpy as np
from scipy import stats

# Population Mean
mu = 10

# Sample Size
N1 = 21

# Degrees of freedom
dof = N1 - 1

# Generate a random sample with mean = 11 and standard deviation = 1
x = np.random.randn(N1) + 11

# Method 1: calculate based on the formula
# Sample Mean
x_bar = x.mean()
# Standard Deviation
std = np.std(x, ddof=1)
# Standard Error
ste = std/np.sqrt(N1)
# Calculating the T-Statistics
t_stat = (x_bar - mu) / ste
# p-value of the t-statistic
p_val = 2*(1 - stats.t.cdf(abs(t_stat), df = dof))

# Method 2: using the Stats library, compute t-statistic and p-value
t_stat, p_val = stats.ttest_1samp(a=x, popmean = mu)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")

# t-statistic = 4.689539773390642
# p-value = 0.0001407967502139183
```

- The p-value=0.0001 is less than the default significance level of 0.05, which indicates that the probability of such an extreme outcome is close to zero and that the null hypothesis can be rejected.

##### Independent two-sample t-test

- The **two-sample t-test** is used to compare the means of two different samples.
- Formula:
  $$t=\frac{\bar{x_1}-\bar{x_2}}{\sqrt{s_1^2/N_1 + s_2^2/N_2}}$$

  - $\bar{x_1}$ first sample mean
  - $\bar{x_2}$ second sample mean
  - $s_1$ first sample standard deviation
  - $s_2$ second sample standard deviation
  - $N_1$ first sample size
  - $N_2$ second sample size

- For example: We need to find out if the ball bearings from the two factories are of different sizes (Null Hypothesis).
  - The first factory shares 21 samples of ball bearings where the mean diameter of the sample comes out to be 10.5 cm.
  - The second factory shares 25 samples with a mean diameter of 9.5 cm.
  - Both have a standard deviation of 1 cm.

```Python
# Sample Sizes
N1, N2 = 21, 25

# Degrees of freedom
dof = min(N1,N2) - 1

# Gaussian distributed data with mean = 10.5 and var = 1
x = np.random.randn(N1) + 10.5

# Gaussian distributed data with mean = 9.5 and var = 1
y = np.random.randn(N2) + 9.5

## Using the internal function from SciPy Package
t_stat, p_val = stats.ttest_ind(x, y)
# t-statistic = 3.18913308431476
# p-value = 0.0026296199823557754
```

- Referring to the p-value of 0.0026 which is less than the significance level of 0.05, we reject the null hypothesis stating that the bearings from the two factories are not identical.

##### Paired-sample t-test

- Paired-sample t-test is used to compare separate means for a group at two different times or under two different conditions.
- Example: A certain manager realized that the productivity level of his employees was trending significantly downwards. This manager decided to conduct a training program for all his employees with the aim of increasing their productivity levels.
  - How will the manager measure if the productivity levels have increased?
    - We are comparing the same sample (the employees) at two different times (before and after the training). This is an example of a paired t-test.
