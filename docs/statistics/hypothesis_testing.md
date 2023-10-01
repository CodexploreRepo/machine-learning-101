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
## Statistical Test
- Reference: [Choosing the Right Statistical Test | Types & Examples](https://www.scribbr.com/statistics/statistical-tests/)
### t-test
- Reference: [Scribbr](https://www.scribbr.com/statistics/t-test/)
- A **t-test** is a statistical test that is used to compare the means of two groups. It is often used in hypothesis testing to determine whether a process or treatment actually has an effect on the population of interest, or whether two groups are different from one another.
#### When to use a t-test
- A t-test can only be used when comparing the means of two groups (a.k.a. pairwise comparison). 
- If you want to compare more than two groups, or if you want to do multiple pairwise comparisons, use an ANOVA test or a post-hoc test
#### Type of t-test
**One-sample, two-sample, or paired t-test**
- If the groups come from a single population (e.g., measuring before and after an experimental treatment), perform a `paired t-test`. This is a within-subjects design.
- If the groups come from two different populations (e.g., two different species, or people from two separate cities), perform a `two-sample t-test` (a.k.a. `independent t-test`). This is a between-subjects design.
- If there is one group being compared against a standard value (e.g., comparing the acidity of a liquid to a neutral pH of 7), perform a `one-sample t test`.

**One-tailed or two-tailed t-test**
- If you only care whether the two populations are different from one another, perform a `two-tailed t-test`.
- If you want to know whether one population mean is greater than or less than the other, perform a `one-tailed t-test`.
- *Example*: You want to know whether the mean petal length of iris flowers differs according to their species. You find two different species of irises growing in a garden and measure 25 petals of each species. You can test the difference between these two groups using a two-tailed t test (as only care whether the two populations are different from one another) and null and alterative hypotheses.
    - The null hypothesis ($H_0$) is that the true difference between these group means is zero.
    - The alternate hypothesis ($H_a$) is that the true difference is different from zero.
#### Results of a t-test
- `t value`
- `p value` related to your sample size, and shows how many ‘free’ data points are available in your test for making comparisons. The greater the degrees of freedom, the better your statistical test will work.
- `degrees of freedom` 
## p-values
- P values are used in hypothesis testing to help decide whether to reject the null hypothesis. 
    - **Tips**: The smaller the p value, the more likely you are to reject the null hypothesis.

## Statistical Significance
- In a hypothesis test, the p value is compared to the **significance** level to decide whether to reject the null hypothesis.
    - If the p value is **higher** than the significance level, the null hypothesis is not refuted (prove a statement or theory to be wrong), and the results are **not statistically significant**.
    - If the p value is **lower** than the significance level, the results are interpreted as refuting (prove a statement or theory to be wrong) the null hypothesis and reported as **statistically significant**
- Example: Through your hypothesis test, you obtain a p value of 0.0029. Since this p value is lower than your significance level of 0.05 (5%), you consider your results statistically significant and reject the null hypothesis.