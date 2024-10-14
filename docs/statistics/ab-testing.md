# A/B Testing

## Design Consideration

- Null and Alternative hypothesis:
  - Null hypothesis $H_0$: from an A/B test perspective, the null hypothesis states that there is no difference between the control and treatment groups.
    - For example: there is no difference in the conversion rate in customers receiving newsletter A and B
  - Alternative hypothesis $H_a$:
    - For example: the conversion rate of newsletter B is higher than those who receive newsletter A
  - This even can be **bi-directional**, it can either increase it or decrease. Even can be `2-tailed hypothesis` test
- Randomisation unit and splitting into groups:
  - Even (50-50) vs Un-even (95-5, makes more sense in a holdout experiment where you are scaling up a new feature cautiously and monitoring it’s impact) Split
- Metrics:
  - **Driver metrics**: the goal that you want to achieve which is aligned with stakeholders
    - For example: increase click through rates, app open rates and thereby conversion.
  - **Guardrail metrics**: used to guard the new treatment from harming the business.
    - For example: if you have added a new feature to the home page of your app, the time to load the app would be a guardrail metric as it should not go above a certain threshold or you will start losing users.
- Test Duration: should be decided during the experiment design phase and before running it as one stakeholder can pre-maturely stop the test when they see the metrics align with their hypothesis wich can lead to false positives and introduce bias in the results,
  - The time duration has to be decided keeping in mind:
    - **Weekend effect**: people tend to make more purchases over the weekend, therefore an experiment should be run at least for a week.
    - **Seasonality effect**
    - **Holiday Season** specially days like Black Friday influence user behaviour in ways that cannot be generalised to non-holiday days.
- Conduct the test:
  - One way to perform the test is to calculate **daily conversion rates** for both the treatment and the control groups. Since the conversion rate in a group on a certain day represents a single data point, the **sample size** is actually the number of days. Thus, we will be testing the difference between the mean of daily conversion rates in each group across the testing period.
  - When we run our experiment for one month, we noticed that the mean conversion rate for the Control group is 16% whereas that for the test Group is 19%.
- Power Analysis: there are two types of errors we need to account for while running A/B tests
  - Type I Error - Significance Level/ α = P(Rejecting null | null True)
    - We reject the null hypothesis when it is true. That is we accept the variant B when it is not performing better than A
  - Type II Error — Statistical Power/ β = P(fail to reject null | null False)
    - We failed to reject the null hypothesis when it is false. It means we conclude variant B is not good when it performs better than A
  - Note: to decide α and β before running the test as to decide these pre-hand to calculate the minimum sample size needed to run the AB test.
  - To avoid these errors we must calculate the statistical significance of our test.
    - An experiment is considered to be statistically significant when we have enough evidence to prove that the result we see in the sample also exists in the population.
    - To prove the statistical significance of our experiment we can use a two-sample T-test.
- Choose the hypothesis test:
  - List of tests: [Cookbook](https://medium.com/@ibtesamahmex/the-ab-testing-cookbook-part-3-3af29b7f9fa7)
  - The **two–sample t–test** is one of the most commonly used hypothesis tests. It is applied to **compare** whether the **average difference** between the two groups.

<p align="center"><img src="../../assets/img/statistical-significance-p-value.png" width=500><br>Statistical Significance & P-Value</p>
- Hypothesis tests: Frequentists vs. Bayesians
  - Frequentist statistics uses hypotheses tests which are based on defining two hypotheses, the null and the alternative hypotheses (or H0 and H1) which are mutually exclusive:
  - [Read more](https://www.linkedin.com/pulse/hypothesis-tests-frequentists-vs-bayesians-miguel-pereira-md-phd/)
- [A/B testing example](https://github.com/renatofillinich/ab_test_guide_in_python/blob/master/AB%20testing%20with%20Python.ipynb)
  - [Confidence Interval](https://medium.com/@ibtesamahmex/confidence-intervals-and-how-to-find-them-42fe4e0900c3)
- [Baysian A/b testing](https://medium.com/@ibtesamahmex/beginners-guide-to-bayesian-ab-testing-22f40988d5e6)
