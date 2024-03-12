# Customer Churn Model

## Introduction

### Customer Lifetime Value (CLV)

- **Customer Lifetime Value** or CLV is considered a really important thing in marketing and ecommerce
- This clever metric tells you the predicted value each customer will bring to your business over their lifetime, and as such requires the ability to detect which customers will churn as well as what they’re likely to spend if they’re retained.

#### How many of your customers are still customers?

- In a [**contractual churn**](https://practicaldatascience.co.uk/machine-learning/how-to-create-a-contractual-churn-model) setting, like a mobile phone network or a broadband provider, you can tell how many of your customers are still customers (or “alive” as they’re called in the world of CLV), because you know how many open contracts you have.
- In [**non-contractual churn**](https://practicaldatascience.co.uk/machine-learning/how-to-create-a-non-contractual-churn-model-for-your-ecommerce-site) settings, like most ecommerce businesses, the time of a customer’s “death” isn’t directly observed, so you never really know which ones are alive or dead.

#### CLV Calculation

- Different models to calculate CLV:
  - Beta Geometric Negative Binomial Distribution (BG/NBD) model
  - Gamma-Gamma model
- These models are to predict the following:
  - Which customers are still customers
  - Who will order again in the next period
  - The number of orders each customer will place
  - The average order value of each customer’s order
  - The total amount each customer will generate for the business

## Reference

- [How to calculate CLV using BG/NBD and Gamma-Gamma](https://practicaldatascience.co.uk/data-science/how-to-calculate-clv-using-bgnbd-and-gamma-gamma)
