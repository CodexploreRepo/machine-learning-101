# Recency, Frequency, and Monetary Value

## CLV

- **Customer lifetime value (CLV)** is the total worth of a customer to a company over the length of his relationship.
  - In practice, this “worth” can be defined as _revenue_, _profit_, _purchasing value_ or other metrics of an analyst’s choosing.

### Why CLV is important ?

- The totality of a company’s **CLV** over its **entire customer base** gives a rough idea of its _market value_.
  - Thus, a company with a high total CLV will appear attractive to investors.
- CLV analysis can guide the formulation of **customer acquisition** and **retention** strategies.
  - For example, special attention could be given to high-value customers to ensure that they stay loyal to the company

### CLV Formula

- A customer’s CLV for a given period can be calculated by multiplying two numbers:

  - The customer's predicted **number of transactions** within this period &#8594; _BG-NBD_ model
  - The predicted **value of each purchase**.
    - Option 1: taking average of all past purchases
    - Option 2: Gamma-Gamma model (a more sophisticated probabilistic model, which was also created by the authors of BG-NBD)
    <p align="center"><img src="../../assets/img/clv-formula.webp" width=300></p>

## R,T,F,M Definition

<p align="center"><img src="../../assets/img/rfm_definition.webp" width=300></p>

- [Recency & T Definition in Lifetimes](https://github.com/CamDavidsonPilon/lifetimes/issues/264): this is to explain why in life times, recency = the age of the latest trans - the first trans.
  - The definition of recency in Lifetimes actually depends on the definition of T.
    - $\text{Recency in Lifetimes} = t_x - t_1$
    - $\text{Actual Recency} = T - \text{Recency in Lifetimes}$
  - T be the age of the customer (or more accurately, time between when we first see them to observation period end)
    - $T = T - t_1$

<p align="center"><img src="../../assets/img/recency_definition_lifetimes.png" width=400></p>

- For frequency & monetary value calculation, the [first transaction is always ignored for the RFM calculations](https://github.com/CamDavidsonPilon/lifetimes/issues/208)

```shell
#summary_data_from_transaction_data() aggregates the rows with the same days.

0 1 2017-11-19 100
1 1 2017-11-19 150 - > 2017-11-19 250

2 1 2017-12-19 200
7 1 2017-12-19 300 - > 2017-12-19 500

6 1 2017-12-20 250 - > 2017-12-20 250

#The first transaction is always ignored for the RFM calculations (Only used for T if it's the only transaction).

# Hence you're only left with averaging 500 and 250, with the frequency of 2. Hence 750/2 = 375.5
```

## Beta Geometric Negative Binomial Distribution (BG-NBD)

- Many CLV models have been developed with different levels of sophistication and accuracy, ranging from rough heuristics to the use of complex probabilistic frameworks & BG-NBD is one of the most influential models in the domain, thanks to its interpretability and accuracy

### Scope of BG-NBD

- The model is only applicable to **non-contractual**, **continuous purchases**.
- The model only tackles 1 component of CLV calculation, which is the prediction of **the number of purchases**.

<p align="center"><img src="../../assets/img/contractual-non-contractual-continous-discrete.webp" width=400><br>The BG-NBD model tackles the non-contractual, continuous situation, which is the most common but most challenging of the four to analyze.</p>

- Under the **non-contractual**, **continuous** setting, customer attrition is not explicitly observable and can happen anytime.
  - This makes it harder to differentiate between customers who have _indefinitely churned_ and those who will return in the future.
  - As we will see later, the BG-NBD model is capable of assigning probabilities to each of these two options.

#### Contractual business vs a Non-contractual business

- A _contractual business_, as its name suggests, is one where the buyer-seller relationship is governed by contracts. When either party no longer wants to continue this relationship, the contract is terminated. Thanks to the contract, there is no ambiguity as to whether someone is a customer of the business at a given point.
- In a _non-contractual business_, on the other hand, purchases are made on a per-need basis without any contract.

#### Continuous and Discrete Settings

- In a _continuous_ setting, purchases can occur at any given moment. The majority of purchase situations (e.g. grocery purchases) fall under this category.
- In a _discrete_ setting, purchases usually occur periodically with some degree of regularity. An example of this is weekly magazine purchase

### Assumptions for BG-NBD

- Each customer has different purchasing rate
- Each customer can stop being your customer at any time
  - At any point, your customer can leave your business for another one a.k.a the **“deactivation” of a previously active customer**.
- To conveniently model a deactivation, we can assume that the _deactivation can only happen after a successful purchase_.
  - That is to say, after every purchase in your store, a customer will decide whether he’ll continue buying at your shop or to abandon it.

## Todo

- [Modeling Customer Lifetime Values with Lifetimes](https://www.aliz.ai/en/blog/part-2-modeling-customer-lifetime-values-with-lifetimes)
- [BTYD Databricks](https://www.databricks.com/notebooks/Customer%20Lifetime%20Value%20Virtual%20Workshop/02%20The%20BTYD%20Models.html)
