# Gradient Boosting

## Introduction

- The gradient boosting ensemble technique consists of three simple steps:
  - Step 1: An initial model $F_0$ is defined to predict the target variable y. This model will be associated with a residual ($y – F_0$)
  - Step 2: A new model $H_1$ is fit to the **residuals** from the previous step.
  - Step 3: Now, $F_0$ and $H_1$ are combined to give $F_1$, the boosted version of $F_0$. The mean squared error from $F_1$ will be lower than that from $F_0$:

> $ F_1 = F_0 + H_1 $

- To improve the performance of $F1$, we could model after the residuals of $F1$ and create a new model $F2$:

> $ F_2 = F_1 + H_2 $

- This can be done for ‘m’ iterations, until residuals have been minimized as much as possible:

> $ F*m = F*{m-1} + H_m $

- In contrast to bagging techniques like Random Forest, in which trees are grown to their maximum extent, boosting makes use of trees with fewer splits. Such small trees, which are not very deep, are highly interpretable. Parameters like the number of trees or iterations, the rate at which the gradient boosting learns, and the depth of the tree, could be optimally selected through validation techniques like k-fold cross validation.
