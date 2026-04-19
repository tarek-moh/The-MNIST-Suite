## The MNIST SUITE
This project is a two phase collaborative project fucosing on the MNIST dataset.

### Phase 1: Binary MNIST Classification
In this phase, we will explore the MNIST dataset and determine if it is linearly seperable.

We will use the following models:
- SVM (Primal and/or Dual)
- Logistic Regression
- [TBD]

### Problem Formulation:
Given a dataset of handwritten digits $D = {(X_i, y_i)}_{i=1}^n$ where $X_i \in \mathbb{R}^d$ and $y_i \in \{1, ... , 10\}$, we want to find a function $f: \mathbb{R}^d \to \{-1, 1\}$ that minimizes the empirical risk, such that:


$$y_i^* = \begin{cases} +1 & \text{if } y_i \in \{1, 4, 7\} \\ -1 & \text{if } y_i \in \{2, 3, 5, 6, 8, 9, 10\} \end{cases}$$

### Phase 2: Multi-Class MNIST Classification
TBD.