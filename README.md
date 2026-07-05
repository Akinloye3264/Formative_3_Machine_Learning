# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

**Course:** Mathematics for Machine Learning
**Team members:** Emmanuel Akinloye, Peggy Dusunge, Junior Nkuba, Janviere Munezero

This project covers four topics: separating mixed data with the EM algorithm, classifying sentiment with Bayes' theorem, and running gradient descent both by hand and in code.

---

## Contents
1. [Overview](#overview)
2. [How to run](#how-to-run)
3. [Part 1 — EM Algorithm](#part-1)
4. [Part 2 — Bayesian Probability](#part-2)
5. [Part 3 — Manual Gradient Descent](#part-3)
6. [Part 4 — Gradient Descent in Code](#part-4)
7. [Presentation notes](#notes)

---

## 1. Overview

The work splits into two themes. Parts 1 and 2 are about **probability** — modelling data with distributions and updating beliefs with Bayes' rule. Parts 3 and 4 are about **optimization** — fitting a model to data with gradient descent, first worked out on paper and then implemented in Python.

| Part | Topic | Files |
|------|-------|-------|
| 1 | Expectation-Maximization on a height mixture | `Em_gmm_heights.py` |
| 2 | Bayes' theorem on IMDb reviews | `main.py` |
| 3 | Manual gradient descent (4 iterations) | handwritten PDF |
| 4 | Gradient descent in Python + plots | `Gradient.py` |

Datasets used: `GaltonFamilies.csv` (Part 1) and `IMDB_Dataset.csv` (Part 2). Place both in the same folder as the scripts.

---

## 2. How to run

Python 3.11 or newer with `numpy`, `scipy`, and `matplotlib`:

```bash
pip install numpy scipy matplotlib
```

Then run each part:

```bash
python Em_gmm_heights.py    # Part 1 — prints the tracking table, then asks for a test height
python main.py              # Part 2 — prints the keyword probability table
python Gradient.py          # Part 4 — prints each iteration and opens the two plots
```

---

## 3. Part 1 — EM Algorithm (Gaussian Mixture Model)

### The problem
We are given a set of heights from a room holding two groups — young children and adults — but with no labels saying who is who. Our task is to recover each group's average height, spread, and proportion using only the numbers.

### Why we don't just split at the global mean
Drawing a single line at the overall mean and averaging each side is tempting, but it is the wrong approach for three reasons:

1. It forces a hard decision on the heights near the boundary. A tall child or a short adult sits right in the overlap, and a single cut-off assigns them with false certainty.
2. It assumes both groups have the same spread. They don't — one group is more varied than the other — and treating them as equal distorts the estimated means.
3. It gives one answer and stops. EM instead assigns each point a *probability* of belonging to each group, re-estimates the groups from those probabilities, and repeats until the fit stops improving.

EM's advantage is these **soft assignments**: a point can be 90% child and 10% adult, and that weighting feeds into the next estimate rather than being thrown away.

### How the algorithm works
We model the data as two Gaussians (bell curves). Each is described by a mean (μ), a variance (σ²), and a mixing weight (π, the share of the data in that group). EM finds these by alternating two steps:

**Initialization.** We place the two starting means at the 25th and 75th percentiles of the data, so one curve begins low and one begins high. Both variances start at the overall variance and both mixing weights at 0.5. A fixed random seed keeps the run reproducible.

**E-step.** For each height, we compute the probability it came from each Gaussian using Bayes' rule:

```
responsibility = (pi × Gaussian(x | mu, sigma²)) / (sum over both components)
```

These responsibilities are soft labels between 0 and 1.

**M-step.** Using those responsibilities as weights, we re-estimate each group's mean (weighted average of the points), variance (weighted spread), and mixing weight (the group's share of the responsibility).

**Convergence.** After each round we compute the log-likelihood, a single number measuring how well the two curves explain the data. When it stops increasing, the algorithm has converged and we stop.

### Tracking table
The program prints the parameters at initialization and after each iteration. We report iteration 0, 1, and 2:

| Iter | μ1 (Child) | μ2 (Adult) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|------|-----------|-----------|-----|-----|-----|-----|----------------|
| 0 (Init) | | | | | 0.5 | 0.5 | |
| 1 | | | | | | | |
| 2 | | | | | | | |

*(Values are filled in from the actual program output. The log-likelihood rises each row, confirming the fit is improving.)*

### Live classification
Given a test height, `classify_height()` runs a single E-step and prints the posterior probabilities:

```
P(Child)     = 0.xxxx
P(Adult)     = 0.xxxx
Classified as: CHILD / ADULT
```

A short height returns almost entirely child, a tall one almost entirely adult, and a height in the overlap returns a genuine split — the soft-assignment behaviour that a hard mean split cannot produce.

---

## 4. Part 2 — Bayesian Probability (IMDb Sentiment)

### The problem
From a large set of labelled movie reviews, we ask: if a review contains a particular keyword, how likely is it to be positive? We answer with Bayes' theorem, written in plain Python with no machine-learning libraries.

### What we compute
We compute **P(Positive | keyword)**. Our keywords are:

- Positive: brilliant, masterpiece, wonderful, excellent
- Negative: boring, terrible, waste, awful

Running all eight through the same formula, the positive words score close to 1 and the negative words close to 0, which confirms the method behaves as expected.

### The four probabilities
For each keyword we report:

| Term | Formula | Meaning |
|------|---------|---------|
| Prior | P(Positive) | share of all reviews that are positive |
| Likelihood | P(keyword \| Positive) | share of positive reviews containing the word |
| Marginal | P(keyword) | share of all reviews containing the word |
| Posterior | P(Positive \| keyword) | the answer |

Bayes' theorem combines them:

```
P(Positive | keyword) = ( P(keyword | Positive) × P(Positive) ) / P(keyword)
```

### How the code works
- **`read_records()`** reads the CSV character by character so it correctly handles the commas and quotation marks inside the reviews, which a simple split on commas would mangle.
- **`tokenize()`** lowercases the text, removes HTML line breaks, and returns the set of distinct words in a review, so each word is counted once per review.
- **`build_counts()`** passes over the reviews once, building dictionaries of how many positive reviews and how many total reviews contain each word, along with the totals.
- **`bayes_positive_given_keyword()`** turns those counts into the four probabilities above.
- **`main()`** prints them in an aligned table.

Every probability is a ratio of counts, so no external libraries are needed.

---

## 5. Part 3 — Manual Gradient Descent

### The problem
We fit a two-feature linear model ŷ = m₁x₁ + m₂x₂ + b to two data points by running gradient descent by hand, using matrix multiplication throughout. There are four iterations, one per team member.

### Setup
- Model: ŷ = Xm + b
- Data: X = [[1, 3], [4, 10]], y = [5, 6]
- Initial values: m = [-1, 2], b = 1
- Learning rate: α = 0.001

Here m is a vector because there are two features, while b is a single shared bias.

### One iteration in full
Taking iteration 1 as the worked example:

1. **Predict.** Xm = [(1)(-1)+(3)(2), (4)(-1)+(10)(2)] = [5, 16], then add b: ŷ = [6, 17].
2. **Error.** e = ŷ − y = [1, 11]; cost MSE = (1/n)Σe² = ½(1 + 121) = 61.
3. **Gradient.** Using the chain rule, ∂J/∂m = (2/n)Xᵀe = [45, 113] and ∂J/∂b = (2/n)Σeᵢ = 12.
4. **Update.** Each parameter moves against its gradient: m₁ = -1 − 0.001(45) = -1.045, m₂ = 2 − 0.001(113) = 1.887, b = 1 − 0.001(12) = 0.988.

### Results across the four iterations
| Iter | m₁ | m₂ | b | MSE |
|------|------|------|------|------|
| 0 (start) | -1.000 | 2.000 | 1.000 | 61.00 |
| 1 | -1.045 | 1.887 | 0.988 | 47.01 |
| 2 | -1.084 | 1.788 | 0.978 | 36.37 |
| 3 | -1.119 | 1.702 | 0.969 | 28.26 |
| 4 | -1.149 | 1.627 | 0.962 | 22.09 |

Each member begins from the previous member's output.

### Trend
The error falls steadily from 61 to 22, and every parameter moves smoothly in one direction. So yes, m and b are moving in a direction that reduces the error, which is exactly what gradient descent should do.

---

## 6. Part 4 — Gradient Descent in Code

This turns the manual work into Python and adds the two required plots. The calculation steps are written out plainly rather than hidden inside a library call.

### How the code is organised
- **Data** matches the worksheet: X, Y, m as a vector, b as a scalar, α = 0.001, four iterations.
- **`predict()`** computes `X @ m + b`, using matrix multiplication so both predictions come out at once.
- **`mse()`** returns the mean of the squared errors.
- **`scipy_derivative()`** wraps SciPy's derivative function, satisfying the requirement to compute the derivative with SciPy.
- **`gradient_wrt_m()` and `gradient_wrt_b()`** express the cost as a function of one parameter and hand it to SciPy to get that parameter's slope.
- **The training loop** repeats four times: compute the gradients, update each parameter against its gradient, and record the values and error.
- **The plots** are drawn side by side.

### The two graphs
**m and b over iterations (left).** The x-axis is the iteration and the y-axis is each parameter's value. m1 and m2 drift downward from -1 and 2, while b stays close to 1. The lines are smooth, showing the learning rate is stable.

**MSE over iterations (right).** The error curve falls from 61 to 22. This is the clearest single picture of the model improving, and it answers the trend question directly.

### Checking the code against the hand calculation
On paper we derived ∂J/∂m = (2/n)Xᵀe. SciPy computes the same slope numerically, and the two agree at every step — so the code confirms the hand-worked calculus and the manual and coded results line up exactly.

---

Documentation used for presentation: https://app.notion.com/p/Formative-3-Probability-Distributions-Bayesian-Probability-Gradient-Descent-Documentation-abac50d040fa473c96dfe130680fceff

*Repository contents: `Em_gmm_heights.py`, `main.py`, `Gradient.py`, this README, the handwritten Part 3 PDF, the contributions document, and the Jupyter notebook with all implementations and plots.*