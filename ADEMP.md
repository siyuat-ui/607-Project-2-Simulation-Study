# Simulation Study: Engression-Based Synthetic Data Generation

## ADEMP Framework

### A - Aims

The [Engression](https://arxiv.org/abs/2307.00835) paper, published in JRSSB, inspired a new method for synthetic data generation, although its primary focus is not on this topic. This github repo applies the engression idea to synthetic data generation, trying to address questions including:

1. Can engression-based neural networks effectively learn to generate samples from different probability distributions?
2. How does the quality of generated samples vary across different distribution types (normal, exponential, uniform, lognormal, chi-square)?
3. How does sample size affect the quality of distribution learning and sample generation?

We first introduce engression-based synthetic data generation. The goal is to construct a neural network $g$ such that

$$
g(\epsilon) \overset{d}{=} X,
$$

where $X \in \mathbb{R}^{d}$ is the random vector we would like to generate, and $\epsilon$ is an easy-to-generate $p$-dimensional random vector independent of $X$. The notation $\overset{d}{=}$ means "equal in distribution".

The population version of the loss function is given by

$$
\mathcal{L}_{engression}(g) = \mathbb{E} \left[ \lVert X - g(\epsilon) \rVert_2 - \frac{1}{2} \lVert g(\epsilon) - g(\epsilon^\prime) \rVert_2 \right],
$$

where $\epsilon^\prime$ is an independent copy of $\epsilon$.

This loss function can be justified as follows. Suppose there exists a network $\tilde{g}$ such that $\tilde{g}(\epsilon) \overset{d}{=} X$, then we have

$$
\tilde{g} \in \mathop{\arg\min}_{g} ~ \mathcal{L}_{engression}(g).
$$

In fact, for any $g$ with $\mathbb{P}\left\{ g(\epsilon) \neq \tilde{g}(\epsilon) \right\} > 0$, it holds that $\mathcal{L}_{engression}(\tilde{g}) < \mathcal{L}_{engression}(g)$.

Given a training set $\{X_i\}_{i=1}^n$, where $X_1, \ldots, X_n \overset{i.i.d.}{\sim} X$, we can define the empirical version of engression as

$$
\hat{g} \in \mathop{\mathrm{argmin}}_{g \in \mathcal{NN}} ~ \hat{\mathcal{L}}_{engression}(g),
$$

where $\mathcal{NN}$ is a class of neural networks, and the empirical loss function is given by

$$
\hat{\mathcal{L}}_{engression}(g) = \frac{1}{n} \sum_{i=1}^n \left[ \frac{1}{m} \sum_{j=1}^m \lVert X_i - g(\epsilon_{i,j}) \rVert_2 - \frac{1}{2m(m-1)} \sum_{j=1}^{m} \sum_{j\prime =1}^{m} \lVert g(\epsilon_{i,j}) - g(\epsilon_{i,j\prime}) \rVert_2 \right].
$$

Note that for each $X_i$, we generate $m$ variants $\left\{ g(\epsilon_{i,j}) \right\}_{j=1}^m$ to ensure that the empirical loss provides a good estimate of the population loss.

For simplicity, we consider only the case $d=1$ in this simulation study; that is, $X$ is an one-dimensional random variable.

### D - Data-Generating Mechanisms

**Data-Generating Processes:**

We evaluate five distinct probability distributions:

1. **Normal Distribution**: $\mathcal{N}(\mu=0, \sigma^2=1)$
   - Symmetric, unimodal
   - Standard Gaussian baseline
   
2. **Exponential Distribution**: $\text{Exp}(\lambda = 1)$
   - Right-skewed (skewness = 2)
   - Models waiting times, heavy right tail
   
3. **Uniform Distribution**: $\text{Unif}(a=0, b=2)$
   - Bounded support $[0, 2]$
   - Zero skewness, finite range
   
4. **Lognormal Distribution**: $\text{LogNormal}(\mu=0, \sigma^2=1)$
   - Right-skewed, positive values only
   - Mean = $\exp(0.5) \approx 1.65$, Median = $\exp(0) = 1$
   
5. **Chi-Square Distribution**: $\chi^2({\text{df}=5})$
   - Right-skewed (skewness = $\sqrt{8/5} \approx 1.26$)
   - Mean = $5$, Variance = $10$

**Simulation Factors:**

| Factor | Levels | Description |
|--------|--------|-------------|
| Distribution Type | 5 | Normal, Exponential, Uniform, Lognormal, Chi-Square |
| Sample Size (n) | 3 | 100, 500, 1000 |
| Replications | 10 | Independent runs per configuration |

**Total Experiments:** 5 distributions $\times$ 3 sample sizes $\times$ 10 replications = 150 experiments

### E - Estimands/Targets

**Target Quantity:**
The goal is to learn a mapping $g: \mathbb{R}^d \rightarrow \mathbb{R}^p$ where:
- $d = 128$ (dimension of input noise $\epsilon \sim \mathcal{N}(0, I_{d})$)
- $p$ = dimension of data $X$ ($p = 1$ for univariate distributions)
- $g(\epsilon) \approx X$ in distribution

**Specific Estimands:**
1. **Distributional Match**: Does $g(\epsilon) \sim P^\ast$ where $P^\ast$ is the true data distribution?
2. **Moment Preservation**: $\mathbb{E}[g(\epsilon)] \approx \mathbb{E}[X]$ and $\text{Var}(g(\epsilon)) \approx \text{Var}(X)$
3. **Shape Preservation**: Does $g(\epsilon)$ preserve skewness, kurtosis, and tail behavior?

### M - Methods

**Engression Neural Network Architecture:**
- **Input layer**: 128-dimensional standard Gaussian noise $\epsilon \sim N(0, I_{128})$
- **Hidden layers**: 3 fully connected layers with 64 units each
- **Activation**: ReLU
- **Output layer**: 1-dimensional (matching data dimension)
- **Total parameters**: $\approx 12,800$

**Training Procedure:**
- **Loss function**: Engression loss = $\mathbb{E} \left[ \lvert X - g(\epsilon) \rvert - \frac{1}{2} \lvert g(\epsilon) - g(\epsilon^\prime) \rvert \right]$
  - Term 1: Distance from data to generated samples (reconstruction)
  - Term 2: Diversity penalty to avoid mode collapse
- **Optimizer**: Adam with learning rate 1e-4
- **Batch size**: 128
- **Maximum epochs**: 200
- **Early stopping**: Patience of 20 epochs (stops if no improvement)
- **Samples per batch**: m = 50 epsilon samples per data point

**Sample Generation:**
After training, generate $k=1000$ samples by:
1. Sample $\epsilon_1, \ldots, \epsilon_k \sim N(0, I_{128})$ independently
2. Compute $X_i = g(\epsilon_i)$ for $i = 1, \ldots, k$
3. Return $\{ X_1, \ldots, X_k \}$ as generated samples

### P - Performance Measures

**Primary Metrics:**

1. **Maximum Mean Discrepancy (MMD)**
   - Measures distance between distributions in RKHS
   - Uses RBF kernel with $\gamma = 1.0$
   - **Target**: MMD < 0.1 (good match), MMD < 0.05 (excellent match)
   - Lower is better (0 = perfect match)

2. **Two-Sample Test p-values**
   - Kolmogorov-Smirnov test for each dimension
   - Reports: min p-value, mean p-value, number of rejections at $\alpha = 0.05$
   - **Target**: Mean p-value > 0.05 (fail to reject $H_0$: same distribution)
   - Higher p-values indicate better match

3. **Mean Distance**
   - $L_2$ distance between sample means: $\vert \hat{\mu}_x - \hat{\mu}_g \vert$
   - **Target**: < 0.1 for standardized data
   - Measures first moment preservation

**Secondary Metrics:**

4. **Covariance Distance**
   - Frobenius norm: $\lvert \widehat{\text{Var}(X)} - \widehat{\text{Var}(g(\epsilon))} \rvert$
   - Measures second moment preservation

5. **Training Efficiency**
   - Training time (seconds)
   - Number of epochs until convergence
   - Final loss value

## Simulation Design Matrix

| Condition | Distribution | Parameters | Sample Size | Replications |
|-----------|--------------|------------|-------------|--------------|
| 1 | Normal | $\mu=0, \sigma^2=1$ | 100 | 10 |
| 2 | Normal | $\mu=0, \sigma^2=1$ | 500 | 10 |
| 3 | Normal | $\mu=0, \sigma^2=1$ | 1000 | 10 |
| 4 | Exponential | $\lambda=1$ | 100 | 10 |
| 5 | Exponential | $\lambda=1$ | 500 | 10 |
| 6 | Exponential | $\lambda=1$ | 1000 | 10 |
| 7 | Uniform | $a=0, b=2$ | 100 | 10 |
| 8 | Uniform | $a=0, b=2$ | 500 | 10 |
| 9 | Uniform | $a=0, b=2$ | 1000 | 10 |
| 10 | Lognormal | $\mu=0, \sigma^2=1$ | 100 | 10 |
| 11 | Lognormal | $\mu=0, \sigma^2=1$ | 500 | 10 |
| 12 | Lognormal | $\mu=0, \sigma^2=1$ | 1000 | 10 |
| 13 | Chi-Square | $\text{df}=5$ | 100 | 10 |
| 14 | Chi-Square | $\text{df}=5$ | 500 | 10 |
| 15 | Chi-Square | $\text{df}=5$ | 1000 | 10 |

**Total**: 15 unique conditions × 10 replications = 150 experiments