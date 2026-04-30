# Monte Carlo & Simulation — ENSAE Paris M1

A comprehensive Jupyter notebook covering Monte Carlo methods, simulation techniques, and Bayesian inference, completed as part of the M1 curriculum at ENSAE Paris.

---

## Quick Setup

```bash
git clone https://github.com/ralphnader/Monte-Carlo-exercises.git
cd Monte-Carlo-exercises
pip install numpy matplotlib scipy
jupyter notebook tutorial.ipynb
```

---

## What's Inside

The notebook is structured as 13 exercises, each tackling a different simulation or inference problem.

### 1 · RANDU — A Broken Random Number Generator
Analysis of the infamous RANDU linear congruential generator. We prove analytically that the triples $(u_{i-2}, u_i{i-1}, u_i)$ lie on exactly **15 parallel hyperplanes**, then verify it empirically via 3D scatter plots and bit-pattern analysis. Compared against Python's Mersenne Twister.

**Key result:** $u_i = 6u_{i-1} - 9u_{i-2} + k$, $k \in \{-5, \ldots, 9\}$

---

### 2 · Laplace Distribution & Rejection Sampling
- Inverse CDF derivation and implementation of a Laplace generator
- Rejection sampler for $\mathcal{N}(0,1)$ using Laplace as proposal
- Optimal constant $c = \sqrt{2e/\pi} \approx 1.316$, acceptance rate $\approx 76\%$
- Discussion of why sampling Laplace from Normal is impossible (tail behaviour)

---

### 3 · Improved Box-Muller (Polar Method)
Proof that the polar Box-Muller algorithm produces exact $\mathcal{N}(0,1)$ samples via a Brownian bridge argument. Performance comparison against standard Box-Muller (trigonometric vs rejection cost).

---

### 4 · Geometric Distribution
Three sampling algorithms compared:
| Method | Complexity |
|--------|-----------|
| Direct Bernoulli simulation | $O(1/p)$ |
| Inverse CDF: $\lceil \log U / \log(1-p) \rceil$ | $O(1)$ |
| Exponential flooring | $O(1)$ |

**Winner:** Inverse CDF — vectorized and exact.

---

### 5 · Variance Reduction: Control Variates, Antithetic Variables, QMC
Computing $P(X \in A)$ where $X \sim \mathcal{N}_d(0, I_d)$ and $A = \{|\prod x_i| \leq c\}$.

- **Control variate:** $Z = |\prod X_i|$ with known mean $(\sqrt{2/\pi})^d$
- **Antithetic variables:** symmetry of the indicator — works but gives zero gain since $f(X) = f(-X)$ exactly
- **QMC (Sobol sequences):** faster convergence for smooth integrands; limited gain for discontinuous indicators

---

### 6 · Importance Sampling
Critique of a naive adaptive IS scheme (circular dependency). Fix via **two-batch adaptive IS**: use one batch to estimate the proposal $\mathcal{N}(\hat\mu, \hat\Sigma)$, a fresh batch to compute self-normalized weights.

---

### 7 · Metropolis-Hastings on a Bivariate Gaussian
Target: $\mathcal{N}_2(0, \Sigma_\rho)$ with correlation $\rho$.

- **Gibbs sampler:** exact conditionals, poor mixing as $\rho \to 1$
- **RWHM:** effect of step size $\tau$ — optimal acceptance rate ~23%
- **Key insight:** isotropic proposal fails for $\rho \approx 1$; proposal $\propto \Sigma$ fixes mixing

---

### 8 · Ising Model on a Grid
Gibbs sampler for $p(x) \propto \exp\{\alpha \sum x_i + \beta \sum_{i\sim j} \mathbf{1}[x_i=x_j]\}$.

- Full conditional: $x_i | x_{-i} \sim \text{Bernoulli}(\sigma(\alpha + \beta(n_1 - n_0)))$
- Mixing diagnostics: magnetization traces, autocorrelation, grid snapshots
- **Critical value:** $\beta_c = \ln(1+\sqrt{2}) \approx 0.88$ — phase transition beyond which mixing collapses

---

### 9 · Hierarchical Bayesian Model
Model: $y_i|\theta_i \sim \text{Bin}(n_i, \theta_i)$, $\theta_i \sim \text{Beta}(\alpha,\beta)$, $\alpha,\beta \sim \text{Gamma}(a,b)$.

- $\theta_i | \text{rest} \sim \text{Beta}(y_i+\alpha, n_i-y_i+\beta)$ — conjugate
- $\alpha, \beta | \text{rest}$ — non-standard, MH within Gibbs
- **Debugging trick:** freeze variables one at a time to isolate bugs (single-component debugging)

---

### 10 · Gaussian Mixture Models
Gibbs sampler with latent variables $z_i$ for $K$-component mixture $\frac{1}{K}\sum \phi(y;\mu_k,1)$.

- **Label switching:** when means are close, the posterior has $K!$ symmetric modes
- Illustrated by scatter plots of $(\mu_1, \mu_2)$ under different separation regimes
- Extended to unknown variances (Inverse-Gamma prior) and unknown weights (Dirichlet prior)

---

### 11 · Discretization Error in Option Pricing
Black-Scholes European put: $V = E[e^{-rT}(K-S(T))^+]$.

- **Q11.1:** Brownian bridge $W_t | W_a, W_b \sim \mathcal{N}\!\left(\frac{(b-t)W_a+(t-a)W_b}{b-a}, \frac{(t-a)(b-t)}{b-a}\right)$
- **Q11.2:** Direct MC estimate using closed-form $S(T)$
- **Q11.3:** Adaptive Euler-Maruyama refinement — recycle simulations via Brownian bridge, stop when consecutive confidence intervals overlap

---

### 12 · Cross-Entropy Optimization
CE algorithm applied to the **Rosenbrock function** $S(x) = \sum [100(x_{i+1}-x_i^2)^2 + (x_i-1)^2]$.

At each iteration: sample from $\mathcal{N}(\mu, \Sigma)$, keep elite fraction $\rho$, refit distribution. Exponential convergence toward $x^* = (1,\ldots,1)$, tested for $d \in \{2, 5, 10\}$.

---

### 13 · ABC for Exponential Random Graph Models
Network model $p(x|\theta) \propto \exp\{\theta^T S(x)\}$ with $S(x) = (\text{edges},\ \#\text{nodes with deg}\geq 3)$.

- **Q13.1:** Gibbs sampler for edge updates — $P(x_{ij}=1|\text{rest}) = \sigma(\theta^T \Delta S_{ij})$
- **Q13.2:** ABC-SMC to estimate $\theta$ — progressively tightening $\varepsilon$ sequence

**Key limitation:** ABC struggles to identify $\theta_2$ because $S_2$ is not sufficiently informative — illustrates why summary statistic choice is critical in ABC.

---

## Key Takeaways

- Rejection sampling requires the proposal to have **heavier tails** than the target
- Gibbs samplers mix poorly when parameters are highly correlated — adapt the proposal
- Variance reduction methods (CV, QMC) work best for **smooth** integrands
- ABC is exact-free but summary statistics must be **informative** — poor statistics = unidentifiable parameters
- The critical temperature in Ising ($\beta_c \approx 0.88$) marks a phase transition where MCMC mixing collapses
