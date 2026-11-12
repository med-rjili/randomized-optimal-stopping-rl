# Randomized Optimal Stopping in Continuous Time and Reinforcement Learning

A numerical study of **optimal stopping for an American put option**, with a focus on randomized stopping rules, entropy regularization, and reinforcement learning.

This repository brings together three complementary numerical approaches:

1. **Finite differences for the exploratory HJB equation** — a deterministic grid-based method for the entropy-regularized randomized stopping problem.
2. **Longstaff–Schwartz least-squares Monte Carlo** — a simulation-based benchmark for the classical Bermudan/American put stopping problem.
3. **Neural fitted policy iteration** — a learning-based approximation of the randomized stopping value and its stopping intensity.

The purpose is not simply to produce three prices. It is to understand how the same financial stopping problem can be approached through a nonlinear partial differential equation, regression on simulated paths, and a neural approximation of a dynamic programming relation.

> **Important distinction:** the positive-temperature exploratory value and the classical American put value are different quantities. Longstaff–Schwartz is included as a classical stopping benchmark; the finite-difference and neural methods target the entropy-regularized randomized formulation. They should not be expected to return identical values when the temperature is positive.

## 1. The financial problem

### 1.1 American put option

An American put gives its holder the right, but not the obligation, to sell an asset for a fixed strike price at any exercise time before maturity.

Let:

- `S_t` be the underlying asset price at time `t`;
- `K` be the strike price;
- `r` be the constant risk-free interest rate;
- `sigma` be the volatility;
- `T` be the maturity.

Under the risk-neutral Black–Scholes model, the asset follows a geometric Brownian motion:

```text
dS_t = r S_t dt + sigma S_t dW_t
```

Here, `W_t` is a standard Brownian motion. The holder's immediate exercise payoff is:

```text
g(S) = max(K - S, 0)
```

The classical optimal stopping problem asks for the exercise time that maximizes the expected discounted payoff. In simplified notation:

```text
V_classical(S,t) = sup over stopping times tau in [t,T]
                   E[ exp(-r (tau-t)) g(S_tau) | S_t = S ]
```

The central difficulty is that the decision to exercise depends on the current asset price, the remaining time, and the expected future value of continuing to hold the option.

### 1.2 From a deterministic exercise time to randomized stopping

In a classical stopping strategy, exercise is described by a stopping time: the option is either exercised or kept alive.

The randomized formulation replaces this all-or-nothing decision by a non-negative **stopping intensity** `pi_t`. Informally, the intensity controls the instantaneous rate at which the option is stopped.

- A low intensity means that the policy tends to continue.
- A high intensity means that the policy tends to stop more quickly.
- The intensity is allowed to depend on the current state and time.

Let `p_t` denote the probability that the option has not yet been stopped. Its continuous-time survival equation is:

```text
dp_t = -pi_t p_t dt
```

Over a short time interval `dt`, the probability mass assigned to stopping is approximately `p_t pi_t dt`, while the remaining survival mass is approximately `p_t (1 - pi_t dt)`.

Randomization makes the control smoother than a strict exercise/continue decision. It also creates a continuous control variable, the intensity, which can be optimized.

### 1.3 Entropy regularization

The randomized problem in the reference paper uses the regularization function:

```text
R(pi) = pi - pi log(pi),       pi > 0
```

The positive parameter `lambda` (also called temperature) controls the strength of the regularization.

The regularizer changes the optimization problem: the controller is not maximizing only the expected exercise payoff. It also receives the regularization contribution associated with the stopping intensity.

The interpretation of the temperature is important:

- A larger temperature gives the regularization term more influence and generally produces a more exploratory, less sharply concentrated stopping policy.
- A smaller positive temperature reduces that influence.
- In the zero-temperature limit, the exploratory formulation is related to the classical optimal stopping problem.

The numerical value at positive temperature must therefore be interpreted as an **exploratory/regularized value**, not automatically as the classical American put price.

## 2. Mathematical formulation

### 2.1 The HJB equation

Write `V(S,t)` for the exploratory value function after using the homogeneous representation with respect to the survival probability.

The risk-neutral Black–Scholes generator, acting on a function of the asset price, is:

```text
L f(S) = r S f_S(S) + 0.5 sigma^2 S^2 f_SS(S) - r f(S)
```

The randomized stopping HJB equation is:

```text
V_t + L V
    + sup_{pi >= 0} { (g(S) - V) pi + lambda R(pi) } = 0

V(S,T) = g(S)
```

The term `g(S) - V(S,t)` compares immediate exercise with continuation:

- If immediate exercise is attractive relative to continuation, the optimal intensity increases.
- If continuation is more attractive, the optimal intensity decreases.

Substituting the entropy regularizer gives the Hamiltonian:

```text
H(V,pi) = (g - V) pi + lambda (pi - pi log(pi))
```

Differentiating with respect to `pi` gives the first-order condition:

```text
dH/dpi = g - V - lambda log(pi) = 0
```

Consequently, the optimal intensity is:

```text
pi*(S,t) = exp( (g(S) - V(S,t)) / lambda )
```

At this intensity, the optimized Hamiltonian is:

```text
sup H(V,pi) = lambda exp( (g(S) - V(S,t)) / lambda )
```

The nonlinear HJB equation can therefore be written as:

```text
V_t + L V + lambda exp((g - V)/lambda) = 0
V(S,T) = g(S)
```

These equations are deliberately written as plain text in code blocks. This avoids dependence on GitHub's mathematical rendering and keeps every symbol readable in the repository.

### 2.2 Log-price coordinates

The finite-difference solver works with the log-price variable:

```text
x = log(S)
mu = r - 0.5 sigma^2
```

In these coordinates, the differential operator becomes:

```text
L V = mu V_x + 0.5 sigma^2 V_xx - r V
```

This transformation is useful because the log of a geometric Brownian motion has constant drift and volatility. It also makes it natural to construct a uniform spatial grid in log-price rather than in asset price.

## 3. Method 1 — Finite differences for the exploratory HJB equation

**Implementation:** `src/finite_difference.py`  
**Main function:** `solve_exploratory_put(...)`

### 3.1 Objective

This method approximates the solution of the nonlinear HJB partial differential equation on a two-dimensional time/space grid.

Unlike Monte Carlo methods, it does not estimate the value by averaging simulated option payoffs. It approximates the value function directly at a collection of asset prices and times.

### 3.2 Discretization

The solver constructs a uniform grid in log-price:

```text
x_i = x_min + i * dx
```

and a uniform time grid:

```text
t_n = n * dt
dt = T / number_of_time_steps
```

The terminal condition is known exactly:

```text
V(S,T) = max(K-S, 0)
```

The algorithm starts from this terminal payoff and moves backward in time to obtain the value at time zero.

The spatial derivatives are approximated by finite differences:

- The second derivative is approximated by a centered difference.
- The drift derivative is approximated with an upwind difference. The direction is selected according to the sign of the log-price drift.
- The discounting term `-r V` is included in the discrete operator.

The upwind treatment of the drift is chosen to improve numerical stability and preserve the monotonicity properties expected from this type of discretization.

### 3.3 Implicit time stepping and nonlinear iteration

At each time step, the solver must handle the nonlinear term:

```text
lambda * exp((g - V)/lambda)
```

It uses an implicit time discretization and a policy/Newton-style iteration.

Given a current guess `V_guess`, it computes the associated intensity:

```text
pi_guess = exp((g - V_guess)/lambda)
```

The exponential term is linearized around the current guess. This produces a sparse linear system for the updated value. The process is repeated until the relative change in the value function is smaller than the prescribed tolerance, or until the maximum number of iterations is reached.

The implementation clips the exponent before evaluating the exponential. This is a numerical protection against overflow; it does not change the mathematical formula being approximated in ordinary ranges.

### 3.4 Boundary conditions

The log-price domain is finite, even though the theoretical asset-price domain is positive and unbounded.

The implementation uses put-option asymptotics at the two ends of the truncated domain:

- At very low asset prices, the put is deep in the money and the boundary is approximated by the discounted strike.
- At very high asset prices, the put payoff and value approach zero.

The quality of the approximation depends on choosing a sufficiently wide spatial domain. A grid refinement study should therefore vary not only the number of grid points but also the domain limits.

### 3.5 Output and interpretation

The function returns:

- the asset-price grid;
- the log-price grid;
- the time grid;
- the computed value at every time/space grid point;
- the interpolated value at the requested initial asset price;
- the temperature used in the calculation.

This method is particularly useful for visualizing the value function and studying how the solution changes with the temperature, volatility, interest rate, and numerical grid.

**Main limitations:** finite differences are affected by spatial truncation, grid resolution, boundary approximations, and nonlinear solver tolerance. A single numerical run is not sufficient to establish convergence.

## 4. Method 2 — Longstaff–Schwartz least-squares Monte Carlo

**Implementation:** `src/longstaff_schwartz.py`  
**Main function:** `longstaff_schwartz_put(...)`

### 4.1 Objective

Longstaff–Schwartz (LSM) is a simulation-based method for estimating the value of an American-style option.

In this repository, it provides a classical Bermudan put benchmark. A Bermudan option can be exercised only on a finite set of exercise dates. Increasing the number of dates makes this exercise schedule a closer approximation to continuous exercise.

This method solves a different numerical problem from the positive-temperature exploratory HJB solver: it estimates a classical exercise policy rather than an entropy-regularized randomized intensity.

### 4.2 Simulating asset-price paths

The algorithm first simulates many risk-neutral paths of the underlying asset.

For a time step `dt`, the exact geometric Brownian motion transition is:

```text
S_(t+dt) = S_t * exp(
    (r - 0.5 sigma^2) dt + sigma sqrt(dt) Z
)
Z ~ Normal(0,1)
```

Using this exact transition avoids introducing an Euler discretization error into the GBM simulation at the exercise dates.

Each simulated path represents one possible evolution of the underlying asset under the risk-neutral model.

### 4.3 Backward estimation of continuation value

At maturity, the value is the terminal put payoff:

```text
payoff = max(K - S_T, 0)
```

The algorithm then works backward through the exercise dates.

At a given date and for each in-the-money path, it estimates the **continuation value**: the conditional expected discounted payoff from keeping the option alive.

The conditional expectation is not known explicitly. LSM approximates it by a regression on functions of the current asset price.

The current implementation uses a polynomial basis in moneyness:

```text
x = S / K

basis(x) = [1, x, x^2, ..., x^degree]
```

The regression coefficients are estimated by ordinary least squares. The code uses `numpy.linalg.lstsq`, rather than explicitly inverting a matrix, which is generally more numerically robust.

### 4.4 Exercise decision

Once the continuation estimate has been computed, the algorithm compares it with immediate exercise:

```text
if immediate_payoff >= estimated_continuation:
    exercise
else:
    continue
```

If exercise is selected, the path's cash flow and stopping date are updated. Otherwise, the previously determined future cash flow is retained.

After processing all exercise dates backward, each path has one estimated exercise date and one discounted payoff. The option value is estimated by averaging these discounted payoffs.

### 4.5 Uncertainty and limitations

The implementation reports:

- the estimated option value;
- a Monte Carlo standard error;
- a nominal 95% interval based on that standard error;
- the stopping dates and discounted path payoffs.

The reported standard error measures the dispersion of the discounted payoffs for the fitted policy on the simulated sample. Because the same paths are used to fit and evaluate the exercise rule, this interval should not be treated as a fully independent out-of-sample confidence interval. For a more rigorous study, fit the exercise policy on one set of paths and evaluate it on a separate, independent set.

LSM is sensitive to the number of paths, the number of exercise dates, and the regression basis. Too few paths or an unsuitable basis can lead to a poor continuation estimate.

## 5. Method 3 — Neural fitted policy iteration

**Implementation:** `src/reinforcement_learning.py`  
**Main functions:** `train_policy_iteration(...)` and `evaluate_policy(...)`  
**Dependency:** PyTorch (optional)

### 5.1 Objective

The third method uses a neural network to approximate the exploratory value function and derive a randomized stopping intensity.

The motivation is to avoid representing the value function only on a fixed spatial grid. A neural network can instead learn a function of the state and time from simulated samples.

The implementation follows a backward-in-time fitted policy-iteration structure inspired by Algorithm 4.1 in Dong (2023). It is an experimental numerical implementation, not a guarantee of convergence for arbitrary parameters.

### 5.2 Neural representation

A separate neural network is created for each time step.

The network receives three normalized inputs:

- the asset price divided by the strike, `S/K`;
- the immediate payoff divided by the strike, `g(S)/K`;
- the normalized time coordinate.

The network predicts a correction to the immediate payoff. Its output is rescaled by the strike to obtain the approximate value:

```text
V_theta(S,t) = K * NN_theta(S/K, g(S)/K, t/T)
```

This parameterization provides a useful payoff-based baseline while allowing the network to learn the difference between immediate exercise and the exploratory value.

### 5.3 Backward fitted policy iteration

The algorithm generates batches of risk-neutral GBM paths. It then visits the time steps backward, from the last time step to the first.

At each time step:

1. The network estimates the current value from the current simulated states.
2. The optimal-intensity formula is applied to this estimated value:

```text
pi = exp((g - V_theta)/lambda)
```

3. The intensity is used to form a one-step target consisting of:
   - the immediate exercise reward weighted by the stopping intensity;
   - the entropy-regularization reward;
   - the discounted next-step value weighted by the probability of surviving the time step.
4. The network is trained to reduce the squared difference between its current value and this target.
5. The procedure continues backward through the remaining time steps.

In the discrete implementation, the one-step target has the form:

```text
target =
    g * pi * dt
    + lambda * (pi - pi log(pi)) * dt
    + exp(-r dt) * V_next * (1 - pi dt)
```

This is a time-discretized approximation of the continuous-time reward and survival relation. The network parameters are optimized with Adam, and gradient clipping is used to limit excessively large updates.

### 5.4 Intensity cap: an implementation safeguard

The continuous-time intensity is non-negative and has no fixed upper bound.

However, the discrete survival approximation used in this implementation is:

```text
survival_next = survival_now * (1 - pi * dt)
```

To prevent a negative survival factor, the implementation caps the intensity so that:

```text
0 < pi * dt < 1
```

This cap is a numerical safeguard for the chosen discrete update. It is not part of the unconstrained continuous-time optimal-intensity formula. Its effect should be assessed when changing the time step or temperature.

### 5.5 Independent policy evaluation

After training, `evaluate_policy(...)` generates a new set of GBM paths using a separate random seed.

For each path, it applies the learned intensity, accumulates the discounted randomized exercise and regularization rewards, updates the survival probability, and adds the discounted terminal payoff of the paths that survive to maturity.

The function returns the average estimated reward and its Monte Carlo standard error.

Using new paths for evaluation is important: it gives a more meaningful assessment of the learned policy than reporting only the training loss.

### 5.6 Limitations

The neural method is sensitive to the number of training paths, batch size, number of training iterations, network width, learning rate, temperature, time discretization, and random seed.

The training batches are simulated from the specified initial asset price. Consequently, the distribution of states seen during training may not cover every state at which the network is later evaluated.

A decreasing training loss does not by itself demonstrate that the learned policy is accurate or that the method has converged. Research-grade comparisons should use multiple seeds, independent evaluation paths, parameter sensitivity tests, and comparisons with the finite-difference solution.

## 6. How the three methods fit together

| Method | Problem approximated | Main numerical tool | Main output |
|---|---|---|---|
| Finite differences | Entropy-regularized randomized stopping | Nonlinear HJB on a grid | Exploratory value function |
| Longstaff–Schwartz | Classical Bermudan stopping | GBM simulation and regression | Classical option-price estimate |
| Neural policy iteration | Entropy-regularized randomized stopping | Neural networks and fitted targets | Learned value approximation and policy reward |

The finite-difference and neural approaches target the same exploratory formulation, but use different approximations:

- Finite differences approximate the HJB solution on a structured grid.
- Neural policy iteration approximates the value through learned functions on simulated states.

Longstaff–Schwartz is a complementary classical benchmark. It is useful for checking the scale of the classical put value and for understanding the exercise/continuation decision, but it does not directly validate the positive-temperature exploratory value.

A meaningful numerical comparison should therefore be organized as follows:

1. Compare finite differences and neural policy iteration at the same positive temperature and with consistent model parameters.
2. Compare Longstaff–Schwartz with a classical American/Bermudan put benchmark.
3. Study the exploratory formulation as the temperature decreases, while checking numerical stability and convergence.
4. Do not interpret agreement between two numbers as proof of correctness without grid refinement, independent simulation, and uncertainty analysis.

## 7. Repository structure

```text
.
├── README.md
├── projet_RL.ipynb
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── simulation.py
│   ├── finite_difference.py
│   ├── longstaff_schwartz.py
│   └── reinforcement_learning.py
└── tests/
    └── test_numerics.py
```

- `src/simulation.py`: risk-neutral GBM simulation and put payoff.
- `src/finite_difference.py`: finite-difference HJB solver.
- `src/longstaff_schwartz.py`: least-squares Monte Carlo.
- `src/reinforcement_learning.py`: neural fitted policy iteration and evaluation.
- `projet_RL.ipynb`: notebook demonstrating the methods.
- `tests/test_numerics.py`: smoke tests and basic numerical invariants.

## 8. Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/med-rjili/randomized-optimal-stopping-rl.git
cd randomized-optimal-stopping-rl

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev,rl]"
```

The `rl` extra installs PyTorch for the neural method. To install only the core numerical dependencies and development tools, use:

```bash
pip install -e ".[dev]"
```

## 9. Quick start

### Finite differences

```python
from src.finite_difference import solve_exploratory_put

result = solve_exploratory_put(
    s0=40,
    strike=40,
    rate=0.06,
    volatility=0.40,
    maturity=1.0,
    temperature=0.1,
    space_steps=400,
    time_steps=400,
)

print("Exploratory value:", result["price"])
```

### Longstaff–Schwartz

```python
from src.longstaff_schwartz import longstaff_schwartz_put

result = longstaff_schwartz_put(
    s0=40,
    strike=40,
    rate=0.06,
    volatility=0.40,
    maturity=1.0,
    steps=50,
    paths=100_000,
    seed=1234,
)

print("Bermudan estimate:", result["price"])
print("95% interval:", result["confidence_interval_95"])
```

### Neural policy iteration

```python
from src.reinforcement_learning import (
    train_policy_iteration,
    evaluate_policy,
)

model = train_policy_iteration(
    s0=40,
    strike=40,
    rate=0.06,
    volatility=0.40,
    maturity=1.0,
    steps=50,
    batch_size=1024,
    iterations=500,
    temperature=0.1,
    seed=1234,
)

evaluation = evaluate_policy(
    model,
    s0=40,
    strike=40,
    rate=0.06,
    volatility=0.40,
    maturity=1.0,
    steps=50,
    paths=50_000,
    seed=9876,
)

print(evaluation)
```

## 10. Tests

Run the test suite from the repository root:

```bash
pytest -q
```

The tests check reproducibility and shape of GBM simulation, payoff and basis construction, and basic finite numerical outputs from the Longstaff–Schwartz and finite-difference routines.

These are smoke tests and numerical sanity checks. They do not prove theoretical convergence, establish error bounds, or validate the neural method's performance.

## 11. Recommended numerical experiments

For a careful study, the following experiments are recommended:

1. **Finite-difference convergence:** increase the number of spatial and time steps and widen the log-price domain. Check whether the value at the initial asset price stabilizes.
2. **Temperature sensitivity:** repeat the exploratory calculation for several positive temperatures. Track both the value and the optimal intensity.
3. **Monte Carlo stability:** increase the number of paths and the number of exercise dates in Longstaff–Schwartz. Change the polynomial degree and assess sensitivity.
4. **Out-of-sample evaluation:** fit an LSM policy on one sample and evaluate it on another independent sample. Keep neural training and evaluation paths separate.
5. **Neural seed sensitivity:** train with several random seeds and report the distribution of evaluation rewards, not only the best run.
6. **Cross-method comparison:** compare the HJB and neural methods at the same temperature and model parameters. Use Longstaff–Schwartz as a classical benchmark, not as a direct positive-temperature target.

## 12. Reference and authorship

**Reference paper**

Yuchao Dong (2023), *Randomized Optimal Stopping Problem in Continuous Time and Reinforcement Learning Algorithm*, arXiv:2208.02409v3.

https://arxiv.org/abs/2208.02409

**Project author:** Mohamed Rejili.

This repository provides a numerical implementation and an educational comparison of methods inspired by the reference paper.
