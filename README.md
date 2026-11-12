# Randomized Optimal Stopping in Continuous Time and Reinforcement Learning

A reproducible numerical companion project for Yuchao Dong's paper, *Randomized Optimal Stopping Problem in Continuous Time and Reinforcement Learning Algorithm* (2023). The project studies an American put under the Black–Scholes risk-neutral model and compares three numerical approaches:

1. an implicit finite-difference / policy-iteration solver for the exploratory HJB equation;
2. Longstaff–Schwartz least-squares Monte Carlo for the classical Bermudan put;
3. neural fitted policy iteration following Algorithm 4.1 of Dong (2023).

> **Research note.** The classical American-put benchmark is not the same object as the entropy-regularized exploratory value for positive temperature. As temperature tends to zero, the exploratory formulation approaches the classical stopping problem. Neural results are stochastic estimates and must be reported with independent evaluation samples and uncertainty estimates.

## Mathematical model

Under the risk-neutral measure,
$
 dS_t = rS_t\,dt + \sigma S_t\,dW_t, \qquad g(S)=(K-S)^+.
$

A randomized stopping policy is represented by an intensity $\pi_t\geq 0$, with survival probability satisfying $dp_t=-\pi_t p_t\,dt$. The regularizer is
$
 R(\pi)=\pi-\pi\log\pi, \qquad \lambda>0.
$

Writing the value in homogeneous form as $V(S,p,t)=pV(S,t)$, the HJB equation is
$
 \partial_t V + \mathcal L V + \sup_{\pi\geq0}\{(g-V)\pi+\lambda R(\pi)\}=0,
 \qquad V(S,T)=g(S),
$
where, in log-price coordinates $x=\log S$,
$
 \mathcal L=(r-\tfrac12\sigma^2)\partial_x
 +\tfrac12\sigma^2\partial_{xx}-r.
$

The first-order condition gives the **optimal randomized stopping intensity**
$
 \boxed{\pi^*(S,t)=\exp\!\left(\frac{g(S)-V(S,t)}{\lambda}\right)}.
$
This sign follows by differentiating the Hamiltonian with respect to $\pi$.

## Repository layout

```text
.
├── README.md
├── projet_RL.ipynb
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── finite_difference.py
│   ├── longstaff_schwartz.py
│   ├── reinforcement_learning.py
│   └── simulation.py
└── tests/
    └── test_numerics.py
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/med-rjili/randomized-optimal-stopping-rl.git
cd randomized-optimal-stopping-rl
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,rl]"
```

For only the finite-difference and Longstaff–Schwartz methods, install core dependencies with `pip install -e .`. PyTorch is optional and required only for the neural method.

## Quick start

### Finite differences

```python
from src.finite_difference import solve_exploratory_put

result = solve_exploratory_put(
    s0=40, strike=40, rate=0.06, volatility=0.40,
    maturity=1.0, temperature=0.1,
    space_steps=400, time_steps=400,
)
print("Exploratory value at S0:", result["price"])
```

### Longstaff–Schwartz

```python
from src.longstaff_schwartz import longstaff_schwartz_put

result = longstaff_schwartz_put(
    s0=40, strike=40, rate=0.06, volatility=0.40,
    maturity=1.0, steps=50, paths=100_000, seed=1234,
)
print(result["price"], result["confidence_interval_95"])
```

The regression uses `numpy.linalg.lstsq` on in-the-money paths instead of explicitly inverting the empirical Gram matrix. The convenience function's standard error is conditional on the fitted exercise policy; use an independent sample to evaluate a policy without training-sample reuse.

### Neural policy iteration

```python
from src.reinforcement_learning import train_policy_iteration, evaluate_policy

model = train_policy_iteration(
    s0=40, strike=40, rate=0.06, volatility=0.40,
    maturity=1.0, steps=50, batch_size=1024,
    iterations=500, temperature=0.1, seed=1234,
)
print(evaluate_policy(model, paths=50_000))
```

Install the optional RL dependency first. The implementation uses backward-in-time neural policy evaluation and a discrete survival update. Intensity is capped so that `1 - pi * dt` remains positive; this is a numerical safeguard, not the unconstrained continuous-time first-order condition. Treat this method as experimental: tune hyperparameters, compare multiple seeds and report independent evaluation uncertainty.

## Run tests

```bash
pytest -q
```

The tests are smoke and invariant tests (simulation reproducibility, payoff/basis shape and finite numerical outputs). They do not prove convergence or establish publication-grade error bounds.

## Numerical choices and limitations

- The finite-difference solver uses log-price coordinates, an upwind drift discretization, implicit time stepping and sparse linear solves. The spatial domain is truncated and uses asymptotic put boundary values; perform time/space/domain refinement before drawing convergence conclusions.
- Longstaff–Schwartz uses least-squares regression on in-the-money paths and returns a conditional Monte Carlo standard error. For research-grade estimates, split policy fitting and policy evaluation samples.
- The neural implementation is an experimental finite-step approximation. Its survival update and intensity cap should be stress-tested as the time step and temperature change.
- The original notebook is retained as a research record; the modular Python implementations are the recommended entry points.

## Reproducibility checklist

1. Fix and report model parameters and random seeds.
2. Refine both time and space grids and report domain sensitivity for the PDE.
3. Increase Monte Carlo path counts and use independent training/evaluation samples.
4. Report confidence intervals and results across multiple neural-network seeds.
5. Compare the classical American-put limit with a trusted benchmark while keeping positive-temperature exploratory values separate.

## Reference

Yuchao Dong (2023), “Randomized Optimal Stopping Problem in Continuous Time and Reinforcement Learning Algorithm,” arXiv:2208.02409v3: https://arxiv.org/abs/2208.02409.

Original student project: Aymen Ben Ammar and Mohamed Rejili. Numerical refactoring and documentation build on the cited paper.
