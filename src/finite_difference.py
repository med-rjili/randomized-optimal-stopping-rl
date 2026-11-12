"""Implicit finite-difference solver for the exploratory American put HJB."""
from __future__ import annotations
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from .simulation import put_payoff


def solve_exploratory_put(*, s0: float = 40.0, strike: float = 40.0,
                          rate: float = 0.06, volatility: float = 0.40,
                          maturity: float = 1.0, temperature: float = 0.1,
                          x_min: float | None = None, x_max: float | None = None,
                          space_steps: int = 400, time_steps: int = 400,
                          max_policy_iterations: int = 100,
                          tolerance: float = 1e-9) -> dict[str, np.ndarray | float]:
    r"""Solve V_t + L V + lambda exp((g-V)/lambda)=0 backward in time.

    x=log(S), L=(r-sigma^2/2)d_x + sigma^2/2 d_xx - r.
    A monotone upwind drift discretization and implicit time stepping are used.
    Boundary values use limiting put asymptotics on a truncated domain.
    """
    if temperature <= 0 or volatility <= 0 or strike <= 0 or s0 <= 0:
        raise ValueError("temperature, volatility, strike and s0 must be positive")
    if maturity <= 0 or space_steps < 3 or time_steps < 1:
        raise ValueError("invalid maturity or grid sizes")
    x_min = float(np.log(strike) - 4.0 if x_min is None else x_min)
    x_max = float(np.log(max(s0, strike)) + 3.0 if x_max is None else x_max)
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    x = np.linspace(x_min, x_max, space_steps + 1)
    dx, dt = x[1] - x[0], maturity / time_steps
    mu, var = rate - 0.5 * volatility**2, volatility**2
    if mu >= 0:
        lower = var / (2 * dx**2) - mu / dx
        upper = var / (2 * dx**2)
        main = -var / dx**2 + mu / dx - rate
    else:
        lower = var / (2 * dx**2)
        upper = var / (2 * dx**2) + mu / dx
        main = -var / dx**2 - mu / dx - rate
    n = space_steps + 1
    L = diags([lower*np.ones(n-1), main*np.ones(n), upper*np.ones(n-1)],
              offsets=[-1, 0, 1], format="lil")
    L[0, :] = 0; L[0, 0] = 1
    L[-1, :] = 0; L[-1, -1] = 1
    L = L.tocsr()
    A = (eye(n, format="csr") / dt - L).tolil()
    A[0, :] = 0; A[0, 0] = 1
    A[-1, :] = 0; A[-1, -1] = 1
    A = A.tocsr()
    stock = np.exp(x)
    payoff = put_payoff(stock, strike)
    values = np.empty((time_steps + 1, n), dtype=float)
    values[-1] = payoff
    for k in range(time_steps - 1, -1, -1):
        tau = maturity - k * dt
        boundary = np.array([strike*np.exp(-rate*tau), 0.0])
        guess = values[k+1].copy()
        for _ in range(max_policy_iterations):
            intensity = np.exp(np.clip((payoff - guess) / temperature, -60.0, 60.0))
            mat = (A + diags(intensity, 0, format="csr")).tolil()
            rhs = values[k+1] / dt + temperature*intensity + intensity*guess
            mat[0, :] = 0; mat[0, 0] = 1; rhs[0] = boundary[0]
            mat[-1, :] = 0; mat[-1, -1] = 1; rhs[-1] = boundary[1]
            updated = spsolve(mat.tocsr(), rhs)
            err = np.max(np.abs(updated - guess)) / max(1.0, np.max(np.abs(updated)))
            guess = updated
            if err < tolerance:
                break
        values[k] = guess
    return {"stock": stock, "log_stock": x, "times": np.linspace(0, maturity, time_steps+1),
            "values": values, "price": float(np.interp(np.log(s0), x, values[0])),
            "temperature": float(temperature)}
