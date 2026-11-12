"""Longstaff-Schwartz least-squares Monte Carlo for Bermudan put options."""
from __future__ import annotations
import numpy as np
from .simulation import simulate_gbm, put_payoff


def polynomial_basis(stock: np.ndarray, strike: float, degree: int = 3) -> np.ndarray:
    """Scaled polynomial basis in moneyness, including a constant."""
    x = np.asarray(stock, dtype=float) / strike
    return np.column_stack([x**j for j in range(degree + 1)])


def longstaff_schwartz_put(*, s0: float = 40.0, strike: float = 40.0,
                           rate: float = 0.06, volatility: float = 0.40,
                           maturity: float = 1.0, steps: int = 50,
                           paths: int = 100_000, degree: int = 3,
                           seed: int = 1234) -> dict[str, float | np.ndarray]:
    """Estimate a Bermudan put value using OLS on in-the-money paths.

    The reported Monte Carlo standard error is conditional on the fitted
    exercise policy; use an independent sample for final policy evaluation.
    """
    if steps < 1 or paths < 2 or degree < 0:
        raise ValueError("invalid steps, paths or polynomial degree")
    paths_s = simulate_gbm(s0, rate, volatility, maturity, steps, paths, seed)
    dt = maturity / steps
    cashflow = put_payoff(paths_s[-1], strike)
    stopping = np.full(paths, steps, dtype=int)
    for t in range(steps - 1, 0, -1):
        immediate = put_payoff(paths_s[t], strike)
        itm = immediate > 0
        if not np.any(itm):
            continue
        continuation_realized = cashflow * np.exp(-rate * dt * (stopping - t))
        X = polynomial_basis(paths_s[t, itm], strike, degree)
        beta, *_ = np.linalg.lstsq(X, continuation_realized[itm], rcond=None)
        continuation = polynomial_basis(paths_s[t], strike, degree) @ beta
        exercise = itm & (immediate >= continuation)
        cashflow[exercise] = immediate[exercise]
        stopping[exercise] = t
    discounted = cashflow * np.exp(-rate * dt * stopping)
    se = discounted.std(ddof=1) / np.sqrt(paths)
    return {"price": float(discounted.mean()), "standard_error": float(se),
            "confidence_interval_95": np.array([discounted.mean()-1.96*se, discounted.mean()+1.96*se]),
            "stopping_times": stopping, "discounted_payoffs": discounted}
