"""Risk-neutral geometric Brownian motion simulation utilities."""
from __future__ import annotations
import numpy as np


def simulate_gbm(s0: float, rate: float, volatility: float, maturity: float,
                 steps: int, paths: int, seed: int | None = None) -> np.ndarray:
    """Simulate exact GBM transitions; return an array shaped (steps+1, paths)."""
    if s0 <= 0 or volatility < 0 or maturity <= 0 or steps < 1 or paths < 1:
        raise ValueError("Require s0>0, volatility>=0, maturity>0, steps>=1, paths>=1")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    z = rng.standard_normal((steps, paths))
    increments = (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
    log_paths = np.vstack([np.full(paths, np.log(s0)), np.log(s0) + np.cumsum(increments, axis=0)])
    return np.exp(log_paths)


def put_payoff(stock: np.ndarray, strike: float) -> np.ndarray:
    """Undiscounted European put payoff at the supplied stock values."""
    if strike <= 0:
        raise ValueError("strike must be positive")
    return np.maximum(strike - np.asarray(stock), 0.0)
