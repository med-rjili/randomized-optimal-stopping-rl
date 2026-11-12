"""Neural fitted policy iteration following Dong (2023), Algorithm 4.1.

Research implementation; convergence is not guaranteed for arbitrary settings.
"""
from __future__ import annotations
import numpy as np


def train_policy_iteration(*, s0: float = 40.0, strike: float = 40.0,
                           rate: float = 0.06, volatility: float = 0.40,
                           maturity: float = 1.0, steps: int = 50,
                           batch_size: int = 1024, iterations: int = 500,
                           temperature: float = 0.1, learning_rate: float = 5e-4,
                           hidden_width: int = 21, seed: int = 1234,
                           max_intensity: float | None = None) -> dict:
    """Train a backward-in-time neural policy iteration scheme."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'rl' extra: pip install .[rl]") from exc
    if temperature <= 0 or steps < 1 or batch_size < 2 or iterations < 1:
        raise ValueError("invalid training parameters")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dt = maturity / steps
    intensity_cap = (0.95 / dt) if max_intensity is None else float(max_intensity)
    if intensity_cap <= 0 or intensity_cap * dt >= 1:
        raise ValueError("max_intensity must satisfy 0 < max_intensity*dt < 1")

    class ValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, hidden_width), nn.ReLU(),
                                     nn.Linear(hidden_width, hidden_width), nn.ReLU(),
                                     nn.Linear(hidden_width, 1))
        def forward(self, s_scaled, payoff_scaled, tau_scaled):
            return payoff_scaled + self.net(torch.cat([s_scaled, payoff_scaled, tau_scaled], dim=1))

    networks, optimizers = [], []
    for _ in range(steps):
        net = ValueNet()
        networks.append(net)
        optimizers.append(torch.optim.Adam(net.parameters(), lr=learning_rate))
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(iterations):
        z = rng.standard_normal((steps, batch_size))
        log_s = np.log(s0) + np.cumsum((rate - 0.5*volatility**2)*dt + volatility*np.sqrt(dt)*z, axis=0)
        stock = np.vstack([np.full(batch_size, s0), np.exp(log_s)])
        batch_losses = []
        for t in range(steps-1, -1, -1):
            s = torch.as_tensor(stock[t], dtype=torch.float32).reshape(-1, 1)
            s_next = torch.as_tensor(stock[t+1], dtype=torch.float32).reshape(-1, 1)
            g = torch.clamp(strike - s, min=0.0)
            g_next = torch.clamp(strike - s_next, min=0.0)
            tau = torch.full_like(s, t/steps)
            tau_next = torch.full_like(s, (t+1)/steps)
            net, opt = networks[t], optimizers[t]
            v = strike * net(s/strike, g/strike, tau)
            with torch.no_grad():
                v_next = g_next if t == steps-1 else strike * networks[t+1](s_next/strike, g_next/strike, tau_next)
                pi = torch.exp(torch.clamp((g-v)/temperature, -40.0, 40.0))
                pi = torch.clamp(pi, max=intensity_cap)
                reg = pi - pi * torch.log(torch.clamp(pi, min=1e-12))
                target = (g*pi*dt + temperature*reg*dt
                          + np.exp(-rate*dt)*v_next*(1.0-pi*dt))
            loss = torch.mean(((target-v)/strike).square())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)))
    return {"networks": networks, "losses": np.asarray(losses), "temperature": temperature,
            "intensity_cap": intensity_cap, "dt": dt, "seed": seed, "device": "cpu"}


def evaluate_policy(model: dict, *, s0: float = 40.0, strike: float = 40.0,
                    rate: float = 0.06, volatility: float = 0.40,
                    maturity: float = 1.0, steps: int = 50,
                    paths: int = 100_000, seed: int = 9876) -> dict[str, float]:
    """Estimate exploratory reward on independent risk-neutral GBM paths."""
    import torch
    from .simulation import simulate_gbm, put_payoff
    stock = simulate_gbm(s0, rate, volatility, maturity, steps, paths, seed)
    dt = maturity / steps
    survival, reward = np.ones(paths), np.zeros(paths)
    with torch.no_grad():
        for t in range(steps):
            s = torch.as_tensor(stock[t], dtype=torch.float32).reshape(-1, 1)
            g = torch.clamp(strike-s, min=0.0)
            tau = torch.full_like(s, t/steps)
            v = strike * model["networks"][t](s/strike, g/strike, tau)
            pi = torch.exp(torch.clamp((g-v)/model["temperature"], -40.0, 40.0)).numpy().ravel()
            pi = np.minimum(pi, model["intensity_cap"])
            reg = pi - pi*np.log(np.maximum(pi, 1e-12))
            reward += np.exp(-rate*t*dt)*survival*(put_payoff(stock[t], strike)*pi + model["temperature"]*reg)*dt
            survival *= (1.0-pi*dt)
    reward += np.exp(-rate*maturity)*survival*put_payoff(stock[-1], strike)
    return {"mean_reward": float(reward.mean()),
            "standard_error": float(reward.std(ddof=1)/np.sqrt(paths))}
