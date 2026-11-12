import numpy as np
from src.simulation import simulate_gbm, put_payoff
from src.longstaff_schwartz import longstaff_schwartz_put, polynomial_basis
from src.finite_difference import solve_exploratory_put


def test_gbm_shape_and_reproducibility():
    a = simulate_gbm(100, .03, .2, 1, 10, 50, seed=7)
    b = simulate_gbm(100, .03, .2, 1, 10, 50, seed=7)
    assert a.shape == (11, 50)
    assert np.array_equal(a, b)
    assert np.all(a > 0)


def test_put_payoff_and_basis():
    np.testing.assert_allclose(put_payoff(np.array([30., 40., 50.]), 40), [10, 0, 0])
    assert polynomial_basis(np.array([40., 80.]), 40).shape == (2, 4)


def test_longstaff_schwartz_smoke():
    result = longstaff_schwartz_put(paths=3000, steps=12, seed=11)
    assert 0 < result["price"] < 40
    assert result["standard_error"] > 0


def test_finite_difference_smoke():
    result = solve_exploratory_put(space_steps=80, time_steps=30, temperature=.1)
    assert np.isfinite(result["price"])
    assert result["values"].shape == (31, 81)
    assert result["price"] > 0
