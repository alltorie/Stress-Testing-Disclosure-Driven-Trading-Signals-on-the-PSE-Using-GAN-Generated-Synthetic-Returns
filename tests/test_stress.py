import numpy as np
import pandas as pd
import pytest
from pse import stress
from pse.config import Config
from pse.costs import CostModel

NOEX = pd.DataFrame(columns=["cmpy_id", "ex_date", "amount"])


def test_metrics_known_values():
    s = np.array([[0.01, -0.02, 0.01, -0.01, 0.02]])
    m = stress.metrics(s, ann_days=252, level=0.2)
    assert m.loc[0, "sharpe"] == pytest.approx(s.mean() / s.std(ddof=1) * np.sqrt(252))
    assert m.loc[0, "max_drawdown"] == pytest.approx(0.02)      # peak 1.01 to trough 0.99
    assert m.loc[0, "cvar"] == pytest.approx(0.02)              # worst 1 of 5 days
    assert m.loc[0, "total_return"] == pytest.approx(0.01)


def test_flat_strategy_has_nan_sharpe_and_zero_drawdown():
    m = stress.metrics(np.zeros((1, 50)))
    assert np.isnan(m.loc[0, "sharpe"]) and m.loc[0, "max_drawdown"] == 0


def _world(n=120, hold=2):
    cal = pd.bdate_range("2026-01-05", periods=n)
    psei = pd.Series(6000.0, index=cal)
    px = pd.DataFrame({"OPEN": 50.0, "CLOSE": 50.0, "VALUE": 5e6}, index=cal)
    trades = pd.DataFrame({"cmpy_id": [1, 1], "event_type": ["dividend"] * 2,
                           "t0": [cal[60], cal[80]], "p_in": [50.0, 50.0], "p_out": [50.0, 50.0]})
    return cal, psei, px, trades


def test_portfolio_contributions_sum_to_trade_net_return_matching_step3():
    cal, psei, px, trades = _world()
    cost = CostModel()
    spec = stress.trade_specs(trades, psei, {1: px}, NOEX, Config(), cost, hold=2)
    gross_log = np.array([[0.01, 0.02]] * 2)                      # same path for both trades
    series = stress._portfolio(spec, gross_log[None, :, :], costs_on=True)[0]
    net_step3 = cost.net_return(gross_log.sum(axis=1)[0], 50.0, 50.0)
    assert series.sum() == pytest.approx(spec["w"] * 2 * net_step3, rel=1e-9)
    gross_series = stress._portfolio(spec, gross_log[None, :, :], costs_on=False)[0]
    assert gross_series.sum() == pytest.approx(spec["w"] * 2 * np.expm1(0.03), rel=1e-9)


def test_weight_never_overallocates_capital():
    cal, psei, px, _ = _world()
    trades = pd.DataFrame({"cmpy_id": [1] * 3, "event_type": ["dividend"] * 3,
                           "t0": [cal[60]] * 3, "p_in": [50.0] * 3, "p_out": [50.0] * 3})
    spec = stress.trade_specs(trades, psei, {1: px}, NOEX, Config(), CostModel(), hold=2)
    assert spec["max_load"] == 3 and spec["w"] == pytest.approx(1 / 3)


def test_simulation_is_reproducible_and_scenarios_share_idiosyncratic_draws():
    cal, psei, px, trades = _world()
    spec = stress.trade_specs(trades, psei, {1: px}, NOEX, Config(), CostModel(), hold=2)
    ar = {"dividend": [0.004, 0.002]}
    resid = {1: np.random.default_rng(0).normal(0, 0.02, 500)}
    calm = np.zeros((40, spec["T"] + 1))
    a = stress.simulate_series(spec, calm, {1: 1.0}, resid, resid[1], ar, seed=7, costs_on=False)
    b = stress.simulate_series(spec, calm, {1: 1.0}, resid, resid[1], ar, seed=7, costs_on=False)
    assert np.array_equal(a, b)
    shock = calm.copy(); shock[:, :] = 0.01
    c = stress.simulate_series(spec, shock, {1: 1.0}, resid, resid[1], ar, seed=7, costs_on=False)
    assert not np.allclose(a, c)                                  # market changes the result
    # with no residual noise and a flat market the strategy earns exactly the abnormal return
    z = stress.simulate_series(spec, calm, {1: 1.0}, {1: np.zeros(5)}, np.zeros(5), ar, seed=7, costs_on=False)
    assert z.sum(axis=1)[0] == pytest.approx(spec["w"] * 2 * np.expm1(0.006), rel=1e-9)


def test_compare_scenarios_detects_a_shifted_group():
    rng = np.random.default_rng(0)
    f = lambda m: pd.DataFrame({"sharpe": rng.normal(m, 1, 200), "max_drawdown": rng.normal(0.1, .01, 200),
                                "cvar": rng.normal(0.01, .001, 200)})
    out = stress.compare_scenarios({"a": f(0), "b": f(0), "c": f(2)})
    row = out[out.measure == "sharpe"].iloc[0]
    assert row["kruskal_p"] < 1e-6 and row["p_holm[a vs c]"] < 1e-6
    assert out[out.measure == "cvar"].iloc[0]["kruskal_p"] > 0.001
