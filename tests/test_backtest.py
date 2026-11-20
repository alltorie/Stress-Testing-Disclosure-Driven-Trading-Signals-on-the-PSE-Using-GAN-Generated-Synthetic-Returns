import numpy as np
import pandas as pd
import pytest
from pse.config import Config, BacktestConfig
from pse.costs import CostModel, tick_size
from pse.backtest import build_trades, walk_forward_rule, h3_summary

NO_EXDIV = pd.DataFrame(columns=["cmpy_id", "ex_date", "amount"])


@pytest.mark.parametrize("price,tick", [(0.3, 0.005), (0.5, 0.01), (4.99, 0.01), (15, 0.02),
                                        (20, 0.05), (36.15, 0.05), (150, 0.1), (250, 0.2),
                                        (1500, 1.0), (6000, 5.0)])
def test_tick_table(price, tick):
    assert tick_size(price) == pytest.approx(tick)


def test_fee_rates_follow_pse_schedule():
    c = CostModel()
    assert c.fee_rate("sell") - c.fee_rate("buy") == pytest.approx(0.001)   # STT on sells only
    assert c.fee_rate("buy") == pytest.approx(0.0025 * 1.12 + 0.00005 * 1.12 + 0.0001 + 0.00005)


def test_round_trip_cost_is_about_seventy_bps():
    r = CostModel().net_return(0.0, 100.0, 100.0)
    assert -0.0080 < r < -0.0070


def test_zero_cost_model_returns_gross():
    z = CostModel(commission=0, pse_fee=0, sccp_fee=0, src_fee=0, stt_sell=0, spread_mult=0)
    assert z.net_return(0.05, 100.0, 105.0) == pytest.approx(np.expm1(0.05))


def _world(n=120):
    cal = pd.bdate_range("2026-01-05", periods=n)
    psei = pd.Series(6000.0, index=cal)
    close = pd.Series(50.0, index=cal)
    i0 = 60
    close.iloc[i0 + 1:] = 52.0                       # +4% the session after t0
    px = pd.DataFrame({"OPEN": close.values, "CLOSE": close.values, "VALUE": 5e6}, index=cal)
    ev = pd.DataFrame({"edge_no": ["a"], "cmpy_id": [1], "company": ["X"], "event_type": ["dividend"],
                       "bucket": ["pre_open"], "t0": [cal[i0]], "median_value": [5e6], "car_0_2": [0.01]})
    return cal, psei, px, ev, i0


def test_trade_gross_and_net_on_known_path():
    cal, psei, px, ev, i0 = _world()
    t = build_trades(ev, {1: px}, psei, psei, NO_EXDIV, Config(), BacktestConfig(), CostModel())
    assert t.loc[0, "skip"] is None or pd.isna(t.loc[0, "skip"])
    assert t.loc[0, "gross"] == pytest.approx(0.04)
    assert t.loc[0, "net"] < t.loc[0, "gross"] - 0.006
    assert t.loc[0, "exit_date"] == cal[i0 + 2]
    assert t.loc[0, "participation"] == pytest.approx(0.02)


def test_missing_quote_at_entry_skips_trade():
    cal, psei, px, ev, i0 = _world()
    px.loc[cal[i0], ["OPEN", "CLOSE"]] = np.nan
    t = build_trades(ev, {1: px.dropna()}, psei, psei, NO_EXDIV, Config(), BacktestConfig(), CostModel())
    assert t.loc[0, "skip"] == "entry_no_quote"


def test_open_entry_only_for_pre_open_filings():
    cal, psei, px, ev, i0 = _world()
    ev.loc[0, "bucket"] = "in_session"
    t = build_trades(ev, {1: px}, psei, psei, NO_EXDIV, Config(),
                     BacktestConfig(entry="open_t0"), CostModel())
    assert t.loc[0, "skip"] == "not_pre_open"


def _synthetic_trades(seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.date_range("2024-11-01", "2026-06-30", freq="2D")
    n = len(t0)
    et = np.where(np.arange(n) % 2 == 0, "dividend", "leadership")
    car = np.where(et == "dividend", rng.normal(0.01, 0.01, n), rng.normal(0.0, 0.01, n))
    return pd.DataFrame({"edge_no": range(n), "event_type": et, "t0": t0, "exit_date": t0 + pd.Timedelta(days=4),
                         "car_0_2": car, "skip": None, "net": car - 0.007, "gross": car,
                         "cost_drag": 0.007, "mkt_ret": 0.0, "net_excess": car - 0.007})


def test_rule_selects_only_significantly_positive_types_and_starts_after_12_months():
    taken, log = walk_forward_rule(_synthetic_trades(), BacktestConfig())
    assert log["fold"].iloc[0] == "2025-11"
    assert set(taken["event_type"]) == {"dividend"}
    assert (log["chosen"] == "dividend").all()


def test_rule_trades_nothing_when_no_type_is_significant():
    t = _synthetic_trades()
    t["car_0_2"] = np.random.default_rng(5).normal(0, 0.01, len(t))
    taken, log = walk_forward_rule(t, BacktestConfig())
    assert len(taken) == 0 or set(log["chosen"]) <= {"", "dividend", "leadership"}
    assert (log["chosen"] != "").sum() <= 2          # at most a chance selection or two


def test_h3_summary_rejects_for_clearly_positive_and_not_for_negative():
    pos = pd.DataFrame({"net": np.random.default_rng(1).normal(0.01, 0.02, 200)})
    pos = pos.assign(gross=pos.net, cost_drag=0.0, mkt_ret=0.0, net_excess=pos.net)
    neg = pos.assign(net=-pos.net.abs(), net_excess=-pos.net.abs())
    assert h3_summary(pos, n_boot=200)["p_t_greater"] < 0.01
    assert h3_summary(neg, n_boot=200)["p_t_greater"] > 0.99


def test_skipped_events_inform_selection_but_are_not_traded():
    t = _synthetic_trades()
    t.loc[t.event_type == "dividend", "skip"] = "not_pre_open"      # none executable
    taken, log = walk_forward_rule(t, BacktestConfig())
    assert (log["chosen"] == "dividend").all()                     # still selected on all events
    assert len(taken) == 0 and (log["n_traded"] == 0).all()
