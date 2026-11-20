import numpy as np
import pandas as pd
import pytest
from pse.config import Config
from pse.eventstudy import compute_cars


def test_car_is_stock_minus_market_over_window():
    cal = pd.bdate_range("2026-03-02", periods=120)
    ln_stock = pd.Series(0.0, index=cal)
    ln_mkt = pd.Series(0.0, index=cal)
    i0 = 80
    ln_stock.iloc[i0:] += 0.05                  # +5% on t0, flat afterwards
    ln_mkt.iloc[i0:] += 0.01                    # market +1% on t0
    psei = pd.Series(np.exp(ln_mkt.values) * 6000, index=cal)
    prices = {7: pd.DataFrame({"CLOSE": np.exp(ln_stock.values) * 50, "VALUE": 5e6}, index=cal)}
    ev = pd.DataFrame({"edge_no": ["a"], "cmpy_id": [7], "company": ["X"],
                       "event_type": ["buyback"], "t0": [cal[i0]]})
    exdiv = pd.DataFrame(columns=["cmpy_id", "ex_date", "amount"])
    out = compute_cars(ev, prices, psei, exdiv, Config())
    assert out.loc[0, "car_0_0"] == pytest.approx(0.04)
    assert out.loc[0, "car_0_2"] == pytest.approx(0.04)   # nothing more after t0
    assert bool(out.loc[0, "liquid"])
    assert not out.loc[0, "suspect"]


def test_window_past_end_of_data_is_excluded():
    cal = pd.bdate_range("2026-03-02", periods=10)
    psei = pd.Series(6000.0, index=cal)
    prices = {7: pd.DataFrame({"CLOSE": 50.0, "VALUE": 5e6}, index=cal)}
    ev = pd.DataFrame({"edge_no": ["a"], "cmpy_id": [7], "company": ["X"],
                       "event_type": ["buyback"], "t0": [cal[-1]]})
    out = compute_cars(ev, prices, psei,
                       pd.DataFrame(columns=["cmpy_id", "ex_date", "amount"]), Config())
    assert out.loc[0, "status"] == "window_outside_data"
