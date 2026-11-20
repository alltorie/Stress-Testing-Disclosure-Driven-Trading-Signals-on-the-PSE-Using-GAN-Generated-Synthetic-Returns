from datetime import time
import numpy as np
import pandas as pd
import pytest
from pse.config import Config
from pse.events import first_tradable_session, classify_template, session_bucket
from pse.returns import parse_amount, ex_dates, adjusted_log_price
from pse.eventstudy import holm

# Fri 2026-11-13, then Mon-Tue; Wed 11-18 is a holiday gap in this toy calendar.
CAL = pd.DatetimeIndex(["2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17", "2026-11-19"])


@pytest.mark.parametrize("ts,expected", [
    ("2026-11-13 07:40", "2026-11-13"),   # before open: same day
    ("2026-11-13 12:30", "2026-11-13"),   # in session: same day
    ("2026-11-13 15:29", "2026-11-13"),   # just before cutoff
    ("2026-11-13 15:30", "2026-11-16"),   # at cutoff: next session
    ("2026-11-13 17:12", "2026-11-16"),   # after close on Friday: Monday
    ("2026-11-14 10:00", "2026-11-16"),   # Saturday: Monday
    ("2026-11-17 16:00", "2026-11-19"),   # next session skips the holiday
    ("2026-11-18 09:00", "2026-11-19"),   # holiday morning: next session
])
def test_first_tradable_session(ts, expected):
    got = first_tradable_session(pd.Timestamp(ts), CAL, time(15, 30))
    assert got == pd.Timestamp(expected)


def test_after_calendar_end_is_nat():
    assert pd.isna(first_tradable_session(pd.Timestamp("2026-11-19 16:00"), CAL))


def test_classify_templates():
    assert classify_template("Declaration of Cash Dividends") == ("dividend", False)
    assert classify_template("[Amend-2]Declaration of Cash Dividends") == ("dividend", True)
    assert classify_template("Share Buy-Back Transactions") == ("buyback", False)
    assert classify_template(
        "Change in Directors and/or Officers (Resignation, Removal or Appointment, "
        "Election and/or Promotion)") == ("leadership", False)
    # shareholdings of directors is NOT a leadership change
    assert classify_template(
        "Change in Shareholdings of Directors and Principal Officers")[0] is None


def test_session_bucket():
    assert session_bucket(pd.Timestamp("2026-11-13 09:00")) == "pre_open"
    assert session_bucket(pd.Timestamp("2026-11-13 10:00")) == "in_session"
    assert session_bucket(pd.Timestamp("2026-11-13 16:00")) == "post_close"


@pytest.mark.parametrize("raw,val", [
    ("Php1.00 per share", 1.0), ("PhP0.45", 0.45), ("P17.75800", 17.758),
    ("Php 18.9625", 18.9625), ("COMMON AT P0.50 PER SHARE", 0.5),
    ("Php 1,250.5", 1250.5)])
def test_parse_amount(raw, val):
    assert parse_amount(raw) == pytest.approx(val)


@pytest.mark.parametrize("raw", ["Fifty-five Centavos", "US$0.02 per share", None])
def test_parse_amount_unreadable(raw):
    assert np.isnan(parse_amount(raw))


def test_ex_date_is_trading_day_before_record():
    ex = ex_dates(pd.to_datetime(["2026-11-16"]), CAL)
    assert ex[0] == pd.Timestamp("2026-11-13")


def test_dividend_adjustment_removes_ex_date_drop():
    cal = pd.bdate_range("2026-01-05", periods=10)
    close = pd.Series(100.0, index=cal)
    close.iloc[5:] = 98.0                       # 2 peso dividend, price falls on ex-date
    prices = pd.DataFrame({"CLOSE": close, "VALUE": 1e7})
    ex = pd.DataFrame({"cmpy_id": [1], "ex_date": [cal[5]], "amount": [2.0]})
    adj, age, bad = adjusted_log_price(prices, cal, ex, Config())
    assert abs(adj.iloc[5] - adj.iloc[4]) < 1e-9   # no spurious drop after adjustment
    assert not bad.any()


def test_untrusted_dividend_is_flagged_not_adjusted():
    cal = pd.bdate_range("2026-01-05", periods=10)
    prices = pd.DataFrame({"CLOSE": 100.0, "VALUE": 1e7}, index=cal)
    ex = pd.DataFrame({"cmpy_id": [1], "ex_date": [cal[5]], "amount": [np.nan]})
    adj, age, bad = adjusted_log_price(prices, cal, ex, Config())
    assert bad.iloc[5] and not bad.iloc[4]


def test_split_like_jump_is_flagged():
    cal = pd.bdate_range("2026-01-05", periods=10)
    close = pd.Series(100.0, index=cal)
    close.iloc[6:] = 40.0
    prices = pd.DataFrame({"CLOSE": close, "VALUE": 1e7})
    adj, age, bad = adjusted_log_price(prices, cal, pd.DataFrame(
        columns=["cmpy_id", "ex_date", "amount"]), Config())
    assert bad.iloc[6]


def test_stale_age_counts_missing_sessions():
    cal = pd.bdate_range("2026-01-05", periods=8)
    close = pd.Series(100.0, index=cal)
    close.iloc[3:5] = np.nan
    prices = pd.DataFrame({"CLOSE": close, "VALUE": 1e7})
    _, age, _ = adjusted_log_price(prices, cal, pd.DataFrame(
        columns=["cmpy_id", "ex_date", "amount"]), Config())
    assert list(age.iloc[2:6]) == [0, 1, 2, 0]


def test_holm_matches_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
