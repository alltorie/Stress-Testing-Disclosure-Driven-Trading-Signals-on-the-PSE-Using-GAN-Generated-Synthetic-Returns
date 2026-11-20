import numpy as np
import pandas as pd
import pytest
from pse.features import clean_dividend_text
from pse.validation import walk_forward_splits, paired_auc_bootstrap


def test_clean_text_strips_numbers_dates_company_and_keeps_wording():
    raw = ("Subject of the Disclosure Declaration of Cash Dividends Acme Holdings 2025 "
           "Type of Securities Common Source of Dividend Payment From the unrestricted "
           "retained earnings as of December 31, 2024 Php 1.50 Filed on behalf Atty X")
    out = clean_dividend_text(raw, "Acme Holdings, Inc.")
    assert "acme" not in out and "holdings" not in out
    assert not any(ch.isdigit() for ch in out)
    assert "december" not in out
    assert "unrestricted retained earnings" in out
    assert "atty" not in out and "common" not in out


def test_clean_text_handles_missing():
    assert clean_dividend_text(None, "X") == ""


def test_walk_forward_has_no_leakage_and_starts_after_12_months():
    t0 = pd.date_range("2024-11-01", "2026-06-30", freq="3D")
    label_end = t0 + pd.Timedelta(days=5)
    folds = list(walk_forward_splits(t0, label_end, 12))
    assert folds and folds[0][0] == pd.Period("2025-11")
    t0s = pd.Series(t0)
    for m, tr, te in folds:
        start = m.start_time
        assert (t0s.iloc[te] >= start).all() and (t0s.iloc[te] <= m.end_time).all()
        assert (t0s.iloc[tr] < start).all()
        assert (pd.Series(label_end).iloc[tr] < start).all()       # purge
        assert set(tr).isdisjoint(te)


def test_purge_drops_training_events_whose_window_crosses_test_start():
    t0 = pd.to_datetime(["2024-11-05", "2025-10-30", "2025-11-10"])
    le = pd.to_datetime(["2024-11-08", "2025-11-04", "2025-11-13"])   # 2nd overlaps Nov 2025
    (m, tr, te), = list(walk_forward_splits(t0, le, 12))
    assert list(tr) == [0] and list(te) == [2]


def test_bootstrap_detects_better_model_and_zero_for_identical():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.8, 400)
    noise = rng.normal(size=400)
    r = paired_auc_bootstrap(y, good, noise, n_boot=300)
    assert r["auc_diff"] > 0.2 and r["p_one_sided"] < 0.05
    same = paired_auc_bootstrap(y, good, good, n_boot=100)
    assert same["auc_diff"] == 0 and same["p_one_sided"] == 1.0
