import re
import numpy as np
import pandas as pd

_PESO = re.compile(r"(?:php|p|\u20b1)\s*([0-9][0-9,]*\.?[0-9]*)", re.I)
_USD = re.compile(r"us\$|usd|\$", re.I)


def parse_amount(raw):
    """Peso dividend per share from free text. NaN if in words, USD or unreadable."""
    if not isinstance(raw, str) or _USD.search(raw):
        return np.nan
    m = _PESO.search(raw)
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return np.nan


def ex_dates(record_dates, calendar):
    """Ex-date is the trading day before the record date (PSE convention)."""
    out = []
    for r in record_dates:
        if pd.isna(r):
            out.append(pd.NaT)
            continue
        pos = calendar.searchsorted(r, side="left")
        out.append(calendar[pos - 1] if 0 < pos <= len(calendar) else pd.NaT)
    return pd.DatetimeIndex(out)


def build_exdiv_table(dividends, calendar):
    d = dividends.copy()
    d["amount"] = d["amount_raw"].map(parse_amount)
    d["ex_date"] = ex_dates(d["record_date"], calendar)
    d = d.dropna(subset=["ex_date"])
    # one row per company and ex-date (amended filings repeat the same declaration)
    d = d.sort_values("ts").drop_duplicates(["cmpy_id", "ex_date"], keep="first")
    return d[["cmpy_id", "ex_date", "amount"]]


def adjusted_log_price(prices, calendar, exdiv_rows, cfg):
    """Dividend-adjusted log close on the market calendar, plus flags.

    Close is forward-filled up to cfg.max_stale_days sessions; `age` counts sessions
    since the last quoted close. The ex-date drop is added back as ln(1 + D / ex-date close),
    cumulated from the ex-date on, so gaps in thin names do not distort the return.
    Returns (adj_lnP, age, bad) where `bad` marks sessions touched by a possible split
    or an untrusted dividend.
    """
    close = prices["CLOSE"].reindex(calendar)
    quoted = close.notna()
    age = (~quoted).groupby(quoted.cumsum()).cumsum()
    close = close.ffill(limit=cfg.max_stale_days)
    ln = np.log(close)

    bad = pd.Series(False, index=calendar)
    bad |= ln.diff().abs() > cfg.jump_threshold

    adj = pd.Series(0.0, index=calendar)
    for _, r in exdiv_rows.iterrows():
        ex = r["ex_date"]
        if ex not in calendar:
            continue
        pos = calendar.get_loc(ex)
        p_ex = close.iloc[pos]
        if r["amount"] != r["amount"] or p_ex != p_ex:
            bad.iloc[pos:pos + 1] = True
            continue
        ratio = r["amount"] / p_ex
        if ratio > cfg.max_div_ratio:
            bad.iloc[pos:pos + 1] = True
            continue
        adj.iloc[pos:] += np.log1p(ratio)
    return ln + adj, age, bad
