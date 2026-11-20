import re
from datetime import time
import pandas as pd

AMEND = re.compile(r"^\[Amend-\d+\]")


def classify_template(template):
    """Return (event_type or None, is_amendment)."""
    amended = bool(AMEND.match(template))
    base = AMEND.sub("", template)
    if base == "Declaration of Cash Dividends":
        return "dividend", amended
    if base == "Share Buy-Back Transactions":
        return "buyback", amended
    if base.startswith("Change in Directors and/or Officers"):
        return "leadership", amended
    return None, amended


def session_bucket(ts, open_t=time(9, 30), cutoff=time(15, 30)):
    t = ts.time()
    if t < open_t:
        return "pre_open"
    if t < cutoff:
        return "in_session"
    return "post_close"


def first_tradable_session(ts, calendar, cutoff=time(15, 30)):
    """First session in `calendar` in which the market can react to a filing at `ts`.

    A filing before the cutoff on a trading day maps to that day. A filing at or
    after the cutoff, or on a non-trading day, maps to the next trading day.
    Returns NaT if the calendar ends first.
    """
    d = ts.normalize()
    pos = calendar.searchsorted(d, side="left")
    if pos >= len(calendar):
        return pd.NaT
    if calendar[pos] == d and ts.time() >= cutoff:
        pos += 1
    return calendar[pos] if pos < len(calendar) else pd.NaT


def build_events(listing, dividends, calendar, cfg):
    """One row per event with type, filing time, bucket and t0."""
    df = listing.copy()
    kinds = df["template"].map(classify_template)
    df["event_type"] = [k[0] for k in kinds]
    df["is_amend"] = [k[1] for k in kinds]
    df = df[df["event_type"].notna()].copy()

    # Dividends: one event per declaration (company, board date, record date),
    # taking the earliest filing, original or amended.
    div = df[df.event_type == "dividend"].merge(
        dividends[["edge_no", "board_date", "record_date", "amount_raw", "div_type"]],
        on="edge_no", how="left")
    div["decl_key"] = (div["cmpy_id"].astype(str) + "|" + div["board_date"].astype(str)
                       + "|" + div["record_date"].astype(str))
    div.loc[div["board_date"].isna(), "decl_key"] = div["edge_no"]
    div = div.sort_values("ts").drop_duplicates("decl_key", keep="first")

    # Buy-back and leadership: originals only, first filing of each cluster.
    other = df[(df.event_type != "dividend") & (~df.is_amend)].copy()
    other["t0_tmp"] = [first_tradable_session(t, calendar, cfg.session_cutoff) for t in other["ts"]]
    other = other.dropna(subset=["t0_tmp"]).sort_values("ts")
    keep = []
    for _, g in other.groupby(["cmpy_id", "event_type"]):
        pos = calendar.searchsorted(g["t0_tmp"].values)
        prev = -10**9
        for idx, p in zip(g.index, pos):
            if p - prev > cfg.cluster_gap_days:
                keep.append(idx)
            prev = p
    other = other.loc[keep].drop(columns="t0_tmp")

    ev = pd.concat([div, other], ignore_index=True)
    ev["bucket"] = [session_bucket(t, cfg.session_open, cfg.session_cutoff) for t in ev["ts"]]
    ev["t0"] = [first_tradable_session(t, calendar, cfg.session_cutoff) for t in ev["ts"]]
    ev = ev.dropna(subset=["t0"]).sort_values("ts").reset_index(drop=True)
    return ev[["edge_no", "cmpy_id", "company", "event_type", "template", "ts",
               "bucket", "t0", "board_date", "record_date", "amount_raw", "div_type"]]
