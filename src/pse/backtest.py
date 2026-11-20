import numpy as np
import pandas as pd
from scipy import stats
from .returns import adjusted_log_price
from .validation import walk_forward_splits


def build_trades(events, prices, psei_close, psei_open, exdiv, cfg, bt, cost):
    """Simulate one long trade per event. Skipped events keep a reason in `skip`.

    Entry at the close of t0 (or the open of t0 for pre-open filings), exit at the close of
    t0 + hold_days. Both ends need a quote that day, otherwise the trade is not executable.
    Gross return uses dividend-adjusted prices. Costs come from `cost`.
    """
    cal = psei_close.index
    ln_m = np.log(psei_close)
    ln_mo = np.log(psei_open.reindex(cal))
    rows = []
    for cid, g in events.groupby("cmpy_id"):
        if cid not in prices:
            rows += [{"edge_no": e, "skip": "no_prices"} for e in g["edge_no"]]
            continue
        adj, age, bad = adjusted_log_price(prices[cid], cal, exdiv[exdiv.cmpy_id == cid], cfg)
        close = prices[cid]["CLOSE"].reindex(cal)
        open_ = prices[cid]["OPEN"].reindex(cal)
        for _, e in g.iterrows():
            i0 = cal.get_loc(e["t0"])
            ie = i0 + bt.hold_days
            r = {"edge_no": e["edge_no"], "skip": None,
                 "exit_date": cal[ie] if ie < len(cal) else pd.NaT}
            use_open = bt.entry == "open_t0"
            if ie >= len(cal):
                r["skip"] = "outside_data"
            elif age.iloc[i0] > 0:
                r["skip"] = "entry_no_quote"
            elif age.iloc[ie] > 0:
                r["skip"] = "exit_no_quote"
            elif use_open and e["bucket"] != "pre_open":
                r["skip"] = "not_pre_open"
            elif use_open and not (open_.iloc[i0] > 0):
                r["skip"] = "no_open"
            elif bad.iloc[i0 + 1: ie + 1].any():
                r["skip"] = "suspect"
            else:
                offset = adj.iloc[i0] - np.log(close.iloc[i0])
                p_in = open_.iloc[i0] if use_open else close.iloc[i0]
                ln_in = np.log(p_in) + offset
                p_out = close.iloc[ie]
                gross = adj.iloc[ie] - ln_in
                m0 = ln_mo.iloc[i0] if use_open else ln_m.iloc[i0]
                r.update(p_in=p_in, p_out=p_out, gross_log=gross,
                         mkt_ret=np.expm1(ln_m.iloc[ie] - m0),
                         exdiv_in_hold=abs((adj.iloc[ie] - np.log(close.iloc[ie])) - offset) > 1e-12)
            rows.append(r)
    res = pd.DataFrame(rows)
    keep = [c for c in ["edge_no", "cmpy_id", "company", "event_type", "bucket", "t0",
                        "median_value", "car_0_2"] if c in events.columns]
    out = events[keep].merge(res, on="edge_no", how="left")
    return apply_costs(out, cost, bt)


def apply_costs(trades, cost, bt):
    t = trades.copy()
    for c in ["gross_log", "p_in", "p_out", "mkt_ret", "exit_date", "exdiv_in_hold"]:
        if c not in t:
            t[c] = np.nan
    ok = t["skip"].isna()
    t["gross"] = np.nan
    t["net"] = np.nan
    t.loc[ok, "gross"] = np.expm1(t.loc[ok, "gross_log"])
    t.loc[ok, "net"] = cost.net_return(t.loc[ok, "gross_log"].values,
                                       t.loc[ok, "p_in"].values, t.loc[ok, "p_out"].values)
    t["cost_drag"] = t["gross"] - t["net"]
    t["net_excess"] = t["net"] - t["mkt_ret"]
    if "median_value" in t:
        t["participation"] = bt.notional / t["median_value"]
    return t


def walk_forward_rule(trades, bt, car_col="car_0_2"):
    """Trade an event type in a test month only if, in the purged training data, its mean
    market-adjusted CAR is positive with t above bt.select_t. The evidence pool is every event
    with a valid CAR (executable or not), so the choice of types does not depend on which
    entry style is simulated. Only executable events of a chosen type are traded. Selection
    never sees net returns or test-period prices. Returns (trades taken, selection log)."""
    pool = trades[trades[car_col].notna() & trades["exit_date"].notna()]
    pool = pool.sort_values("t0").reset_index(drop=True)
    taken, log = [], []
    for month, tri, tei in walk_forward_splits(pool["t0"], pool["exit_date"], bt.train_months):
        train, test = pool.iloc[tri], pool.iloc[tei]
        chosen = []
        for et, g in train.groupby("event_type"):
            x = g[car_col].dropna()
            if len(x) >= bt.min_train_events and x.mean() > 0 and \
                    stats.ttest_1samp(x, 0.0).statistic > bt.select_t:
                chosen.append(et)
        sel = test[test["event_type"].isin(chosen) & test["skip"].isna()]
        log.append({"fold": str(month), "n_train": len(train), "n_test": len(test),
                    "chosen": ",".join(chosen), "n_traded": len(sel)})
        taken.append(sel.assign(fold=str(month)))
    taken = pd.concat(taken, ignore_index=True) if taken else pool.iloc[0:0]
    return taken, pd.DataFrame(log)


def h3_summary(t, n_boot=2000, seed=0):
    """H3: mean net return per trade is zero or negative (one-sided, Ha: positive)."""
    x = t["net"].dropna().values
    n = len(x)
    if n < 3:
        return {"n": n}
    rng = np.random.default_rng(seed)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    tt = stats.ttest_1samp(x, 0.0, alternative="greater")
    try:
        pw = stats.wilcoxon(x, alternative="greater").pvalue
    except ValueError:
        pw = np.nan
    xe = t["net_excess"].dropna().values
    return {"n": n, "mean_gross": t["gross"].mean(), "mean_cost": t["cost_drag"].mean(),
            "mean_net": x.mean(), "median_net": np.median(x), "hit_rate": (x > 0).mean(),
            "std_net": x.std(ddof=1), "t_stat": tt.statistic, "p_t_greater": tt.pvalue,
            "p_wilcoxon_greater": pw, "ci_low": np.percentile(boots, 2.5),
            "ci_high": np.percentile(boots, 97.5), "mean_mkt": t["mkt_ret"].mean(),
            "mean_net_excess": xe.mean(),
            "p_excess_greater": stats.ttest_1samp(xe, 0.0, alternative="greater").pvalue}
