"""Run the disclosure strategy on simulated background markets.

Model per trade (day k after entry):  log return = beta * market(k) + AR(k) + eps
  market(k): the scenario's index return (what changes between scenarios)
  AR(k):     estimated mean abnormal return of that event type on day k (fixed)
  eps:       stock-specific residual, resampled from the stock's own history (same draws in every
             scenario, so scenarios differ only in the market path)
The real disclosure dates, hold period, position weights and costs are held fixed.
"""
import numpy as np
import pandas as pd
from scipy import stats
from .returns import adjusted_log_price
from .eventstudy import holm


def estimate_stock_models(cids, prices, psei, exdiv, cfg, min_obs=250):
    """Beta to the PSEi and the stock's residual returns, from the full price history."""
    cal = psei.index
    r_m = np.log(psei).diff()
    betas, resids = {}, {}
    for cid in cids:
        if cid not in prices:
            continue
        adj, age, bad = adjusted_log_price(prices[cid], cal, exdiv[exdiv.cmpy_id == cid], cfg)
        r = adj.diff()
        ok = (age == 0) & (age.shift(1) == 0) & (~bad) & r.notna() & r_m.notna()
        x, y = r_m[ok].values, r[ok].values
        if len(x) < min_obs or x.var() == 0:
            continue
        b = np.cov(x, y)[0, 1] / x.var()
        a = y.mean() - b * x.mean()
        betas[cid] = b
        resids[cid] = y - a - b * x
    pooled = np.concatenate(list(resids.values())) if resids else np.array([0.0])
    return betas, resids, pooled


def abnormal_return_by_day(events, hold=2):
    """Mean market-adjusted return on days +1 and +2 after t0, per event type, from CAR windows."""
    assert hold == 2, "built for the primary 2-session hold"
    d = events.copy()
    d["ar1"] = d["car_0_1"] - d["car_0_0"]
    d["ar2"] = d["car_0_2"] - d["car_0_1"]
    return {et: [g["ar1"].mean(), g["ar2"].mean()] for et, g in d.groupby("event_type")}


def trade_specs(trades, psei, prices, exdiv, cfg, cost, hold=2):
    """Fixed trade layout for the stress test: entry day, weight, cost fractions, real daily returns."""
    cal = psei.index
    t = trades.sort_values("t0").reset_index(drop=True)
    e_idx = np.array([cal.get_loc(x) for x in pd.to_datetime(t["t0"])])
    start, end = e_idx.min(), e_idx.max() + hold
    T = end - start + 1
    # weight so that capital is never over-allocated at the busiest moment
    load = np.zeros(T)
    for e in e_idx:
        load[e - start: e - start + hold + 1] += 1
    w = 1.0 / load.max()
    p_in, p_out = t["p_in"].values, t["p_out"].values
    f_in = 1 - 1 / ((1 + cost.half_spread(p_in)) * (1 + cost.fee_rate("buy")))
    f_out = 1 - (1 - cost.half_spread(p_out)) * (1 - cost.fee_rate("sell"))
    real_r = np.full((len(t), hold), np.nan)
    for cid, g in t.groupby("cmpy_id"):
        adj, _, _ = adjusted_log_price(prices[cid], cal, exdiv[exdiv.cmpy_id == cid], cfg)
        for i in g.index:
            for k in range(1, hold + 1):
                real_r[i, k - 1] = adj.iloc[e_idx[i] + k] - adj.iloc[e_idx[i] + k - 1]
    return {"trades": t, "e": e_idx - start, "start": start, "T": int(T), "hold": hold,
            "w": w, "f_in": f_in, "f_out": f_out, "real_r": real_r, "max_load": int(load.max())}


def _portfolio(spec, daily_log_r, costs_on=True):
    """daily_log_r: (P, n_trades, hold) stock log returns. Returns (P, T) portfolio daily returns."""
    P, n, hold = daily_log_r.shape
    out = np.zeros((P, spec["T"]))
    for i in range(n):
        e = spec["e"][i]
        f_in = spec["f_in"][i] if costs_on else 0.0
        f_out = spec["f_out"][i] if costs_on else 0.0
        v_prev = np.ones(P)
        v = np.full(P, 1.0 - f_in)
        out[:, e] += spec["w"] * (v - v_prev)
        for k in range(hold):
            v_prev = v
            v = v * np.exp(daily_log_r[:, i, k])
            if k == hold - 1:
                v = v * (1.0 - f_out)
            out[:, e + k + 1] += spec["w"] * (v - v_prev)
    return out


def real_series(spec, costs_on=True):
    return _portfolio(spec, spec["real_r"][None, :, :], costs_on)[0]


def simulate_series(spec, market_paths, betas, resids, pooled, ar_table, seed, costs_on=True):
    """Strategy daily returns on each simulated market path. market_paths: (P, >=T) log returns."""
    t = spec["trades"]
    P, n, hold = market_paths.shape[0], len(t), spec["hold"]
    rng = np.random.default_rng(seed)               # identical idiosyncratic draws for every scenario
    daily = np.empty((P, n, hold))
    for i, row in t.iterrows():
        beta = betas.get(row["cmpy_id"], 1.0)
        pool = resids.get(row["cmpy_id"], pooled)
        ar = ar_table[row["event_type"]]
        for k in range(hold):
            eps = pool[rng.integers(0, len(pool), P)]
            daily[:, i, k] = beta * market_paths[:, spec["e"][i] + k + 1] + ar[k] + eps
    return _portfolio(spec, daily, costs_on)


def metrics(series, ann_days=252, level=0.05):
    """Per-path Sharpe (annualized), max drawdown of the fixed-notional equity curve, daily CVaR."""
    s = np.atleast_2d(series)
    sd = s.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, s.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(ann_days), np.nan)
    eq = 1.0 + np.cumsum(s, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(s), 1)), eq], axis=1), axis=1)[:, 1:]
    mdd = (peak - eq).max(axis=1)
    k = max(1, int(np.ceil(level * s.shape[1])))
    cvar = -np.sort(s, axis=1)[:, :k].mean(axis=1)
    return pd.DataFrame({"sharpe": sharpe, "max_drawdown": mdd, "cvar": cvar,
                         "total_return": eq[:, -1] - 1.0})


def compare_scenarios(metric_frames, cols=("sharpe", "max_drawdown", "cvar")):
    """H5: Kruskal-Wallis across scenario types, then pairwise Mann-Whitney with Holm."""
    names = list(metric_frames)
    rows = []
    for c in cols:
        groups = [metric_frames[n][c].dropna().values for n in names]
        kw = stats.kruskal(*groups)
        pairs, ps = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append(f"{names[i]} vs {names[j]}")
                ps.append(stats.mannwhitneyu(groups[i], groups[j]).pvalue)
        adj = holm(ps)
        row = {"measure": c, "kruskal_H": kw.statistic, "kruskal_p": kw.pvalue}
        for pr, p, a in zip(pairs, ps, adj):
            row[f"p[{pr}]"] = p
            row[f"p_holm[{pr}]"] = a
        rows.append(row)
    return pd.DataFrame(rows)
