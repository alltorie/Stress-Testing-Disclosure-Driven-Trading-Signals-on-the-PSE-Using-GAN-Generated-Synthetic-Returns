import numpy as np
import pandas as pd
from scipy import stats
from .returns import adjusted_log_price


def compute_cars(events, prices, psei, exdiv, cfg):
    """Market-adjusted cumulative abnormal returns for every event and window.

    CAR(a, b) = [lnP_adj(t0+b) - lnP_adj(t0+a-1)] - [lnPSEi(t0+b) - lnPSEi(t0+a-1)]
    Prices are dividend-adjusted. Rows carry flags so the analysis can filter.
    """
    calendar = psei.index
    ln_m = np.log(psei)
    windows = [cfg.window] + [w for w in cfg.extra_windows if w != cfg.window]
    rows = []
    for cid, g in events.groupby("cmpy_id"):
        if cid not in prices:
            for _, e in g.iterrows():
                rows.append({"edge_no": e["edge_no"], "status": "no_prices"})
            continue
        adj, age, bad = adjusted_log_price(
            prices[cid], calendar, exdiv[exdiv.cmpy_id == cid], cfg)
        val = prices[cid]["VALUE"].reindex(calendar).fillna(0.0)
        med = (val.rolling(cfg.liquidity_lookback, min_periods=20).median().shift(1))
        for _, e in g.iterrows():
            i0 = calendar.get_loc(e["t0"])
            r = {"edge_no": e["edge_no"], "status": "ok",
                 "median_value": med.iloc[i0]}
            for a, b in windows:
                s, t = i0 + a - 1, i0 + b
                if s < 0 or t >= len(calendar):
                    r[f"car_{a}_{b}"] = np.nan
                    r["status"] = "window_outside_data"
                    continue
                car = (adj.iloc[t] - adj.iloc[s]) - (ln_m.iloc[t] - ln_m.iloc[s])
                r[f"car_{a}_{b}"] = car
                if bad.iloc[s + 1:t + 1].any():
                    r["suspect"] = True
                if age.iloc[s] > 0 or age.iloc[t] > 0:
                    r["stale_anchor"] = True
            rows.append(r)
    res = pd.DataFrame(rows)
    for c in ["suspect", "stale_anchor"]:
        if c not in res:
            res[c] = False
        res[c] = res[c].fillna(False).astype(bool)
    out = events.merge(res, on="edge_no", how="left")
    out["liquid"] = out["median_value"] >= cfg.min_median_value
    return out


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(cars, cfg, sample="liquid"):
    """H1: mean CAR is zero for each disclosure type. t-test, Wilcoxon, Holm across types."""
    d = cars[(cars.status == "ok") & (~cars.suspect)]
    if sample == "liquid":
        d = d[d.liquid]
    out = []
    windows = [cfg.window] + [w for w in cfg.extra_windows if w != cfg.window]
    for a, b in windows:
        col = f"car_{a}_{b}"
        block = []
        for et, g in d.groupby("event_type"):
            x = g[col].dropna().values
            if len(x) < 3:
                block.append({"event_type": et, "window": f"[{a},{b}]", "n": len(x)})
                continue
            t = stats.ttest_1samp(x, 0.0)
            try:
                w = stats.wilcoxon(x).pvalue
            except ValueError:
                w = np.nan
            block.append({"event_type": et, "window": f"[{a},{b}]", "n": len(x),
                          "mean_car": x.mean(), "median_car": np.median(x),
                          "std": x.std(ddof=1), "t_stat": t.statistic,
                          "p_t": t.pvalue, "p_wilcoxon": w})
        bdf = pd.DataFrame(block)
        if "p_t" in bdf:
            bdf["p_t_holm"] = holm(bdf["p_t"].fillna(1.0))
        out.append(bdf)
    return pd.concat(out, ignore_index=True)
