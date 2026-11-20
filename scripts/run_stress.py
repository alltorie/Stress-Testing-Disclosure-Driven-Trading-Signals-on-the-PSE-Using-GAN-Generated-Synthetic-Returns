"""Step 4: GAN, GARCH and bootstrap stress test of the disclosure strategy (H4, H5).

Run Steps 1 and 3 first. Usage:
python scripts/run_stress.py --data pse_research/run_2026-11-15 --out outputs [--retrain]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from scipy import stats
from pse.config import Config, StressConfig
from pse.costs import CostModel
from pse import data, returns, scenarios, stress
from pse.eventstudy import holm


def get_gan(r, scfg, path, retrain):
    import torch
    from pse.gan import QuantGAN
    gan = QuantGAN(scfg, seed=scfg.seed)
    if os.path.exists(path) and not retrain:
        st = torch.load(path, weights_only=False)
        gan.G.load_state_dict(st["G"]); gan.D.load_state_dict(st["D"])
        gan.pre, gan.history = st["pre"], st["history"]
        print("Loaded trained GAN from", path)
    else:
        print(f"Training Quant GAN: {scfg.gan_iters} iterations on {len(r)} PSEi returns")
        gan.fit(r, log_every=500)
        torch.save({"G": gan.G.state_dict(), "D": gan.D.state_dict(),
                    "pre": gan.pre, "history": gan.history}, path)
    return gan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    cfg, scfg = Config(), StressConfig()
    pd.set_option("display.width", 220)
    os.makedirs(args.out, exist_ok=True)

    psei = data.load_psei(f"{args.data}/psei_clean.csv")
    r = scenarios.log_returns(psei).values
    print(f"PSEi: {len(psei)} days, {len(r)} returns, "
          f"{psei.index.min().date()} to {psei.index.max().date()}")

    # ---------- generators ----------
    garch = scenarios.fit_garch(r)
    print("GARCH(1,1)-t:", {k: round(v, 4) for k, v in garch.params.items()})
    gan = get_gan(r, scfg, f"{args.out}/quantgan.pt", args.retrain)
    print(f"Lambert W delta = {gan.pre['delta']:.2f}")

    # ---------- H4: stylized facts ----------
    L, P = scfg.window_len, scfg.n_paths
    real_w = scenarios.historical_windows(r, L, step=5)
    gens = {"bootstrap": scenarios.bootstrap_paths(r, P, L, scfg.seed + 1),
            "garch": scenarios.garch_paths(garch, P, L, scfg.seed + 2),
            "gan": gan.sample(P, L, seed=scfg.seed + 3)}
    real_stats = scenarios.stylized_stats(r[None, :])
    ci = {"excess_kurtosis": scenarios.stationary_bootstrap_ci(
              r, scenarios.excess_kurtosis, n_boot=300, seed=1),
          "acf_sq_lag1": scenarios.stationary_bootstrap_ci(
              r, lambda x: scenarios.acf_squared(x, 1), n_boot=300, seed=2)}
    rw = scenarios.stylized_stats(real_w)
    rows = []
    for name, paths in gens.items():
        gs = scenarios.stylized_stats(paths)
        for stat in ["excess_kurtosis", "acf_sq_lag1"]:
            ks = stats.ks_2samp(rw[stat], gs[stat])
            rows.append({"generator": name, "statistic": stat,
                         "real_full_sample": float(real_stats[stat][0]),
                         "real_ci_low": ci[stat][0], "real_ci_high": ci[stat][1],
                         "gen_median": float(np.median(gs[stat])),
                         "real_window_median": float(np.median(rw[stat])),
                         "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                         "std_ratio_gen_to_real": float(paths.std() / r.std())})
    h4 = pd.DataFrame(rows)
    h4["ks_p_holm"] = np.nan
    for name in gens:
        m = h4.generator == name
        h4.loc[m, "ks_p_holm"] = holm(h4.loc[m, "ks_p"].values)
    print("\nH4: stylized facts, windows of", L, "days (real windows overlap, so KS p-values are optimistic)")
    print(h4.round(4).to_string(index=False))
    h4.to_csv(f"{args.out}/h4_stylized_facts.csv", index=False)

    # ---------- strategy ----------
    divs = data.load_dividends(f"{args.data}/dividends_parsed.csv")
    exdiv = returns.build_exdiv_table(divs, psei.index)
    ev = pd.read_csv(f"{args.out}/events_with_cars.csv", parse_dates=["ts", "t0"])
    ev = ev[(ev.status == "ok") & (~ev.suspect.astype(bool)) & ev.liquid.astype(bool)]
    trades = pd.read_csv(f"{args.out}/trades_primary.csv", parse_dates=["t0"])
    cids = pd.concat([ev["cmpy_id"], trades["cmpy_id"]]).unique()
    prices = data.load_prices(f"{args.data}/prices_10y", cids)
    betas, resids, pooled = stress.estimate_stock_models(trades["cmpy_id"].unique(), prices,
                                                         psei, exdiv, cfg)
    ar_table = stress.abnormal_return_by_day(ev[ev.event_type.isin(trades.event_type.unique())])
    print("\nFixed event effect (mean abnormal return on day +1, +2):",
          {k: [round(x, 4) for x in v] for k, v in ar_table.items()})
    print(f"Stock models: {len(betas)} of {trades.cmpy_id.nunique()} traded companies have their own "
          f"beta (mean {np.mean(list(betas.values())):.2f}); the rest use beta 1 and pooled residuals")

    results, real_rows, summary = {}, [], []
    for label, costs_on in [("before_costs", False), ("after_costs", True)]:
        spec = stress.trade_specs(trades, psei, prices, exdiv, cfg, CostModel(), hold=2)
        T = spec["T"]
        if label == "before_costs":
            print(f"\nStrategy window: {T} sessions, {len(spec['trades'])} trades, "
                  f"position weight {spec['w']:.3f} (max {spec['max_load']} open at once)")
        mk = {"bootstrap": scenarios.bootstrap_paths(r, P, T + 1, scfg.seed + 11),
              "garch": scenarios.garch_paths(garch, P, T + 1, scfg.seed + 12),
              "gan": gan.sample(P, T + 1, seed=scfg.seed + 13)}
        hist = scenarios.historical_windows(r, T + 1, step=5)
        frames = {n: stress.metrics(stress.simulate_series(spec, m, betas, resids, pooled, ar_table,
                                                          seed=scfg.seed + 99, costs_on=costs_on),
                                    scfg.ann_days, scfg.cvar_level) for n, m in mk.items()}
        hist_frame = stress.metrics(stress.simulate_series(spec, hist, betas, resids, pooled, ar_table,
                                                           seed=scfg.seed + 99, costs_on=costs_on),
                                    scfg.ann_days, scfg.cvar_level)
        real = stress.metrics(stress.real_series(spec, costs_on), scfg.ann_days, scfg.cvar_level).iloc[0]
        h5 = stress.compare_scenarios(frames)
        print(f"\n===== {label.upper()} =====")
        print("Real history (single value):", real.round(4).to_dict())
        for n, f in list(frames.items()) + [("historical_windows", hist_frame)]:
            q = f.quantile([0.05, 0.5, 0.95])
            tail5 = f["total_return"].nsmallest(max(1, int(0.05 * len(f)))).mean()
            tail1 = f["total_return"].nsmallest(max(1, int(0.01 * len(f)))).mean()
            print(f"{n:>18}: Sharpe med {q.loc[0.5,'sharpe']:.2f} [{q.loc[0.05,'sharpe']:.2f},{q.loc[0.95,'sharpe']:.2f}]  "
                  f"MDD med {q.loc[0.5,'max_drawdown']:.4f} [{q.loc[0.05,'max_drawdown']:.4f},{q.loc[0.95,'max_drawdown']:.4f}]  "
                  f"CVaR med {q.loc[0.5,'cvar']:.4f} [{q.loc[0.05,'cvar']:.4f},{q.loc[0.95,'cvar']:.4f}]  "
                  f"path-tail loss 5%/1%: {tail5:.4f}/{tail1:.4f}")
            summary.append({"costs": label, "scenario": n, **{f"{c}_p50": q.loc[0.5, c] for c in q.columns},
                            **{f"{c}_p05": q.loc[0.05, c] for c in q.columns},
                            **{f"{c}_p95": q.loc[0.95, c] for c in q.columns},
                            "tail_total_return_5pct": tail5, "tail_total_return_1pct": tail1,
                            "real_sharpe_pctile": float((f["sharpe"] < real["sharpe"]).mean())})
        print("\nH5 (Kruskal-Wallis across bootstrap, GARCH, GAN; pairwise Mann-Whitney, Holm):")
        print(h5.round(4).T.to_string(header=False))
        h5.to_csv(f"{args.out}/h5_{label}.csv", index=False)
        real_rows.append({"costs": label, **real.to_dict()})
        results[label] = frames
    pd.DataFrame(summary).to_csv(f"{args.out}/stress_summary.csv", index=False)
    pd.DataFrame(real_rows).to_csv(f"{args.out}/real_strategy_metrics.csv", index=False)


if __name__ == "__main__":
    main()
