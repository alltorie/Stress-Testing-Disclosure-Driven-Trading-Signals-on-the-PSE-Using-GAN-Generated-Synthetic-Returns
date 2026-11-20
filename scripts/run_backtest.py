"""Step 3: cost-aware walk-forward backtest of the disclosure signal (H3).

Run Step 1 first. Usage:
python scripts/run_backtest.py --data pse_research/run_2026-11-15 --out outputs
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from pse.config import Config, BacktestConfig
from pse.costs import CostModel
from pse import data, returns, backtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    cfg, base_bt = Config(), BacktestConfig()
    pd.set_option("display.width", 220)

    psei = data.load_psei(f"{args.data}/psei_clean.csv")
    psei_open = data.load_psei_open(f"{args.data}/psei_clean.csv")
    divs = data.load_dividends(f"{args.data}/dividends_parsed.csv")
    exdiv = returns.build_exdiv_table(divs, psei.index)
    ev = pd.read_csv(f"{args.out}/events_with_cars.csv", parse_dates=["ts", "t0"])
    ev = ev[(ev.status == "ok") & (~ev.suspect.astype(bool)) & ev.liquid.astype(bool)].copy()
    prices = data.load_prices(f"{args.data}/prices_10y", ev["cmpy_id"].unique())

    costs = {"full_costs": CostModel(),
             "no_commission": CostModel(commission=0.0),
             "no_spread": CostModel(spread_mult=0.0),
             "zero_costs": CostModel(commission=0.0, pse_fee=0, sccp_fee=0, src_fee=0,
                                     stt_sell=0, spread_mult=0.0)}
    grid, primary = [], None
    for entry in ["close_t0", "open_t0"]:
        for hold in [2, 5]:
            bt = dataclasses.replace(base_bt, entry=entry, hold_days=hold)
            trades = backtest.build_trades(ev, prices, psei, psei_open, exdiv, cfg, bt, costs["full_costs"])
            for cname, cm in costs.items():
                t2 = backtest.apply_costs(trades, cm, bt)
                taken, log = backtest.walk_forward_rule(t2, bt)
                row = {"entry": entry, "hold": hold, "costs": cname,
                       **backtest.h3_summary(taken)}
                grid.append(row)
                if entry == "close_t0" and hold == 2 and cname == "full_costs":
                    primary = (t2, taken, log)

    t2, taken, log = primary
    print("Primary: entry at close of t0, hold 2 sessions, full Philippine costs, liquid sample")
    print("Skip reasons:\n", t2["skip"].fillna("traded").value_counts().to_string())
    print("\nSelection by month (types traded):\n", log.to_string(index=False))
    print("\nTest-period trades by type:\n", taken["event_type"].value_counts().to_string())
    prim = pd.DataFrame([backtest.h3_summary(taken)])
    print("\nH3 (one-sided, Ha: mean net return per trade > 0):")
    print(prim.round(5).T.to_string(header=False))
    print(f"\nMedian participation of a P{base_bt.notional:,.0f} trade: "
          f"{taken['participation'].median():.1%}; ex-dividend inside hold: "
          f"{int(taken['exdiv_in_hold'].sum())} trades")
    print("\nSensitivity (walk-forward rule):")
    g = pd.DataFrame(grid)
    print(g[["entry", "hold", "costs", "n", "mean_gross", "mean_cost", "mean_net",
             "p_t_greater", "mean_net_excess"]].round(5).to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    taken.to_csv(f"{args.out}/trades_primary.csv", index=False)
    log.to_csv(f"{args.out}/selection_log.csv", index=False)
    prim.to_csv(f"{args.out}/h3_primary.csv", index=False)
    g.to_csv(f"{args.out}/h3_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
