"""Step 1: PSEi-adjusted event study for dividends, buy-backs and leadership changes.

Usage: python scripts/run_event_study.py --data pse_research/run_2026-11-15 --out outputs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from pse.config import Config
from pse import data, events, returns, eventstudy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    cfg = Config()
    os.makedirs(args.out, exist_ok=True)

    listing = data.load_listing(f"{args.data}/listing_all.csv")
    divs = data.load_dividends(f"{args.data}/dividends_parsed.csv")
    psei = data.load_psei(f"{args.data}/psei_clean.csv")
    directory = data.load_directory(f"{args.data}/company_directory.csv")
    calendar = psei.index

    ev = events.build_events(listing, divs, calendar, cfg)
    prices = data.load_prices(f"{args.data}/prices_10y", ev["cmpy_id"].unique())
    exdiv = returns.build_exdiv_table(divs, calendar)
    cars = eventstudy.compute_cars(ev, prices, psei, exdiv, cfg)
    cars = cars.merge(directory[["cmpy_id", "symbol", "sector"]], on="cmpy_id", how="left")

    cars.to_csv(f"{args.out}/events_with_cars.csv", index=False)
    for sample in ["liquid", "all"]:
        summ = eventstudy.summarize(cars, cfg, sample)
        summ.to_csv(f"{args.out}/h1_summary_{sample}.csv", index=False)

    print("Disclosures loaded:", len(listing))
    print("Events by type:\n", ev["event_type"].value_counts().to_string())
    print("Session bucket:\n", ev["bucket"].value_counts().to_string())
    print("Status:\n", cars["status"].value_counts().to_string())
    ok = cars[cars.status == "ok"]
    print(f"suspect={int(ok.suspect.sum())}  stale_anchor={int(ok.stale_anchor.sum())}  "
          f"liquid={int(ok.liquid.sum())} of {len(ok)}")
    pd.set_option("display.width", 200)
    print("\nH1 summary (liquid sample, primary window first):")
    print(eventstudy.summarize(cars, cfg, "liquid").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
