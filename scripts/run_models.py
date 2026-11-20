"""Step 2: does TF-IDF text add out-of-sample AUC over an event-type-only baseline?

Run Step 1 first (outputs/events_with_cars.csv). Usage:
python scripts/run_models.py --data pse_research/run_2026-11-15 --out outputs [--sample all]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from pse.config import Config
from pse import data, features, models, validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--sample", choices=["liquid", "all"], default="liquid")
    ap.add_argument("--min_months", type=int, default=12)
    args = ap.parse_args()
    cfg = Config()
    a, b = cfg.window
    car_col = f"car_{a}_{b}"

    psei = data.load_psei(f"{args.data}/psei_clean.csv")
    cal = psei.index
    ev = pd.read_csv(f"{args.out}/events_with_cars.csv", parse_dates=["ts", "t0"])
    ok = (ev.status == "ok") & (~ev.suspect.astype(bool)) & ev[car_col].notna()
    if args.sample == "liquid":
        ok &= ev.liquid.astype(bool)
    ev = ev[ok].copy()
    ev["y"] = (ev[car_col] > 0).astype(int)
    ev["label_end"] = [cal[min(cal.get_loc(t) + b, len(cal) - 1)] for t in ev["t0"]]
    print(f"Sample={args.sample}: {len(ev)} events, share positive CAR = {ev.y.mean():.3f}")

    # Part A: pooled, all three types, type-only baseline
    predsA = models.run_walk_forward(ev, {"B0": ("B0", ["event_type"], [])}, train_months=args.min_months)
    print("\nPart A. All types, event-type-only baseline (walk-forward, pooled):")
    print(models.score(predsA, ["B0"]).round(4).to_string(index=False))

    # Part B: cash dividends with parsed text
    div = ev[ev.event_type == "dividend"].copy()
    texts = features.load_dividend_texts(f"{args.data}/text_dividends.jsonl")
    div = div.merge(texts, on="edge_no", how="left")
    div["text_clean"] = [features.clean_dividend_text(t, c) for t, c in zip(div["text"], div["company"])]
    prices = data.load_prices(f"{args.data}/prices_10y", div["cmpy_id"].unique())
    div["yield"] = features.add_dividend_yield(div, prices, cal)
    div["log_yield"] = np.log1p(div["yield"])
    div["div_type"] = div["div_type"].fillna("Unknown")
    div = div.sort_values("t0").reset_index(drop=True)

    specs = {"B0": ("B0", ["div_type"], []),
             "S": ("S", ["div_type", "bucket"], ["log_yield"]),
             "T": ("T", ["div_type"], []),
             "G": ("G", ["div_type", "bucket"], ["log_yield"])}
    predsB = models.run_walk_forward(div, specs, train_months=args.min_months)
    sc = models.score(predsB, list(specs))
    print(f"\nPart B. Cash dividends: {len(div)} events, {len(predsB)} out-of-sample, "
          f"{predsB.fold.nunique()} test months")
    print(sc.round(4).to_string(index=False))

    print("\nH2: paired bootstrap of AUC difference (one-sided, Ha: text raises AUC)")
    rows = []
    for new, base in [("T", "B0"), ("T", "S"), ("G", "B0"), ("G", "S")]:
        r = validation.paired_auc_bootstrap(predsB["y"], predsB[f"p_{new}"], predsB[f"p_{base}"])
        rows.append({"new": new, "base": base, **r})
    h2 = pd.DataFrame(rows)
    print(h2.round(4).to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    predsA.to_csv(f"{args.out}/oos_predictions_all_types.csv", index=False)
    predsB.to_csv(f"{args.out}/oos_predictions_dividends.csv", index=False)
    sc.to_csv(f"{args.out}/model_scores_dividends.csv", index=False)
    h2.to_csv(f"{args.out}/h2_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
