# PSE disclosure event study (Step 1)

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_event_study.py --data <path>/pse_research/run_2026-11-15 --out outputs

Modules in src/pse: config (all study parameters), data (loaders), events (event types,
first tradable session), returns (dividend parsing and adjustment), eventstudy (CAR, H1 tests).
Outputs: outputs/events_with_cars.csv, outputs/h1_summary_liquid.csv, outputs/h1_summary_all.csv

## Step 2 (prediction models)

    python scripts/run_models.py --data <path>/pse_research/run_2026-11-15 --out outputs
    python scripts/run_models.py --data <path>/pse_research/run_2026-11-15 --out outputs_all --sample all

Needs outputs/events_with_cars.csv from Step 1 and text_dividends.jsonl in the data folder.
Outputs: model_scores_dividends.csv, h2_bootstrap.csv, oos_predictions_*.csv

## Step 3 (cost-aware backtest, H3)

    python scripts/run_backtest.py --data <path>/pse_research/run_2026-11-15 --out outputs

Needs outputs/events_with_cars.csv from Step 1. Costs follow the PSE fee schedule (src/pse/costs.py).
Outputs: trades_primary.csv, selection_log.csv, h3_primary.csv, h3_sensitivity.csv

## Step 4 (GAN, GARCH and bootstrap stress test, H4 and H5)

    pip install torch arch
    python scripts/run_stress.py --data <path>/pse_research/run_2026-11-15 --out outputs

Needs outputs/events_with_cars.csv (Step 1) and outputs/trades_primary.csv (Step 3). Add --retrain to
retrain the Quant GAN, otherwise outputs/quantgan.pt is reused. Training takes about 10 minutes on one CPU.
Outputs: h4_stylized_facts.csv, h5_before_costs.csv, h5_after_costs.csv, stress_summary.csv,
real_strategy_metrics.csv, quantgan.pt
