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
