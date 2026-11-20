# PSE disclosure event study (Step 1)

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_event_study.py --data <path>/pse_research/run_2026-11-15 --out outputs

Modules in src/pse: config (all study parameters), data (loaders), events (event types,
first tradable session), returns (dividend parsing and adjustment), eventstudy (CAR, H1 tests).
Outputs: outputs/events_with_cars.csv, outputs/h1_summary_liquid.csv, outputs/h1_summary_all.csv
