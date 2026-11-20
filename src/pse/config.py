"""Study parameters. Fix these BEFORE looking at results (pre-specification)."""
from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class Config:
    # Filings stamped at or after this clock time can only be traded next session.
    session_cutoff: time = time(15, 30)
    session_open: time = time(9, 30)
    # Primary event window in trading days relative to t0 (first tradable session).
    window: tuple = (0, 2)
    # Robustness windows reported next to the primary one.
    extra_windows: tuple = ((0, 0), (0, 1), (0, 5))
    # Buy-back and leadership filings repeat; keep only the first filing of a cluster.
    cluster_gap_days: int = 10
    # Thin-trading filter: median daily traded value (PHP) over the prior 60 sessions.
    min_median_value: float = 1_000_000.0
    liquidity_lookback: int = 60
    # Anchor prices older than this many sessions are treated as stale.
    max_stale_days: int = 2
    # A one-day move above this (log) is treated as a possible split or bad tick.
    jump_threshold: float = 0.40
    # Dividend larger than this share of the prior close is not trusted for adjustment.
    max_div_ratio: float = 0.20
    alpha: float = 0.05
