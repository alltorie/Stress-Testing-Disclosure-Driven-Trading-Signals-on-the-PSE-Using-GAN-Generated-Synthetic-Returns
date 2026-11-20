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


@dataclass(frozen=True)
class BacktestConfig:
    # Long-only. Entry at the close of t0 is the conservative, always-tradable choice.
    entry: str = "close_t0"            # "close_t0" or "open_t0" (pre-open filings only)
    hold_days: int = 2                 # exit at the close of t0 + hold_days
    notional: float = 100_000.0        # peso size per trade, used for participation only
    train_months: int = 12
    min_train_events: int = 20
    select_t: float = 1.96             # a type is traded if its training mean CAR is positive with t above this


@dataclass(frozen=True)
class StressConfig:
    n_paths: int = 500                 # simulated paths per scenario type
    window_len: int = 250              # length of windows used for the stylized-fact tests (H4)
    seed: int = 42
    ann_days: int = 252                # annualization for the Sharpe ratio
    cvar_level: float = 0.05
    # Quant GAN settings, fixed before looking at stylized-fact results
    gan_hidden: int = 48
    gan_layers: int = 7                # kernel 2, dilations 1..64, receptive field 128
    gan_noise_dim: int = 3
    gan_train_len: int = 255
    gan_iters: int = 3000
    gan_batch: int = 32
    gan_lr: float = 2e-4
