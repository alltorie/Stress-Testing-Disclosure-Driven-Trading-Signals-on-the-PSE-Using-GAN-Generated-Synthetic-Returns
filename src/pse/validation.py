import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def walk_forward_splits(t0, label_end, train_months=12):
    """Expanding-window walk-forward with purging.

    The first test month comes `train_months` after the first event. Training uses only
    events with t0 before the test month AND whose label window ended before it, so no
    training label overlaps test-period prices.
    """
    t0 = pd.Series(pd.to_datetime(np.asarray(t0))).reset_index(drop=True)
    le = pd.Series(pd.to_datetime(np.asarray(label_end))).reset_index(drop=True)
    first = t0.min().to_period("M") + train_months
    last = t0.max().to_period("M")
    for m in pd.period_range(first, last, freq="M"):
        start, end = m.start_time, m.end_time
        test = np.where((t0 >= start) & (t0 <= end))[0]
        train = np.where((t0 < start) & (le < start))[0]
        if len(test) and len(train):
            yield m, train, test


def paired_auc_bootstrap(y, p_new, p_base, n_boot=2000, seed=0):
    """Paired bootstrap of AUC(new) - AUC(base) over test events.

    One-sided p = share of resampled differences that are <= 0 (Ha: new raises AUC).
    """
    y, p_new, p_base = map(np.asarray, (y, p_new, p_base))
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = roc_auc_score(y, p_new) - roc_auc_score(y, p_base)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i]))
    d = np.array(diffs)
    return {"auc_diff": obs, "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
            "p_one_sided": float((d <= 0).mean()), "n_boot_used": len(d)}
