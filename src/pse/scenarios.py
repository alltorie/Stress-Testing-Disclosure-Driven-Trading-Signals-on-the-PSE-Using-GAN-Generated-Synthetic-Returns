"""Background-market generators (bootstrap, GARCH) and stylized-fact statistics."""
import numpy as np
from scipy import stats


def log_returns(prices):
    return np.log(prices).diff().dropna()


def bootstrap_paths(r, n_paths, T, seed):
    """Resample real daily returns with replacement (iid)."""
    rng = np.random.default_rng(seed)
    return np.asarray(r)[rng.integers(0, len(r), (n_paths, T))]


def fit_garch(r):
    """GARCH(1,1) with Student-t innovations on log returns (decimal). Plain GARCH: o=0."""
    from arch.univariate import ConstantMean, GARCH, StudentsT
    am = ConstantMean(np.asarray(r) * 100.0)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = StudentsT()
    return am.fit(disp="off")


def garch_paths(res, n_paths, T, seed, burn=500):
    """Simulate from the fitted GARCH. A burn-in removes dependence on the starting state."""
    from arch.univariate import StudentsT
    am = res.model
    ss = np.random.SeedSequence(seed).spawn(n_paths)
    out = np.empty((n_paths, T))
    for i, s in enumerate(ss):
        am.distribution = StudentsT(seed=np.random.default_rng(s))
        out[i] = am.simulate(res.params, nobs=T, burn=burn)["data"].values / 100.0
    return out


def historical_windows(r, T, step=5):
    """Contiguous windows of the real history (overlapping): historical simulation."""
    r = np.asarray(r)
    return np.stack([r[i:i + T] for i in range(0, len(r) - T + 1, step)])


def excess_kurtosis(x):
    return stats.kurtosis(x, axis=-1, fisher=True)


def acf_squared(x, lag=1):
    """Autocorrelation of squared returns at `lag`, computed row by row."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    s = x ** 2 - (x ** 2).mean(axis=1, keepdims=True)
    num = (s[:, lag:] * s[:, :-lag]).sum(axis=1)
    den = (s ** 2).sum(axis=1)
    return num / den


def stylized_stats(paths):
    paths = np.atleast_2d(paths)
    return {"excess_kurtosis": excess_kurtosis(paths), "acf_sq_lag1": acf_squared(paths, 1),
            "acf_sq_mean_1_10": np.mean([acf_squared(paths, k) for k in range(1, 11)], axis=0)}


def stationary_bootstrap_ci(r, stat_fn, n_boot=1000, mean_block=20, seed=0, level=0.95):
    """Politis-Romano stationary bootstrap CI for a statistic of one long return series."""
    r = np.asarray(r)
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    vals = []
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        starts = rng.integers(0, n, n)
        for t in range(1, n):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
        vals.append(float(np.ravel(stat_fn(r[idx][None, :]))[0]))
    a = (1 - level) / 2
    return float(np.percentile(vals, 100 * a)), float(np.percentile(vals, 100 * (1 - a)))
