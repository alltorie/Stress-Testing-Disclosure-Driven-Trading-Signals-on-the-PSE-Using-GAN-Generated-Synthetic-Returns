import numpy as np
import pytest
from pse import scenarios as sc


def test_bootstrap_shape_and_membership():
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    p = sc.bootstrap_paths(r, 20, 60, seed=1)
    assert p.shape == (20, 60) and np.isin(p, r).all()
    assert np.array_equal(p, sc.bootstrap_paths(r, 20, 60, seed=1))


def test_excess_kurtosis_of_normal_is_near_zero_and_fat_tails_positive():
    rng = np.random.default_rng(1)
    assert abs(sc.excess_kurtosis(rng.normal(size=200000))) < 0.1
    assert sc.excess_kurtosis(rng.standard_t(4, 200000)) > 1.5


def test_acf_squared_detects_volatility_clustering():
    rng = np.random.default_rng(2)
    iid = rng.normal(size=(1, 20000))
    vol = np.ones(20000)
    for t in range(1, 20000):                       # simple ARCH(1)
        vol[t] = np.sqrt(0.2 + 0.7 * (vol[t - 1] * rng.normal()) ** 2)
    arch_like = vol * rng.normal(size=20000)
    assert abs(sc.acf_squared(iid)[0]) < 0.03
    assert sc.acf_squared(arch_like[None, :])[0] > 0.2


def test_historical_windows_are_contiguous():
    r = np.arange(100.0)
    w = sc.historical_windows(r, 10, step=5)
    assert w.shape[1] == 10 and (np.diff(w, axis=1) == 1).all()


def test_garch_paths_are_reproducible_fat_tailed_and_clustered():
    pytest.importorskip("arch")
    rng = np.random.default_rng(3)
    vol = np.ones(3000); x = np.empty(3000)
    for t in range(3000):                           # data with real clustering
        x[t] = vol[t] * rng.standard_t(6) * 0.006
        if t + 1 < 3000:
            vol[t + 1] = np.sqrt(0.1 + 0.15 * (x[t] / 0.006) ** 2 + 0.75 * vol[t] ** 2)
    res = sc.fit_garch(x)
    a = sc.garch_paths(res, 6, 200, seed=5)
    assert a.shape == (6, 200) and np.array_equal(a, sc.garch_paths(res, 6, 200, seed=5))
    long = sc.garch_paths(res, 10, 1500, seed=6)
    assert np.median(sc.excess_kurtosis(long)) > 0.3
