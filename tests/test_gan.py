import numpy as np
import pytest
from pse.config import StressConfig
from pse.gan import fit_preprocessing, to_returns

torch = pytest.importorskip("torch")
from pse.gan import QuantGAN


def test_preprocessing_roundtrip_and_gaussianizes_tails():
    from scipy import stats
    r = np.random.default_rng(0).standard_t(3, 3000) * 0.01
    pre, u = fit_preprocessing(r)
    assert abs(u.mean()) < 1e-9 and u.std() == pytest.approx(1.0)
    assert abs(stats.kurtosis(u)) < abs(stats.kurtosis(r))
    assert np.allclose(to_returns(u, pre), r)


def test_gan_trains_and_samples_requested_shape():
    cfg = StressConfig(gan_iters=6, gan_batch=8, gan_hidden=8, gan_layers=5, gan_train_len=70)
    r = np.random.default_rng(1).normal(0, 0.01, 400)
    gan = QuantGAN(cfg, seed=0).fit(r)
    x = gan.sample(5, 40, seed=1)
    assert x.shape == (5, 40) and np.isfinite(x).all()
    assert np.array_equal(x, gan.sample(5, 40, seed=1))
