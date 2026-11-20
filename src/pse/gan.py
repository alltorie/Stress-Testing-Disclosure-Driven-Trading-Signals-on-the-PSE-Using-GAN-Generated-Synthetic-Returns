"""Quant GAN (Wiese et al., 2020): temporal convolutional generator and discriminator trained on
Lambert W-Gaussianized returns. Univariate, trained on PSEi log returns."""
import numpy as np
from scipy.special import lambertw


def lw_forward(z, delta):
    """Gaussianize heavy tails: w = sign(z) * sqrt(W(delta z^2) / delta)."""
    if delta <= 0:
        return z.copy()
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)


def lw_inverse(w, delta):
    """Exact inverse of lw_forward: z = w * exp(delta w^2 / 2)."""
    return w if delta <= 0 else w * np.exp(delta * w ** 2 / 2.0)


def fit_preprocessing(r):
    """Standardize, pick the Lambert W delta that brings excess kurtosis closest to zero, standardize again."""
    from scipy import stats
    r = np.asarray(r, dtype=float)
    mu, sd = r.mean(), r.std()
    z = (r - mu) / sd
    grid = np.linspace(0.0, 2.0, 81)
    kurt = [abs(stats.kurtosis(lw_forward(z, d))) for d in grid]
    delta = float(grid[int(np.argmin(kurt))])
    w = lw_forward(z, delta)
    mw, sw = w.mean(), w.std()
    return {"mu": mu, "sd": sd, "delta": delta, "mw": mw, "sw": sw}, (w - mw) / sw


def to_returns(u, pre):
    w = u * pre["sw"] + pre["mw"]
    return lw_inverse(w, pre["delta"]) * pre["sd"] + pre["mu"]


def _tcn(torch, c_in, hidden, c_out, n_layers, k=2):
    nn = torch.nn
    layers, c = [], c_in
    for l in range(n_layers):
        layers += [nn.Conv1d(c, hidden, k, dilation=2 ** l), nn.PReLU()]
        c = hidden
    return nn.Sequential(*layers, nn.Conv1d(hidden, c_out, 1))


class QuantGAN:
    def __init__(self, scfg, seed=0):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.torch, self.cfg, self.seed = torch, scfg, seed
        self.rf = 2 ** scfg.gan_layers
        self.G = _tcn(torch, scfg.gan_noise_dim, scfg.gan_hidden, 1, scfg.gan_layers)
        self.D = _tcn(torch, 1, scfg.gan_hidden, 1, scfg.gan_layers)
        self.pre = None
        self.history = []

    def fit(self, r, log_every=0):
        torch, cfg = self.torch, self.cfg
        self.pre, u = fit_preprocessing(r)
        u = torch.tensor(u, dtype=torch.float32)
        Lw, B = cfg.gan_train_len, cfg.gan_batch
        assert Lw > self.rf, "training window must exceed the receptive field"
        gopt = torch.optim.Adam(self.G.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))
        dopt = torch.optim.Adam(self.D.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))
        bce = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed)
        n = len(u)
        for it in range(cfg.gan_iters):
            starts = torch.randint(0, n - Lw, (B,), generator=g)
            real = torch.stack([u[s:s + Lw] for s in starts]).unsqueeze(1)
            z = torch.randn(B, cfg.gan_noise_dim, Lw + self.rf - 1, generator=g)
            fake = self.G(z)
            dreal = self.D(real).mean(dim=-1)
            dfake = self.D(fake.detach()).mean(dim=-1)
            dloss = bce(dreal, torch.full_like(dreal, 0.9)) + bce(dfake, torch.zeros_like(dfake))
            dopt.zero_grad(); dloss.backward(); dopt.step()
            gloss = bce(self.D(fake).mean(dim=-1), torch.ones(B, 1))
            gopt.zero_grad(); gloss.backward(); gopt.step()
            if it % 100 == 0:
                self.history.append((it, float(dloss.detach()), float(gloss.detach())))
            if log_every and it % log_every == 0:
                print(f"  iter {it}: D {float(dloss.detach()):.3f} G {float(gloss.detach()):.3f}", flush=True)
        return self

    def sample(self, n_paths, T, seed=0):
        """Generate n_paths return series of length T, in the original (decimal log return) scale."""
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        self.G.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n_paths, 100):
                m = min(100, n_paths - i)
                z = torch.randn(m, self.cfg.gan_noise_dim, T + self.rf - 1, generator=g)
                out.append(self.G(z).squeeze(1).numpy())
        self.G.train()
        return to_returns(np.concatenate(out), self.pre)
