"""
amortized_ff.py
Fourier-feature amortized wave-field operator — fixes the spectral-bias mean-collapse of the
raw-linear-coordinate PosteriorNet/NeuralOperator (d52/d56/d58/d67/d68).

Root cause (verified 2026-07-02): raw Linear(x,y,t) embedding cannot represent oscillatory wave
fields (spectral bias, Tancik et al. 2020) -> the net collapses to the mean (nRMSE 0.25 = predict-
zero). With Fourier features + pre-norm transformer the same net overfits (train-rmse 0.43->0.036)
and generalizes. This module is the corrected operator.
"""
import numpy as np, torch, torch.nn as nn

XS, YS, TS = 120.0, 100.0, 10.0


class FourierFeatures(nn.Module):
    """Fourier features with frequencies MATCHED to the wave band (fixes the aliasing bug:
    old scale 0.3 gave effective wavelength ~3m << obs spacing ~9m -> net couldn't use coords).
    Spatial: k = 2*pi*|B_xy| log-spaced in [0.02, 0.6] rad/m, random directions.
    Temporal: omega = 2*pi*B_t log-spaced in [0.3, 3] rad/s."""
    def __init__(self, n=64, seed=0, learnable=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        mag = torch.exp(torch.linspace(np.log(0.02 / (2 * np.pi)), np.log(0.6 / (2 * np.pi)), n))
        ang = torch.rand(n, generator=g) * 2 * np.pi
        Bt = torch.exp(torch.linspace(np.log(0.3 / (2 * np.pi)), np.log(3.0 / (2 * np.pi)), n))
        B0 = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang), Bt])  # (3,n)
        if learnable:
            self.B = nn.Parameter(B0)     # net tunes the frequencies from the wave-band init
        else:
            self.register_buffer('B', B0)

    def forward(self, xyt):
        proj = xyt @ self.B
        return torch.cat([torch.sin(2 * np.pi * proj), torch.cos(2 * np.pi * proj)], -1)


class AmortizedFF(nn.Module):
    """obs (x,y,t,z) -> per-query (x,y,t) mean(,log_var). Fourier-feature coords + pre-norm transformer."""
    def __init__(self, d=160, heads=4, enc=4, dec=3, nff=32, uq=True, learnable_ff=True):
        super().__init__()
        self.uq = uq; self.ff = FourierFeatures(nff, learnable=learnable_ff); fdim = 2 * nff
        self.oenc = nn.Sequential(nn.Linear(fdim + 1, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc)
        self.qenc = nn.Sequential(nn.Linear(fdim, d), nn.GELU(), nn.Linear(d, d))
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu', norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 if uq else 1))

    def forward(self, obs, q):
        oin = torch.cat([self.ff(obs[..., :3]), obs[..., 3:4]], -1)
        ctx = self.enc(self.oenc(oin))
        h = self.dec(self.qenc(self.ff(q)), ctx)
        out = self.head(h)
        if self.uq:
            return out[..., 0], out[..., 1]
        return out[..., 0]
