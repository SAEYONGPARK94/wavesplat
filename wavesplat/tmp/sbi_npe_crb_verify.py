"""SBI/NPE keystone(3) empirical verification of the Gaussian-approximation CRB
(E-51/52), coordinator-assigned 2026-07-22.

Question: E-51/52's Cramer-Rao bound on the 5-dim regime-shape parameter
theta=(d1,Tp1,d2,Tp2,w) is a LOCAL, LINEARIZED Gaussian approximation (FIM via
central finite differences of C(theta) around the true theta). Does that
approximation's implied posterior WIDTH match the TRUE posterior width/shape
one gets from simulation-based inference (many (theta,z) draws, amortized
neural posterior) under the EXACT SAME assumed model z ~ N(0, C(theta))?

Scope (explicit, to avoid overclaiming): this tests the CRB's local-Gaussian
approximation quality UNDER A WELL-SPECIFIED MODEL (data really drawn from the
same ensemble-Gaussian z~N(0,C(theta)) the CRB analysis assumes) -- it does
NOT test model misspecification against the true nonlinear random-phase
physical process (a separate, already partially addressed question, see
tmp/F1_gaussianity_2026-07-22.md).

Tp1 note (MEASURED via code trace, important scope limitation): build_sea_crosstp
(the arena X scene generator, tmp/e37_bimodal_ladder.py) holds Tp1 FIXED at 7.0
across all scenes -- it is not a free/varying generative parameter in this
arena. NPE (a population-based estimator) cannot learn anything about the
Tp1 posterior from data with zero population variation in Tp1. This script
therefore restricts the empirical NPE comparison to the 4 dims that DO vary:
(d1, d2, Tp2, w). CRB(Tp1) remains a well-defined LOCAL finite-difference
sensitivity in E-51/52 but is not empirically testable here without
introducing an artificial prior not present in the arena's own generator.

No sbi package installed (checked) -> self-built Mixture Density Network
(MDN) amortized posterior estimator (standard SBI building block predating
normalizing-flow NPE, Papamakarios & Murray 2016), per instructions' fallback.

Reuses tmp/e51_crb.py's build_C/fim_crb/JointEngine/build_sea_crosstp DIRECTLY
(not reimplemented) so the z~N(0,C(theta)) simulator and the analytic CRB
comparison target are guaranteed to be the exact same model (this also means
this script inherits whatever internal angle-unit convention e51_crb.py uses
without needing to resolve it -- see report for a units-consistency note found
while tracing the code).

Usage: python3 tmp/sbi_npe_crb_verify.py
Out:   results/direction3/sbi_npe_crb_verify.json
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'tmp')

import e51_crb as E  # noqa: E402

DEVICE = E.DEVICE
N_TRAIN = 4000
N_HELDOUT = 64
K_MIX = 4
HID = 256
STEPS = 3000
BATCH = 128
GEOMS = [700, 750]          # primary + one robustness-check geometry
OUT = 'results/direction3/sbi_npe_crb_verify.json'


def sample_theta(rng, n):
    d1 = rng.uniform(0, 360, n)
    sign = rng.choice([-1.0, 1.0], n)
    gap = rng.uniform(60, 150, n)
    d2 = d1 + sign * gap
    Tp2 = 7.0 * rng.uniform(0.75, 1.4, n)
    w = rng.uniform(0.2, 0.5, n)
    return d1, d2, Tp2, w


def gen_dataset(eng, n, seed):
    """z ~ N(0, C(theta)) for theta ~ prior, EXACT model e51_crb.py assumes."""
    rng = np.random.default_rng(seed)
    d1, d2, Tp2, w = sample_theta(rng, n)
    Z = torch.empty(n, eng.n, dtype=torch.float64, device=DEVICE)
    for i in range(n):
        theta5 = [float(d1[i]), 7.0, float(d2[i]), float(Tp2[i]), float(w[i])]
        C = E.build_C(eng, theta5)
        L = torch.linalg.cholesky(C)
        eps = torch.randn(eng.n, dtype=torch.float64, device=DEVICE)
        Z[i] = L @ eps
    theta4 = np.stack([d1, d2, Tp2, w], axis=1)
    return theta4, Z


class MDN(nn.Module):
    """Small embedding MLP + Gaussian-mixture density head (diag cov)."""
    def __init__(self, n_in, n_out=4, hid=HID, k=K_MIX):
        super().__init__()
        self.k, self.n_out = k, n_out
        self.embed = nn.Sequential(
            nn.Linear(n_in, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, hid // 2), nn.ReLU())
        self.head = nn.Linear(hid // 2, k * (2 * n_out + 1))

    def forward(self, z):
        h = self.embed(z)
        out = self.head(h)
        k, d = self.k, self.n_out
        logits = out[:, :k]
        mu = out[:, k:k + k * d].view(-1, k, d)
        logsig = out[:, k + k * d:].view(-1, k, d).clamp(-5, 3)
        return logits, mu, logsig


def mdn_nll(logits, mu, logsig, target):
    """target: (B, d). Mixture NLL, diag-Gaussian components."""
    logpi = torch.log_softmax(logits, dim=1)              # (B,k)
    sig = torch.exp(logsig)                                # (B,k,d)
    t = target[:, None, :]                                  # (B,1,d)
    logN = (-0.5 * ((t - mu) / sig) ** 2 - logsig
           - 0.5 * np.log(2 * np.pi)).sum(-1)               # (B,k)
    logmix = torch.logsumexp(logpi + logN, dim=1)            # (B,)
    return -logmix.mean()


def mixture_mean_std(logits, mu, logsig):
    """Marginal mean/std per output dim from the mixture (closed form)."""
    pi = torch.softmax(logits, dim=1)                        # (B,k)
    sig2 = torch.exp(2 * logsig)                              # (B,k,d)
    mean = (pi[:, :, None] * mu).sum(1)                        # (B,d)
    var = (pi[:, :, None] * (sig2 + mu ** 2)).sum(1) - mean ** 2
    return mean, torch.sqrt(torch.clamp(var, min=1e-12))


def mixture_pit(pi, mu, logsig, target):
    """Probability-integral-transform (SBC rank statistic) of `target` under
    the fitted 1-D marginal Gaussian mixture, per dim. Well-calibrated =>
    PIT ~ Uniform(0,1) across held-out examples. pi:(B,k) mu,logsig:(B,k,d)
    target:(B,d) -- all numpy."""
    from scipy.stats import norm
    sig = np.exp(logsig)
    z = (target[:, None, :] - mu) / sig                       # (B,k,d)
    cdf_k = norm.cdf(z)                                        # (B,k,d)
    pit = (pi[:, :, None] * cdf_k).sum(1)                       # (B,d)
    return pit


def train_mdn(theta_train, Z_train, mu_norm, sd_norm, steps=STEPS, batch=BATCH):
    n_in = Z_train.shape[1]
    net = MDN(n_in).to(DEVICE).double()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Tn = torch.tensor((theta_train - mu_norm) / sd_norm, dtype=torch.float64, device=DEVICE)
    n = Z_train.shape[0]
    Zn = (Z_train - Z_train.mean(0, keepdim=True)) / (Z_train.std(0, keepdim=True) + 1e-6)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEVICE)
        logits, mu, logsig = net(Zn[idx])
        loss = mdn_nll(logits, mu, logsig, Tn[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            losses.append((step, float(loss)))
            print(f"    step {step:5d} nll {float(loss):+.4f}", flush=True)
    return net, Zn.mean(0), Zn.std(0) + 1e-6


def run_geom(si, seed_train, seed_held):
    t0 = time.time()
    print(f"\n=== geometry si={si} ===", flush=True)
    S0 = E.build_sea_crosstp(si)
    eng = E.JointEngine(S0)
    print(f"  eng.n={eng.n} kp={eng.kp:.4f}", flush=True)

    theta_train, Z_train = gen_dataset(eng, N_TRAIN, seed_train)
    print(f"  train set generated [{time.time()-t0:.0f}s]", flush=True)
    mu_norm = theta_train.mean(0); sd_norm = theta_train.std(0) + 1e-6

    net, zmean, zstd = train_mdn(theta_train, Z_train, mu_norm, sd_norm)
    print(f"  MDN trained [{time.time()-t0:.0f}s]", flush=True)

    theta_held, Z_held = gen_dataset(eng, N_HELDOUT, seed_held)
    Zh_n = (Z_held - zmean) / zstd
    with torch.no_grad():
        logits, mu, logsig = net(Zh_n)
        pi = torch.softmax(logits, dim=1).cpu().numpy()
        mean_n, std_n = mixture_mean_std(logits, mu, logsig)
    std_npe = (std_n.cpu().numpy() * sd_norm)               # de-normalize std
    mean_npe = (mean_n.cpu().numpy() * sd_norm + mu_norm)

    # ---- calibration (SBC-style PIT, coordinator-requested confound check) ----
    # de-normalize per-component mu/sigma to raw theta units for the PIT integral
    mu_np = mu.cpu().numpy() * sd_norm[None, None, :] + mu_norm[None, None, :]
    logsig_np = logsig.cpu().numpy() + np.log(sd_norm)[None, None, :]
    pit = mixture_pit(pi, mu_np, logsig_np, theta_held)      # (N_HELDOUT, 4)
    cover_1s = np.abs(theta_held - mean_npe) <= std_npe       # crude +-1sigma coverage
    calib = {}
    for j, dname in enumerate(['d1', 'd2', 'Tp2', 'w']):
        ks = stats.kstest(pit[:, j], 'uniform')
        calib[dname] = dict(pit_ks_stat=float(ks.statistic), pit_ks_p=float(ks.pvalue),
                            cover_1sigma=float(cover_1s[:, j].mean()),
                            pit_vals=pit[:, j].tolist())
        print(f"    calib[{dname}] PIT KS-stat {ks.statistic:.3f} p={ks.pvalue:.3g}  "
              f"cover(+-1sig) {cover_1s[:, j].mean():.2f} (want ~0.68 if Gaussian-like)", flush=True)

    # analytic CRB at each held-out true theta (fim_crb reused verbatim)
    crb_rows = []
    for i in range(N_HELDOUT):
        d1, d2, Tp2, w = theta_held[i]
        theta5 = [float(d1), 7.0, float(d2), float(Tp2), float(w)]
        sc, cond, _ = E.fim_crb(S0, theta5)
        crb_rows.append(dict(crb_d1=float(sc[0]), crb_d2=float(sc[2]),
                             crb_Tp2=float(sc[3]), crb_w=float(sc[4]), cond=float(cond)))
    print(f"  analytic CRB done [{time.time()-t0:.0f}s]", flush=True)

    dims = ['d1', 'd2', 'Tp2', 'w']
    out = dict(si=si, eng_n=int(eng.n), kp=float(eng.kp), per_dim={})
    for j, dname in enumerate(dims):
        crb = np.array([r[f'crb_{dname}'] for r in crb_rows])
        npe = std_npe[:, j]
        rho, p = stats.spearmanr(crb, npe)
        ratio = npe / np.clip(crb, 1e-9, None)
        out['per_dim'][dname] = dict(
            crb_median=float(np.median(crb)), npe_std_median=float(np.median(npe)),
            spearman_crb_npe=float(rho), spearman_p=float(p),
            ratio_median=float(np.median(ratio)), ratio_iqr=[float(np.percentile(ratio, 25)),
                                                              float(np.percentile(ratio, 75))],
            crb_vals=crb.tolist(), npe_vals=npe.tolist())
        print(f"  {dname:5s} CRB med {np.median(crb):.4f}  NPE-std med {np.median(npe):.4f}  "
              f"ratio(NPE/CRB) med {np.median(ratio):.2f}  Spearman {rho:+.3f} (p={p:.3g})", flush=True)

    # multimodality diagnostic: fraction of held-out cases where mixture is
    # NOT effectively single-mode (top component weight < 0.9 AND >1 component
    # with weight > 0.15 -- a crude but explicit pre-registered-style rule)
    top_w = pi.max(axis=1)
    n_active = (pi > 0.15).sum(axis=1)
    multimodal_frac = float(np.mean((top_w < 0.9) & (n_active > 1)))
    out['multimodal_frac'] = multimodal_frac
    out['top_weight_median'] = float(np.median(top_w))
    print(f"  multimodal_frac (top_w<0.9 & >=2 active comps) = {multimodal_frac:.3f}  "
          f"median top weight = {np.median(top_w):.3f}", flush=True)

    out['theta_held'] = theta_held.tolist()
    out['mean_npe'] = mean_npe.tolist()
    out['std_npe'] = std_npe.tolist()
    out['pi'] = pi.tolist()
    out['calib'] = calib
    # structural periodicity check (bit-level, independent of MDN quality):
    # does C(theta) repeat exactly every 2*pi in the RAW d1/d2 values?
    theta0 = [float(S0['d1']), 7.0, float(S0['d2']), float(S0['Tp2']), float(S0['r2'])]
    C0 = E.build_C(eng, theta0)
    per = {}
    for pname, idx in [('d1', 0), ('d2', 2)]:
        th2pi = list(theta0); th2pi[idx] += 2 * np.pi
        th360 = list(theta0); th360[idx] += 360.0
        d2pi = (C0 - E.build_C(eng, th2pi)).abs().max().item()
        d360 = (C0 - E.build_C(eng, th360)).abs().max().item()
        per[pname] = dict(diff_at_2pi_shift=d2pi, diff_at_360_shift=d360)
    out['periodicity_check'] = per
    print(f"  periodicity check: d1 2pi-shift diff={per['d1']['diff_at_2pi_shift']:.2e} "
          f"(360-shift diff={per['d1']['diff_at_360_shift']:.2e})", flush=True)
    out['wall_s'] = time.time() - t0
    return out


def main():
    t0 = time.time()
    results = []
    for si in GEOMS:
        try:
            results.append(run_geom(si, seed_train=si * 13 + 1, seed_held=si * 13 + 2))
        except Exception as ex:
            print(f"  si={si} FAILED ({type(ex).__name__}: {ex})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(results=results, config=dict(N_TRAIN=N_TRAIN, N_HELDOUT=N_HELDOUT,
                   K_MIX=K_MIX, steps=STEPS, geoms=GEOMS)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"\n[{time.time()-t0:.0f}s] saved -> {OUT}")


if __name__ == '__main__':
    main()
