"""A/B diagnostic: is the learned prior's 'complexity win' actually a CURRENT-DOPPLER
(model-misspecification) robustness win?

Same 8 nc=30 scenes as the figure; toggle ambient-current Doppler (om += k·Uc, applied to
BOTH obs and true field, exactly as gen_sample does; uc_mag~U(0,0.4) fixed per scene).
bw kept 0 in both arms so Doppler is the ONLY toggle. Paired hand-vs-learned per condition.
"""
import numpy as np
from scipy import stats as ss
import d3_prior_net_data as G
from wavesim.wave_generator import jonswap_components
from d3_sa_nomp import mmse_bank_estimate, beamform_top
from d3_prior_net_eval import get_net, net_gamma_mmse

net = get_net('results/direction3/prior_net_2ndlobeTF.pt')
NC = 30


def build(seed, doppler):
    rng = np.random.default_rng(seed)
    Tp = rng.uniform(5.0, 11.0); Hs = rng.uniform(1.0, 3.0)
    gam = rng.uniform(1.5, 6.0); spread = rng.uniform(15.0, 45.0)
    mdir = rng.uniform(0.0, 360.0)
    cseed = int(rng.integers(0, 2**31 - 1))
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=NC, depth=G.DEPTH, gamma=gam,
                           direction_deg=mdir, direction_spread_deg=spread, seed=cseed)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    A = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    k = np.hypot(kx, ky); om = np.sqrt(9.81 * k * np.tanh(k * G.DEPTH) + 1e-12)
    uc_mag = rng.uniform(0.0, 0.4); uc_ang = rng.uniform(0, 2 * np.pi)
    if doppler:
        om = om + kx * uc_mag * np.cos(uc_ang) + ky * uc_mag * np.sin(uc_ang)
    lam = G.dispersion_lam(Tp, G.DEPTH); cp = lam / Tp; T = 10 * Tp
    side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    r_fp = 0.18 * lam / 2
    ucp = rng.uniform(0.1, 1.2)
    P = G.sample_obs('moving', ucp * cp, lam, cp, T, gx, gy, r_fp, rng)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3] + ph[None]
    nfrac = rng.uniform(0.02, 0.10)
    z = (A[None]*np.cos(psi)).sum(1) + rng.normal(0, nfrac*Hs, len(P))
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), G.DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), G.DEPTH, klim=3.5*kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    Xg, Yg = np.meshgrid(gx, gy)
    psi_g = (kx[None]*Xg.ravel()[:, None] + ky[None]*Yg.ravel()[:, None] - om[None]*T + ph[None])
    gt = (A[None]*np.cos(psi_g)).sum(1)
    Pq = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, T)])
    return P, z, Pq, gt, kp, Hs


def nr(f, gt): return float(np.sqrt(np.mean((f - gt)**2)) / max(np.std(gt), 1e-9))


for dop in (False, True):
    ds = []
    for i in range(8):
        P, z, Pq, gt, kp, Hs = build(31000 + i, dop)
        s0 = 0.05 * Hs
        nh = nr(mmse_bank_estimate(P, z, G.DEPTH, Pq, sigma0=s0, kp_hint=kp, seed=i), gt)
        nl = nr(net_gamma_mmse(net, P, z, G.DEPTH, Pq, sigma0=s0, seed=i, kp_hint=kp), gt)
        ds.append(nh - nl)
        print(f"[dop={dop} {i}] hand {nh:.3f} learned {nl:.3f} d {nh-nl:+.3f}", flush=True)
    ds = np.array(ds)
    w = ss.wilcoxon(ds).pvalue if np.any(ds != 0) else 1.0
    print(f"== Doppler={dop}: mean d {ds.mean():+.4f}  learned wins {int((ds>0).sum())}/8  p={w:.3f}\n", flush=True)
print("⟹ if d jumps positive ONLY with Doppler=True → the 'complexity win' is largely a "
      "current-misspecification robustness win (thesis wording must change).")
