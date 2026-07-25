"""Visual evidence for the Item-1 complexity win: does the learned prior beat the
analytic prior on complex (nc=30) seas — SHOW it.

Honesty rules: field scene = same gen_sample family (same param distributions, obs model,
grid side=2λ, N_OBS) but rebuilt here so the TRUE components are known for exact GT
rendering (bw_inject=0, no current — disclosed); scene selection = MEDIAN learned-vs-hand
delta among 8 fixed-seed candidates (no cherry-picking). Chart = the pre-registered sweep
numbers from item1.log.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import d3_prior_net_data as G
from wavesim.wave_generator import jonswap_components
from d3_sa_nomp import mmse_bank_estimate, beamform_top
from d3_prior_net_eval import get_net, net_gamma_mmse

net = get_net('results/direction3/prior_net_2ndlobeTF.pt')
NC = 30


def build_scene(seed):
    """Mirror gen_sample (d3_prior_net_data.py:45-119) with known components, bw=0, uc=0."""
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
    lam = G.dispersion_lam(Tp, G.DEPTH); cp = lam / Tp; T = 10 * Tp
    side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    r_fp = 0.18 * lam / 2
    ucp = rng.uniform(0.1, 1.2)
    P = G.sample_obs('moving', ucp * cp, lam, cp, T, gx, gy, r_fp, rng)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3] + ph[None]
    nfrac = rng.uniform(0.02, 0.10)
    z = (A[None]*np.cos(psi)).sum(1) + rng.normal(0, nfrac*Hs, len(P))
    # kp exactly as at inference
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), G.DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), G.DEPTH, klim=3.5*kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    # render grid at t=T (nowcast, end of window)
    Xg, Yg = np.meshgrid(gx, gy)
    psi_g = (kx[None]*Xg.ravel()[:, None] + ky[None]*Yg.ravel()[:, None]
             - om[None]*T + ph[None])
    gt = (A[None]*np.cos(psi_g)).sum(1).reshape(Xg.shape)
    Pq = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, T)])
    return dict(P=P, z=z, Pq=Pq, gt=gt, gx=gx, gy=gy, kp=kp, Hs=Hs, Tp=Tp, shape=Xg.shape)


def nr(f, gt): return float(np.sqrt(np.mean((f - gt)**2)) / max(np.std(gt), 1e-9))


cands = []
for i in range(8):
    sc = build_scene(31000 + i)
    s0 = 0.05 * sc['Hs']
    fh = mmse_bank_estimate(sc['P'], sc['z'], G.DEPTH, sc['Pq'], sigma0=s0, kp_hint=sc['kp'], seed=i)
    fl = net_gamma_mmse(net, sc['P'], sc['z'], G.DEPTH, sc['Pq'], sigma0=s0, seed=i, kp_hint=sc['kp'])
    nh, nl = nr(fh, sc['gt'].ravel()), nr(fl, sc['gt'].ravel())
    cands.append((nh - nl, nh, nl, sc, fh, fl))
    print(f"[cand {i}] hand {nh:.3f} learned {nl:.3f} d {nh-nl:+.3f}", flush=True)

cands.sort(key=lambda x: x[0])
d, nh, nl, sc, fh, fl = cands[len(cands)//2]   # MEDIAN delta — no cherry-picking
print(f"MEDIAN scene: hand {nh:.3f} learned {nl:.3f} d {d:+.3f}")

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.35)
ext = [sc['gx'][0], sc['gx'][-1], sc['gy'][0], sc['gy'][-1]]
vmax = np.abs(sc['gt']).max()
for j, (arr, ti) in enumerate([(sc['gt'], 'Ground truth (nc=30)'),
                               (fh.reshape(sc['shape']), f'Analytic prior  nRMSE {nh:.3f}'),
                               (fl.reshape(sc['shape']), f'Learned prior  nRMSE {nl:.3f}')]):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(arr, origin='lower', extent=ext, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(ti, fontsize=11)
    ax.plot(sc['P'][:, 0], sc['P'][:, 1], '.', ms=0.8, color='k', alpha=0.15)
fig.colorbar(im, ax=fig.axes[:3], shrink=0.75, label='η [m]')

# chart: pre-registered sweep numbers (item1.log)
axc = fig.add_subplot(gs[1, :])
ncs = [5, 15, 30]; hand = [0.877, 0.934, 0.965]; lrn = [0.831, 0.880, 0.860]
pv = ['p=0.0042', 'p=0.0023', 'p=0.0004']; wins = ['15/20', '17/20', '18/20']
x = np.arange(3); w = 0.32
axc.bar(x - w/2, hand, w, label='Analytic prior (hand)', color='#D08770')
axc.bar(x + w/2, lrn, w, label='Learned prior (net)', color='#4C9A6B')
for i in range(3):
    axc.text(x[i], max(hand[i], lrn[i]) + 0.012,
             f'Δ{hand[i]-lrn[i]:+.3f}  {wins[i]}  {pv[i]}', ha='center', fontsize=10)
axc.set_xticks(x); axc.set_xticklabels([f'{n} components' for n in ncs])
axc.set_ylabel('nRMSE (lower better)'); axc.set_ylim(0.75, 1.03)
axc.set_title('Complexity sweep — paired, in-distribution, n=20/condition (pre-registered numbers)')
axc.legend(loc='upper left')
fig.suptitle('Does the learned prior beat the analytic prior on COMPLEX seas?  (median-delta scene shown, no cherry-pick)',
             fontsize=12)
out = 'results/gifs/item1_complexity_win.png'
fig.savefig(out, dpi=110, bbox_inches='tight')
print(f"-> {out}")
