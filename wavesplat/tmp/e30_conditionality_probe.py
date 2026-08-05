"""e30 — IS THE INCUMBENT A CONSTANT PRIOR? (conditionality probe)

The gate-0 ladder (e29_gate0_ladder.json) reports:
    rung1 cur (learned CNN)                 +0.4032
    rung2 ens_blind (hand constant + mu_hat) +0.3914   (-0.0118)
and this is read as "perfect ensemble knowledge reaches only +0.391, the
incumbent is at +0.403, therefore the gamma class is SATURATED".

That reading assumes rung2 upper-bounds the CONDITIONAL class. It does not.
rung2 bounds the *input-independent* (constant) sub-class only. If the
incumbent is ITSELF ~a constant prior + a direction read-off, then rung2
~= rung1 is a TAUTOLOGY (two constants tie) and says nothing about
information.

DECISIVE TEST (inference only, no training, no GT peek beyond what the
ladder already used):
  A  cur_own   : the incumbent's own per-sea gamma, ladder arena
  B  cur_mean  : the incumbent's gamma REPLACED by the across-sea MEAN of its
                 own aligned (rotated to mu_hat=0) output, rotated back to
                 this sea's mu_hat.  => zero per-sea info except direction.
  C  cur_noalign: the across-sea mean WITHOUT rotating back (pure global
                 constant, no direction) -> control; must be much worse.
  If A ~= B, the incumbent is a constant prior + direction => ladder is
  a tautology, the "information wall" reading dies.

Also (probe 2) rung2's nuisance sensitivity: build_sea_c hardcodes
Tp=7.0/gamma=3.3/spread=30 while the incumbent TRAINED on Tp~U(5,11),
gamma~U(1.5,6), spread~U(15,45). So rung2's "universal radial" is handed the
true nuisance parameters. Score rung2 with MISMATCHED nuisance and with the
honest MARGINAL (averaged over the training distribution) to see how much of
rung2's +0.391 is bought with nuisance-oracle information.
"""
import os, sys, json
import numpy as np
import torch
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, _KXN, _KYN, REGIMES,
                          load_incumbent, maps6, NGRID, KREL)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid, SMULT
from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/e30_conditionality.json'


def rotate_map(M, ang):
    """N(k) = M(R(-ang)k): content rotated by +ang about the bank origin."""
    ca, sa = np.cos(ang), np.sin(ang)
    sx = ca * _KXN + sa * _KYN
    sy = -sa * _KXN + ca * _KYN
    sc = (NGRID - 1) / (2 * KREL)
    u = (sx + KREL) * sc
    v = (sy + KREL) * sc
    return map_coordinates(M, [v, u], order=1, mode='constant', cval=0.0)


def net_gamma_map(S):
    """The incumbent's log-gamma map (51x51) for this sea, its own recipe."""
    net, head = load_incumbent()
    X = maps6(S['P'], S['z'], S['kp'])
    with torch.no_grad():
        Ylog = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
    return Ylog


def radial_of(Tp, gam_j):
    c = jonswap_components(Hs=2.0, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=gam_j, direction_deg=0.0,
                           direction_spread_deg=1e-6, seed=1234)
    kmag = np.array([np.hypot(x.kx, x.ky) for x in c])
    A = np.array([x.A for x in c])
    e = A ** 2 / 2
    kp = kmag[np.argmax(e)]
    return kmag / kp, e / e.sum()          # NORMALIZED radial (Tp-invariant-ish)


def ens_grid_norm(kn, en, mu, kp, spread_deg):
    """Ensemble grid from a NORMALIZED radial kn=k/kp, placed at this sea's kp."""
    s = np.deg2rad(spread_deg)
    th = np.linspace(mu - 4 * s, mu + 4 * s, 181)
    w = np.exp(-0.5 * ((th - mu) / s) ** 2); w /= w.sum()
    kxs = np.concatenate([k * kp * np.cos(th) for k in kn])
    kys = np.concatenate([k * kp * np.sin(th) for k in kn])
    ens = np.concatenate([e * w for e in en])
    return deposit_grid(kxs, kys, ens, kp)


def main(n_seas=16):
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    seas, maps, mus = [], [], []
    print('--- pass 1: incumbent gamma maps ---', flush=True)
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        Y = net_gamma_map(S)
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        mu = np.arctan2(by, bx)
        seas.append(S); maps.append(Y); mus.append(mu)
        print(f'  sea{si:2d} mu_hat {np.rad2deg(mu):7.1f}', flush=True)

    # align each map to mu_hat = 0 and average -> the net's own "constant prior"
    aligned = np.stack([rotate_map(maps[i], -mus[i]) for i in range(n_seas)])
    mean_aligned = aligned.mean(0)

    # variance decomposition inside the band
    band = aligned[:, _MASK]                       # (n, NB) log-gamma
    tot = band.var(0).mean()                       # per-cell across-sea variance
    print(f'\n--- aligned log-gamma across-sea variance (band) = {tot:.4f} ---')
    print(f'    per-sea std of band-mean (level)  = {band.mean(1).std():.4f}')
    resid = band - band.mean(0, keepdims=True)
    print(f'    residual (shape) std              = {resid.std():.4f}')

    rows = []
    print('\n--- pass 2: solve ---', flush=True)
    for i in range(n_seas):
        S, Y, mu = seas[i], maps[i], mus[i]
        g_own = np.maximum(10.0 ** Y, 1e-8)
        g_mean = np.maximum(10.0 ** rotate_map(mean_aligned, mu), 1e-8)
        g_noal = np.maximum(10.0 ** mean_aligned, 1e-8)
        a = solve_with_grid(S, g_own)
        b = solve_with_grid(S, g_mean)
        c = solve_with_grid(S, g_noal)
        rows.append(dict(si=i, cur_own=a, cur_mean=b, cur_noalign=c,
                         cur_ladder=ref[i]['cur'], mu_hat=float(mu)))
        print(f'  sea{i:2d} own {a:+.3f} | mean {b:+.3f} ({b-a:+.3f}) '
              f'| noalign {c:+.3f} | ladder_cur {ref[i]["cur"]:+.3f}', flush=True)

    A = np.array([r['cur_own'] for r in rows])
    B = np.array([r['cur_mean'] for r in rows])
    C = np.array([r['cur_noalign'] for r in rows])
    d = B - A
    print(f'\n=== MEAN-COLLAPSE TEST (n={n_seas}) ===')
    print(f'  A cur_own    (net per-sea gamma) : {A.mean():+.4f}')
    print(f'  B cur_mean   (net MEAN + dir)    : {B.mean():+.4f}   B-A {d.mean():+.4f}'
          f'  (median {np.median(d):+.4f}, B>=A in {int((d>=0).sum())}/{n_seas})')
    print(f'  C cur_noalign(MEAN, no dir)      : {C.mean():+.4f}   C-A {(C-A).mean():+.4f}')
    se = d.std(ddof=1) / np.sqrt(n_seas)
    print(f'  B-A: {d.mean():+.4f} +- {se:.4f} (t={d.mean()/se:+.2f})')
    print(f'  -> conditioning is worth {-d.mean():+.4f} corr of the +0.182 to rung4'
          f' = {100*(-d.mean())/0.1824:.1f}%')

    # ---- probe 2: rung2 nuisance sensitivity ----
    print('\n=== PROBE 2: rung2 nuisance sensitivity ===', flush=True)
    variants = {
        'true(Tp7,g3.3,s30)': (7.0, 3.3, 30.0),
        'Tp5':                (5.0, 3.3, 30.0),
        'Tp11':               (11.0, 3.3, 30.0),
        'gam1.5':             (7.0, 1.5, 30.0),
        'gam6':               (7.0, 6.0, 30.0),
        'spread15':           (7.0, 3.3, 15.0),
        'spread45':           (7.0, 3.3, 45.0),
    }
    p2 = {}
    for name, (tp, gj, sp) in variants.items():
        kn, en = radial_of(tp, gj)
        v = []
        for i in range(n_seas):
            v.append(solve_with_grid(seas[i], ens_grid_norm(kn, en, mus[i],
                                                            seas[i]['kp'], sp)))
        p2[name] = float(np.mean(v))
        print(f'  rung2[{name:20s}] {np.mean(v):+.4f}  vs cur {A.mean():+.4f} '
              f'-> {np.mean(v)-A.mean():+.4f}', flush=True)

    # honest MARGINAL prior: average the ensemble grid over the TRAINING nuisance
    print('  building honest MARGINAL prior (avg over Tp~U(5,11),gam~U(1.5,6),spread~U(15,45))',
          flush=True)
    rng = np.random.default_rng(0)
    NS = 24
    draws = [(rng.uniform(5, 11), rng.uniform(1.5, 6), rng.uniform(15, 45))
             for _ in range(NS)]
    marg = []
    for i in range(n_seas):
        acc = np.zeros((NGRID, NGRID))
        for (tp, gj, sp) in draws:
            kn, en = radial_of(tp, gj)
            acc += ens_grid_norm(kn, en, mus[i], seas[i]['kp'], sp)
        acc /= NS
        marg.append(solve_with_grid(seas[i], acc))
    marg = np.array(marg)
    p2['MARGINAL(honest uncond)'] = float(marg.mean())
    print(f'  rung2[MARGINAL honest    ] {marg.mean():+.4f}  vs cur {A.mean():+.4f} '
          f'-> {marg.mean()-A.mean():+.4f}')

    json.dump(dict(rows=rows, probe2=p2,
                   A=float(A.mean()), B=float(B.mean()), C=float(C.mean()),
                   BA=float(d.mean()), BA_se=float(se),
                   var_band=float(tot)), open(OUT, 'w'), indent=1)
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
