"""Is the "net = constant prior + direction read-off" closure an artifact of a
DEGENERATE evaluation regime?

e30_conditionality found, in the synthetic arena (n=16), that replacing the net's
per-sea gamma with its OWN across-sea mean map (rotated to mu_hat) costs only
    -0.0129 +- 0.0242 (t=-0.53, p=0.604, 6/16)   -> indistinguishable
and concluded the learned conditioning is worthless => the prior-net direction is
CLOSED. But `build_sea_c` HARDCODES
    Tp=7.0, gamma=3.3, direction_spread_deg=30.0, n_components=200
so every eval sea is a SINGLE-MODE JONSWAP with the SAME directional spread and the
SAME (kp-normalized) radial profile. Across seas only Hs (a pure scale => no-op),
the mean direction (a rotation), and the observation geometry change. A constant
prior rotated to mu_hat is then almost by construction sufficient.

Counter-evidence from REAL data (tmp/e37_real_conditionality.py): on the real
encounter-forecast task the same probe gives const-net = -0.0620 (t=-2.07, p=0.050)
and the constant prior LOSES the net's whole edge over the hand prior
(const-hand -0.011 vs net-hand +0.051). Real seas are broadband / multi-directional.

THIS SCRIPT: re-run the conditionality probe with the eval regime OPENED to the
net's own TRAINING distribution (d3_prior_net_data.py:46):
    Tp ~ U(5,11), gamma_J ~ U(1.5,6), spread ~ U(15,45), Hs ~ U(1,3)
If const-net becomes significantly negative here, the closure is an artifact of
regime degeneracy and the direction re-opens. If const-net stays ~0, the closure
survives the fairest synthetic test and the real-data signal needs another
explanation.

PRE-REGISTERED (before running):
  primary  : const - net on the OPEN regime, n=32.
             |const-net| <= 0.02 with CI inside +-0.03  -> closure SURVIVES
             const-net <= -0.03 with CI excluding 0     -> closure is REGIME ARTIFACT
             anything else                              -> inconclusive, report as such
  control  : the same probe on the CLOSED regime (Tp=7 etc.), n=32 -> must reproduce
             e30's ~-0.013 n.s., otherwise the comparison is invalid.
  All arms: fixed smult=3, no GT-peeked selection, identical solver/arena.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (deposit_grid, corr, DEVICE, DEPTH, REGIMES,
                          load_incumbent, maps6, two_stage_kp)
from e29_gate0_ladder import solve_with_grid
from e30_conditionality_probe import rotate_map
from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, N_OBS)
from d18_relax_estimator import beamform_top

OUT = 'results/direction3/e38_open_regime_conditionality.json'


def build_sea(si, regime, open_reg):
    """build_sea_c, with the nuisance axes OPENED when open_reg=True."""
    R = REGIMES[regime]
    rng = np.random.default_rng(7000 + 31 * si + hashabs(regime))
    Hs = rng.uniform(*R['hs'])
    if open_reg:
        Tp = rng.uniform(5.0, 11.0)
        gam_j = rng.uniform(1.5, 6.0)
        spread = rng.uniform(15.0, 45.0)
    else:
        Tp, gam_j, spread = 7.0, 3.3, 30.0
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=gam_j, direction_deg=rng.uniform(0, 360),
                           direction_spread_deg=spread, seed=4000 + si)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)), False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(*R['U']) * cp
    nfrac = rng.uniform(*R['nf'])
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    kp = two_stage_kp(P, z)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                Hs=float(Hs), Tp=float(Tp), gam_j=float(gam_j),
                spread=float(spread), m_kx=kx, m_ky=ky, m_A=Am, m_ph=ph,
                bw=None, bw_sc=0.0)


def net_map(S):
    net, _ = load_incumbent()
    X = maps6(S['P'], S['z'], S['kp'])
    with torch.no_grad():
        return net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()


def run(open_reg, n_seas):
    tag = 'OPEN ' if open_reg else 'CLOSED'
    seas = [build_sea(si, 'spread', open_reg) for si in range(n_seas)]
    maps, mus = [], []
    for S in seas:
        maps.append(net_map(S))
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32), DEPTH,
                              klim=3.5 * S['kp'], n=120)
        mus.append(float(np.arctan2(by, bx)))
    aligned = np.stack([rotate_map(maps[i], -mus[i]) for i in range(n_seas)])
    mean_aligned = aligned.mean(0)
    rows = []
    for i, S in enumerate(seas):
        g_own = np.maximum(10.0 ** maps[i], 1e-8)
        g_con = np.maximum(10.0 ** rotate_map(mean_aligned, mus[i]), 1e-8)
        gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        rows.append(dict(si=i, Tp=S['Tp'], gam_j=S['gam_j'], spread=S['spread'],
                         own=solve_with_grid(S, g_own),
                         const=solve_with_grid(S, g_con),
                         real=solve_with_grid(S, gf)))
        r = rows[-1]
        print(f"  [{tag}] sea{i:2d} Tp {S['Tp']:4.1f} g {S['gam_j']:3.1f} "
              f"spr {S['spread']:4.1f} | own {r['own']:+.3f} const {r['const']:+.3f} "
              f"({r['const']-r['own']:+.3f}) real {r['real']:+.3f}", flush=True)
    return rows


def summarize(rows, tag):
    from scipy import stats
    a = lambda k: np.array([r[k] for r in rows], float)
    own, con, real = a('own'), a('const'), a('real')
    d = con - own
    t, p = stats.ttest_rel(con, own)
    se = d.std(ddof=1) / np.sqrt(len(d))
    tq = stats.t.ppf(0.975, len(d) - 1)
    ci = (d.mean() - tq * se, d.mean() + tq * se)
    print(f"\n=== {tag} regime (n={len(rows)}) ===")
    print(f"  net own gamma : {own.mean():+.4f}")
    print(f"  const (평균맵) : {con.mean():+.4f}")
    print(f"  oracle real   : {real.mean():+.4f}   (헤드룸 {(real-own).mean():+.4f})")
    print(f"  ★const − own = {d.mean():+.4f} ± {se:.4f}  t={t:+.2f} p={p:.3f}  "
          f"승 {int((con>own).sum())}/{len(d)}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]")
    return dict(tag=tag, own=float(own.mean()), const=float(con.mean()),
                real=float(real.mean()), d=float(d.mean()), se=float(se),
                t=float(t), p=float(p), ci=[float(ci[0]), float(ci[1])],
                n=len(rows))


def main(n_seas=32):
    print(f"사전등록: |const−own| ≤ 0.02 & CI ⊂ ±0.03 -> 폐쇄 생존 / "
          f"const−own ≤ −0.03 & CI가 0 배제 -> regime 아티팩트\n")
    cl = run(False, n_seas)
    op = run(True, n_seas)
    s_cl = summarize(cl, 'CLOSED (Tp=7, g=3.3, spread=30 — e30 재현)')
    s_op = summarize(op, 'OPEN (Tp~U(5,11), g~U(1.5,6), spread~U(15,45) = 학습분포)')
    print(f"\n★★판정")
    print(f"  CLOSED regime const−own = {s_cl['d']:+.4f} (p={s_cl['p']:.3f})  "
          f"<- e30의 −0.013 재현 여부")
    print(f"  OPEN   regime const−own = {s_op['d']:+.4f} (p={s_op['p']:.3f})  "
          f"CI[{s_op['ci'][0]:+.4f},{s_op['ci'][1]:+.4f}]")
    if s_op['d'] <= -0.03 and s_op['ci'][1] < 0:
        v = "★regime 아티팩트 — 열린 regime서 조건화가 값을 함. 방향 재개방."
    elif abs(s_op['d']) <= 0.02 and s_op['ci'][0] > -0.03 and s_op['ci'][1] < 0.03:
        v = "폐쇄 생존 — 가장 공정한 합성 시험서도 조건화가 무가치."
    else:
        v = "미결 — 사전등록 어느 쪽도 미충족. 검정력/n 보고할 것."
    print(f"  ⟹ {v}")
    json.dump(dict(closed=cl, open=op, s_closed=s_cl, s_open=s_op,
                   verdict=v), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
