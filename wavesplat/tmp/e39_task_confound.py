"""Is the CLOSURE an artifact of the TASK the synthetic arena scores?

Facts to reconcile:
  synthetic arena (field reconstruction at held-out pixels, SAME time T):
      const - net = -0.012 (n=32, p=0.380)   -> conditioning worthless => CLOSED
  real data (encounter forecast, leads 0-10 s BEYOND the obs window):
      const - net = -0.062 (n=24, p=0.050)   -> conditioning IS the net's edge
Rejected explanation: "real gamma maps are more diverse". Measured the opposite --
the net's mu-aligned maps are LESS diverse on real (RMS resid/mean 0.557, 1-R^2 0.229)
than on synthetic (0.695 / 0.315).

Remaining hypothesis: the TASK. The synthetic arena queries Pq = (x, y, T) -- held-out
pixels at the SAME instant as the end of the obs window: pure SPATIAL interpolation.
The real task queries the ship track at t0 + lead for lead in [0, 10] s: TEMPORAL
EXTRAPOLATION past the data. Under temporal extrapolation a small error in the prior's
(k, omega) support turns into a phase error growing like d(omega)*dt, so gamma accuracy
matters far more. If so, the whole closure was measured on a task that is NOT the
deployment task.

TEST: the SAME synthetic seas, the SAME solver, the SAME arms -- only the QUERY changes.
  arm 'now'  : Pq = held-out pixels at time T             (the arena's task; control)
  arm 'fc'   : Pq = a forward track at T + lead, lead in [0, 10] s   (deployment task)
For each: net's own gamma vs its own across-sea mean map rotated to mu_hat (const).
PRE-REGISTERED:
  If (const-net)|fc is more negative than (const-net)|now by >= 0.03 with the fc CI
  excluding 0  -> the closure is a TASK artifact; conditioning matters on the
                  deployment task and the direction re-opens.
  If (const-net)|fc ~ (const-net)|now ~ 0 -> the task is not the confound; closure holds
                  and the real-data signal needs another explanation.
Fixed smult=3, no GT-peeked selection, identical arena.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (deposit_grid, corr, DEVICE, DEPTH, REGIMES,
                          load_incumbent, maps6, two_stage_kp, bank_atoms,
                          design_t, _MASK)
from e29_gate0_ladder import SMULT
from e30_conditionality_probe import rotate_map
from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, N_OBS)
from d18_relax_estimator import beamform_top

OUT = 'results/direction3/e39_task_confound.json'
LEADS = np.arange(0.0, 10.0 + 1e-9, 1.0)


def build_sea(si):
    """build_sea_c ('spread' regime) + an extra FORECAST query along a fwd track."""
    R = REGIMES['spread']
    rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
    Tp = 7.0
    Hs = rng.uniform(*R['hs'])
    dirdeg = rng.uniform(0, 360)
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=dirdeg,
                           direction_spread_deg=30.0, seed=4000 + si)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH); cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam; r_fp = fp / 2; side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)), False)
    Xs, Ys = Xs[take], Ys[take]
    gt_now = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Pq_now = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(*R['U']) * cp
    nfrac = rng.uniform(*R['nf'])
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    kp = two_stage_kp(P, z)
    # ---- TEMPORAL-EXTRAPOLATION queries: the SAME held-out pixels, but at T+dt.
    # Isolates "extrapolate in time" from "interpolate in space" while keeping the
    # low-variance 1200-pixel metric (the track version was too noisy: 11 points).
    out = dict(P=P, z=z, kp=kp, sigma0=float(nfrac * Hs), Hs=float(Hs),
               Pq_now=Pq_now, gt_now=gt_now, m_kx=kx, m_ky=ky, m_A=Am, Tp=Tp)
    for dt in (5.0, 10.0, 20.0):
        out[f'Pq_fc{int(dt)}'] = np.column_stack([Xs, Ys, np.full(len(Xs), T + dt)])
        out[f'gt_fc{int(dt)}'] = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T + dt)
    return out


def solve(S, g_grid, which):
    Pq = S['Pq_now'] if which == 'now' else S[f'Pq_{which}']
    gt = S['gt_now'] if which == 'now' else S[f'gt_{which}']
    g = np.concatenate([g_grid[_MASK], np.zeros(int(np.asarray(_MASK).sum()))])
    g = np.maximum(g, 1e-12); g = g / g.sum()
    kxb, kyb, om = bank_atoms(S['kp'])
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(S['P']))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    gam = t(np.concatenate([g, g]) * float(np.var(S['z'])))
    K = (Phi * gam[None]) @ Phi.T + (SMULT * S['sigma0']) ** 2 * torch.eye(
        len(S['z']), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(S['z']))
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), gt))


def net_map(S):
    net, _ = load_incumbent()
    X = maps6(S['P'], S['z'], S['kp'])
    with torch.no_grad():
        return net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()


def main(n_seas=32):
    seas = [build_sea(si) for si in range(n_seas)]
    maps, mus = [], []
    for S in seas:
        maps.append(net_map(S))
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32), DEPTH,
                              klim=3.5 * S['kp'], n=120)
        mus.append(float(np.arctan2(by, bx)))
    mean_aligned = np.stack([rotate_map(maps[i], -mus[i])
                             for i in range(n_seas)]).mean(0)
    rows = []
    for i, S in enumerate(seas):
        g_own = np.maximum(10.0 ** maps[i], 1e-8)
        g_con = np.maximum(10.0 ** rotate_map(mean_aligned, mus[i]), 1e-8)
        gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        r = dict(si=i)
        for wh in ('now', 'fc5', 'fc10', 'fc20'):
            r[f'own_{wh}'] = solve(S, g_own, wh)
            r[f'con_{wh}'] = solve(S, g_con, wh)
            r[f'real_{wh}'] = solve(S, gf, wh)
        rows.append(r)
        print(f"  sea{i:2d} | NOW {r['own_now']:+.3f}/{r['con_now']:+.3f}"
              f"({r['con_now']-r['own_now']:+.3f}) | +5s {r['own_fc5']:+.3f}/"
              f"{r['con_fc5']:+.3f}({r['con_fc5']-r['own_fc5']:+.3f}) | +10s "
              f"{r['own_fc10']:+.3f}/{r['con_fc10']:+.3f}"
              f"({r['con_fc10']-r['own_fc10']:+.3f}) | +20s {r['own_fc20']:+.3f}/"
              f"{r['con_fc20']:+.3f}({r['con_fc20']-r['own_fc20']:+.3f})", flush=True)

    from scipy import stats
    a = lambda k: np.array([r[k] for r in rows], float)
    print(f"\n=== 과제 교란 시험 (같은 바다·같은 solver, 질의만 다름, n={n_seas}) ===")
    res = {}
    for wh, lab in (('now', 'NOW   = 같은 시각 T, 미관측 픽셀 (아레나 과제)'),
                    ('fc5', '+5s   = 같은 픽셀, T+5s  (시간 외삽)'),
                    ('fc10', '+10s  = 같은 픽셀, T+10s (배포 리드)'),
                    ('fc20', '+20s  = 같은 픽셀, T+20s (긴 리드)')):
        own, con, real = a(f'own_{wh}'), a(f'con_{wh}'), a(f'real_{wh}')
        d = con - own
        t, p = stats.ttest_rel(con, own)
        se = d.std(ddof=1) / np.sqrt(len(d)); tq = stats.t.ppf(0.975, len(d) - 1)
        print(f"\n  {lab}")
        print(f"    net own {own.mean():+.4f} | const {con.mean():+.4f} | "
              f"oracle {real.mean():+.4f} (헤드룸 {(real-own).mean():+.4f})")
        print(f"    ★const − net = {d.mean():+.4f} ± {se:.4f}  t={t:+.2f} p={p:.3f} "
              f"승 {int((con>own).sum())}/{len(d)}  "
              f"95%CI[{d.mean()-tq*se:+.4f},{d.mean()+tq*se:+.4f}]")
        res[wh] = dict(own=float(own.mean()), const=float(con.mean()),
                       real=float(real.mean()), d=float(d.mean()),
                       se=float(se), p=float(p),
                       ci=[float(d.mean()-tq*se), float(d.mean()+tq*se)])
    dn = a('con_now') - a('own_now'); df = a('con_fc10') - a('own_fc10')
    t, p = stats.ttest_rel(df, dn)
    print(f"\n  ★★+10s vs NOW (대응, 같은 바다·같은 픽셀): 차이 = {(df-dn).mean():+.4f}  "
          f"t={t:+.2f} p={p:.3f}")
    if res['fc10']['d'] <= res['now']['d'] - 0.03 and res['fc10']['ci'][1] < 0:
        v = "★과제 아티팩트 — 배포 과제(예보)선 조건화가 값을 함. 방향 재개방."
    elif abs(res['fc10']['d']) < 0.02:
        v = "과제는 교란이 아님 — 예보 과제서도 조건화 무가치. 폐쇄 유지."
    else:
        v = "미결 — 사전등록 미충족. 검정력 보고."
    print(f"  ⟹ {v}")
    json.dump(dict(rows=rows, res=res, verdict=v), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
