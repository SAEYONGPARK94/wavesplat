"""e44 probe (Designer N2) — DOES THE REGIME PRIZE TRANSFER TO THE
DEPLOYMENT TASK (short-horizon encounter forecast at the platform track)?

WHY. N2's structural argument: under the arena's generative model any blind
z->field estimator is dominated by the Bayes predictor, whose form IS the
"infer theta then average" family; the realization channel is closed (E-3x RL
deconv @ infinite SNR ties, -0.0003). So the only OUTSIDE-family question
left is whether the TASK FUNCTIONAL itself (domain-wide field corr at t=T)
is the right score. Deployment target = short-horizon encounter forecasting
for control. If oracle-theta buys ~0 on the encounter task, the +0.078 field
prize is control-irrelevant and the STOP verdict gets an independent leg; if
it transfers, the E-43 external-info spec inherits control relevance.

DESIGN (probe, n=16, si=900..915 -- PROBE-ONLY BLOCK, burned for future
pre-registration; the pre-registered version must use a fresh block):
  arena  build_sea_crosstp (arena X, cross-Tp)
  query  platform track extrapolated past T=70 s (same deterministic
         motion law as sample_obs: ping-pong diagonal at speed U, or the
         fixed disc center), NO jitter; t in (T, T+10] s, 40 points.
         gt rendered from the true modes (clean, same convention as gt_u).
  arms   cur    : incumbent net gamma (10**net_gamma_map, e43c recipe)
         orc    : supplied_grid at TRUE theta (e43 dose-0 oracle)
  anchor field corr on the standard Pq (must reproduce orc-cur ~ +0.07)
  score  corr on encounter points, horizons (0,10], (0,5], (5,10] s

No training, no modification of existing scripts. Runtime target: minutes.
Out: results/direction3/e44_probe_N2_encounter_transfer.json
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, net_gamma_map
from e43_external_regime_dose import supplied_grid
from d3_sa_fim_identifiability import omega_of
from d3_sa_achievability_gap import dispersion_lam
from e29_gennet_C import DEPTH

OUT = 'results/direction3/e44_probe_N2_encounter_transfer.json'
TP1 = 7.0
T_END = 10 * TP1          # 70 s, matches build_sea_crosstp
HORIZON = 10.0            # s
N_ENC = 40


def track_future(S, te):
    """Deterministic platform position at future times te (no jitter).
    Replicates the motion law of d3_sa_achievability_gap.sample_obs with the
    geometry hardcoded in build_sea_crosstp (side=2*lam, r_fp=0.09*lam)."""
    lam = dispersion_lam(TP1, DEPTH)
    side = 2.0 * lam
    r_fp = (0.18 * lam) / 2
    U = S['U']
    if U < 1e-6:
        cx = cy = side / 2
        return np.column_stack([np.full(len(te), cx), np.full(len(te), cy), te])
    P0 = np.array([r_fp, r_fp])
    P1 = np.array([side - r_fp, side - r_fp])
    L = float(np.linalg.norm(P1 - P0))
    u = (P1 - P0) / L
    s = np.mod(U * te, 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    xy = P0[None] + d[:, None] * u[None]
    return np.column_stack([xy[:, 0], xy[:, 1], te])


def render_at(S, Pq):
    kx, ky, Am, ph = S['m_kx'], S['m_ky'], S['m_A'], S['m_ph']
    om = omega_of(kx, ky, DEPTH)
    psi = (kx[None] * Pq[:, 0:1] + ky[None] * Pq[:, 1:2]
           - om[None] * Pq[:, 2:3] + ph[None])
    return (Am[None] * np.cos(psi)).sum(1)


def main(n=16, si0=900):
    t0 = time.time()
    rows = []
    for si in range(si0, si0 + n):
        S = build_sea_crosstp(si)
        g_cur = np.maximum(10.0 ** net_gamma_map(S), 1e-8)
        g_orc = supplied_grid(S, S['d1'], S['d2'], S['Tp2'], S['r2'])

        te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
        Pq_e = track_future(S, te)
        gt_e = render_at(S, Pq_e)

        row = dict(si=si, U=float(S['U']),
                   fixed=bool(S['U'] < 1e-6), r2=float(S['r2']),
                   Tp2=float(S['Tp2']))
        # anchor: standard field task
        row['fld_cur'] = solve_with_grid(S, g_cur)
        row['fld_orc'] = solve_with_grid(S, g_orc)
        # encounter task, three horizon windows
        for tag, sel in [('enc10', slice(None)),
                         ('enc05', slice(0, N_ENC // 2)),
                         ('enc510', slice(N_ENC // 2, N_ENC))]:
            S2 = dict(S, Pq=Pq_e[sel], gt_u=gt_e[sel])
            row[f'{tag}_cur'] = solve_with_grid(S2, g_cur)
            row[f'{tag}_orc'] = solve_with_grid(S2, g_orc)
        rows.append(row)
        print(f"si{si} U{S['U']:5.1f} | fld {row['fld_cur']:+.3f}/"
              f"{row['fld_orc']:+.3f} d{row['fld_orc']-row['fld_cur']:+.3f}"
              f" | enc10 {row['enc10_cur']:+.3f}/{row['enc10_orc']:+.3f}"
              f" d{row['enc10_orc']-row['enc10_cur']:+.3f}"
              f" | enc05 d{row['enc05_orc']-row['enc05_cur']:+.3f}"
              f" | enc510 d{row['enc510_orc']-row['enc510_cur']:+.3f}"
              f" [{time.time()-t0:.0f}s]", flush=True)

    def summ(tag):
        d = np.array([r[f'{tag}_orc'] - r[f'{tag}_cur'] for r in rows])
        d = d[np.isfinite(d)]
        se = d.std(ddof=1) / np.sqrt(len(d))
        return dict(mean=float(d.mean()), se=float(se), sd=float(d.std(ddof=1)),
                    win=int((d > 0).sum()), n=len(d))

    out = dict(rows=rows, summary={t: summ(t) for t in
                                   ['fld', 'enc10', 'enc05', 'enc510']})
    print('\n=== e44 N2 probe: oracle-theta minus cur, per task ===')
    for t, s in out['summary'].items():
        print(f"  {t:7s}: {s['mean']:+.4f} +- {s['se']:.4f} "
              f"(sd {s['sd']:.3f}, win {s['win']}/{s['n']})")
    a, e = out['summary']['fld'], out['summary']['enc10']
    tr = e['mean'] / a['mean'] if abs(a['mean']) > 1e-9 else float('nan')
    print(f"  transfer ratio enc10/fld = {tr:.2f}")
    out['transfer_ratio'] = float(tr)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
