"""Rx-4 STAGE 2 analysis -- reads tmp/Rx4_tracker_init_2026-08-06/diag.jsonl only."""
import json
import os
import numpy as np
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, 'diag.jsonl')) if l.strip()]
ok = [r for r in rows if 'error' not in r]
print(f'rows={len(rows)} ok={len(ok)}  errors={len(rows)-len(ok)}')
for r in rows:
    if 'error' in r:
        print('  ERR', r['job_id'], r['error'])

EPS = sorted({r['episode'] for r in ok})
SEEDS = sorted({r['seed'] for r in ok})


def sel(nt):
    return [r for r in ok if r['notrack'] == nt]


print('\n=== 0. control: probe reproduces B3 AUTO? (per run nRMSE) ===')
for r in sorted(sel(False), key=lambda r: (r['episode'], r['seed'])):
    print(f"  {r['episode'][8:-3]:28s} s{r['seed']} route={r['auto_selected']:9s} k={r['ensemble_k']} "
          f"N={r['N']:2d} init={r['init_nrmse']:.4f} final={r['nRMSE']:.4f} "
          f"d={r['nRMSE']-r['init_nrmse']:+.4f}")
print('  --- NOTRACK ---')
for r in sorted(sel(True), key=lambda r: (r['episode'], r['seed'])):
    print(f"  {r['episode'][8:-3]:28s} s{r['seed']} route={r['auto_selected']:9s} k={r['ensemble_k']} "
          f"N={r['N']:2d} init={r['init_nrmse']:.4f} final={r['nRMSE']:.4f} "
          f"d={r['nRMSE']-r['init_nrmse']:+.4f}")

print('\n=== 1. DISPERSION / WAVENUMBER FREEZE (is the hard constraint ever broken?) ===')
dr = [t['disp_resid_max'] for r in ok for t in r['traj']]
dk = [t['dk_l2'] for r in ok for t in r['traj']]
dom = [t['domega_max'] for r in ok for t in r['traj']]
print(f'  frames inspected           : {len(dr)}')
print(f'  max |omega_model - sqrt(g k tanh kd)| over ALL frames/atoms : {max(dr):.3e}')
print(f'  max ||dk||_2 (kx,ky drift) over ALL frames                  : {max(dk):.3e}')
print(f'  max |omega_t - omega_0|    over ALL frames                  : {max(dom):.3e}')

print('\n=== 2. TIME STRUCTURE (AUTO): nRMSE vs frame, quartiles of the tracking window ===')
print(f"  {'episode':28s} {'init':>7s} {'q1':>7s} {'q2':>7s} {'q3':>7s} {'q4':>7s} {'last':>7s} {'slope/frame':>12s}")
for ep in EPS:
    for nt, tag in ((False, 'AUTO'), (True, 'NOTR')):
        rs = [r for r in ok if r['episode'] == ep and r['notrack'] == nt]
        if not rs:
            continue
        qs = []
        slopes = []
        lasts = []
        inits = []
        for r in rs:
            v = np.array([t['nrmse_actual'] for t in r['traj'] if t['phase'] == 'track'])
            n = len(v)
            qs.append([float(np.mean(v[i * n // 4:(i + 1) * n // 4])) for i in range(4)])
            slopes.append(float(np.polyfit(np.arange(n), v, 1)[0]))
            lasts.append(float(np.mean(v[-10:])))
            inits.append(r['init_nrmse'])
        q = np.median(np.array(qs), axis=0)
        print(f'  {ep[8:-3]:22s}{tag:>6s} {st.median(inits):7.4f} '
              f'{q[0]:7.4f} {q[1]:7.4f} {q[2]:7.4f} {q[3]:7.4f} {st.median(lasts):7.4f} '
              f'{st.median(slopes)*1e4:9.2f}e-4')

print('\n=== 3. CHANNEL (AUTO vs NOTRACK): decorr / shrink, mean over tracking frames ===')
print(f"  {'episode':28s} {'cell':>6s} {'m':>8s} {'decorr':>8s} {'shrink':>8s} {'rho':>7s} {'VR':>7s}")
for ep in EPS:
    for nt, tag in ((False, 'AUTO'), (True, 'NOTR')):
        rs = [r for r in ok if r['episode'] == ep and r['notrack'] == nt]
        if not rs:
            continue
        def agg(k):
            return st.median([float(np.mean([t[k] for t in r['traj'] if t['phase'] == 'track']))
                              for r in rs])
        d, s2, rh, vr = agg('decorr'), agg('shrink'), agg('rho'), agg('vr')
        print(f'  {ep[8:-3]:22s}{tag:>6s} {d+s2:8.4f} {d:8.4f} {s2:8.4f} {rh:7.4f} {vr:7.4f}')

print('\n=== 4. COUNTERFACTUAL REPLAY (AUTO runs only) ===')
print('  nRMSE with (w,phi) taken from: frozen=(w0,phi0)  w_only=(wt,phi0)  '
      'phi_only=(w0,phit)  actual=(wt,phit)')
print('  NOTE: this decomposes the REALISED damage along the ACTUAL joint trajectory.')
print('        It is NOT a prediction of what a one-channel-frozen RUN would do.')
print(f"  {'episode':24s} {'frozen':>8s} {'w_only':>8s} {'phi_only':>9s} {'actual':>8s}"
      f" {'dmg_w':>8s} {'dmg_phi':>8s} {'dmg_tot':>8s} {'interact':>9s}")
tot = {'w': [], 'p': [], 't': []}
for ep in EPS:
    rs = [r for r in ok if r['episode'] == ep and not r['notrack']]
    def agg(k):
        return st.median([float(np.mean([t[k] for t in r['traj'] if t['phase'] == 'track']))
                          for r in rs])
    fz, wo, po, ac = agg('nrmse_frozen'), agg('nrmse_w_only'), agg('nrmse_phi_only'), agg('nrmse_actual')
    # per-run paired damages, then median (never "difference of medians")
    dw = st.median([float(np.mean([t['nrmse_w_only'] - t['nrmse_frozen']
                                   for t in r['traj'] if t['phase'] == 'track'])) for r in rs])
    dp = st.median([float(np.mean([t['nrmse_phi_only'] - t['nrmse_frozen']
                                   for t in r['traj'] if t['phase'] == 'track'])) for r in rs])
    dt = st.median([float(np.mean([t['nrmse_actual'] - t['nrmse_frozen']
                                   for t in r['traj'] if t['phase'] == 'track'])) for r in rs])
    ia = st.median([float(np.mean([t['nrmse_actual'] - t['nrmse_w_only']
                                   - t['nrmse_phi_only'] + 2 * t['nrmse_frozen'] - t['nrmse_frozen']
                                   for t in r['traj'] if t['phase'] == 'track'])) for r in rs])
    tot['w'].append(dw); tot['p'].append(dp); tot['t'].append(dt)
    print(f'  {ep[8:-3]:24s} {fz:8.4f} {wo:8.4f} {po:9.4f} {ac:8.4f} '
          f'{dw:+8.4f} {dp:+8.4f} {dt:+8.4f} {ia:+9.4f}')
print(f"  {'MEDIAN(4 ep)':24s} {'':8s} {'':8s} {'':9s} {'':8s} "
      f"{st.median(tot['w']):+8.4f} {st.median(tot['p']):+8.4f} {st.median(tot['t']):+8.4f}")

print('\n=== 5. PARAMETER DRIFT (AUTO): growth of ||dw||/||w0||, RMS dphi, sum|w| ratio ===')
print(f"  {'episode':24s} {'q1':>25s} {'q4':>25s}")
print(f"  {'':24s} {'dw_rel  dphi  sumw':>25s} {'dw_rel  dphi  sumw':>25s}")
for ep in EPS:
    for nt, tag in ((False, 'AUTO'), (True, 'NOTR')):
        rs = [r for r in ok if r['episode'] == ep and r['notrack'] == nt]
        if not rs:
            continue
        out = []
        for frac in (0, 3):
            vals = []
            for r in rs:
                tr = [t for t in r['traj'] if t['phase'] == 'track']
                n = len(tr)
                seg = tr[frac * n // 4:(frac + 1) * n // 4]
                vals.append([np.mean([t['dw_rel_l2'] for t in seg]),
                             np.mean([t['dphi_rms'] for t in seg]),
                             np.mean([t['sumabs_w_ratio'] for t in seg])])
            out.append(np.median(np.array(vals), axis=0))
        print(f'  {ep[8:-3]:18s}{tag:>6s} {out[0][0]:7.3f} {out[0][1]:6.3f} {out[0][2]:6.3f}   '
              f'    {out[1][0]:7.3f} {out[1][1]:6.3f} {out[1][2]:6.3f}')

print('\n=== 6. IS dphi A COHERENT RAMP? (r^2 of dphi_rms vs frame index, AUTO) ===')
for ep in EPS:
    rs = [r for r in ok if r['episode'] == ep and not r['notrack']]
    r2s, lin = [], []
    for r in rs:
        tr = [t for t in r['traj'] if t['phase'] == 'track']
        y = np.array([t['dphi_rms'] for t in tr]); x = np.arange(len(y))
        c = np.corrcoef(x, y)[0, 1]
        r2s.append(float(c ** 2))
        lin.append(float(np.polyfit(x, y, 1)[0]))
    print(f'  {ep[8:-3]:24s} r^2(dphi_rms,frame)={st.median(r2s):.3f}  '
          f'slope={st.median(lin):.4f} rad/frame')

print('\n=== 7. CONCENTRATION: is the drift on weak atoms or strong atoms? (last frame) ===')
print('  Spearman-ish: split atoms at the median |w0| and compare drift magnitudes.')
for ep in EPS:
    rs = [r for r in ok if r['episode'] == ep and not r['notrack']]
    rows_out = []
    for r in rs:
        tr = [t for t in r['traj'] if t['phase'] == 'track' and 'per_atom' in t]
        if not tr:
            continue
        pa = tr[-1]['per_atom']
        w0 = np.abs(np.array(pa['w0_per_atom']))
        dw = np.abs(np.array(pa['dw_per_atom']))
        dphi = np.abs(np.array(pa['dphi_per_atom']))
        if len(w0) < 4:
            continue
        med = np.median(w0)
        strong = w0 >= med; weak = ~strong
        rows_out.append([
            float(np.mean(dw[strong] / np.maximum(w0[strong], 1e-9))),
            float(np.mean(dw[weak] / np.maximum(w0[weak], 1e-9))),
            float(np.mean(dphi[strong])), float(np.mean(dphi[weak])),
            # share of total |dw| carried by the weakest half
            float(dw[weak].sum() / max(dw.sum(), 1e-12)),
        ])
    if rows_out:
        a = np.median(np.array(rows_out), axis=0)
        print(f'  {ep[8:-3]:24s} |dw|/w0 strong={a[0]:6.3f} weak={a[1]:6.3f} | '
              f'|dphi| strong={a[2]:5.3f} weak={a[3]:5.3f} | weak-half share of |dw|={a[4]:.2f}')
