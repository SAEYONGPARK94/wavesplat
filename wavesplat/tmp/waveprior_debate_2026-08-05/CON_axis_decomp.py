"""CON-2: axis decomposition -- what did the LEARNED proposal actually change?
Re-uses the *verbatim* matcher from tmp/s1s2_2026-08-05/run_s1s2.py so learned atoms are
scored on the SAME ruler as S1-A0's `spec` block. No wave runs; reads cached atoms + h5 GT."""
import json, os, statistics as st
import numpy as np, h5py

HERE = '/home/vexit/wavesplat/wavesplat/tmp'
DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
EPS = ['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']

# ---- verbatim matcher (run_s1s2.py lines 113-168) ----
def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    N_est = len(est_kx); N_gt = len(gt_kx)
    if N_est == 0 or N_gt == 0:
        return []
    order = np.argsort(-np.asarray(gt_A))
    used = set(); pairs = []
    for gi in order:
        d = np.hypot(np.asarray(est_kx) - gt_kx[gi], np.asarray(est_ky) - gt_ky[gi])
        for ei in np.argsort(d):
            if ei in used:
                continue
            kmag = np.hypot(gt_kx[gi], gt_ky[gi])
            if d[ei] <= 0.5 * kmag:
                used.add(int(ei)); pairs.append((int(gi), int(ei))); break
            break
    return pairs

def spec_of(kx, ky, amp, gt):
    gt_kx, gt_ky, gt_A = gt['kx'], gt['ky'], gt['A']
    pairs = match_components(kx, ky, amp, gt_kx, gt_ky, gt_A)
    n_gt = len(gt_A)
    E = gt_A**2
    r = dict(n_gt=n_gt, n_est=len(kx), n_matched=len(pairs),
             detection_rate=len(pairs)/n_gt if n_gt else 0.0)
    if pairs:
        kerr, derr = [], []
        for gi, ei in pairs:
            gm = np.hypot(gt_kx[gi], gt_ky[gi]); em = np.hypot(kx[ei], ky[ei])
            kerr.append(abs(em-gm)/max(gm,1e-9))
            ga = np.degrees(np.arctan2(gt_ky[gi], gt_kx[gi]))
            ea = np.degrees(np.arctan2(ky[ei], kx[ei]))
            derr.append(abs((ea-ga+180) % 360 - 180))
        r['k_mag_rel_error_mean'] = float(np.mean(kerr))
        r['direction_error_deg_mean'] = float(np.mean(derr))
        r['energy_detection_rate'] = float(E[[g for g,_ in pairs]].sum()/E.sum())
    else:
        r['k_mag_rel_error_mean'] = float('nan'); r['direction_error_deg_mean'] = float('nan')
        r['energy_detection_rate'] = 0.0
    return r

# ---- GT ----
GT = {}
for ep in EPS:
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        wc = np.array(f['config/wave_components'])   # (N,6) [A, w, kx, ky, phi, Q]
    GT[ep] = dict(A=wc[:,0], w=wc[:,1], kx=wc[:,2], ky=wc[:,3])

# ---- learned atoms (5 diffusion seeds each) ----
atoms = {}
for l in open(f'{HERE}/wp_integration_2026-07-25/atoms_cache.jsonl'):
    d = json.loads(l)
    if d['episode'] in EPS and not d.get('err'):
        atoms.setdefault(d['episode'], {})[d['diff_seed']] = d

# ---- S1-A0 / oracle spec from today's raw ----
s1s2 = {}
for l in open(f'{HERE}/s1s2_2026-08-05/results.jsonl'):
    d = json.loads(l); s1s2.setdefault((d['arm'], d['episode']), {})[d['seed']] = d

def medspec(arm, ep, key):
    return st.median([v['spec'][key] for v in s1s2[(arm, ep)].values()])

print('=== LEARNED PROPOSAL scored on the SAME spectral ruler as S1-A0 (median over 5 diff seeds) ===')
print(f"{'episode':26s} {'arm':14s} {'n_est':>5s} {'det':>6s} {'Edet':>6s} {'k_err':>7s} {'dir_deg':>8s}")
res = {}
for ep in EPS:
    rows = [spec_of(np.array(a['kx']), np.array(a['ky']), np.array(a['energy']), GT[ep])
            for a in atoms[ep].values()]
    L = {k: st.median([r[k] for r in rows]) for k in
         ['n_est','detection_rate','energy_detection_rate','k_mag_rel_error_mean','direction_error_deg_mean']}
    A0 = {k: medspec('S1-A0', ep, k) for k in
          ['n_est','detection_rate','energy_detection_rate','k_mag_rel_error_mean','direction_error_deg_mean']}
    OR = {k: medspec('S2-T1f', ep, k) for k in
          ['n_est','detection_rate','energy_detection_rate','k_mag_rel_error_mean','direction_error_deg_mean']}
    res[ep] = dict(learned_proposal=L, S1_A0=A0, oracle=OR)
    for tag, r in [('learned-atoms', L), ('S1-A0 (v2)', A0), ('oracle T1f', OR)]:
        print(f"{ep[8:-3]:26s} {tag:14s} {r['n_est']:5.0f} {r['detection_rate']:6.3f} "
              f"{r['energy_detection_rate']:6.3f} {r['k_mag_rel_error_mean']:7.3f} {r['direction_error_deg_mean']:8.2f}")
    print()

def mean_over(tag, key):
    return st.mean([res[ep][tag][key] for ep in EPS])
print('MEAN4:')
for tag in ['learned_proposal','S1_A0','oracle']:
    print(f"  {tag:18s} n_est {mean_over(tag,'n_est'):5.1f}  det {mean_over(tag,'detection_rate'):.3f}  "
          f"Edet {mean_over(tag,'energy_detection_rate'):.3f}  k_err {mean_over(tag,'k_mag_rel_error_mean'):.3f}  "
          f"dir {mean_over(tag,'direction_error_deg_mean'):.2f}")
d_l = mean_over('learned_proposal','detection_rate'); d_a = mean_over('S1_A0','detection_rate')
print(f"\n  PLACEMENT-AXIS recovery (detection_rate): ({d_l:.3f}-{d_a:.3f})/(1.000-{d_a:.3f}) = "
      f"{100*(d_l-d_a)/(1.0-d_a):.1f}%")
e_l = mean_over('learned_proposal','energy_detection_rate'); e_a = mean_over('S1_A0','energy_detection_rate')
print(f"  PLACEMENT-AXIS recovery (energy_detection): ({e_l:.3f}-{e_a:.3f})/(1.000-{e_a:.3f}) = "
      f"{100*(e_l-e_a)/(1.0-e_a):.1f}%")
json.dump(res, open(f'{HERE}/waveprior_debate_2026-08-05/CON_axis_decomp.json','w'), indent=1, default=float)
