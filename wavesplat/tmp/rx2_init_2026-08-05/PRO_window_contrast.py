"""PRO supplement: supply the IN-WINDOW vs OUT-OF-WINDOW contrast the design was said to lack.

PREREG AMENDMENT 1 (neutral audit) is right that `init_nrmse` is scored on a single frame
INSIDE the fitting window while `nRMSE` is a 120-160 frame OUT-OF-SAMPLE mean, so the design
has no control separating window overfit from real improvement.

It can be supplied WITHOUT new runs. compute_all_metrics gives, per frame,
    rho = corr(pred, gt)   and   v = Var(pred)/Var(gt)
and for zero-mean-difference fields
    RMSE^2 / Var(gt) = 1 + v - 2*rho*sqrt(v) = (1-rho^2) + (sqrt(v)-rho)^2 =: m
so   nRMSE_f = sqrt(m_f) * sigma_gt(f) / Hs.
The driver stored rho and v for EVERY tracking frame, so a per-frame nRMSE trajectory is
reconstructable. Step 1 below VALIDATES the reconstruction against the recorded frame-mean
nRMSE; the trajectory is only used if that check passes.

Then:  init_nrmse (in-window)  ->  nRMSE at tracking frame 1 (just outside)  ->  frame-mean.
An intervention whose init gain is gone by frame 1 was buying window fit, not physics.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
EPISODES = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
DATA = os.path.join(HERE, '..', '..', 'data_finemesh_2026-07-23')


def gt_scale():
    """sigma_gt(f)/Hs per episode: mean and spread over the tracking frames."""
    from wavesplat.data_loader import EpisodeDataLoader
    out = {}
    for e in EPISODES:
        L = EpisodeDataLoader(os.path.join(DATA, e))
        s = [float(np.std(f.gt_field.cpu().numpy())) / L.Hs_true
             for f in L.get_tracking_frames(tp_factor=2.0)]
        out[e] = (float(np.mean(s)), float(np.std(s)), np.array(s))
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
    rows = [r for r in rows if 'error' not in r]
    sc = gt_scale()

    # --- step 1: validate the reconstruction on every run we have -------------------
    err = []
    for r in rows:
        v = np.array(r['vr_frames'], float); rho = np.array(r['rho_frames'], float)
        if v.size == 0:
            continue
        m = (1 - rho ** 2) + (np.sqrt(np.clip(v, 0, None)) - rho) ** 2
        s = sc[r['episode']][2][:m.size]
        rec = float(np.mean(np.sqrt(np.clip(m, 0, None)) * s))
        err.append((rec - r['nRMSE']) / max(r['nRMSE'], 1e-9))
    err = np.array(err)
    print(f'reconstruction check on {err.size} runs: '
          f'median rel err {np.median(err):+.4f}  '
          f'p95 |rel err| {np.percentile(np.abs(err), 95):.4f}')
    ok = np.percentile(np.abs(err), 95) < 0.05
    print('  => reconstruction ' + ('USABLE' if ok else 'NOT USABLE (do not use below)'))
    if not ok:
        return

    # --- step 2: in-window -> just-outside -> long-horizon --------------------------
    by = defaultdict(dict)
    for r in rows:
        by[(r['cell'], r['episode'])][r['seed']] = r
    cells = sorted({c for c, _ in by})

    def traj(c, e, upto):
        d = by[(c, e)]
        vals = []
        for s in sorted(d):
            r = d[s]
            v = np.array(r['vr_frames'], float); rho = np.array(r['rho_frames'], float)
            m = (1 - rho ** 2) + (np.sqrt(np.clip(v, 0, None)) - rho) ** 2
            sg = sc[e][2][:m.size]
            n = np.sqrt(np.clip(m, 0, None)) * sg
            vals.append(float(np.mean(n[:upto])))
        return float(np.median(vals))

    print(f"\n{'cell':<9}{'init(in-win)':>13}{'frame1':>9}{'frames1-5':>11}"
          f"{'frame-mean':>12}   (median over 4 eps)")
    base = 'X-BASE'
    rowsum = {}
    for c in cells:
        if any((c, e) not in by for e in EPISODES):
            continue
        i = np.median([np.median([by[(c, e)][s]['init_nrmse'] for s in by[(c, e)]])
                       for e in EPISODES])
        f1 = np.median([traj(c, e, 1) for e in EPISODES])
        f5 = np.median([traj(c, e, 5) for e in EPISODES])
        fm = np.median([np.median([by[(c, e)][s]['nRMSE'] for s in by[(c, e)]])
                        for e in EPISODES])
        rowsum[c] = (i, f1, f5, fm)
        print(f'{c:<9}{i:>13.4f}{f1:>9.4f}{f5:>11.4f}{fm:>12.4f}')

    if base in rowsum:
        print('\npaired delta vs X-BASE at each horizon '
              '(if the init gain dies by frame 1 it was window fit, not physics)')
        for c in cells:
            if c == base or c not in rowsum:
                continue
            di = np.median([np.median([by[(c, e)][s]['init_nrmse'] for s in by[(c, e)]]) -
                            np.median([by[(base, e)][s]['init_nrmse'] for s in by[(base, e)]])
                            for e in EPISODES])
            d1 = np.median([traj(c, e, 1) - traj(base, e, 1) for e in EPISODES])
            d5 = np.median([traj(c, e, 5) - traj(base, e, 5) for e in EPISODES])
            dm = np.median([np.median([by[(c, e)][s]['nRMSE'] for s in by[(c, e)]]) -
                            np.median([by[(base, e)][s]['nRMSE'] for s in by[(base, e)]])
                            for e in EPISODES])
            keep = (d1 / di * 100) if abs(di) > 1e-9 else float('nan')
            print(f'  {c:<9} d_init {di:+.4f}   d_frame1 {d1:+.4f} ({keep:+.0f}% of the init '
                  f'gain survives one frame)   d_f1-5 {d5:+.4f}   d_mean {dm:+.4f}')


if __name__ == '__main__':
    main()
