"""F-5 (coordinator-assigned, 2026-07-22) — independent re-derivation of the E-59
"level-calibration" (magnitude, not rank) result cited in E59_AUDIT_findings.md's
RESCUE section (level-rho AA15a +1.00 / AA14 +0.92 / YS17 +0.89, 5-6x dynamic range).

That number was produced via the RAW_SINK hook in tmp/e59v2_real_anchor.py, which
is NOT persisted in results/direction3/e59v2_real_anchor.json (only summary rows
are saved -- same "ad hoc analysis, no artifact on disk" pattern flagged elsewhere
in this project, e.g. E60K audit finding F3). This script re-runs process_record()
for mode='real' on the 3 primary census records (AA15a, AA14, YS17) with RAW_SINK
captured, then bins sigma_CRB into deciles and reports median sigma_actual per
decile -- reusing 100% existing code/data (no retraining, no new simulation, no
network calls). Does NOT overwrite results/direction3/e59v2_real_anchor.json.

Usage: python3 tmp/f5_level_calibration.py
Out:   results/direction3/f5_level_calibration.json
"""
import os
import sys
import json
import time

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'tmp')
sys.path.insert(0, 'scripts')
import e59v2_real_anchor as E

OUT = 'results/direction3/f5_level_calibration.json'
RECS = ['AA15a', 'AA14', 'YS17']
NBIN = 10


def decile_calibration(crb_all, act_all, nbin=NBIN):
    """Bin by sigma_CRB decile, return per-bin median sigma_actual + Spearman(bin
    rank, median act) as 'level-rho', + dynamic range (max/min bin median)."""
    m = np.isfinite(crb_all) & np.isfinite(act_all)
    crb, act = crb_all[m], act_all[m]
    order = np.argsort(crb)
    crb_s, act_s = crb[order], act[order]
    edges = np.linspace(0, len(crb_s), nbin + 1).astype(int)
    bin_med_crb, bin_med_act = [], []
    for i in range(nbin):
        sl = slice(edges[i], edges[i + 1])
        if edges[i + 1] <= edges[i]:
            continue
        bin_med_crb.append(np.median(crb_s[sl]))
        bin_med_act.append(np.median(act_s[sl]))
    bin_med_crb, bin_med_act = np.array(bin_med_crb), np.array(bin_med_act)
    level_rho = stats.spearmanr(np.arange(len(bin_med_act)), bin_med_act).correlation
    dyn_range = float(bin_med_act.max() / max(bin_med_act.min(), 1e-9))
    return bin_med_crb, bin_med_act, float(level_rho), dyn_range, int(m.sum())


def run_mode(mode):
    E.RAW_SINK = []
    rows = []
    for rc in RECS:
        rows += E.process_record(rc, mode, 128, False)
    raw = E.RAW_SINK
    assert len(raw) == len(rows), "RAW_SINK/rows length mismatch -- ordering assumption broken"
    for r, rw in zip(raw, rows):
        r['rec'] = rw['rec']; r['lead_fr'] = rw['lead_fr']; r['window'] = rw['window']
    return raw


def main():
    t0 = time.time()
    out = {}
    for mode in ('real', 'twin'):
        raw = run_mode(mode)
        print(f"\n=== {mode.upper()}: level-calibration (decile-binned sigma_CRB vs "
              f"median sigma_actual) [{time.time()-t0:.0f}s] ===")
        out[mode] = {}
        for rc in RECS:
            rr = [r for r in raw if r['rec'] == rc]
            crb_all = np.concatenate([r['crb'] for r in rr])
            act_all = np.concatenate([r['act'] for r in rr])
            bmc, bma, level_rho, dyn_range, n = decile_calibration(crb_all, act_all)
            out[mode][rc] = dict(level_rho=level_rho, dyn_range=dyn_range, n_q=n,
                                 bin_med_crb=bmc.tolist(), bin_med_act=bma.tolist())
            print(f"  {rc:6s} n_q={n:6d}  level-rho={level_rho:+.3f}  "
                  f"dyn_range={dyn_range:.1f}x  "
                  f"act[lo-dec..hi-dec]={bma[0]:.3f}->{bma[-1]:.3f}")
        crb_all = np.concatenate([r['crb'] for r in raw])
        act_all = np.concatenate([r['act'] for r in raw])
        bmc, bma, level_rho, dyn_range, n = decile_calibration(crb_all, act_all)
        out[mode]['POOLED'] = dict(level_rho=level_rho, dyn_range=dyn_range, n_q=n,
                                   bin_med_crb=bmc.tolist(), bin_med_act=bma.tolist())
        print(f"  {'POOLED':6s} n_q={n:6d}  level-rho={level_rho:+.3f}  dyn_range={dyn_range:.1f}x")

    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.0f}s)")
    return out


if __name__ == '__main__':
    main()
