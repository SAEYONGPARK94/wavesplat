"""DESIGN-PHASE (pre-freeze): beamform dominant-peak count as a GT-free routing signal.

The learned-proposal atoms are unreliable exactly on the OOD edge (BF2_1comp slope smears
30 atoms with no dominant peak). The v2 beamform power map is computed directly from the
init-window observations, independent of the diffusion proposal, and is deterministic.

For each episode init window we run _beamform_coarse (v2 convention), zero DC, then greedily
extract peaks with MIN_DK suppression, and count how many peaks exceed a fraction of the
dominant peak power. This is the 'dominant-peak structure' signal. We check whether it
cleanly separates BF2_1comp (single swell -> few) from genuine crowding (15/30comp -> many).

Reads episodes read-only. One beamform per episode (diff_seed-independent).
"""
import json, re, os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.'))
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import (
    _beamform_coarse, _collect_observations, MIN_DK)

FINE = 'data_finemesh_2026-07-23/'
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'


def comp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


def beamform_peak_profile(frames, depth, n_coarse=81, n_extract=30):
    """Return list of peak powers (descending, MIN_DK-suppressed) from the v2 coarse map."""
    x, y, z, t = _collect_observations(frames)
    if x is None or len(x) < 10:
        return []
    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z, t, depth, (-4.0, 4.0), (-4.0, 4.0), n_coarse, n_coarse)
    dkx = kx_axis[1] - kx_axis[0]; dky = ky_axis[1] - ky_axis[0]
    cx = np.argmin(np.abs(kx_axis)); cy = np.argmin(np.abs(ky_axis))
    power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0
    supx = int(MIN_DK/dkx)+1; supy = int(MIN_DK/dky)+1
    peaks = []
    P = power.copy()
    for _ in range(n_extract):
        idx = np.argmax(P.ravel())
        iy, ix = np.unravel_index(idx, P.shape)
        pk = P[iy, ix]
        if pk < 1e-12:
            break
        peaks.append(float(pk))
        yl, yh = max(0, iy-supy), min(P.shape[0], iy+supy+1)
        xl, xh = max(0, ix-supx), min(P.shape[1], ix+supx+1)
        P[yl:yh, xl:xh] = 0.0
    return peaks


def main():
    eps = sorted(set(json.loads(l)['episode'] for l in open(BATCHC)))
    assert len(eps) == 100
    out = []
    t0 = time.time()
    for i, ep in enumerate(eps):
        L = EpisodeDataLoader(FINE + ep)
        fr = L.get_init_window(2.0)
        peaks = beamform_peak_profile(fr, L.depth)
        pk = np.array(peaks)
        rec = dict(ep=ep, comp=comp(ep), n_peaks_found=len(peaks))
        if len(pk) > 0:
            pmax = pk[0]
            for frac in (0.1, 0.2, 0.3, 0.5):
                rec[f'npk_{int(frac*100)}'] = int((pk >= frac * pmax).sum())
            rec['p2_over_p1'] = float(pk[1]/pmax) if len(pk) > 1 else 0.0
            rec['p3_over_p1'] = float(pk[2]/pmax) if len(pk) > 2 else 0.0
            rec['p8_over_p1'] = float(pk[7]/pmax) if len(pk) > 7 else 0.0
        out.append(rec)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/100  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('tmp/unified_guard_2026-07-26/explore_beamform.json', 'w'), indent=0)

    by = {}
    for r in out:
        by.setdefault(r['comp'], []).append(r)

    def stat(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return f"med={np.median(vals):.2f} [{np.min(vals):.0f},{np.max(vals):.0f}]" if vals else 'n/a'

    print("\n" + "="*110)
    print("Beamform dominant-peak count by comp: how many peaks >= frac * dominant peak power")
    print("="*110)
    print(f"{'comp':>4} {'N':>3} | {'npk>=10%':>16} | {'npk>=20%':>16} | {'npk>=30%':>16} | {'npk>=50%':>16} | {'p2/p1':>14}")
    for c in sorted(by):
        rs = by[c]
        print(f"{c:>4} {len(rs):>3} | {stat([r.get('npk_10') for r in rs]):>16} | {stat([r.get('npk_20') for r in rs]):>16} | "
              f"{stat([r.get('npk_30') for r in rs]):>16} | {stat([r.get('npk_50') for r in rs]):>16} | {stat([r.get('p2_over_p1') for r in rs]):>14}")

    # Separation check: sparse (1,2) vs crowded (15,30)
    print("\n" + "="*110)
    print("SEPARATION: npk>=frac distribution, sparse(1,2comp) vs crowded(15,30comp)")
    print("="*110)
    for frac in (10, 20, 30, 50):
        key = f'npk_{frac}'
        sp = sorted(r[key] for r in out if r['comp'] in (1,2))
        cr = sorted(r[key] for r in out if r['comp'] in (15,30))
        md = sorted(r[key] for r in out if r['comp'] == 5)
        print(f"  {key}: sparse max={max(sp)} (median {np.median(sp):.0f}); "
              f"moderate5 [{min(md)},{max(md)}] med {np.median(md):.0f}; "
              f"crowded min={min(cr)} (median {np.median(cr):.0f})  "
              f"-> clean gap if sparse.max < crowded.min: {max(sp)} < {min(cr)} = {max(sp) < min(cr)}")

    # BF2_1comp specifically (the characterized misroute)
    print("\n" + "="*110)
    print("BF2_1comp episodes (the characterized misroute cluster): beamform peak counts")
    print("="*110)
    for r in out:
        if r['comp'] == 1 and 'BF2' in r['ep']:
            print(f"  {r['ep']:34s} npk10={r.get('npk_10')} npk20={r.get('npk_20')} npk30={r.get('npk_30')} "
                  f"npk50={r.get('npk_50')} p2/p1={r.get('p2_over_p1',0):.3f} p3/p1={r.get('p3_over_p1',0):.3f}")
    print(f"\ndone {time.time()-t0:.0f}s; wrote explore_beamform.json")


if __name__ == '__main__':
    main()
