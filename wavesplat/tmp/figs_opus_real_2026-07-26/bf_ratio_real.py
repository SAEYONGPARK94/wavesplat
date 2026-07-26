"""Compute the shipped guard signal (bf_peak_ratio) on all 30 real windows.
Prereg: PREREG_bf_ratio_real.md (predictions written first). Run from package root."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from wavesplat.optimizer.initializer import _beamform_coarse, MIN_DK

RECORDS = ['AA14', 'AA15a', 'AA15b', 'LJ18g', 'YS17']
COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}
GUARD = 0.05


def peak_ratio(x, y, z, t, depth, kx_range=(-4.0, 4.0), n_coarse=81, n_extract=30):
    """Verbatim logic of initializer.beamform_peak_ratio, taking raw arrays."""
    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z, t, depth, kx_range, kx_range, n_coarse, n_coarse)
    dkx = kx_axis[1] - kx_axis[0]; dky = ky_axis[1] - ky_axis[0]
    cx = int(np.argmin(np.abs(kx_axis))); cy = int(np.argmin(np.abs(ky_axis)))
    power[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0
    supx = int(MIN_DK / dkx) + 1; supy = int(MIN_DK / dky) + 1
    peaks = []
    P = power.copy()
    for _ in range(n_extract):
        idx = int(np.argmax(P.ravel()))
        iy, ix = np.unravel_index(idx, P.shape)
        pk = float(P[iy, ix])
        if pk < 1e-12:
            break
        peaks.append(pk)
        P[max(0, iy - supy):iy + supy + 1, max(0, ix - supx):ix + supx + 1] = 0.0
    if len(peaks) < 2 or peaks[0] <= 0:
        return 0.0
    return float(peaks[1] / peaks[0])


def main():
    import d3_compare_gifs as cg
    rows = []
    for rec in RECORDS:
        for win in range(6):
            sc = cg.scene_real(rec, win=win)
            P, z = sc['P'], sc['z']
            r = peak_ratio(P[:, 0], P[:, 1], z - z.mean(), P[:, 2], sc['depth'])
            rows.append(dict(record=rec, win=win, bf_ratio=r,
                             collapse=(rec, win) in COLLAPSE,
                             veto=r < GUARD))
            print(f'{rec}_w{win}: bf_ratio {r:.4f} -> '
                  f'{"VETO->analytic" if r < GUARD else "learned would run"}'
                  f'{"  [collapse win]" if (rec, win) in COLLAPSE else ""}', flush=True)
    norm = [r for r in rows if not r['collapse']]
    coll = [r for r in rows if r['collapse']]
    n_veto = sum(r['veto'] for r in norm)
    cell = 'A' if n_veto >= 20 else ('C' if n_veto <= 6 else 'B')
    summary = dict(n_normal=len(norm), n_veto_normal=n_veto,
                   ratios_normal=sorted(round(r['bf_ratio'], 4) for r in norm),
                   collapse=[(r['record'], r['win'], round(r['bf_ratio'], 4), r['veto'])
                             for r in coll],
                   guard=GUARD, prereg_cell=cell)
    json.dump(dict(rows=rows, summary=summary),
              open(os.path.join(HERE, 'bf_ratio_real.json'), 'w'), indent=1)
    print('\nSUMMARY:', json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
