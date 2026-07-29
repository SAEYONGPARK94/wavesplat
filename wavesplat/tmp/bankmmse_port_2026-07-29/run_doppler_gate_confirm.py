"""PREREG-2 run: false-fire rate of the record-coherence Doppler gate on fresh
no-current windows.  Protocol/bars: PREREG2_DOPPLER_GATE.md (frozen)."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

OUT = 'tmp/bankmmse_port_2026-07-29/doppler_gate_confirm.json'
NULLSET = {'AA15a': [0, 1, 2, 4], 'AA15b': [0, 2, 4, 5],
           'AA14': [0, 1, 2, 3, 4, 5], 'LJ18g': [0, 1, 2, 3, 4, 5]}

SPEEDS = np.arange(0.15, 1.20 + 1e-9, 0.15)
DIRS = np.arange(12) * (2 * np.pi / 12)
GRID = [(0.0, 0.0)] + [(sp * np.cos(th), sp * np.sin(th))
                       for sp in SPEEDS for th in DIRS]
FLOOR = 0.15 + 1e-9


def select_window(rec, win):
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    sc = cg.scene_real(rec, win)
    s0 = 0.2 * float(np.std(sc['z']))
    Pq_tiny = sc['Pq'][:2]
    vrs = []
    for Uc in GRID:
        d = {}
        mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq_tiny, sigma0=s0,
                           kp_hint=None, seed=42, Uc=Uc, diag_out=d)
        vrs.append(float(d['vr']))
    j = int(np.argmin(vrs))
    U = GRID[j]
    return dict(U=list(U), speed=float(np.hypot(*U)),
                dir_deg=float(np.degrees(np.arctan2(U[1], U[0]))),
                vr0=vrs[0], vr_sel=vrs[j])


def circ_spread_deg(degs):
    if len(degs) < 2:
        return 0.0
    d = np.radians(np.asarray(degs, float))
    md = 0.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            md = max(md, abs(np.degrees(np.angle(np.exp(1j * (d[i] - d[j]))))))
    return float(md)


def gate(sels):
    nf = [s for s in sels if s['speed'] > FLOOR]
    spread = circ_spread_deg([s['dir_deg'] for s in nf])
    med_sp = float(np.median([s['speed'] for s in nf])) if nf else 0.0
    fired = bool(len(nf) >= 3 and spread <= 45.0 and med_sp >= 0.30)
    return dict(fired=fired, n_nonfloor=len(nf), dir_spread_deg=spread,
                median_speed=med_sp)


def main():
    results = {'records': {}}
    for rec, wins in NULLSET.items():
        sels = []
        for w in wins:
            t0 = time.time()
            s = select_window(rec, w)
            sels.append(dict(window=w, **s))
            print(f'{rec} w{w}: |U|={s["speed"]:.2f} dir={s["dir_deg"]:.0f} '
                  f'vr {s["vr0"]:.4f}->{s["vr_sel"]:.4f} '
                  f'[{time.time()-t0:.0f}s]', flush=True)
        g = gate(sels)
        results['records'][rec] = dict(selections=sels, gate=g)
        print(f'>> {rec} gate: {g}', flush=True)
        json.dump(results, open(OUT, 'w'), indent=1)

    fired = [r for r, v in results['records'].items() if v['gate']['fired']]
    # Y1 derived statistic (consumed YS17 windows, non-confirmatory)
    ys17 = dict(dirs=[150.0] * 6, speeds=[1.05, 1.2, 1.2, 0.75, 0.75, 1.05])
    y1 = gate([dict(speed=s, dir_deg=d)
               for s, d in zip(ys17['speeds'], ys17['dirs'])])
    results['verdict'] = dict(
        N1=dict(passed=len(fired) == 0, fired_records=fired),
        Y1_derived_nonconfirmatory=y1)
    json.dump(results, open(OUT, 'w'), indent=1)
    print('\nVERDICT:', json.dumps(results['verdict'], indent=1), flush=True)


if __name__ == '__main__':
    main()
