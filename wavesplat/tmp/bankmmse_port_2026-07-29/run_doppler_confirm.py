"""Pre-registered confirmation run for record-conditional Doppler Uc.
Protocol and bars: PREREG_DOPPLER_CONFIRM.md (frozen before this run).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

sys.path.insert(0, 'tmp/bankmmse_port_2026-07-29')
from measure_blind_vs_frozen import dense_gt_std, score

OUT = 'tmp/bankmmse_port_2026-07-29/doppler_confirm.json'
SEEDS = [42, 1042, 2042]
TEST = [('YS17', w) for w in (1, 2, 3, 4, 5)]
NULLC = [('AA15a', 3), ('AA15b', 1), ('AA15a', 5), ('AA15b', 3)]

SPEEDS = np.arange(0.15, 1.20 + 1e-9, 0.15)
DIRS = np.arange(12) * (2 * np.pi / 12)
GRID = [(0.0, 0.0)] + [(sp * np.cos(th), sp * np.sin(th))
                       for sp in SPEEDS for th in DIRS]


def run_window(rec, win):
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate

    sc = cg.scene_real(rec, win)
    sg = dense_gt_std(rec, win)
    npx, nl = sc['npx'], len(sc['leads'])
    s0 = 0.2 * float(np.std(sc['z']))
    Pq_tiny = sc['Pq'][:2]

    t0 = time.time()
    vrs = []
    for Uc in GRID:
        d = {}
        mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq_tiny, sigma0=s0,
                           kp_hint=None, seed=42, Uc=Uc, diag_out=d)
        vrs.append(float(d['vr']))
    j = int(np.argmin(vrs))
    Usel = GRID[j]
    sel = dict(U=list(Usel), speed=float(np.hypot(*Usel)),
               dir_deg=float(np.degrees(np.arctan2(Usel[1], Usel[0]))),
               vr0=vrs[0], vr_sel=vrs[j], landscape_s=round(time.time() - t0, 1))

    out = {}
    for tag, Uc in (('U0', (0.0, 0.0)), ('Usel', Usel)):
        fs = []
        for s in SEEDS:
            fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                    sigma0=s0, kp_hint=None, seed=s, Uc=Uc)
            fs.append(fq[:npx * nl].reshape(nl, *sc['Xq'].shape))
        out[tag] = score(np.median(np.stack(fs, 0), 0), sc, sg)
    return dict(selection=sel, U0=out['U0'], Usel=out['Usel'],
                d_now=out['Usel']['nowcast_nrmse'] - out['U0']['nowcast_nrmse'],
                d_mean=out['Usel']['mean_nrmse'] - out['U0']['mean_nrmse'])


def circ_spread_deg(degs):
    """Max pairwise angular distance of the direction set [deg]."""
    if len(degs) < 2:
        return 0.0
    d = np.radians(np.asarray(degs, float))
    md = 0.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            a = abs(np.degrees(np.angle(np.exp(1j * (d[i] - d[j])))))
            md = max(md, a)
    return float(md)


def main():
    results = {'test': {}, 'null': {}}
    for rec, win in TEST + NULLC:
        key = f'{rec}_w{win}'
        kind = 'test' if (rec, win) in TEST else 'null'
        r = run_window(rec, win)
        results[kind][key] = r
        s = r['selection']
        print(f"[{kind}] {key}: Usel |U|={s['speed']:.2f} dir={s['dir_deg']:.0f} "
              f"vr {s['vr0']:.4f}->{s['vr_sel']:.4f} | "
              f"now {r['U0']['nowcast_nrmse']:.4f}->{r['Usel']['nowcast_nrmse']:.4f} "
              f"({r['d_now']:+.4f}) mean ({r['d_mean']:+.4f})", flush=True)
        json.dump(results, open(OUT, 'w'), indent=1)

    T = results['test']
    d_now = [T[k]['d_now'] for k in T]
    d_mean = [T[k]['d_mean'] for k in T]
    n_win = sum(1 for v in d_now if v < 0)
    floors = [k for k in T if abs(T[k]['selection']['speed'] - 0.15) < 1e-9]
    dirs = [T[k]['selection']['dir_deg'] for k in T
            if abs(T[k]['selection']['speed'] - 0.15) > 1e-9]
    spread = circ_spread_deg(dirs)
    N = results['null']
    d_now_null = [N[k]['d_now'] for k in N]

    P1 = bool(np.median(d_now) <= -0.02 and n_win >= 3)
    P2 = bool(len(floors) < 3 and spread <= 60.0)
    P3 = bool(np.median(d_mean) <= 0.01)
    P4 = bool(np.median(d_now_null) <= 0.01 and max(d_now_null) <= 0.05)
    verdict = dict(
        P1_win=dict(passed=P1, median_d_now=float(np.median(d_now)),
                    n_improved=n_win),
        P2_mechanism=dict(passed=P2, dir_spread_deg=spread,
                          n_floor=len(floors), dirs=dirs),
        P3_lead=dict(passed=P3, median_d_mean=float(np.median(d_mean))),
        P4_null=dict(passed=P4, median_d_now_null=float(np.median(d_now_null)),
                     max_d_now_null=float(max(d_now_null))),
        ALL_PASS=bool(P1 and P2 and P3 and P4))
    results['verdict'] = verdict
    json.dump(results, open(OUT, 'w'), indent=1)
    print('\nVERDICT:', json.dumps(verdict, indent=1), flush=True)


if __name__ == '__main__':
    main()
