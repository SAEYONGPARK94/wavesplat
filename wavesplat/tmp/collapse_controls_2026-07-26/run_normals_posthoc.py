"""EXPLORATORY / POST-HOC (NOT in PREREG; added after the frozen scoring, and labelled as
such everywhere). The frozen run found E2-double (kp_hint x 2) to be the strongest cheap
escape. That leaves one interpretation open that the frozen grid cannot close: is kp_hint x 2
a *collapse escape*, or simply a better default for this arena? This runs the single config
kp_hint x 2 on the 60 NORMAL windows (hand < 1.0) to see what it costs there.

Result is diagnostic only; it may not be used to satisfy or fail any bar, and no default is
changed by it. CPU only.
"""
import os, sys, json, time, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'heldout_real_2026-07-26'))
OUT = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26/normals_posthoc.json')


def job(args):
    rec, win = args
    import heldout_cfg
    heldout_cfg.install()
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    sc = cg.scene_real(rec, win=win)
    npx = sc['npx']; nl = len(sc['leads'])
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))

    def score(fq):
        m0 = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)[0]
        m = np.isfinite(gt[0]) & np.isfinite(m0)
        return float(np.sqrt(np.mean((m0[m] - gt[0][m]) ** 2)) / sg) if m.sum() >= 8 else None
    base = score(mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                    sigma0=0.05 * sc['Hs'], kp_hint=sc['kp'], seed=sc['seed']))
    e2d = score(mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                   sigma0=0.05 * sc['Hs'], kp_hint=2.0 * sc['kp'], seed=sc['seed']))
    return dict(record=rec, win=int(win), base=base, e2_double=e2d)


def main():
    hv = json.load(open(os.path.join(ROOT, 'tmp/heldout_real_2026-07-26/harvest_heldout.json')))
    hd = json.load(open(os.path.join(ROOT, 'tmp/unified_2026-07-26/hand_amp.json')))
    tg = [(r['record'], r['win']) for r in hv['rows']
          if r.get('hand_now') is not None and r['hand_now'] < 1.0]
    tg += [(r['record'], r['win']) for r in hd['rows'] if r['hand_now'] < 1.0]
    print(f'# POST-HOC: kp_hint x2 on {len(tg)} NORMAL windows, 6 workers', flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(job, j): j for j in tg}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:
                print(f'  {futs[fu]}: FAIL {e!r}', flush=True); traceback.print_exc(); continue
            rows.append(r)
            print(f"  {r['record']}_w{r['win']} base={r['base']:.3f} kpx2={r['e2_double']:.3f} "
                  f"d={r['base']-r['e2_double']:+.3f}", flush=True)
    rows.sort(key=lambda r: (r['record'], r['win']))
    json.dump(dict(rows=rows, note='POST-HOC, not pre-registered', elapsed_s=time.time() - t0),
              open(OUT, 'w'), indent=1)
    d = np.array([r['base'] - r['e2_double'] for r in rows])
    print(f'\n n={len(d)}  improved {int((d>0).sum())}  worsened {int((d<0).sum())}  '
          f'median d {np.median(d):+.4f}  mean d {d.mean():+.4f}')
    print(f' newly ABOVE 1.0 (was normal, becomes collapse): '
          f'{sum(1 for r in rows if r["e2_double"] >= 1.0)}')
    print(f'wrote {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
