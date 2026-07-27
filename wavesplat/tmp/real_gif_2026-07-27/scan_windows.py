"""Score the DEPLOYED path (make_updater -> init_method='auto' -> real-domain gate ->
single v2_konly) on candidate real stereo windows, to pick one representative good
normal window for the animation. Read-only w.r.t. repo code."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0, ROOT); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT,'scripts'))
sys.path.insert(0, os.path.join(ROOT,'tmp/real_gif_2026-07-27'))

CAND = [('AA15a',5),('AA15b',1),('AA14',2),('AA15b',5),('AA15a',0),
        ('AA15a',2),('AA15b',3),('AA15b',4),('YS17',1),('YS17',0)]
OUT = 'tmp/real_gif_2026-07-27/scan.jsonl'


def _work(job):
    rec, win = job
    import numpy as np
    from probe_est import run
    t = time.time()
    try:
        r = run(rec, win)
        return dict(record=rec, win=win, nowcast_nrmse=r['nr'][0], nowcast_corr=r['cc'][0],
                    lead_nrmse=float(np.mean(r['nr'])), lead_corr=float(np.mean(r['cc'])),
                    per_lead_nrmse=r['nr'], per_lead_corr=r['cc'], n_gauss=r['n_gauss'],
                    route=r['route'], real_gate=r['real_gate'], k=r['k'],
                    sec=time.time()-t, err=None)
    except Exception as e:
        return dict(record=rec, win=win, err=repr(e), sec=time.time()-t)


if __name__ == '__main__':
    with open(OUT,'w',buffering=1) as f, Pool(5) as p:
        for r in p.imap_unordered(_work, CAND):
            f.write(json.dumps(r)+'\n')
            if r.get('err'):
                print(f"{r['record']} w{r['win']} ERR {r['err'][:80]}", flush=True)
            else:
                print(f"{r['record']:6s} w{r['win']} now {r['nowcast_nrmse']:.3f}/"
                      f"{r['nowcast_corr']:.3f}  lead {r['lead_nrmse']:.3f}/"
                      f"{r['lead_corr']:.3f}  N={r['n_gauss']} ({r['sec']:.0f}s)", flush=True)
    print('done', flush=True)
