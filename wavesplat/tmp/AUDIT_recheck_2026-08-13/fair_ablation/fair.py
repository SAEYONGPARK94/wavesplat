import json, os, sys, glob, collections, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat')
import numpy as np, torch
from wavesplat.optimizer import sequential_update as SU
from wavesplat.model.dispersion import dispersion_omega
_orig = SU.SequentialUpdater._build_optimizer
def patched(self):
    m = getattr(self, 'model', None)
    if m is not None and hasattr(m, 'omega_free') and not getattr(self, '_om_seeded', False):
        with torch.no_grad():
            k = torch.sqrt(m.kx**2 + m.ky**2 + 1e-12)
            m.omega_free.data.copy_(dispersion_omega(k, m.depth))
        self._om_seeded = True
    return _orig(self)
SU.SequentialUpdater._build_optimizer = patched
from wavesplat.run import run_estimation, DEFAULT_CONFIG
rows=[json.loads(l) for l in open('tmp/S13_oos_2026-08-12/results.jsonl')]
D=collections.defaultdict(dict)
for r in rows: D[r['episode']][r['arm']]=r
files={os.path.basename(p)[len('episode_'):-3]:p for d in
       ['data_s9_sparse_2026-08-11','data_s8_crowded_2026-08-10','data_s9_crowded_2026-08-11']
       for p in glob.glob(d+'/*.h5')}
OUT='/tmp/claude-1000/-home-vexit-wavesplat/39cf8928-7b64-4a9a-9499-4aaea3ab3771/scratchpad/fair.jsonl'
done=set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)['ep'])
with open(OUT,'a') as fh:
    for strat in ('sparse','crowded'):
        eps=sorted([e for e in D if D[e]['SHIP']['stratum']==strat])
        for e in eps[::4][:16]:
            if e in done: continue
            t0=time.time()
            cfg=dict(DEFAULT_CONFIG); cfg['use_dispersion']=False
            r=run_estimation(files[e], config=cfg, verbose=False)
            rec=dict(ep=e, strat=strat, fair_init=float(r['init_nrmse']), fair_fin=float(r['avg_nRMSE']),
                     ship_init=D[e]['SHIP']['init_nrmse'], ship_fin=D[e]['SHIP']['avg_nRMSE'],
                     nod_init=D[e]['SHIP_NODISP']['init_nrmse'], nod_fin=D[e]['SHIP_NODISP']['avg_nRMSE'])
            fh.write(json.dumps(rec)+'\n'); fh.flush()
            print(f"{strat:8s} {e:42s} FAIR {rec['fair_fin']:.4f} | SHIP {rec['ship_fin']:.4f} | NODISP {rec['nod_fin']:.4f} [{time.time()-t0:.0f}s]", flush=True)
print('DONE')
