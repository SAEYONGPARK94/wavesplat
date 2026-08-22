"""P1b: 실측 경로(GT-free)를 전역 RNG 를 명시 고정하고 재측정 — 비트 비교."""
import sys, os, json, warnings
import numpy as np, torch
TAG = sys.argv[1]
from wavesplat.data_loader import FrameData
from wavesplat.optimizer.sequential_update import make_updater
from wavesplat.run import DEFAULT_CONFIG

def frames(n=64, nf=3):
    g = torch.Generator().manual_seed(0)
    return [FrameData(t=float(i), x_obs=torch.rand(n, generator=g)*100+10,
                      y_obs=(torch.rand(n, generator=g)-0.5)*100,
                      z_obs=torch.randn(n, generator=g)*0.1,
                      confidence=torch.ones(n),
                      incidence_angles=torch.zeros(n), gt_field=None) for i in range(nf)]

out = {}
for est in ('bank_mmse', 'v2_konly'):
    torch.manual_seed(2026); np.random.seed(2026)
    cfg = dict(DEFAULT_CONFIG, depth=50.0, seed=42, real_estimator=est)
    u = make_updater(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = u.initialize(frames(), x_range=(10.0, 110.0), y_range=(-50.0, 50.0))
    m = getattr(u, 'model', None)
    par = None if m is None else {k: getattr(m,k).detach().cpu().numpy().astype(np.float64).tolist()
                                  for k in ('kx','ky','w','phi')}
    fld = u.predict_field(torch.linspace(10.,110.,9), torch.linspace(-50.,50.,9), 2.0)
    fld = fld.detach().cpu().numpy() if hasattr(fld,'detach') else np.asarray(fld)
    out[est] = {'gate': bool(getattr(u,'_auto_real_gate',None)),
                'init_method': u.init_method, 'final_n': r.get('final_n'),
                'ensemble_k': r.get('ensemble_k'), 'params': par,
                'field': np.asarray(fld,dtype=np.float64).ravel().tolist()}
json.dump(out, open(f'v1_probe_real_{TAG}.json','w'), indent=1)
print(TAG,'ok')
