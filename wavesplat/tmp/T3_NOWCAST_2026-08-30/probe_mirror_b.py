"""G-MIRROR-b 실현가능성 탐침 — ⛔판정 자료 아님. 채점(GT 대조) 0회.

물음: `SHIP-K1M3` 의 세 멤버(k=1, seed 42/1042/2042)가 **서로 다른 장**을 내는가.
36 셀 전건에서 멤버 장 md5 를 세고, 그것이 `SHIP-K3`(k=3 median) 과 일치하는지 본다.
⛔GT 를 읽지 않는다 ⟹ 어떤 판정량도 노출되지 않는다.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib, json, sys, time, warnings          # noqa: E402
warnings.filterwarnings('ignore')
import numpy as np                                 # noqa: E402
import torch                                       # noqa: E402

ROOT = '/home/vexit/wavesplat/wavesplat'
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'T3_NOWCAST_2026-08-30'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)

import gates_t3 as G                               # noqa: E402
from scene_station import scene_station            # noqa: E402
import make_gif_bank_deployed as MK                # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater   # noqa: E402
from wavesplat.run import DEFAULT_CONFIG, set_seed               # noqa: E402


def md5(Z):
    return hashlib.md5(np.ascontiguousarray(np.asarray(Z, np.float32)).tobytes()
                       ).hexdigest()[:10]


def field(sc, frames, t0, arm, off):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), device='cpu')
    cfg.update(G.arm_config(arm))
    cfg.update(G.member_overrides(arm, off))
    set_seed(int(cfg['seed']))
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False) or {}
    Z = upd.predict_field(torch.tensor(np.asarray(sc['gxq']), dtype=torch.float32),
                          torch.tensor(np.asarray(sc['gyq']), dtype=torch.float32),
                          t0).cpu().numpy().reshape(len(sc['gyq']), len(sc['gxq']))
    hp = None
    b = getattr(upd, '_bank', None)
    if b:
        hp = [(round(float(f.pexp), 6), round(float(f.smult), 6),
               int(f.n_candidates)) for f in b]
    return Z, hp


out = []
t0all = time.time()
for rec in G.RECORDS:
    for win in G.WINDOWS:
        sc = scene_station(rec, win=win, noise=G.NOISE)
        t_shift = float(np.asarray(sc['P'])[:, 2].min())
        t0 = float(sc['Pq'][0, 2]) - t_shift
        frames = MK.frames_from_scene(sc, t_shift, dtype=torch.float32)
        F1, hp1 = [], []
        for off in G.ARMS['SHIP-K1M3']['MEM']:
            Z, hp = field(sc, frames, t0, 'SHIP-K1M3', off)
            F1.append(Z); hp1.append(hp[0])
        Z3, hp3 = field(sc, frames, t0, 'SHIP-K3', 0)
        med = np.median(np.stack([np.asarray(f, np.float32) for f in F1], 0), 0)
        r = dict(record=rec, win=win,
                 member_md5=[md5(z) for z in F1], n_distinct=len({md5(z) for z in F1}),
                 k3_md5=md5(Z3), mirror_ok=(md5(med) == md5(Z3)),
                 hp_k1=hp1, hp_k3=hp3,
                 n_obs=int(len(sc['z'])))
        out.append(r)
        print(f'{rec:6s} w{win}  distinct={r["n_distinct"]}/3  mirror={r["mirror_ok"]}  '
              f'k1={r["member_md5"]}  k3={r["k3_md5"]}  hp_k1={hp1}', flush=True)

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'probe_mirror_b.json'), 'w'), indent=1)
n = len(out)
print(f'\n=== {n} 셀, {time.time()-t0all:.0f}s ===')
print('멤버 3개가 서로 다른 셀:', sum(1 for r in out if r['n_distinct'] == 3), '/', n)
print('멤버 2개만 다른 셀   :', sum(1 for r in out if r['n_distinct'] == 2), '/', n)
print('멤버 전부 같은 셀    :', sum(1 for r in out if r['n_distinct'] == 1), '/', n)
print('거울(median==k=3) 성립:', sum(1 for r in out if r['mirror_ok']), '/', n)
