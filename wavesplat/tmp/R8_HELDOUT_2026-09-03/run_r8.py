"""`R8-HELDOUT` 러너 — 추정기를 **한 번** 돌리고 **질의점 부분집합 둘**로 채점한다.

★그래서 `IN`/`OUT` 은 개구·관측수·시드·초기화가 **비트 동일**이고 채점 영역만 다르다.
⛔추정 체인은 `tmp/R1B_SCORINGFRAME_2026-09-02/run_r1b.py` 를 축자로 옮긴다(팔은 임포트).
"""
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for _p in (HERE, B2DIR, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_r8 as G                                                 # noqa: E402
from scene_r8 import scene_heldout, M_CAP                            # noqa: E402
from scene_station import scene_station                              # noqa: E402
import make_gif_bank_deployed as M                                   # noqa: E402

OUT = os.path.join(HERE, 'results_r8.jsonl')
META = os.path.join(HERE, 'meta_r8.json')

_MD5 = [os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py'),
        os.path.join(ROOT, 'wavesplat', 'optimizer', 'learned_init.py'),
        os.path.join(ROOT, 'wavesplat', 'run.py'),
        os.path.join(B2DIR, 'scene_station.py'),
        os.path.join(HERE, 'scene_r8.py'),
        os.path.join(HERE, 'gates_r8.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
            for p in _MD5}


def _h(a):
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64)).hexdigest()[:12]


def bit_check(record):
    """★`G-BIT` — `region='ALL'` 이 `scene_station` 과 비트 동일한가."""
    a = scene_heldout(record, win=G.WIN, region='ALL')
    b = scene_station(record, win=G.WIN)
    same = (_h(np.column_stack([a['P'], a['z']])) == _h(np.column_stack([b['P'], b['z']]))
            and _h(a['Pq']) == _h(b['Pq'])
            and _h(a['maps_gt']) == _h(b['maps_gt']))
    return bool(same), int(len(a['z']))


def run(record, arm):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    bit_ok, n_all = bit_check(record)
    sc = scene_heldout(record, win=G.WIN, region='LEFT')
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=sc['depth'], seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm])
    set_seed(G.SEED)
    upd = make_updater(cfg)
    upd.initialize(frames,
                   x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                   y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                   verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                          torch.tensor(gyq, dtype=torch.float32),
                          t0).cpu().numpy().reshape(len(gyq), len(gxq))
    g = np.asarray(sc['maps_gt'][0], float) + sc['delta_q']          # ★R1b 수리 규약
    fin = np.isfinite(g)
    sg = sc['sg']

    def score(mask):
        mm = fin & mask
        if mm.sum() < 1:
            return float('nan'), float('nan'), 0, float('nan')
        rm = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sg)
        z0 = float(np.sqrt(np.mean(g[mm] ** 2)) / sg)
        return rm, z0, int(mm.sum()), float(np.median(sc['gap'][mm]) / sc['lam'])

    rin, zin, nin, gin = score(sc['in_mask'])
    rout, zout, nout, gout = score(sc['out_mask'])

    eff = getattr(upd, 'config', {}) or {}
    exp = G._shipping_expect()
    arm_ok = (all(eff.get(k) == v for k, v in G.ARMS[arm].items())
              and {k: eff.get(k) for k in G.SHIPPING_KEYS} == exp)
    return dict(cell=arm, record=record, win=G.WIN, seed=G.SEED,
                scene_seed=sc['scene_seed'], convention='R1b_mu_mem',
                bit_identical_all=bit_ok, n_obs=int(len(sc['z'])), n_obs_all=n_all,
                x_split=sc['x_split'], lam=sc['lam'], kp=sc['kp'],
                rmse_in=rin, rmse_out=rout, zero_in=zin, zero_out=zout,
                ratio_in=(rin / zin if zin else float('nan')),
                ratio_out=(rout / zout if zout else float('nan')),
                npx_in=nin, npx_out=nout, gap_in_lam=gin, gap_out_lam=gout,
                arm_ok=bool(arm_ok),
                shipping={k: eff.get(k) for k in G.SHIPPING_KEYS},
                n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                code_md5=code_md5(), pid=os.getpid(),
                wall_s=round(time.time() - t0w, 1))


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                done.add((r.get('cell'), r.get('record')))
    with open(OUT, 'a', encoding='utf-8') as fh:
        for arm in G.ARM_ORDER:
            for rec in G.RECORDS:
                if (arm, rec) in done:
                    continue
                try:
                    r = run(rec, arm)
                except Exception as exc:                             # noqa: BLE001
                    r = dict(cell=arm, record=rec,
                             error=f'{type(exc).__name__}: {exc}')
                fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
                if 'error' in r:
                    print(f"  {arm:5s} {rec:6s} ⛔{r['error']}", flush=True)
                else:
                    print(f"  {arm:5s} {rec:6s} bit={r['bit_identical_all']} "
                          f"n={r['n_obs']} in={r['ratio_in']:.3f} out={r['ratio_out']:.3f} "
                          f"gap {r['gap_in_lam']:.3f}→{r['gap_out_lam']:.3f} λ "
                          f"({r['wall_s']}s)", flush=True)
    json.dump(dict(m_cap=M_CAP, shipping_expect=G._shipping_expect(),
                   code_md5=code_md5()),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
