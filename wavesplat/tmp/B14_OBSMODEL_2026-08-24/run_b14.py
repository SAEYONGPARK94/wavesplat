"""B-14 러너 — 관측 모형 개입. **게이트·팔·문턱은 `gates_b14.py` 하나에서만 온다.**

축 A: 관측 모형 4종 × 잡음 3수준 × 레코드 12 = 144런
축 B: 밀도 4수준 × 레코드 12 = 48런 (전부 무잡음)

행에 싣는 것(사후 추론 금지 — 전부 **되읽기**):
  · `effective` = updater 객체에서 되읽은 배포 상수 4개
  · `n_obs_used` · `z_hash`(개입 발화 증거) · `zero_ref`(참 제로필드, 상수 아님)
"""
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)

import gates_b14 as G                                # noqa: E402
from obs_models import scene_obs                     # noqa: E402
import make_gif_bank_deployed as M                   # noqa: E402

OUT = os.path.join(HERE, 'results_b14.jsonl')


def _one(record, model, noise, m_scan, axis):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    cap = 600 if axis == 'A' else m_scan * G.DENSITY_CAP_PER_FRAME
    sc = scene_obs(record, win=G.WIN, model=model, noise=noise, m_scan=m_scan,
                   m_cap=cap)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.PROP_CFG)
    set_seed(G.SEED)
    upd = make_updater(cfg)
    upd.initialize(frames,
                   x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                   y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                   verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    g = np.asarray(sc['maps_gt'][0], float)
    mm = np.isfinite(g)
    Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                          torch.tensor(gyq, dtype=torch.float32),
                          t0).cpu().numpy().reshape(len(gyq), len(gxq))
    now = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
    # ★참 제로필드 = Z≡0 을 **같은 채점식**에 통과시킨 값 (상수 금지)
    zero = float(np.sqrt(np.mean(g[mm] ** 2)) / sc['sg'])
    # 배포 상수 되읽기 (의도 dict 이 아니라 객체에서)
    ucfg = getattr(upd, 'config', {}) or {}
    eff = {k: ucfg.get(k) for k in ('varpro_steps', 'learned_init_iters',
                                    'auto_switch_thresh', 'lambda_l1',
                                    'init_method', 'real_domain_gate')}
    zh = hashlib.sha256(np.asarray(sc['z'], float).tobytes()).hexdigest()[:16]
    return dict(axis=axis, record=record, obs_model=model, noise=noise,
                m_scan=m_scan, nowcast=now, zero_ref=zero,
                beats_zero=bool(now < zero), sigma=float(sc['sg']),
                nsr=(noise / float(sc['sg'])), lam=float(2 * np.pi / sc['kp']),
                n_obs_used=int(sc['n_obs_used']), z_hash=zh,
                n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                effective=eff, wall_s=time.time() - t0w)


def cells():
    for rec in G.RECORDS:
        for mdl in G.OBS_MODELS:
            for nz in G.NOISE_LEVELS:
                yield ('A', rec, mdl, nz, G.M_SCAN_A)
    for rec in G.RECORDS:
        for m_ in G.DENSITIES:
            yield ('B', rec, 'IID', G.NOISE_B, m_)


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['axis'], r['record'], r['obs_model'], r['noise'], r['m_scan']))
    todo = [c for c in cells() if c not in done]
    print(f'B-14: {len(todo)} 셀 남음 (완료 {len(done)})', flush=True)
    with open(OUT, 'a') as fh:
        for i, (axis, rec, mdl, nz, ms) in enumerate(todo, 1):
            try:
                row = _one(rec, mdl, nz, ms, axis)
            except Exception as e:                    # noqa: BLE001
                row = dict(axis=axis, record=rec, obs_model=mdl, noise=nz,
                           m_scan=ms, error=f'{type(e).__name__}: {e}')
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            tag = row.get('error') or (f"nRMSE {row['nowcast']:.4f} "
                                       f"(zero {row['zero_ref']:.4f}) "
                                       f"{'WIN' if row['beats_zero'] else '   '}")
            print(f'[{i}/{len(todo)}] {axis} {rec:6s} {mdl:5s} σ={nz:<5} m={ms:<3} {tag}',
                  flush=True)


if __name__ == '__main__':
    main()
