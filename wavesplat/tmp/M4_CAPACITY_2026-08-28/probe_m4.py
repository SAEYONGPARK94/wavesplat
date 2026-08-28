"""M4 **Phase A** — 실측 12 레코드의 확산 제안 원자를 **한 번** 만들어 동결한다.

    CUDA_VISIBLE_DEVICES= python3 tmp/M4_CAPACITY_2026-08-28/probe_m4.py

⛔**GPU 금지.** 이 파일은 import 전에 `CUDA_VISIBLE_DEVICES=''` 를 스스로 박고
`torch.cuda.is_available()` 이 False 임을 **assert** 한다. 그래도 외부에서 넘겨받은 값이
있으면 그것을 존중하지 않고 덮어쓴다(다른 세션의 M1 21시간을 지키기 위해).

## 이 파일이 만드는 것
  · `atoms_real/<record>__s<seed>.json` — 원시 원자 128개 (nsamp=4 × M=32)
  · `probe_m4.json`  — ★**실측 용량 헤드룸**: 팔별 설치 기대 원자 수 K + 원자 md5
  · `FROZEN_md5_phaseA.txt` — 위 파일들의 md5 (⟹ Phase B 는 **이 목록에 묶인다**)

## 왜 Phase A 가 따로인가
확산 제안은 이 배치에서 **유일하게 비싼 것**이고 팔마다 반복할 이유가 없다. 한 번 만들어
동결하면 (i) 모든 팔이 **같은 원자 풀**을 나눠 쓰고(⟹ 개입이 정확히 선택 노브 셋),
(ii) Phase B 는 확산을 **0회** 부른다(`G-17` 이 강제), (iii) GPU 가 필요 없다.
"""
import os
import sys

# ★import 순서가 중요하다 — torch 보다 먼저.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')

import hashlib      # noqa: E402
import json         # noqa: E402
import time         # noqa: E402
import warnings     # noqa: E402

warnings.filterwarnings('ignore')
import numpy as np  # noqa: E402
import torch        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import gates_m4 as G                         # noqa: E402
from scene_station import scene_station      # noqa: E402
import make_gif_bank_deployed as M           # noqa: E402

ATOMS = os.path.join(HERE, 'atoms_real')
OUT = os.path.join(HERE, 'probe_m4.json')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_phaseA.txt')

assert not torch.cuda.is_available(), 'GPU 가 보인다 — 중단. (M1 이 GPU 를 쓰고 있다)'


def _md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def raw_atoms(record, seed):
    """`_auto_select_init:1017-1022` 축자와 같은 호출 (gates_b22.learned_raw_atoms 규약)."""
    from wavesplat.optimizer.learned_init import propose_atoms, DEFAULT_CKPT
    from wavesplat.run import set_seed
    sc = scene_station(record, win=G.WIN)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    set_seed(int(seed))                      # 러너와 같은 전역 시드 상태에서 부른다
    kx, ky, en = propose_atoms(frames, float(sc['depth']), ckpt=DEFAULT_CKPT,
                               nsamp=4, diffusion_seed=int(seed))
    return dict(kx=np.asarray(kx, float).tolist(),
                ky=np.asarray(ky, float).tolist(),
                energy=np.asarray(en, float).tolist()), sc


def main():
    os.makedirs(ATOMS, exist_ok=True)
    table, t0 = [], time.time()
    for rec in G.RECORDS:
        for s in G.EST_SEEDS:
            p = os.path.join(ATOMS, f'{rec}__s{s}.json')
            if os.path.exists(p):
                d = json.load(open(p))
                atoms, sc = d['atoms'], None
            else:
                tt = time.time()
                atoms, sc = raw_atoms(rec, s)
                json.dump(dict(atoms=atoms, record=rec, seed=int(s),
                               wall_s=time.time() - tt), open(p, 'w'))
            row = dict(record=rec, seed=int(s), n_raw=len(atoms['kx']),
                       atoms_md5=_md5(atoms),
                       K={a: G.pooled_K(atoms, a) for a in G.LADDER})
            row['K_uncapped_at_shipped_knobs'] = row['K']['K-128']
            table.append(row)
            print(f"  {rec:7s} s{s}  raw={row['n_raw']:4d}  "
                  f"K(base)={row['K'][G.BASE]:3d}  K(max)={row['K'][G.PRIMARY_ARM]:3d}",
                  flush=True)
    # ── 헤드룸 요약 ─────────────────────────────────────────────────────────
    import statistics as st
    summ = {}
    for a in G.LADDER:
        v = [r['K'][a] for r in table]
        summ[a] = dict(median=st.median(v), min=min(v), max=max(v))
    base = [r['K'][G.BASE] for r in table]
    e_th, ddk, max_n = G.shipped_knobs()
    out = dict(
        note='M4 Phase A. GPU 0회 · 생성 0회 · 추정 0회 (확산 제안만).',
        cuda_available=bool(torch.cuda.is_available()),
        n_cells=len(table), records=G.RECORDS, seeds=list(G.EST_SEEDS),
        shipped_knobs=dict(energy_thresh=e_th, dedup_dk=ddk, max_n=max_n),
        K_by_arm=summ,
        cap_binds=sum(1 for x in base if x >= max_n),
        headroom_above_cap=sum(1 for r in table
                               if r['K']['K-128'] > max_n),
        primary_strictly_larger=sum(1 for r in table
                                    if r['K'][G.PRIMARY_ARM] > r['K'][G.BASE]),
        wall_s=time.time() - t0, table=table)
    json.dump(out, open(OUT, 'w'), indent=1, sort_keys=True)
    # ── Phase A 동결 (Phase B 결과 행이 하나라도 생기기 **전에** 쓴다) ────────
    import glob
    lines = ['# M4 Phase A 동결 (2026-08-28). Phase B 결과 0행 시점에 기록.',
             '# ⚠경로는 저장소 상대. 이 목록은 PREREG §7 이 요구하는 "원자 풀 동결" 이다.', '']
    for f in sorted(glob.glob(os.path.join(ATOMS, '*.json'))) + [OUT]:
        h = hashlib.md5(open(f, 'rb').read()).hexdigest()
        lines.append(f'{h}  {os.path.relpath(f, ROOT)}')
    open(FROZEN_A, 'w').write('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'table'},
                     indent=1, ensure_ascii=False))
    print('->', OUT, '\n->', FROZEN_A)


if __name__ == '__main__':
    main()
