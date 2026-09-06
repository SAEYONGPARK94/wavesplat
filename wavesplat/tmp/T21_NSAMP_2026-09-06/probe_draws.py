#!/usr/bin/env python3
"""`T21-NSAMP` **동결 전 프로브** — 집합값 제안의 draw 다양성을 잰다.

⛔⛔**이것은 판정이 아니다.** 사전등록 **미작성**, 게이트·분석기·러너 **0개**.
이 스크립트는 `T17` 이 동결 전에 했고 `T19` 초안이 하지 못해 스스로 *"가장 큰 결함"* 이라
적은 그 **결과-전 프로브**다. 여기서 나오는 수는 **설계를 정하는 데만** 쓰고
어떤 주장도 지탱하지 않는다.

## 왜 이 프로브인가

살아남은 신규성 주장은 하나다 — ***집합값(set-valued) 사전***: 제안망이 관측창에 조건화된
원자 집합을 **여러 벌**(`learned_nsamp=4`) 뽑고 적합기가 고른다. 최근접 선행
(Barthelme 2021 · Schieler 2024 · AAETR)은 전부 **결정적 회귀 = 한 벌**이다.
⟹ 우리를 그들과 가르는 성질이 **이것 하나**다.

⛔**그런데 저장소 전역 grep 결과 `learned_nsamp` 에 대한 개입 배치가 0건이다.**
`M4/PREREG.md:246` 은 축자로 *"제안 자체의 용량(`learned_nsamp=4`)은 고정"* 이라 적고 지나갔고
`W5/AUDIT_NEUTRAL.md:76` 은 *"단 하나의 미고정 의존"* 이라 지적만 했다.
⟹ ***"네 벌이 한 벌보다 나은가" 를 이 프로젝트는 한 번도 물어본 적이 없다.***

## ★설계 전에 반드시 알아야 할 교락 하나

`learned_init.py:412-419` → `propose_atoms`(원자 `nsamp × M`) → `pool_and_select`
(상대 에너지 문턱 `0.03` · dedup `dk<0.03` · 상한 `pool_cap=30`).

동결 `T17` 행 실측: `n_proposal_raw` = **128 전건**(=4×32), `n_proposal_sel` 중앙 **30**
= **상한이 문다**. ⟹ `nsamp` 를 1 로 내리면 풀이 32 가 되어 상한이 **거의 물지 않는다**.
★***그러므로 `nsamp` 단독 처치는 「뽑는 벌 수」와 「선택 압력」을 동시에 움직인다.***
단일요인 설계를 하려면 이 프로브가 먼저 분해해야 한다.

## 무엇을 재는가 (추정기 적합 0회 — 제안 단계까지만)

1. `nsamp ∈ {1,2,4,8}` 별로 `n_raw → 문턱 통과 → dedup 통과 → 상한 적용` 의 **4단 감쇠**
2. `nsamp=4` 에서 선택된 원자의 **draw 출처 분포** (몇 벌이 기여하는가)
3. **draw 간 일치도** — draw 0 의 원자 중 draw b 에 `dedup_dk` 이내 짝이 있는 비율
   ⟹ 이 값이 1 에 가까우면 ***집합값이 장식***이라는 뜻이다
4. 장면당 벽시계 (설계 비용 산정)

★그리고 그것을 **개구가 학습 지지집합 안인 장면과 밖인 장면**에서 각각 잰다
(`tmp/_audit/APERTURE_GAP.md`: 실측 11/12 가 학습 p5 아래).
가설: 분포 안에서는 네 벌이 서로 같고(제안이 확신), 밖에서는 갈린다.
⛔**가설이지 예측 등록이 아니다** — 이 문서는 동결물이 아니다.

## 사용
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
        python3 tmp/T21_NSAMP_2026-09-06/probe_draws.py --scenes BS13c --nsamp 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp/B2_REAL_2026-08-23')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

# `T17` 과 비트 동일한 장면 규약 (⛔손복사 금지 — 게이트 모듈에서 임포트한다)
sys.path.insert(0, os.path.join(ROOT, 'tmp/T17_REALSCALE_2026-09-05'))

M_ATOMS = 32          # 체크포인트의 슬롯 수 — 아래에서 되읽어 assert 한다


def _select_with_provenance(kx, ky, en, energy_thresh, dedup_dk, max_n):
    """`learned_init.pool_and_select` 와 **같은 논리**로 고르되 원본 인덱스를 함께 낸다.

    ⛔재구현이므로 아래에서 원본 함수 출력과 **비트 동일**을 assert 한다.
    """
    kx = np.asarray(kx, float); ky = np.asarray(ky, float); en = np.asarray(en, float)
    if en.size == 0 or en.max() <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([], int), 0, 0
    keep = en > energy_thresh * en.max()
    idx_keep = np.flatnonzero(keep)
    n_after_thresh = int(idx_keep.size)
    kxk, kyk, enk = kx[keep], ky[keep], en[keep]
    order = np.argsort(-enk)
    sk, sy, se, si = [], [], [], []
    n_dup = 0
    for i in order:
        dup = any(abs(kxk[i] - jx) < dedup_dk and abs(kyk[i] - jy) < dedup_dk
                  for jx, jy in zip(sk, sy))
        if not dup:
            sk.append(kxk[i]); sy.append(kyk[i]); se.append(enk[i])
            si.append(int(idx_keep[i]))
        else:
            n_dup += 1
        if len(sk) >= max_n:
            break
    return (np.array(sk), np.array(sy), np.array(se), np.array(si, int),
            n_after_thresh, n_dup)


def _draw_agreement(kx, ky, nsamp, m, dedup_dk):
    """draw 0 의 원자 중 draw b 안에 `dedup_dk` 이내 짝이 있는 비율 (b=1..nsamp-1)."""
    out = []
    a0 = np.column_stack([kx[:m], ky[:m]])
    for b in range(1, nsamp):
        ab = np.column_stack([kx[b * m:(b + 1) * m], ky[b * m:(b + 1) * m]])
        hit = 0
        for p in a0:
            if np.any((np.abs(ab[:, 0] - p[0]) < dedup_dk)
                      & (np.abs(ab[:, 1] - p[1]) < dedup_dk)):
                hit += 1
        out.append(hit / len(a0))
    return out


def real_scene(rec, win, G, scene_station, R2):
    sc = scene_station(rec, win=win, W=G.W_OBS, m_scan=G.M_SCAN, m_cap=G.M_CAP,
                       noise=G.NOISE, cap_mode=G.CAP_MODE)
    P = np.asarray(sc['P'], float)
    t_shift = float(P[:, 2].min())
    return R2.frames_from(sc, t_shift), float(sc['depth']), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=['BS13c'],
                    help='실측 레코드 이름 (T17 RECORDS)')
    ap.add_argument('--win', type=int, default=0)
    ap.add_argument('--nsamp', type=int, nargs='+', default=[4])
    ap.add_argument('--dseed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'probe_draws.jsonl'))
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    import gates_t17 as G
    from scene_station import scene_station
    import run_b2 as R2
    os.chdir(ROOT)                       # run_b2 임포트가 chdir 한다
    from wavesplat.optimizer import learned_init as LI

    e_th, ddk = 0.03, 0.03               # `learned_init_params` 기본값
    pool_cap = int(LI.POOL_CAP_DEFAULT)
    print(f'pool_cap(POOL_CAP_DEFAULT) = {pool_cap} · energy_thresh={e_th} · dedup_dk={ddk}')
    print(f'ckpt = {LI.DEFAULT_CKPT}')

    rows = []
    for rec in a.scenes:
        frames, depth, sc = real_scene(rec, a.win, G, scene_station, R2)
        P = np.asarray(sc['P'], float)
        L_obs = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])))
        for ns in a.nsamp:
            t0 = time.time()
            rkx, rky, ren = LI.propose_atoms(frames, depth, nsamp=ns,
                                             diffusion_seed=a.dseed)
            wall = time.time() - t0
            n_raw = int(len(rkx))
            m = n_raw // ns
            assert n_raw == m * ns, f'원자 수가 nsamp 로 나뉘지 않는다: {n_raw}/{ns}'

            sk, sy, se, si, n_th, n_dup = _select_with_provenance(
                rkx, rky, ren, e_th, ddk, pool_cap)
            # ★재구현 검증 — 원본과 비트 동일이어야 한다
            okx, oky, oen = LI.pool_and_select(rkx, rky, ren, energy_thresh=e_th,
                                               dedup_dk=ddk, max_n=pool_cap)
            assert np.array_equal(sk, okx) and np.array_equal(sy, oky) \
                and np.array_equal(se, oen), '★재구현이 pool_and_select 와 다르다'

            draws = (si // m) if si.size else np.array([], int)
            per_draw = [int((draws == b).sum()) for b in range(ns)]
            agree = _draw_agreement(rkx, rky, ns, m, ddk) if ns > 1 else []
            row = dict(record=rec, win=a.win, L_obs=L_obs, depth=depth,
                       nsamp=ns, M=m, n_raw=n_raw, n_after_thresh=n_th,
                       n_dup_dropped=n_dup, n_sel=int(si.size),
                       cap_bound=bool(si.size >= pool_cap),
                       per_draw=per_draw, n_draws_used=int(sum(1 for v in per_draw if v)),
                       draw_agreement=agree, wall_s=round(wall, 2))
            rows.append(row)
            print(f"  {rec:7s} L={L_obs:7.1f}m nsamp={ns} : raw {n_raw:3d} "
                  f"→thresh {n_th:3d} →dedup -{n_dup:3d} →sel {si.size:2d}"
                  f"{' (CAP)' if row['cap_bound'] else '      '}"
                  f"  draws {per_draw}  agree {[round(v,3) for v in agree]}"
                  f"  {wall:.1f}s", flush=True)

    with open(a.out, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n{len(rows)} 행 → {a.out}')
    print('⛔이것은 동결 전 프로브다 — 어떤 주장도 지탱하지 않는다.')


if __name__ == '__main__':
    main()
