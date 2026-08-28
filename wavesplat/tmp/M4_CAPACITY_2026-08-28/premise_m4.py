"""M4 전제 — **런 0회**. 동결된 파일만 읽어 `premise_m4.json` 을 만든다.

⛔추정기를 돌리지 않는다. ⛔GPU 를 건드리지 않는다. ⛔에피소드를 만들지 않는다.
읽는 것은 넷뿐이다:
  ① `tmp/wp_integration_2026-07-25/atoms_cache.jsonl`
     `tmp/unified_2026-07-26/atoms_cache_extra.jsonl`  — 2026-07 finemesh 100 ep × 5 확산시드
  ② `tmp/B22_PRIOR_ABLATION_2026-08-27/atoms/*__learned__*.json` — ★현 헤드라인 코퍼스
     (`B-12`/`B-16`/`B-22` 의 HOS 혼잡 48 ep × 추정기시드 3)
  ③ `tmp/B2_REAL_2026-08-23/results.jsonl` — 실측 12 레코드의 `PROP` 팔 성분 수
  ④ `wavesplat/optimizer/bank_mmse.py` 의 상수(서명 기본값) — 배포 **실측** 팔의 용량

★이 파일의 목적은 하나다: **PREREG 의 모든 수를 손복사 대신 자료에서 낸다.**
"""
import glob
import inspect
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'tmp', 'B22_PRIOR_ABLATION_2026-08-27')):
    if p not in sys.path:
        sys.path.insert(0, p)

from wavesplat.optimizer.learned_init import pool_and_select   # noqa: E402
from wavesplat.optimizer import bank_mmse as BM                # noqa: E402

B22_ATOMS = os.path.join(ROOT, 'tmp/B22_PRIOR_ABLATION_2026-08-27/atoms')
OLD_CACHES = [os.path.join(ROOT, 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'),
              os.path.join(ROOT, 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl')]
B2_ROWS = os.path.join(ROOT, 'tmp/B2_REAL_2026-08-23/results.jsonl')


def _shipped_knobs():
    """배포 선택 노브 = `pool_and_select` 의 **서명 기본값** (⛔손복사 아님)."""
    s = inspect.signature(pool_and_select).parameters
    return (float(s['energy_thresh'].default), float(s['dedup_dk'].default),
            int(s['max_n'].default))


def _pool(kx, ky, en, e_th, ddk, max_n):
    return len(pool_and_select(np.asarray(kx, float), np.asarray(ky, float),
                               np.asarray(en, float), energy_thresh=e_th,
                               dedup_dk=ddk, max_n=max_n)[0])


def headroom_b22():
    """★현 헤드라인 코퍼스에서 **상한을 풀면 원자가 몇 개까지 늘어나는가.**"""
    fs = sorted(glob.glob(os.path.join(B22_ATOMS, '*__learned__*.json')))
    e_th, ddk, max_n = _shipped_knobs()
    raw = []
    for f in fs:
        a = json.load(open(f))['atoms']
        raw.append((a['kx'], a['ky'], a['energy']))
    grid = {}
    for et in (e_th, 0.0):
        for dd in (ddk, ddk / 2.0, 0.0):
            n = [_pool(k, y, e, et, dd, 10 ** 9) for k, y, e in raw]
            grid[f'e{et}_d{dd}'] = dict(median=st.median(n), min=min(n), max=max(n),
                                        n_gt_shipcap=sum(1 for x in n if x > max_n))
    ship = [_pool(k, y, e, e_th, ddk, max_n) for k, y, e in raw]
    unc = [_pool(k, y, e, e_th, ddk, 10 ** 9) for k, y, e in raw]
    return dict(n_files=len(fs), raw_len=sorted({len(k) for k, _, _ in raw}),
                shipped_knobs=dict(energy_thresh=e_th, dedup_dk=ddk, max_n=max_n),
                shipped_K=dict(median=st.median(ship), min=min(ship), max=max(ship)),
                uncapped_K=dict(median=st.median(unc), min=min(unc), max=max(unc),
                                n_gt_cap=sum(1 for x in unc if x > max_n)),
                grid=grid)


def headroom_2026_07():
    """2026-07 finemesh 캐시 — 성분수 층별로 상한이 무는 비율(레짐 대조)."""
    e_th, ddk, max_n = _shipped_knobs()
    out, agree = {}, [0, 0]
    for f in OLD_CACHES:
        if not os.path.exists(f):
            continue
        for line in open(f):
            d = json.loads(line)
            if d.get('err'):
                continue
            nc = int(d['episode'].split('_')[2].replace('comp', ''))
            k = _pool(d['kx'], d['ky'], d['energy'], e_th, ddk, max_n)
            u = _pool(d['kx'], d['ky'], d['energy'], e_th, ddk, 10 ** 9)
            out.setdefault(nc, []).append((k, u, d['n_selected']))
            agree[1] += 1
            agree[0] += int(k == d['n_selected'])
    res = {}
    for nc, v in sorted(out.items()):
        res[str(nc)] = dict(n=len(v), K_median=st.median([a for a, _, _ in v]),
                            uncapped_median=st.median([b for _, b, _ in v]),
                            uncapped_max=max(b for _, b, _ in v),
                            at_cap=sum(1 for a, _, _ in v if a >= max_n))
    # ★양성대조(런0): 저장된 `n_selected` 를 캐시 원자에서 **재도출**해 일치를 확인한다.
    res['_recompute_matches_stored'] = f'{agree[0]}/{agree[1]}'
    return res


def real_prop_atoms():
    """`B-2` 의 `PROP` 팔(제안 표현 강제)이 실측에서 실제로 세운 성분 수."""
    if not os.path.exists(B2_ROWS):
        return {'error': 'B-2 results.jsonl 없음'}
    rows = [json.loads(l) for l in open(B2_ROWS) if l.strip()]
    prop = {r['record']: r for r in rows if r.get('cell') == 'PROP'}
    ship = {r['record']: r for r in rows if r.get('cell') == 'SHIP'}
    _, _, max_n = _shipped_knobs()
    n = {k: v['n_atoms'] for k, v in prop.items()}
    return dict(n_records=len(n), n_atoms=n,
                at_cap=sum(1 for v in n.values() if v >= max_n),
                npx={k: v['npx'] for k, v in prop.items()},
                n_obs=sorted({v['n_obs'] for v in prop.values()}),
                ship_n_atoms=sorted({v['n_atoms'] for v in ship.values()}),
                prop_beats_ship=sum(1 for k in prop
                                    if prop[k]['nowcast'] < ship[k]['nowcast']))


def bank_capacity():
    """배포 **실측** 팔(`bank_mmse`)의 용량 — 척도 무관 상수임을 자료로 보인다."""
    sig = inspect.signature(BM.fit_bank_mmse).parameters
    band = sig['band'].default
    out = {}
    for kp in (0.05, 0.1, 0.3, 0.7):
        dk = kp / 10.0                       # bank_mmse.py: dk_c is None -> kp/10
        ax = np.arange(-band[1] * kp, band[1] * kp + dk / 2, dk)
        KX, KY = np.meshgrid(ax, ax)
        kk = np.hypot(KX, KY)
        sel = (kk > band[0] * kp) & (kk < band[1] * kp)
        out[str(kp)] = int(sel.sum())
    return dict(band=list(band), dk_c_rule='kp/10',
                B_by_kp=out, B_is_scale_free=len(set(out.values())) == 1,
                n_coefficients=2 * list(out.values())[0])


def main():
    p = dict(
        note='M4 전제. 런 0회 — 동결 파일만 읽었다.',
        shipped_selection_knobs=dict(zip(('energy_thresh', 'dedup_dk', 'max_n'),
                                         _shipped_knobs())),
        P1_headline_corpus_headroom=headroom_b22(),
        P2_2026_07_by_ncomp=headroom_2026_07(),
        P3_real_prop_atoms=real_prop_atoms(),
        P4_deployed_real_bank_capacity=bank_capacity(),
    )
    out = os.path.join(HERE, 'premise_m4.json')
    json.dump(p, open(out, 'w'), indent=1, sort_keys=True)
    print(json.dumps(p, indent=1, sort_keys=True)[:4000])
    print('->', out)


if __name__ == '__main__':
    main()
