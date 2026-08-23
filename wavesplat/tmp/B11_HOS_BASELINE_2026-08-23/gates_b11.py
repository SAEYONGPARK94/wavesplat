"""B-11 단일출처 — ★**헤드라인의 두 단서를 동시에 친다** (R2 변호인의 "이것 하나").

## 왜
논문 제1장 기여 ①의 헤드라인(**혼잡 −38.6%, 62/64, vs `SPEC25`**)에는 **의무 병기 3개**가 붙는다:
① 합성·**선형** GT ② 에피소드가 **체크포인트 학습 분포와 겹친다**(신선하지 않다) ③ 최강 기준선.
⟹ ★**①과 ②를 동시에 없앤 표본이 이미 저장소에 있다**: `HOSGATE`(08-13)의 **HOS M=3 참값 ·
시드 1581–1588** 에피소드. 그런데 `HOSGATE` 는 기준선으로 **`LETKF_PLUS`** 를 썼지 **`SPEC25`**
가 아니었고, `SHIP` 팔은 **2026-08-22 배포 이전 구성**이었다.
⟹ **같은 에피소드에 ⓐ현재 배포 구성 `SHIP` 과 ⓑ논문의 기준선 `SPEC25` 를 새로 돌린다.**

## 표본 (새 생성 0회 — 기존 자산 재사용)
`tmp/HOSGATE_2026-08-13/data_hos/` 의 **혼잡 16 에피소드**(BF4/BF5 × 15/30comp × 시드 1581–1584).
⚠**n=16 이다**(64 가 아니다) — 단측 부호검정 문턱은 아래에 그 n 으로 계산해 적었다.

## 팔
- `SHIP-NOW` : **현재 배포 기본값**(추정기 시드 median-of-3: 42·43·44)
- `SPEC25`   : `S16` 의 정의 그대로(`background='jonswap'`, `loc=25`, `n_ensemble=600`,
               필터 시드 median-of-3: 42·43·44) — `tmp/S16_BASELINE_2026-08-14/letkf_spec.py` 를 **그대로 import**
- `ZERO`     : 각 에피소드의 `zero_nRMSE`(무기술 기준, 기준선 행에서 읽는다)

## 판정 (결과 전 확정) — 짝 부호검정, n=16 (에피소드별 median-of-3)
**PRIMARY** = `SHIP-NOW` < `SPEC25` 인 에피소드 수:
- **≥13/16**(단측 p=0.0106) ⟹ ★★`HEADLINE-SURVIVES-NONLINEAR-FRESH`
  — ***헤드라인의 단서 ①②가 동시에 제거된 표본에서도 부호가 산다.***
- **11~12** ⟹ ⚠`PARTIAL` · **≤10** ⟹ ⛔`HEADLINE-WEAKENS`
⚠**크기는 별도로 읽는다**: 상대 중앙을 보고하되 ⛔**`−38.6%` 와 직접 비교하지 마라**
(다른 표본·다른 GT다). **크기 비교는 "같은 방향인가" 까지만.**

**SECONDARY(음성 대조, 필수)** = `SPEC25` 가 `ZERO` 를 이기는가. **≤10/16 이면 기준선이
이 표본에서 무기술**이고 ⟹ ⛔**PRIMARY 는 무효**(`B-9b` 에서 물린 부류: *비교자가 무기술*).

★**게이트**: 두 팔이 **같은 16 에피소드** · `SHIP` 이 현재 배포값 되읽기 · 시드 3개 전건 ·
`SPEC25` 되읽기(loc 25 · ens 600 · jonswap). 가족단위 α = 0.05/16 = 0.003125.
"""
import os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
HOS = os.path.join(ROOT, 'tmp', 'HOSGATE_2026-08-13', 'data_hos')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
CROWDED_SEEDS = (1581, 1582, 1583, 1584)
EST_SEEDS = (42, 43, 44)
FILTER_SEEDS = (42, 43, 44)
SPEC = dict(background='jonswap', loc=25.0, n_ensemble=600)
NOMINAL, PARTIAL_LO = 13, 11
NEG_MIN = 11              # SPEC25 가 ZERO 를 이겨야 하는 최소 (n=16, 문턱)
FAMILY_ALPHA = 0.05 / 16


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def med_by_episode(rows, arm, field='avg_nRMSE'):
    d = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm or r.get(field) is None:
            continue
        d.setdefault(r['episode'], []).append(r[field])
    return {e: st.median(v) for e, v in d.items()}


def paired(a, b):
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    return {'n': len(ks), 'better': w,
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, len(ks)) if ks else None}


def gates(rows, ship, spec, zero):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}'),
           ('G-1 SHIP 16 에피소드', len(ship) == 16, f'{len(ship)}'),
           ('G-1b SPEC25 16 에피소드', len(spec) == 16, f'{len(spec)}'),
           ('★G-2 두 팔 에피소드 집합 완전 일치', set(ship) == set(spec),
            f'교집합 {len(set(ship) & set(spec))}'),
           ('G-3 ZERO 16 에피소드', len(zero) == 16, f'{len(zero)}')]
    sh = [r for r in ok if r['arm'] == 'SHIP-NOW']
    sp = [r for r in ok if r['arm'] == 'SPEC25']
    out.append(('G-4 SHIP 시드 3개 전건', len(sh) == 48, f'{len(sh)}/48'))
    out.append(('G-4b SPEC25 시드 3개 전건', len(sp) == 48, f'{len(sp)}/48'))
    out.append(('★G-5 SPEC25 되읽기(loc 25 · ens 600 · jonswap)',
                bool(sp) and all(abs(r.get('loc_radius', -1) - 25.0) < 1e-9
                                 and r.get('n_ensemble') == 600
                                 and r.get('background') == 'jonswap' for r in sp), ''))
    out.append(('★G-6 SHIP 이 현재 배포값 되읽기',
                bool(sh) and all(r.get('varpro_steps') == 400
                                 and r.get('learned_init_iters') == 0 for r in sh),
                str(sorted({(r.get('varpro_steps'), r.get('learned_init_iters')) for r in sh}))[:50]))
    return out


def label(prim, neg):
    if neg['better'] < NEG_MIN:
        return ('BASELINE-NO-SKILL-HERE',
                f'⛔기준선이 이 표본에서 무기술이다({neg["better"]}/{neg["n"]}, 문턱 {NEG_MIN}) '
                '⟹ PRIMARY 는 무효다(비교자가 무기술인 대조는 우열을 말하지 않는다).')
    w, n = prim['better'], prim['n']
    tail = (f" [상대중앙 {prim['rel_median_pct']:.2f}% · p={prim['p_one_sided']:.5f}"
            f" · 음성대조 SPEC25<ZERO {neg['better']}/{neg['n']}]")
    if w >= NOMINAL:
        return ('HEADLINE-SURVIVES-NONLINEAR-FRESH',
                f'★★비선형(HOS M=3) 참값 · 신선 시드 표본에서도 배포 구성이 기준선을 {w}/{n} '
                f'이긴다 ⟹ **헤드라인의 단서 ①②가 동시에 제거된 표본에서 부호가 산다.**' + tail)
    if w >= PARTIAL_LO:
        return ('PARTIAL', f'{w}/{n} — 문턱 {NOMINAL} 미달.' + tail)
    return ('HEADLINE-WEAKENS',
            f'⛔{w}/{n} — 비선형·신선 표본에서 부호가 약해진다. **정직한 음성.**' + tail)
