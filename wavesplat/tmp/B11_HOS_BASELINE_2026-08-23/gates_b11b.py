"""B-11b 단일출처 — ★**헤드라인의 두 단서를 동시에 친다** (R2 변호인의 "이것 하나").

## ★B-11b — **희소 층에서 같은 시험을 한다** (`B-11` 의 짝)
`B-11` 이 **혼잡** 층에서 헤드라인의 단서 둘을 떨어뜨렸다. **희소 층은 반대 방향의 시험**이다:
저장소는 *"희소의 큰 상대 이득은 **표현 아티팩트**"* 라 판정했고(`HOSGATE`: 참값을 비선형화하면
우리 오차만 **0.0195 → 0.0700 = 3.6배**, 기준선은 ±0.5% 무감), 그것이 **금지 목록의 근거**다.
⟹ ★***그 판정을 논문의 기준선(`SPEC25`)에 대해 처음 검정한다.***
⚠**이 배치는 우리에게 불리한 방향을 기대하고 돌린다** — 예상은 *"희소 우위가 줄거나 사라진다"* 다.
그 예상을 **결과 전에** 적는다. **예상대로면 아티팩트 판정이 강화되고, 예상과 다르면
그 판정을 다시 봐야 한다.**

## 왜
논문 제1장 기여 ①의 헤드라인(**혼잡 −38.6%, 62/64, vs `SPEC25`**)에는 **의무 병기 3개**가 붙는다:
① 합성·**선형** GT ② 에피소드가 **체크포인트 학습 분포와 겹친다**(신선하지 않다) ③ 최강 기준선.
⟹ ★**①과 ②를 동시에 없앤 표본이 이미 저장소에 있다**: `HOSGATE`(08-13)의 **HOS M=3 참값 ·
시드 1581–1588** 에피소드. 그런데 `HOSGATE` 는 기준선으로 **`LETKF_PLUS`** 를 썼지 **`SPEC25`**
가 아니었고, `SHIP` 팔은 **2026-08-22 배포 이전 구성**이었다.
⟹ **같은 에피소드에 ⓐ현재 배포 구성 `SHIP` 과 ⓑ논문의 기준선 `SPEC25` 를 새로 돌린다.**

## 표본 (새 생성 0회 — 기존 자산 재사용)
`tmp/HOSGATE_2026-08-13/data_hos/` 의 ★**희소 16 에피소드**(BF4/BF5 × **2/5comp** × 시드 **1585–1588**).
⛔**2026-08-23 정정(RV-7 K)**: 이 줄이 원래 *"혼잡 16 에피소드 … 15/30comp … 1581–1584"* 라
적혀 있었다 — `B-11` 에서 복사하며 층을 안 바꾼 것이다. **코드는 처음부터 옳았다**
(`STRATUM='sparse'`, `ncomp < 15`, `SPARSE_SEEDS`) ⟹ **판정 무손상, 자기기술만 반대였다.**
⚠**n=16 이다**(64 가 아니다) — 단측 부호검정 문턱은 아래에 그 n 으로 계산해 적었다.

## 팔
- `SHIP-NOW` : **현재 배포 기본값**(추정기 시드 median-of-3: 42·43·44)
- `SPEC25`   : `S16` 의 정의 그대로(`background='jonswap'`, `loc=25`, `n_ensemble=600`,
               필터 시드 median-of-3: 42·43·44) — `tmp/S16_BASELINE_2026-08-14/letkf_spec.py` 를 **그대로 import**
- `ZERO`     : 각 에피소드의 `zero_nRMSE`(무기술 기준, 기준선 행에서 읽는다)

## 판정 (결과 전 확정) — 짝 부호검정, n=16 (에피소드별 median-of-3)
**PRIMARY** = `SHIP-NOW` < `SPEC25` 인 **희소** 에피소드 수:
- **≥13/16** ⟹ `SPARSE-ADVANTAGE-SURVIVES` — ⚠**아티팩트 판정을 다시 봐야 한다**
  (그 판정은 *"비선형화하면 우리만 나빠진다"* 였는데 기준선 대비 우위가 그대로면 설명이 안 된다)
- **11~12** ⟹ `SPARSE-PARTIAL`
- **≤10** ⟹ ★`SPARSE-ADVANTAGE-SHRINKS` — **아티팩트 판정이 기준선 대비로도 확인된다**
★**부차(결과 전)**: `SHIP` 의 절대 nRMSE 를 `HOSGATE` 의 **선형** 희소 값(중앙 **0.0195**)·
**비선형** 값(**0.0700**)과 나란히 적는다. **3.6배 열화가 재현되는지**가 이 배치의 두 번째 관전점이다.
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
SPARSE_SEEDS = (1585, 1586, 1587, 1588)
STRATUM = 'sparse'          # ★B-11b 의 유일한 변경점
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
    # ⚠B-11b: 방향이 반대다 — '작을수록 아티팩트 판정이 강화된다'.
    if neg['better'] < NEG_MIN:
        return ('BASELINE-NO-SKILL-HERE',
                f'⛔기준선이 이 표본에서 무기술이다({neg["better"]}/{neg["n"]}, 문턱 {NEG_MIN}) '
                '⟹ PRIMARY 는 무효다(비교자가 무기술인 대조는 우열을 말하지 않는다).')
    w, n = prim['better'], prim['n']
    tail = (f" [상대중앙 {prim['rel_median_pct']:.2f}% · p={prim['p_one_sided']:.5f}"
            f" · 음성대조 SPEC25<ZERO {neg['better']}/{neg['n']}]")
    if w >= NOMINAL:
        return ('SPARSE-ADVANTAGE-SURVIVES',
                f'⚠희소 층 우위가 비선형·신선 표본에서도 유지된다({w}/{n}) ⟹ '
                '**"희소는 표현 아티팩트" 판정을 다시 봐야 한다** — 그 판정은 '
                '"비선형화하면 우리만 나빠진다" 였는데 기준선 대비 우위가 그대로면 설명이 안 된다.' + tail)
    if w >= PARTIAL_LO:
        return ('SPARSE-PARTIAL', f'{w}/{n} — 부분적.' + tail)
    return ('SPARSE-ADVANTAGE-SHRINKS',
            f'★{w}/{n} — **희소 우위가 비선형 참값에서 줄어든다** ⟹ '
            '"희소는 표현 아티팩트" 판정이 논문의 기준선 대비로도 확인된다.' + tail)
