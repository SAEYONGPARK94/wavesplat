"""B-7 단일출처 — ★**논문의 중심 주장을 올바른 프로토콜에서 처음 잰다**.

## 왜 이 배치가 필요한가 (오늘 발견한 구멍)
`B-2` 는 제안 표현(`PROP`)과 배포 경로(`SHIP`)를 **lead 0 나우캐스트 한 점**에서만 비교했다 —
저장소 전수 확인 결과 **`B-2` 계열은 추정기의 `update()` 를 한 번도 부르지 않는다**
(`cfg.update(...)` 는 dict 메서드다; 2026-08-23 `DEPLOY_IMPACT.md` 정정).
그런데 **논문이 주장하는 것은 순차 추적·전방 예보**다(`sec:pipeline`, 기여 ②c).
⟹ ★***제안 표현이 "추적하면서 예보할 때" 무엇을 하는지는 아직 한 번도 안 쟀다.***

`B-3`~`B-6` 는 **배포 경로만** 추적시켰다. 이 배치가 그 짝을 채운다.

## 프로토콜 — `B-3` 과 **글자 그대로 동일**
과거 40 s 초기화 → lead 1..14 s 마다 **① 먼저 예측·채점 ② 그 다음 그 시각 관측 투입**.
채점 = σ 정규화 RMSE. **기준은 상수가 아니라 `zero_ref.py` 의 레코드·창·lead 별 참 제로필드.**

## 팔 (12 레코드 × 창 {1,3,5} = 36 짝)
- `PROP-TRACK` : `real_domain_gate=False`, `init_method='learned'` (= `B-2` 의 `PROP` 를 **추적시킨다**)
- `SHIP-TRACK` : 배포 기본값 그대로 (실측 게이트 → `bank_mmse`, **재적합 5 s** = 2026-08-23 배포)
- `FROZEN` 은 `../B3_REFIT_2026-08-23/results_b3.jsonl` 에서 재사용(배포 전 거동 대조).

## 판정 (결과 전 확정) — 짝 부호검정, n=36. 가족단위 α = 0.05/11 = 0.00455
**PRIMARY** = `PROP-TRACK` 이 **참 제로필드**를 이기는 창 수, **lead 1–3 s** 구간:
- **≥26/36** ⟹ ★★★`PROP-HAS-REAL-SKILL` — **논문의 중심 주장이 실측에서 처음 지지된다.**
- **≥24/36** ⟹ ⚠`PROP-SKILL-EXPLORATORY`(명목만)
- **19~23** ⟹ `PROP-UNDECIDED`
- **≤18/36** ⟹ ⛔`PROP-NO-REAL-SKILL` — **정직한 음성. `B-2` 의 나우캐스트 음성이 추적에서도 유지된다.**

**SECONDARY-A** = 같은 것을 **lead 4–14 s** 에서.
**SECONDARY-B** = `PROP-TRACK` vs `SHIP-TRACK` 짝(`fc_mean`, lead 2–14 평균).
  ⚠`B-2` 는 나우캐스트에서 **`SHIP` 이 11/12** 였다. 추적에서 뒤집히는지 본다.

★**개입 계수기(가지 단위)**: `PROP` 팔에서 **추적 경사스텝이 실제로 돌았는가**를 센다.
0 이면 판정은 `PROP-NO-REAL-SKILL` 이 아니라 ⛔**`PROP-NO-INTERVENTION`** 이다
(= 자료가 아니라 설계가 답을 막은 것. 오늘 `B-2f` 에서 같은 함정에 빠졌다).
★**음성 대조**: `SHIP-TRACK` 의 lead 0 이 `FROZEN` 의 lead 0 과 **비트 동일**이어야 한다.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
for p in (HERE, B2, B3):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS, sign_test_one_sided, key, paired  # noqa

ARMS = {'PROP-TRACK': {'real_domain_gate': False, 'init_method': 'learned'},
        'SHIP-TRACK': {}}
ARM_ORDER = ['PROP-TRACK', 'SHIP-TRACK']
REUSED = ['FROZEN']
NOMINAL, CONFIRM = 24, 26
FAMILY_ALPHA = 0.05 / 11
LEAD_NEAR, LEAD_FAR = (1.0, 3.0), (4.0, 14.0)


def beats_zero(rows, arm, zero, lo, hi):
    import statistics as st
    w = n = 0
    for r in rows:
        if r['cell'] != arm or 'error' in r: continue
        pl = r.get('per_lead') or {}
        z = zero[f"{r['record']}|{r['win']}"]
        a = [v for k, v in pl.items() if lo <= float(k) <= hi]
        b = [z[k] for k in pl if lo <= float(k) <= hi]
        if not a: continue
        n += 1; w += (st.mean(a) < st.mean(b))
    return {'arm': arm, 'n': n, 'better': w,
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * N_PAIRS
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    prop = [r for r in ok if r['cell'] == 'PROP-TRACK']
    ship = [r for r in ok if r['cell'] == 'SHIP-TRACK']
    out.append(('G-1 PROP 이 Gaussian 성분을 만들었나',
                all((r.get('n_atoms') or 0) > 0 for r in prop),
                str(sorted({r.get('n_atoms') for r in prop}))[:60]))
    out.append(('G-2 SHIP 은 성분 0개(bank 경로)',
                all((r.get('n_atoms') or 0) == 0 for r in ship), ''))
    out.append(('★G-3 PROP 추적 개입 >0 (가지 계수기)',
                all((r.get('track_steps') or 0) > 0 for r in prop),
                str(sorted({r.get('track_steps') for r in prop}))[:60]))
    out.append(('G-4 SHIP 재적합 발화 (배포 5 s)',
                all((r.get('refit_calls') or 0) > 0 for r in ship),
                str(sorted({r.get('refit_calls') for r in ship}))[:40]))
    out.append(('G-5 36 창 전건', len({key(r) for r in ok}) == N_PAIRS, ''))
    return out


def label(prim, near_n, sec_a, sec_b, track_steps_ok):
    if not track_steps_ok:
        return ('PROP-NO-INTERVENTION',
                '⛔`PROP` 팔에서 추적 경사스텝이 한 번도 돌지 않았다 ⟹ 이 배치는 제안 표현의 '
                '추적을 재지 못했다. 자료가 아니라 설계가 답을 막았다.')
    w = prim['better']
    tail = (f" (lead 4–14: {sec_a['better']}/{sec_a['n']} p={sec_a['p_one_sided']:.5f}"
            f" · PROP vs SHIP: {sec_b['better']}/{sec_b['n']} p={sec_b['p_one_sided']:.5f})")
    if w >= CONFIRM:
        return ('PROP-HAS-REAL-SKILL',
                f'★★★제안 표현이 추적 하에서 실측 제로필드를 lead 1–3 s 에서 {w}/{near_n} 이긴다'
                f'(p={prim["p_one_sided"]:.6f}) — **논문의 중심 주장이 실측에서 처음 지지된다.**' + tail)
    if w >= NOMINAL:
        return ('PROP-SKILL-EXPLORATORY', f'{w}/{near_n} 명목 통과, 가족단위 미달.' + tail)
    if w <= near_n - NOMINAL:
        return ('PROP-NO-REAL-SKILL',
                f'⛔제안 표현이 추적 하에서도 실측 기술이 없다({w}/{near_n}) — **정직한 음성**. '
                f'`B-2` 의 나우캐스트 음성이 추적에서도 유지된다.' + tail)
    return ('PROP-UNDECIDED', f'부호로 갈리지 않는다({w}/{near_n}).' + tail)
