"""B-10 단일출처 — ★**세 번째 GT-free 초기화** (심사 토론 ML 심사의 "이것 하나").

## 왜 표본이 아니라 **다른 초기화**인가
`B-8b` 가 표본을 두 배(n=72)로 늘렸는데도 `REPRESENTATION-LIMITED` 문턱에서 **한 칸** 미달했다.
⛔**세 번째로 표본을 키우면 그것이 p-hacking 이다**(`RESULT_B8B §1` 에 그렇게 적었다).
⟹ ★**증거의 *종류*를 바꾼다**: 같은 표현을 **또 다른 GT-free 초기화**로 돌린다.
- `PROP` = `learned` (합성 학습 체크포인트)
- `V2K`  = `v2_konly` (해석적 빔포밍, **파수만** 적합)
- ★`V2`   = `v2` (해석적 빔포밍, **전 파라미터** 코스투파인) ← **신규**
세 초기화는 **서로 다른 방식으로 같은 Gaussian 성분 표현에 도달**하고 **같은 추적 경사하강**을 쓴다.

## ★판정을 결과 전에 **정확히** 정의한다 (문턱 게임을 하지 않는다)
이 배치는 ⛔**`B-8b` 의 count 문턱을 다시 시도하지 않는다.** 대신 **수렴성**을 본다.
참 제로필드 대비 **lead 1–3 s**, 창 수준 n=36(창 {1,3,5}), 명목 유의 문턱 **24/36**:

- `V2` **≥24/36** ⟹ ★★`THIRD-INIT-WORKS` — **표현은 되고 앞의 두 초기화가 나빴던 것**이다.
  ⟹ 이 경우 `B-7`/`B-8` 의 음성은 **초기화 탓**으로 재해석된다(**가장 큰 반전**).
- `V2` **≤18/36**(= 무기술 쪽 대칭 문턱) ⟹ ★`THREE-INITS-NO-SKILL`
  — ***서로 독립적으로 설계된 세 초기화가 모두 무기술이다.*** count 문턱을 넘지 않아도
  **수렴 증거**로서 `REPRESENTATION-LIMITED` 를 **논증 수준(`ARGUED`)** 에서 지지한다.
  ⛔**그래도 `MEASURED` 로 승격하지 않는다** — 사전등록 문턱은 여전히 미달이다.
- **19~23** ⟹ `THIRD-INIT-UNDECIDED`.

★**부차(결과 전)**: 짝 `V2` vs `V2K` · `V2` vs `PROP`. 그리고 **벽시계**(`v2` 는 전 파라미터라
`v2_konly` 보다 느릴 것으로 예상 — 그 수를 적어 둔다).
★**개입 계수기**: `V2` 팔에서 추적 경사스텝 >0 · 성분 >0 · 재적합 0 이어야 한다.
가족단위 α = 0.05/14 = 0.00357.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
B7 = os.path.abspath(os.path.join(HERE, '..', 'B7_PROP_TRACK_2026-08-23'))
B8 = os.path.abspath(os.path.join(HERE, '..', 'B8_SPLIT_2026-08-23'))
for p in (HERE, B2, B3, B7, B8):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS, sign_test_one_sided, key, paired  # noqa
from gates_b7 import beats_zero, LEAD_NEAR, LEAD_FAR                                           # noqa

ARMS = {'V2-TRACK': {'real_domain_gate': False, 'init_method': 'v2'}}
ARM_ORDER = ['V2-TRACK']
REUSED = ['PROP-TRACK', 'V2K-TRACK']
NOMINAL, NOSKILL = 24, 18
FAMILY_ALPHA = 0.05 / 14


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    out = [('G-0 신규 런 수', len(new) == N_PAIRS, f'{len(new)}/{N_PAIRS}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    for a in REUSED:
        n = sum(1 for r in ok if r['cell'] == a and r['win'] in WINDOWS)
        out.append((f'G-1 재사용 {a} {N_PAIRS}행(창 {WINDOWS})', n == N_PAIRS, f'{n}/{N_PAIRS}'))
    out.append(('★G-2 V2 추적 개입 >0', all((r.get('track_steps') or 0) > 0 for r in new),
                str(sorted({r.get('track_steps') for r in new}))[:40]))
    out.append(('G-3 V2 성분 >0 (음성대조)', all((r.get('n_atoms') or 0) > 0 for r in new),
                str(sorted({r.get('n_atoms') for r in new}))[:60]))
    out.append(('G-4 V2 재적합 0회', all((r.get('refit_calls') or 0) == 0 for r in new), ''))
    # ★G-5 세 초기화가 실제로 서로 다른가 (같으면 이 배치가 공허하다)
    fh = {}
    for r in ok:
        if r['win'] in WINDOWS:
            fh.setdefault(r['cell'], {})[key(r)] = r.get('fc_mean')
    pairs = [('V2-TRACK', 'V2K-TRACK'), ('V2-TRACK', 'PROP-TRACK')]
    same = [(a, b) for a, b in pairs
            if a in fh and b in fh and all(fh[a].get(k) == fh[b].get(k) for k in fh[a])]
    out.append(('★G-5 세 초기화가 서로 다른 결과를 낸다', not same, f'동일 쌍 {same}'))
    out.append(('G-6 36 창 전건', len({key(r) for r in new}) == N_PAIRS, ''))
    return out


def label(v2, v2k, prop, pair_vk, pair_vp, ok_interv):
    if not ok_interv:
        return ('V2-NO-INTERVENTION', '⛔`V2` 팔에서 추적 개입이 0회다 ⟹ 이 배치는 공허하다.')
    w, n = v2['better'], v2['n']
    tail = (f" [`V2` {w}/{n} p={v2['p_one_sided']:.6f} · `V2K` {v2k['better']}/{v2k['n']}"
            f" · `PROP` {prop['better']}/{prop['n']} · 짝 V2<V2K {pair_vk['better']}/{pair_vk['n']}"
            f" · 짝 V2<PROP {pair_vp['better']}/{pair_vp['n']}]")
    if w >= NOMINAL:
        return ('THIRD-INIT-WORKS',
                '★★세 번째 초기화에서 실측 기술이 나타난다 ⟹ `B-7`/`B-8` 의 음성은 '
                '**표현이 아니라 초기화 탓**으로 재해석된다.' + tail)
    if w <= NOSKILL:
        return ('THREE-INITS-NO-SKILL',
                '★**서로 독립적으로 설계된 세 GT-free 초기화가 모두 무기술이다.** '
                '수렴 증거로서 `REPRESENTATION-LIMITED` 를 **논증 수준**에서 지지한다. '
                '⛔사전등록 count 문턱은 여전히 미달이므로 `MEASURED` 로 승격하지 않는다.' + tail)
    return ('THIRD-INIT-UNDECIDED', '부호로 갈리지 않는다.' + tail)
