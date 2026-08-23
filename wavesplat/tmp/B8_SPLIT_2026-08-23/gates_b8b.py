"""B-8b 단일출처 — **`B-8` 의 `SPLIT-UNDECIDED` 를 표본을 늘려 닫는다**.

## 왜
`B-8` 은 `V2K`(학습 0) **14/36** 을 냈다. 사전등록 문턱은
`CHECKPOINT-LIMITED ≥24` (⟹ **반증**) · `REPRESENTATION-LIMITED ≤12` (⟹ **두 칸 미달**).
⟹ 라벨이 `SPLIT-UNDECIDED` 로 남았고, **그 두 칸이 논문의 핵심 문장을 가른다**:
*"표현 자체가 실측의 한계다"* 를 쓸 수 있는가 없는가.

★**닫는 방법은 새 처치가 아니라 표본이다.** `B-3c` 가 이미 연 **창 {0,2,4}** 를 쓴다
(레코드는 같지만 **창은 오늘 `B-3c` 외에 아무도 안 봤다**). 창 6개 ⟹ **n=72**.

## 팔 (신규 72런 = 2팔 × 12 레코드 × 창 {0,2,4})
`V2K-TRACK` · `PROP-TRACK` — 설정·프로토콜·채점 **전부 `B-7`/`B-8` 과 글자 그대로 동일**.
창 {1,3,5} 는 `../B7_.../results_b7.jsonl` 과 `results_b8.jsonl` 에서 재사용 ⟹ **팔당 n=72**.

## 판정 (결과 전 확정) — 참 제로필드 대비, **lead 1–3 s**, **창 수준 n=72**
단측 부호검정 문턱(정확 계산): p<0.05 ⟹ **44/72** · 가족단위 α=0.05/13=0.00385 ⟹ **48/72**.
- `V2K` **≥48/72** ⟹ ★`CHECKPOINT-LIMITED` (확증)
- **44~47** ⟹ ⚠`CHECKPOINT-LIMITED (EXPLORATORY)`
- **29~43** ⟹ `SPLIT-STILL-UNDECIDED`
- **≤28/72** ⟹ ⛔★**`REPRESENTATION-LIMITED`** — *"학습을 빼도, 표본을 두 배로 늘려도
  기술이 없다"* ⟹ **논문이 그 문장을 쓸 수 있다.**

★**보수적 재확인(결과 전 확정)**: **레코드 수준**(창 6개 중앙, n=12)도 함께 낸다.
문턱: p<0.05 ⟹ 10/12 · 가족단위 ⟹ 11/12. **두 수준이 어긋나면 보수적(레코드) 쪽을 채택한다.**

★**개입 계수기**: 두 팔 모두 추적 경사스텝 >0 · 성분 수 >0 이어야 한다(아니면 `NO-INTERVENTION`).
⚠**`PROP` 도 같은 창에서 새로 돌린다** — 한 팔만 확장하면 두 팔이 다른 자료를 보게 된다.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
B7 = os.path.abspath(os.path.join(HERE, '..', 'B7_PROP_TRACK_2026-08-23'))
for p in (HERE, B2, B3, B7):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, SEED, NOISE, sign_test_one_sided, key, paired   # noqa
from gates_b7 import beats_zero, LEAD_NEAR, LEAD_FAR                          # noqa

WINDOWS = [0, 2, 4]                 # ★신규로 도는 창
ALL_WINDOWS = [0, 1, 2, 3, 4, 5]    # 통합 판정에 쓰는 창
ARMS = {'V2K-TRACK': {'real_domain_gate': False, 'init_method': 'v2_konly'},
        'PROP-TRACK': {'real_domain_gate': False, 'init_method': 'learned'}}
ARM_ORDER = ['V2K-TRACK', 'PROP-TRACK']
N_NEW = len(RECORDS) * len(WINDOWS)
N_POOL = len(RECORDS) * len(ALL_WINDOWS)
NOMINAL, CONFIRM, NOSKILL = 44, 48, 28          # /72, 결과 전 정확 계산
REC_NOMINAL, REC_CONFIRM, REC_NOSKILL = 10, 11, 2   # /12
FAMILY_ALPHA = 0.05 / 13


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['win'] in WINDOWS]
    out = [('G-0 신규 런 수', len(new) == len(ARMS) * N_NEW, f'{len(new)}/{len(ARMS)*N_NEW}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    for a in ARM_ORDER:
        n = sum(1 for r in ok if r['cell'] == a)
        out.append((f'G-1 통합 {a} {N_POOL}행', n == N_POOL, f'{n}/{N_POOL}'))
    out.append(('★G-2 두 팔 추적 개입 >0',
                all((r.get('track_steps') or 0) > 0 for r in ok),
                str(sorted({r.get('track_steps') for r in ok}))[:40]))
    out.append(('G-3 두 팔 성분 >0 (음성대조)',
                all((r.get('n_atoms') or 0) > 0 for r in ok), ''))
    out.append(('G-4 재적합 0회 (bank 아님)',
                all((r.get('refit_calls') or 0) == 0 for r in ok), ''))
    out.append(('G-5 창 6개 전건', {r['win'] for r in ok} == set(ALL_WINDOWS), ''))
    return out


def label(v2k_win, v2k_rec, prop_win, ok_intervention):
    if not ok_intervention:
        return ('V2K-NO-INTERVENTION', '⛔개입 계수기가 0 ⟹ 가설을 가르지 못했다.')
    w, n = v2k_win['better'], v2k_win['n']
    rw, rn = v2k_rec['better'], v2k_rec['n']
    tail = (f" [창 {w}/{n} p={v2k_win['p_one_sided']:.6f} · 레코드 {rw}/{rn}"
            f" p={v2k_rec['p_one_sided']:.6f} · 대조 `PROP` {prop_win['better']}/{prop_win['n']}]")
    # ★두 수준이 어긋나면 보수적(레코드) 쪽을 채택한다 (결과 전 확정)
    win_skill = w >= NOMINAL
    rec_skill = rw >= REC_NOMINAL
    if win_skill and rec_skill:
        lab = 'CHECKPOINT-LIMITED' if (w >= CONFIRM and rw >= REC_CONFIRM) \
            else 'CHECKPOINT-LIMITED (EXPLORATORY)'
        return (lab, f'학습을 빼면 실측 기술이 생긴다.' + tail)
    if w <= NOSKILL and rw <= REC_NOSKILL:
        return ('REPRESENTATION-LIMITED',
                '⛔★**학습을 빼도, 표본을 두 배로 늘려도 실측 기술이 없다** ⟹ '
                '`B-7` 의 음성은 체크포인트가 아니라 **표현/옵티마이저 수준**이다.' + tail)
    if w <= NOSKILL or rw <= REC_NOSKILL:
        return ('REPRESENTATION-LIMITED (LEVEL-SPLIT)',
                '⚠두 수준 중 하나만 무기술 문턱을 넘었다 ⟹ **보수적으로 읽는다**.' + tail)
    return ('SPLIT-STILL-UNDECIDED', '표본을 두 배로 늘려도 부호로 갈리지 않는다.' + tail)
