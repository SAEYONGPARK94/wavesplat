"""B-3c 단일출처 — **`B-3` 의 진짜 out-of-sample 복제**: 한 번도 안 본 창 {0,2,4}.

## 왜
`scene_station` 은 레코드마다 창을 **6개**(`np.linspace(lo,hi,6)`) 놓을 수 있다.
오늘의 모든 실측 배치(`B-2`…`B-2f`)는 **창 3 하나만** 썼고 `B-3`/`B-3b` 는 **{1,3,5}** 를 썼다.
⟹ **{0,2,4} 는 이 저장소가 한 번도 연 적이 없다.** 진짜 신선 표본이다.

⚠**저장소의 아픈 곳**: 08-14 페르소나 토론이 *"out-of-sample 은 시드 축에서만"* 이라 판정했고,
S13 의 "신선 시드 재확인" 서술은 감사에서 깎였다. **이 배치는 그 비판에 직접 답한다** —
자료 축에서 신선하고, 판정 규칙은 `B-3` 것을 **글자 그대로 재사용**한다(새로 고르지 않는다).

## 팔·판정 (결과 전 확정 — `B-3` 규칙을 그대로 상속)
`FROZEN` vs `REFIT2`, 12 레코드 × 창 {0,2,4} = **36 짝**, 잡음 0.02, 시드 42.
- **≥26/36** ⟹ ★`B3-REPLICATES (CONFIRMATORY)`
- **≥24/36** ⟹ ⚠`B3-REPLICATES (EXPLORATORY)`
- **19~23** ⟹ `UNDECIDED`
- **≤18/36** ⟹ ⛔`B3-DOES-NOT-REPLICATE` — **`B-3` 의 창 선택이 인자였다는 뜻**이고
  그러면 `B-3` 인용을 창 {1,3,5} 로 좁혀야 한다.

★**부차(결과 전)**: `B-3` 의 상대중앙 **−23.96%** 와 비교한다. ⚠**차이가 ±10%p 안이면
"같은 크기" 라 부르고, 밖이면 크기 주장을 `B-3` 에 한정한다.**
★**음성 대조 재사용**: lead 0·1·2 가 다시 **비트 동일**이어야 한다(`G-3`).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gates_b3 import (RECORDS, SEED, NOISE, FIELD, ARMS, ARM_ORDER,   # noqa: F401
                      sign_test_one_sided, key, paired, label)

WINDOWS = [0, 2, 4]                      # ★한 번도 안 본 창
N_PAIRS = len(RECORDS) * len(WINDOWS)
B3_REL_MEDIAN = -23.96                   # 손복사가 아니라 대조용 기준(분석기가 원본에서 재계산해 assert)
SAME_SIZE_PP = 10.0


def gates(rows):
    from gates_b3 import gates as g3
    out = [(n.replace('36 창', f'{N_PAIRS} 창'), ok, d) for n, ok, d in g3(rows)]
    out.append(('G-6 창이 {0,2,4} 뿐 (오염 없음)',
                {r['win'] for r in rows if 'error' not in r} == set(WINDOWS),
                str(sorted({r.get('win') for r in rows}))))
    return out


def label_c(p, b3_rel):
    lab, txt = label(p)
    lab = lab.replace('REFIT-HELPS', 'B3-REPLICATES').replace(
        'REFIT-NULL-OR-HURTS', 'B3-DOES-NOT-REPLICATE')
    same = abs(p['rel_median_pct'] - b3_rel) <= SAME_SIZE_PP
    txt += (f"  크기: 이 배치 {p['rel_median_pct']:+.2f}% vs B-3 {b3_rel:+.2f}% ⟹ "
            + ('**같은 크기**(±10%p 안).' if same
               else '⛔**다른 크기** — 크기 주장은 B-3 창에 한정할 것.'))
    return (lab, txt)
