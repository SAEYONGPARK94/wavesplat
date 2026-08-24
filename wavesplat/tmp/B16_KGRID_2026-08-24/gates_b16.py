"""B-16 단일출처 — **표현 절제: 연속 파수가 값을 하는가.**

## 왜
논문의 기여 ①은 *"격자 비의존 스펙트럼 표현"* 이다. 그런데 2026-08-24 방법론 심사가
지적했다: ★**표현만 절제한 팔이 저장소에 0건**이다. 헤드라인(`B-12` 43/48)은
*표현·학습·솔버·추적*이 **묶인 채로** 기준선을 이긴 것이고, 이긴 것이 **표현**인지
**적합 절차**인지 분리되지 않았다. 기준선을 더 고쳐서는 닫히지 않는다(그 축은 `S17` 로 폐쇄).

## 개입
배포 파이프라인을 **그대로** 두고 초기화 끝(VARPRO 직후)에 **파수만** 격자에 스냅하고
(w, φ) 를 LS 로 다시 맞춘다(`sequential_update._snap_k_to_grid`, 기본 None = no-op).
⟹ 바뀌는 것은 **파수의 연속성 하나뿐**이다. 추적은 `w_phi` 모드라 파수가 고정되고,
성장/가지치기는 **초기화 안에서만** 돌므로(`:851`,`:906` 둘 다 스냅 **앞**) 누수가 없다.

## 격자 간격 (결과 전 확정, 자료에서 도출)
기준선 격자 = `gt_grid` (x: 10–250 m, y: −100–100 m) ⟹ Δk = 2π/L:
  **Δk_x = 2π/240 = 0.02618 rad/m** · Δk_y = 2π/200 = 0.03142 rad/m
- `KGRID1` = **0.02618** — 두 축 중 **더 고운** 쪽. ★우리 팔에 **유리한** 선택이다
  (해칠 기회를 줄인다) ⟹ 그래도 해치면 주장이 강해진다.
- `KGRID2` = **0.05236** (2배) — **용량-반응**을 본다(W7 이 벽의 인과를 이렇게 샀다).

## 판정 (결과 전, n=48 에피소드, median-of-3, 단측 부호검정)
`31/48 → p=0.0297` · `34/48 → p=0.00276`
- **PRIMARY** = `SHIP-NOW < KGRID1` 인 에피소드 수
  · ≥34 ⟹ ★`CONTINUOUS-K-EARNS`(가족단위 α=0.05/2=0.025 통과)
  · =31–33 ⟹ ⚠명목만 · 18–30 ⟹ `UNDECIDED`
  · ≤17 ⟹ ⛔**`CONTINUOUS-K-DOES-NOT-EARN`** — *"이긴 것은 표현이 아니라 적합 절차다"*
- **SECONDARY(용량-반응)** = `KGRID1 < KGRID2` ≥31/48 이면 `DOSE-RESPONSE`
- ⛔**어느 결과에서도 `B-12` 판정을 대체하지 않는다.** 이 배치는 **귀속**을 재지 크기를 재지 않는다.

★**결과 전에 적는 해석 규칙**: PRIMARY 가 널이어도 그것은 **나쁜 소식이 아니다** —
*"우리 이득의 담지자는 연속 파수가 아니라 적합 절차"* 라는 **더 정직하고 방어 가능한 문장**이 된다.
⛔단 그 경우 **논문의 기여 ①을 다시 써야 한다.**
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
B12 = os.path.abspath(os.path.join(HERE, '..', 'B12_HOS_EXTEND_2026-08-23'))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
for p in (B12, B11):
    if p not in sys.path:
        sys.path.insert(0, p)
from gates_b12 import ALL_SEEDS, N_EP                       # noqa: E402
from gates_b11 import HOS, EST_SEEDS                        # noqa: E402,F401

DK1 = 0.02618          # = 2π/240, 기준선 격자의 고운 축
DK2 = 0.05236          # 2배 (용량-반응)
ARMS = {'SHIP-NOW': {'k_snap_dk': None},
        'KGRID1':   {'k_snap_dk': DK1},
        'KGRID2':   {'k_snap_dk': DK2}}
ARM_ORDER = ['SHIP-NOW', 'KGRID1', 'KGRID2']

NOMINAL, CONFIRM, NOT_EARN = 31, 34, 17                     # /48
FAMILY_ALPHA = 0.05 / 2


def sign_test_one_sided(w, n=N_EP):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def label_primary(w):
    if w >= CONFIRM:
        return ('CONTINUOUS-K-EARNS',
                f'{w}/{N_EP}, p={sign_test_one_sided(w):.4g} — 격자에 스냅하면 나빠진다 '
                f'⟹ 연속 파수가 이득의 담지자 쪽이다')
    if w >= NOMINAL:
        return ('CONTINUOUS-K-EARNS-NOMINAL', f'{w}/{N_EP} — 명목만, 탐색적')
    if w <= NOT_EARN:
        return ('CONTINUOUS-K-DOES-NOT-EARN',
                f'{w}/{N_EP} — ★격자에 스냅해도 나빠지지 않는다 '
                f'⟹ 이긴 것은 표현이 아니라 적합 절차다')
    return ('ATTRIBUTION-UNDECIDED', f'{w}/{N_EP} — 문턱 사이')


def label_dose(w):
    return (('DOSE-RESPONSE', f'{w}/{N_EP} — 간격을 2배로 하면 더 나빠진다')
            if w >= NOMINAL else ('NO-DOSE-RESPONSE', f'{w}/{N_EP}'))
