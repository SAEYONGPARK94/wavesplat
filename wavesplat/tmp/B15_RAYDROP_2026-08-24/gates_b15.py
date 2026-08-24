"""B-15 단일출처 — **센서 모형(raydrop)이 헤드라인을 바꾸는가.**

## 왜 (2026-08-24)
배포 생성기의 기본 raydrop 은 `uniform` 이고, 그 반환 확률은
`P ∝ R_fresnel(θ)·cos θ / d²` 다(`wavesim/raydrop_uniform.py:66`).
★**그 식을 저장소 함수로 직접 계산하면 입사각 80°가 법선보다 3.12배 더 잘 반환된다.**
그런데 논문 제2장(`thesis_main.tex:913-916`)은 **정반대**를 적는다 —
*"고반사율에서 대부분이 정반사되어 수신기로 돌아오지 않으므로 유효 관측점이 소실된다."*
그리고 물리적으로 그 서술과 맞는 구현이 **이미 저장소에 있다**(`wavesim/raydrop_slope.py`
= 반사 광선이 센서로 되돌아오는 기하 조건).

⟹ **헤드라인(`B-1`·`B-11`·`B-12`)이 전부 `uniform` 위에서 측정됐다.**
⚠이것은 음성이 아니라 **양성 결과에 붙는 교락**이므로 더 무겁다.

## ★이 배치가 가능한 이유 — 완벽한 짝이 이미 디스크에 있다
`episode_*_uniform.h5` 와 `episode_*_slope.h5` 는 **같은 파랑장·같은 성분**이고
(`gt_wave_field` max|Δ| = 0.0, `wave_components` 동일 — 런 전 확인)
**센서 모형만 다르다**. ⟹ 생성 0회, 순수 개입.

## 표본
혼잡 층(`ncomp ≥ 15`) × `BF4/BF5` × 시드 0–4 = **20 짝**(uniform 20 + slope 20).
⚠**선형 참값**이다(`B-11`/`B-12` 의 HOS 코퍼스가 아니다) — HOS 는 slope 짝이 없다.
⟹ 이 배치가 재는 것은 *"헤드라인이 센서 모형에 견디는가"* 이지 *"HOS 헤드라인의 재현"* 이 아니다.

## 판정 (결과 전 확정, n=20 짝, 단측 부호검정)
`15/20 → p=0.0207` · `16/20 → p=0.0059` · `17/20 → p=0.0013`
- **PRIMARY** = **slope**(물리적으로 옳은 모형)에서 `SHIP-NOW < SPEC25` 인 에피소드 수
  · ≥16 ⟹ ★`HEADLINE-SURVIVES-CORRECT-SENSOR`(가족단위 α=0.05/2=0.025 통과)
  · =15 ⟹ ⚠명목만 · 6–14 ⟹ `UNDECIDED` · ≤5 ⟹ ⛔`HEADLINE-IS-SENSOR-ARTIFACT`
- **음성 대조(필수)** = slope 에서 `SPEC25 < ZERO` ≥ **15/20**. 미달이면 **PRIMARY 무효**
  (기준선이 무기술이면 이겨도 뜻이 없다 — `B-12` 규약 상속).
- **SECONDARY** = 짝 대비: `uniform` 승수 − `slope` 승수
  · |Δ| ≥ 5 ⟹ `RAYDROP-MATTERS` · |Δ| ≤ 2 ⟹ `RAYDROP-DOES-NOT-CHANGE-HEADLINE` · 그 사이 `RAYDROP-UNDECIDED`
⛔**어느 결과에서도 `B-12` 의 n=48 판정을 대체하지 않는다** — 참값이 다르다(선형 vs HOS).
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
if B11 not in sys.path:
    sys.path.insert(0, B11)
from gates_b11 import EST_SEEDS, FILTER_SEEDS, SPEC        # noqa: E402,F401

BFS = ('BF4', 'BF5')
NCOMPS = (15, 30)
GEN_SEEDS = (0, 1, 2, 3, 4)
RAYDROPS = ('uniform', 'slope')
N_PAIR = len(BFS) * len(NCOMPS) * len(GEN_SEEDS)          # 20

NOMINAL, CONFIRM, ARTIFACT = 15, 16, 5                    # /20
NEG_MIN = 15                                              # /20 음성 대조
SEC_MATTERS, SEC_SAME = 5, 2                              # 짝 대비 판정
FAMILY_ALPHA = 0.05 / 2


def sign_test_one_sided(w, n=N_PAIR):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def label_primary(w, neg_ok):
    if not neg_ok:
        return ('NEG-CONTROL-FAILED',
                'slope 에서 기준선이 무기술 ⟹ PRIMARY 를 읽지 않는다')
    if w >= CONFIRM:
        return ('HEADLINE-SURVIVES-CORRECT-SENSOR',
                f'{w}/{N_PAIR}, p={sign_test_one_sided(w):.4g} — 가족단위 통과')
    if w >= NOMINAL:
        return ('HEADLINE-SURVIVES-NOMINAL', f'{w}/{N_PAIR} — 명목만, 탐색적으로 읽는다')
    if w <= ARTIFACT:
        return ('HEADLINE-IS-SENSOR-ARTIFACT',
                f'{w}/{N_PAIR} — ★물리적으로 옳은 센서 모형에서 헤드라인이 무너진다')
    return ('UNDECIDED-AT-SLOPE', f'{w}/{N_PAIR} — 문턱 사이')


def label_secondary(w_uniform, w_slope):
    d = abs(w_uniform - w_slope)
    if d >= SEC_MATTERS:
        return ('RAYDROP-MATTERS', f'승수 차 {w_uniform}-{w_slope} = {w_uniform - w_slope}')
    if d <= SEC_SAME:
        return ('RAYDROP-DOES-NOT-CHANGE-HEADLINE', f'승수 차 {w_uniform - w_slope}')
    return ('RAYDROP-UNDECIDED', f'승수 차 {w_uniform - w_slope}')
