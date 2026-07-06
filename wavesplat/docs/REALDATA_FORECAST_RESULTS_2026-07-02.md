# 실측 forecast·외삽 결과 (B) + estimator 하드닝 (C) — 2026-07-02

> 헤드라인 결론: **nowcast 단일프레임 복원은 물리≈naive(개구비 한계)지만, naive가 원리적으로 못 하는
> forecast·unobserved 외삽에서 물리가 명확히 이긴다.** 이것이 방어가능한 논문 positive.
> 스크립트: `tmp/bs_forecast_extrap.py`, `tmp/validate_improved_*.py`; estimator: `scripts/estimator_improved.py`.

## B) BS_2011 forecast + 외삽 (fixed estimator: klim=1.5 + 모델차수 제어)
short-crested wind-sea, 패치 14m, Hs 0.23m, Tp 2.4s, 방향분산 27°, 개구비 ~1λ. 4프레임×2seed.

### ① Forecast vs persistence (naive는 시간모델 없어 persist만 가능)
| lead | 물리 nRMSE | persistence | skill | 물리 승률 |
|---|---|---|---|---|
| 0.5s | 0.204 | 0.338 | **0.64** | 100% |
| 1.0s | 0.208 | 0.422 | **0.76** | 100% |
| 2.0s | 0.220 | 0.296 | 0.45 | 100% |
| 3.0s | 0.275 | 0.306 | 0.19 | 88% |
| 5.0s | 0.336 | 0.318 | −0.12 | 25% |
| 8.0s | 0.388 | 0.330 | −0.38 | 0% |

→ **물리 forecast가 persistence를 0.5~3s서 압도**(skill 0.76까지). 유용 horizon ≈ 1~2주기(Tp 비례). 5s+는
Tp2.4s의 2주기 넘어 무의미(파동 decorrelate).

### ② Unobserved 외삽 (관측 hull 밖 영역)
물리 nRMSE **0.206**±0.047 vs naive **0.284**±0.021 → **물리 100% 승, margin +0.078.**
(naive는 hull 밖서 평균으로 붕괴, 물리는 식별한 k·ω로 파동 실어나름.)

### 대조: La Jument (개구비 0.11λ)
forecast도 약함(8s에나 겨우 이김), 외삽도 무의미 → **개구비<1λ면 forecast·외삽마저 무력.** 개구비가 floor.

## C) estimator 하드닝 (`scripts/estimator_improved.py`)
적응 klim(≈3.5×kp, [0.6,2.0]) + CV 모델차수(held-out RMSE로 성분 중단). `relax_step`에 `klim` 파라미터 추가(기본0.6).

**합성 헤드라인 회귀검증 (4프레임×2seed):**
| episode | default nRMSE/corr | improved nRMSE/corr | improved 승률 |
|---|---|---|---|
| BF4_15c | 0.101/0.90 | 0.113/0.89 | 38% |
| BF4_30c | 0.167/0.76 | 0.154/0.82 | 50% |
| BF5_30c | 0.173/0.67 | 0.177/0.66 | 50% |

→ **improved ≈ default (무승부, 회귀 없음). 극적 개선 아님.** 유용한 건 **적응 klim(짧은 파 실측)**, CV 모델차수는
합성서 중립. estimator_improved는 backward-safe 하드닝으로 채택 가능.

## 방법론 교훈 (반복 확인)
**단일프레임 결과 3회 연속 오도** (BS 0.162, C 0.100, La Jument 0.163 전부 multi-frame서 완화). → 모든 판정은
**multi-frame×seed + 공정(in-hull) 채점** 필수. Discord 보고도 이 원칙으로 교정.

## 남은 것
- B를 **다레코드(A 다운로드분: AA·BS·YS·LJ-Grid2)로 확장**해 forecast·외삽 positive 굳히기.
- D: amortized NN + 보정 UQ 배치레이어(정확도 아닌 speed+신뢰도, headline 아님).
