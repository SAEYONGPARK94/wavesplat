# 방법론 종합 검토 — 방향추정의 한계와 올바른 연구 방향

> 생성: 2026-06-26. Direction II PoC(Step1~4) + 추가 실험 종합. 박사 방법론 정리.

## 0. 메타 결론 (가장 중요)

**병목은 알고리즘이 아니라 정보다(information-limited, not algorithm-limited).**
단일 시점/짧은 윈도우의 sparse LiDAR에는 미세 파향 정보가 근본적으로 부족하다
(Step1: 정답 근방 likelihood가 30°에서 0.18 nat/obs로 평평 = 낮은 Fisher 정보 = 높은 CRB).
→ **추론을 아무리 영리하게 바꿔도 없는 정보는 못 만든다.** 방법론 선택은 이 사실에서 출발해야 한다.

## 1. 방법론 옵션 3분류 + 정직한 판정

### 그룹 A — 추론을 재배열 (정보 추가 없음) → 한계효과, 권장 안 함
| 옵션 | 검토 | 판정 |
|---|---|---|
| 더 미세한 격자(128²) | grid loss 줄지만 4× 비용, 근본한계 불변 | △ 보조만 |
| 전역/multi-start 탐색 | likelihood가 평평 → 전역도 신호 약함 | ✗ |
| 조건부 diffusion prior | init은 개선 가능하나 미세방향 정보부재는 동일 | △ |
| prior 스펙트럼을 GP 커널에 직접(peak-pick 회피) | 양자화 일부 완화, 한계 불변 | △ |
- **공통 결론:** Step3/4에서 격자·off-grid refine 모두 실패. 정보가 한계라 추론 재배열은 0.28벽을 못 넘음.

### 그룹 B — 정보를 추가 → 진짜 레버, 단 제한적
| 옵션 | 검토 | 판정 |
|---|---|---|
| **Synthetic aperture(시간 융합)** | 실험: 200점 고정 시 시간폭↑→미세방향(10°) 식별력↑(단순해상 1.1→5.9). **단 ~3초(0.3주기)라 이득 modest, 복잡해상 noisy** | ◯ (winning GP의 근거) |
| 전 에피소드 joint fit(최대 시간개구) | 윈도우보다 나을 것(미검증) | ◯ 후보 |
| 더 긴 관측(~10주기) | predictable-zone 이론상 필요. 실데이터 제약 | ◯ 이론적 |
| 다중센서/더 넓은 개구 | 하드웨어 변경, 범위 밖 | — |
- **핵심:** **ST-DF-GP(0.205)가 single-window(0.28)보다 좋은 이유가 바로 이 시간개구.** 정보를 추가하는 유일한 실용 경로.

### 그룹 C — 불확실성을 끌어안기 → 방법론적으로 정직 + novel
| 옵션 | 검토 | 판정 |
|---|---|---|
| **방향 불확실성 위에서 Bayesian 주변화 → 보정 예측분포** | point 추정이 한계면, *분포*를 정확히 내는 게 정답 | ◎ 강력 |
| **확률적 predictable zone** (UQ로 일반화) | "어디까지·얼마나 확신하고 예보 가능한가" | ◎ novel |
| post-hoc variance scaling (이미 90→91%) | 검증됨, 확장 가능 | ◎ |
- **DSWP 문헌 공백이 바로 여기.** point 예측은 성숙(radar/EnKF), *보정된 phase-resolved UQ*는 비어 있음.

## 2. 방법론 권고 (연구 방향 전환)

1. **방향 point 추정기 개선은 중단** — 정보한계라 수확체감(Step3/4가 증명).
2. **두 축에 집중:**
   - **(B) Synthetic-aperture 정식화** — 시간개구를 방향추정에 *명시적으로* 쓰는 추정(전 윈도우 dispersion-manifold 추정). ST-DF-GP를 이 관점으로 재정식화.
   - **(C) UQ 중심 기여** — 보정된 phase-resolved 예측분포 + 확률적 predictable zone. 정보한계를 *기능*으로 전환.
3. **(I) CRB 이론 = 백본** — 시간개구·SNR·궤적의 함수로 방향추정 근본한계를 유도해 "(B)가 필연이고 (C)가 정직"임을 증명. ST-DF-GP가 한계에 근접함을 보이면 수조 없이도 이론적 무게(Level 2 박사).

## 3. 박사 novelty / contribution (확정 서사)
- **부정 결과들이 토대:** T-NSE(amortization 붕괴) + Step3/4(prior·refine로도 single-window 한계) →
  "왜 영리한 추정기로는 안 되는가"를 실증.
- **긍정 핵심:** (B) 시간개구가 답이고(ST-DF-GP +20%), (C) 보정 UQ가 DSWP 공백을 메우며,
  (I) CRB가 이 모두를 *근본한계*로 묶는다.
- **한 문장:** *"파향은 한 시점 스냅샷으론 알 수 없음을 증명하고(CRB), 시간개구로 얻으며(ST-DF-GP),
  남는 불확실성은 보정해 예보·제어에 쓴다(UQ)."*

## 4. 다음 작업 우선순위
1. **Direction I — CRB 유도** (이론 백본, 저비용 고효과).
2. **전-윈도우 dispersion-manifold 추정** (B 정식화, ST-DF-GP 재해석).
3. **확률적 predictable zone + UQ 전 sea-state** (C, 주력 기여).
4. (T-NSE·Step3/4는 부정결과로 논문 discussion에 편입.)
