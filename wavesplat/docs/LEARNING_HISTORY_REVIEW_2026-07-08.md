# 학습 접근 전체 이력 검토 — 새 학습 설계 전 (2026-07-08)

사용자 지시: "새 학습 시도 전에 이전에 학습했던 것들과 도전 기록 전체를 먼저
검토." 목적 = **같은 실패를 반복하지 않고, 진짜 미탐색 레버를 식별.**
1차 소스: `STUDY_GUIDE_2026-07-06.md` §12–14(학습 아크 권위 종합),
`SESSION_SYNTHESIS_2026-07-02/05/06`, `tnp_pilot_design`, `METHODOLOGY_REVIEW_learned`,
40+ 체크포인트/eval JSON.

---

## A. 시대별 실패·돌파 지도 (시간순)

### 시대 0 — "iterative 위에 NN 덧붙이기" (6월, 전부 실패)
`tnp_pilot_design_20260601.md`. 시도 7갈래: DCSpecNet, USGO, Neural
corrector, Stokes2, DU-RELAX, PINO/FNO/DeepONet, RELAX-standalone.
- **공통 실패**: 기존 iterative(Adam 500 iter)의 local minimum(nRMSE 0.22)을
  못 벗어남. **optimizer의 NN화(USGO, DU-RELAX)조차 실패** — 10 unrolled
  layer가 500 Adam iter를 못 따라감.
- 교훈: "물리 optimizer를 NN으로 대체"는 死路. → paradigm 전환(amortized).
- KalmanNet 파일럿(6/03): Go/No-go #1 통과했으나 streaming/tracking 방향
  (조우예보 본류 아님).

### 시대 1 — mean-collapse (직접 회귀 전멸)
raw 점 회귀 / 판별망 / amortized 직접 회귀 / mode-seeking → **전부
mean-collapse**(출력이 흐릿한 평균≈0). 원인 = 모델 크기 아니라 **목적함수·
표현**: 위상·방향이 다봉인데 픽셀 MSE는 다봉서 평균을 맞히는 게 최적.
- `learned_offgrid`, `mse_operator_fc`, `amortized_ff_mse` 등이 이 시대.

### 시대 2 — 돌파 1: 표현(도메인) 교체
입력을 점구름 → **beamform 파워맵(k-공간)**으로 바꾸니 단순 CNN 검출+잔차가
RELAX급. k-공간에선 에너지 위치가 유일(다봉 제거).
- **Neural-guided RELAX**: net이 전 성분 동시 제안(component-distill, Chamfer)
  + 물리 refine → sparse/고성분 격자서 RELAX 이김(15중 10).
  `component_distill_mixed.pt`, `net_vs_relax.pt`.

### 시대 3 — 돌파 2: 생성 prior (DPS set-diffusion)
성분 집합 분포를 diffusion으로 학습 → 관측 가능도로 조건화(DPS) → 사후 샘플.
복잡/고성분/**HOS 비선형** 강점 + UQ 공짜(Woodbury exact O(mM²)).
- `setdiff_prior_mixed_big.pt`, `dps_on_hos.json`, `symmetric_dps.json`.
- 정리: **"단순=반복 검출, 복잡/비선형=생성 prior"** 상보 구조.

### 시대 4 — sim2real 벽
합성-clean 학습망 실측 직접전이 **실패(≈0 vs RELAX 0.6–0.7)**. 원인=
broadband+노이즈+스케일 gap. 물리(RELAX/MMSE)는 무학습이라 전이됨.
처방 = realistic 분포 + **정규화(레코드 의존성 제거)**.

### 시대 5 — amortized prior net v1→v7 (현 본류, "net은 prior만")
물리(MMSE·분산·Woodbury)는 그대로, net은 **γ(k) prior 맵만** 공급(정규화
k/kp∈[−2.5,2.5]² bank, 6×51×51 beamform 입력, 실패 시 hand로 폴백).

| 버전 | 변경 | 결과 |
|---|---|---|
| v1/v2 | 라벨 회귀(참 γ SmoothL1) | **G1 실패**(이동 동률/−0.05). 라벨=hand 정보 재가공+1-draw 노이즈 |
| **v3** | **task-loss**(γ→미분가능 MMSE→홀드아웃 장오차) | **G1 통과**. "라벨 아닌 태스크에 손실을" |
| v3 실측 | zero-shot | **G3 최초 학습 양성전이**: BS11a 0.579/AA14 0.582(hand 0.474/0.505); 8-record fixed **0.538(+34%, 8/8승)**, supra +48% (SA 프로토콜) |
| v4cur | 해류-증강(γ-prior로 Doppler 보상 시도) | **무효** — γ-prior는 스펙트럼 시프트 보상 불가 |
| **v5fan** | 전방-팬+**선형 이봉** 증강 4k, v3서 FT | **기각**: 실측 0.55 < v3 0.60 (hand 0.57). **"선형 이봉이 실측 실패의 비선형 물리를 안 담음"** |
| v6fan | 20k 샘플 | 실측 0.609 > v3 0.595 (첫 실측-개선, 미미) |
| v7fan | 50k 샘플 | 실측 0.595 ≈ v3 — **스케일 무효**(oracle 갭≈0 예측대로) |

- **⑤ 이중-대역 bank**(비학습): 합성 대역밖 스웰 0.68→0.74 개선, **실측 24창
  탐지 0회** — 실측 이봉은 동일-대역 방향 이봉(환대역 이미 커버).

### 시대 6 — PINO (병렬 갈래)
WavePINO-S4 (physics-informed operator, stage 1–5). 합성 BF3_5c nRMSE
0.177 vs WaveSplat 0.152 / BF4_5c 0.203 — **WaveSplat과 대등~약간 열세**.
방법 novelty 있으나 성능 우위 없음.

---

## B. 관통 법칙 (이력이 가르치는 것)

1. **"학습 가치 = prior 미스매치 크기"** (STUDY_GUIDE §14). 학습은 hand prior
   가정(단봉 JONSWAP)이 깨지는 곳(실측 broadband·교차해·조우 시프트)서만
   이득; 설계점 합성(가정 정확)선 이득 0. **스케일(4k→50k) 무효**가 이 법칙의
   증거.
2. **목적함수·표현이 병목**(모델 크기 아님). 3번의 전환(도메인→beamform,
   회귀→생성, 라벨→task-loss)이 전부 목적/표현 교체로 뚫림.
3. **물리 분업이 sim2real 열쇠**. "net은 prior만 + 물리 폴백"이 G3 통과 구조.
   순수 학습망은 실측서 붕괴.
4. **iterative-NN화·optimizer-NN화는 死路**(시대 0). amortized prior가 유일 생존.

---

## C. 결정적 관찰 — 왜 학습이 오늘의 천장을 아직 못 깼나

- **net은 hand와 동일한 단일대역 자유파 bank 위의 prior**(KREL=2.5). 즉
  **net도 같은 모델 클래스에 갇힘** — γ 재가중만 하지 bound wave(오프-셸)를
  표현 못 함. 오늘 데이터-클래스 사다리(0.19→0.46→0.55)·P5 밀집무효와
  일관: 천장은 모델 클래스이고 net은 그 클래스 안에 있음.
- **v5fan이 이미 이봉 증강을 시도해 실패** — 단 **선형** 이봉이었음. 오늘 K3가
  실측 나쁜창=**비선형 cross-Tp bound wave**로 규명 → v5fan 실패의 원인을
  사후 설명. **미탐색 = HOS 비선형 이봉 학습 데이터.**
- **8-record +34%는 SA/이동개구 nowcast 프로토콜** — 조우예보 요구대역이 아님.
  오늘 처음 net을 조우예보 연산자에 연결(§D).

---

## D. 오늘의 새 baseline — net v3를 조우예보 요구대역에 (2026-07-08 신규)

`d3_falcon_real_operator.py --net prior_net_v3.pt` (연산자에 학습 prior 연결,
프로젝트 최초). 요구대역(0-2s) nRMSE / corr, hand 대비 (W=20):

| 레코드 | hand | net v3 |
|---|---|---|
| YS17 | 0.72 / +0.78 | 0.69 / +0.80 |
| AA15a | 0.48 / +0.88 | 0.43 / +0.90 |
| AA14 | 0.52 / +0.86 | 0.49 / +0.88 |
| AA15b | 0.41 / +0.92 | 0.41 / +0.91 |
| **풍파4 평균** | **0.53 / +0.86** | **0.51 / +0.87** |
| **LJ18g(폭풍)** | **1.31 / +0.09** | **0.70 / +0.72** |
| **5레코드 평균** | **0.69** | **0.55** |

- **풍파4 요구대역: 작지만 일관된 개선**(0.53→0.51, corr +0.86→+0.87,
  3/4 개선). **프로젝트 최초 학습 net을 조우예보에 적용, 첫 양성.**
- **★LJ18g 폭풍: 극적 개선(1.31→0.70, corr 0.09→0.72)** — 폭풍은 broadband라
  hand 단일대역 prior 미스매치가 최대 → 학습이 최대로 도움. **"학습 가치=prior
  미스매치 크기" 법칙의 가장 강한 실증**(비학습 wideband도 LJ 1.31→0.82와 정합).
- 종합: **net은 미스매치 큰 곳(폭풍)서 크게·정합 좋은 곳(풍파)서 marginal.**
  풍파 요구대역 천장(~0.5)은 유지(net도 단일대역 클래스, C절). 0-10s 혼합
  (AA15a/AA14 악화 = 장리드는 net 손해).
- **함의**: zero-shot으로도 풍파 +0.02·폭풍 −0.61. **미스매치를 겨냥한
  재학습(HOS 이봉, E #1)·모델클래스 확장(#2)이면 풍파 천장도 움직일 여지** —
  첫 신호 긍정, 단 풍파 요구대역 근본 돌파는 여전히 #2에 달림.

---

## E. 새 학습 설계 후보 (이력이 남긴 진짜 미탐색 레버)

이력상 **이미 실패로 닫힌 것**(반복 금지): 직접 회귀, iterative-NN화,
optimizer-NN화, 선형 이봉 증강(v5fan), 스케일 증대(v7), 해류 γ-증강(v4),
비학습 이중대역(실측 탐지 0).

**진짜 미탐색 (오늘 K3/P5가 새로 연 것)**:
1. **HOS 비선형 이봉 학습 데이터** — v5fan이 놓친 비선형 물리. 단 net이
   자유파 prior라 표현 한계 잔존(개선 상한 있음).
2. **★모델 클래스 확장 — bank에 2차 bound-wave 후보(k₁±k₂, ω₁±ω₂) 추가**
   → 추정기가 오프-셸 에너지를 표현. 이게 오늘 천장의 유일한 근본 돌파구
   (P5/K3/A1 토론 결론과 일치). net은 그 확장 bank의 prior를 학습.
3. **학습형 2차-피크 배치** — K7 geomean(고정)의 학습판: net이 창별로 kp2를
   검출·배치. 단 kp2=2kp1이 이미 bank 내부라 배치보다 표현이 관건(→ #2).
4. **생성 prior(DPS) 재소환** — 시대 3서 HOS 비선형 강점 확인됨. 조우예보
   프로토콜엔 미적용. #2와 결합 가능(bound-wave 성분 분포 학습).

**권고 우선순위**: #2(모델 클래스 확장) > #1(HOS 학습) > #4(DPS) > #3.
근거: 오늘 전 분석이 "천장=모델 클래스"로 수렴 → prior 재학습(#1,#3)은
상한이 낮고(자유파 안), **표현 확장(#2)만 근본 돌파**. #1은 #2의 학습
데이터로 병용.
