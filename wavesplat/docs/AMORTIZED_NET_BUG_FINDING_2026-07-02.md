# Amortized net 결함 규명 — spectral bias + under-determined 정보한계 (2026-07-02)

> 계기: D vs B GIF에서 D(amortized posterior)가 완전히 빈칸 → 사용자가 "학습 안 됐거나 코드 버그 아니냐"
> 지적. 검증 결과 **두 개의 별개 원인**이 겹쳐 있었음. d52·d56·d58·d67·d68 amortized 결론(전부 nRMSE≈0.25)에
> 직접 영향. 스크립트: `tmp/diagnose_D.py`, `tmp/net_sanity.py`, `tmp/net_fourier_fix.py`,
> `tmp/train_ff*.py`; 수정 모듈: `scripts/amortized_ff.py`.

## 결정적 진단
| 테스트 | 원시 net(d52/d56 아키텍처) | Fourier-feature 수정 |
|---|---|---|
| 단일배치 overfit (8샘플 암기) | **실패** (train-rmse=signal std, 전혀 안 떨어짐) | **성공** (0.43→0.036) |
| 관측점서 z 재현 | 실패 (corr −0.02) | — |
| 일반화(새 64성분, pure-MSE) | mean (0.25, corr~0) | **여전히 mean (0.25, corr~0)** |
| 일반화(NLL) | mean (0.25) | mean (0.25) |

## 두 원인 (겹침)
1. **표현력 버그 (진짜 버그)**: 좌표(x,y,t)를 **원시 Linear로 임베딩** → 진동(파동) 함수 표현 불가 = **spectral
   bias** (Tancik et al. 2020). net이 8샘플조차 암기 못 함. **Fourier features로 수정** (`amortized_ff.py`).
   → **d52~d68 amortized의 "info-limit 0.25" 결론은 상당부분 이 표현력 버그였음.**
2. **일반화 정보한계 (진짜 한계)**: 표현력을 고쳐도 *일반화*는 mean(0.25). 200점→64성분 broadband는
   **under-determined → pixel-MSE 최적해가 prior mean으로 수축.** net이 옳게 학습한 결과. NLL·MSE 공통.

## 핵심 통찰 (세션 전체 관통)
- RELAX/물리는 nRMSE **0.28**(mean 0.25보다 나쁨)인데 **corr 0.7** — 구조·위상 보존, pixel-MSE는 손해.
- pixel-MSE 최적 net은 mean으로 수축(nRMSE 좋고 corr 0, 쓸모없음).
- → **pixel-MSE nowcast가 왜 잘못된 지표인지의 결정적 증거.** 유용성은 구조(forecast·외삽)에 있음.

## 함의
- **NN 방향 정정**: amortized net은 hopeless 아님(FF로 표현 가능). 단 **pixel-MSE/NLL로 훈련하면 붕괴** →
  쓸모있으려면 **구조보존 목적**(성분예측 / RELAX distillation / forecast목적)으로 훈련.
- **기존 결과 재검토 필요**: d52(F4 operator), d56/d58/d67/d68(posterior)의 nRMSE≈0.25는 표현력 버그가
  오염 → FF로 재실행해야 진짜 amortized 성능(및 UQ 유효성) 판정 가능.
- **앞선 "NN dead-end" 추천**(2026-07-02)은 이 버그 위에서 나온 것 → 완화·정정.

## 실행 (2026-07-02, 사용자 지시 "A로 학습해서 보여주고 B는 자료로 남기자")
- **RELAX distillation D 실행**: `tmp/precompute_relax_cache.py`(300 에피소드 RELAX 성분 캐시) →
  `tmp/train_distill_D.py`(obs→RELAX필드 지도학습, `scripts/amortized_ff.py` FF net, 구조보존) →
  `tmp/make_D_vs_B_gifs_v2.py`(distilled D로 GIF 재생성). 결과: `results/direction3/amortized_distill.pt`,
  `results/gifs/DvsB_distill_ep*.gif`.
  - 타깃이 RELAX(obs)=obs의 결정적 함수라 mean-collapse 없이 학습됨 → D가 파동을 실제로 그림(빈칸 아님).
  - 단 정확도는 RELAX 상한(≈0.28, mean보다 나쁨) — "구조 보존은 pixel-MSE 손해"를 그대로 계승.
- **빈칸-D GIF는 증거로 보존**: `results/gifs/why_pixelMSE_fails/`(+README) — "왜 pixel-MSE nowcast가 틀린
  지표인지"의 근거 자료.
## RELAX-distillation 시도 결과 — 실패 (또 하나의 정직한 negative)
`tmp/train_distill_D.py` (300 에피소드, obs→RELAX필드 지도학습, FF net):
- distilled-D std **0.001**, corr(D,RELAX) ~0, rmse-vs-RELAX = signal std → **또 mean 붕괴.**
- **훈련 에피소드(캐시 내부)조차 재현 못 함** = 300개 obs→필드 매핑 memorize도 실패. (8샘플 overfit은 됨.)
- ∴ FF로 *표현력*은 고쳤으나, **이 transformer 아키텍처는 sparse obs→파동장 operator를 학습 자체 못 함**
  (direct GT·distillation·300개 암기 전부 mean 붕괴). 8샘플 이상이면 mean-basin으로 수렴.
- **왜 RELAX는 되고 net은 안 되나**: RELAX는 dispersion 물리 prior(ω=√(gk·tanh(kd)))로 탐색공간을 제약 →
  희소 obs서도 성분 식별. black-box transformer엔 그 inductive bias가 없어 mean으로 붕괴.

## 4개 amortized 변형 전부 mean 붕괴 (로버스트 negative)
| 변형 | 손실/타깃 | 결과 |
|---|---|---|
| black-box 필드 (d52/d56) | GT-필드 MSE/NLL | mean (0.25), overfit조차 실패 |
| FF 필드 (`amortized_ff`) | GT-필드 MSE/NLL | 표현력 회복(overfit OK) but 일반화 mean |
| RELAX distillation | RELAX-필드 MSE | mean, 300개 암기도 실패 |
| **physics-basis 계수 + obs-fit** (`train_spectral`) | 관측 재현 | **mean, obs조차 못 맞춤(계수 안 냄)** |

- 마지막이 결정적: obs-fit은 *determined*(200 obs를 344 계수로 거의 완벽 적합 가능)인데도 net이 mean →
  **정보한계 아니라 set→추론 operator를 trained NN이 학습 못 하는 문제.** T-NSE input-invariant collapse를
  표현력 confound 제거하고 재확인.
- **결과를 내는 방법 = 에피소드별 물리 solver**: F1(dispersion 기저 ridge-LS), RELAX, GP(파동커널). 이들은
  각 에피소드 데이터에 계수를 적합(=추론)해 구조를 냄. **trained NN(amortized)만 붕괴.**

## 결론 & 다음
- **빈칸-D GIF(`results/gifs/why_pixelMSE_fails/`)가 정직한 산출물.** 4개 학습변형 다 실패.
- **trained-NN이 결과를 내려면 = deep-unrolled optimization**: ridge-LS/RELAX 반복을 미분가능 net으로 unroll
  (물리가 각 스텝에 내장 → 구조 강제 + 학습가능). black-box·amortized-회귀는 이 문제에 안 됨. 유일한 유망 경로.
- 지금 "학습이 결과 내는" 것을 보고 싶으면 = **F1/GP(물리 solver)가 그것**(B 패널). trained-NN 데모는
  deep-unroll 구현(별도 규모) 필요.
- d56/d58 UQ 결론도 이 붕괴 위 → 재검증 필요.

## 추가 버그 발견 (2026-07-02, 사용자 "학습 안 된다고 하지말고 면밀히" 지적 후)
**Fourier feature 주파수 스케일 버그** — 이번 세션 여러 "collapse" 결론을 뒤집음.
- **격리 테스트**: 단일시간 순수 공간보간(dispersion 없음, 4comp·600obs)에서도 FF net이 corr 0.004로 stuck.
  600점 스냅샷 보간은 GP도 쉬운데 안 됨 → 정보한계·dispersion 아니라 **훈련 병리** 확정.
- **원인**: FourierFeatures scale=0.3, 좌표 x∈[10,250]m → 유효파장 ~3m ≪ obs 간격 ~9m → **aliasing** →
  net이 좌표를 못 씀 → mean 붕괴.
- **수정**: FF 주파수를 파동 대역에 정합 (공간 k=2π|B_xy|∈[0.02,0.6], 시간 ω∈[0.3,3], log-spaced) →
  단일시간 net corr **0.004→0.39(계속 상승)**. `scripts/amortized_ff.py` FourierFeatures 수정 완료.
- **함의**: (1) 내 FF 넷들의 "일반화 collapse"가 이 버그 탓일 수 있음(전부 scale 0.3). (2) 원본 d52~d68은 raw
  선형좌표라 더 심각. (3) **"학습 안 됨"은 성급한 오류였음** — 제대로 스케일 맞추면 학습됨.
- **교훈(중대)**: "안 된다" 결론 전에 **격리 실험(trivial 하위문제가 되는가)** + **하이퍼파라미터(FF스케일 등)
  점검** 필수. 사용자 지적이 이 버그를 드러냄.
- 다음: multi-time/dispersion 전체 문제로 fixed-FF 확장 검증 중.

## Step 1-2 재검증·튜닝 결과 (2026-07-02, 차례대로 진행)
- **Step 1 (fixed-FF 재검증)**: determined(4comp·600obs) net corr **0.004(buggy 0.3)→0.60(fixed 파동대역 FF)**.
  → 학습 re-open 확정. d52~d68은 이 좌표버그로 오염(재실행 필요).
- **Step 2 (RELAX 0.98 추격, FF튜닝)**: learnable FF = **0.49 (fixed 0.60보다 나쁨)** — loss↓인데 field corr↓ =
  주파수를 훈련fit에 옮겨 과적합. → **FF튜닝은 레버 아님. net ~0.6 plateau, RELAX 격차는 아키텍처 문제.**
- **정직한 위치**: determined-저성분서 물리(0.98)>학습(0.6). 학습 값어치는 물리 약한 고성분/복잡해(DPS>RELAX 사례)
  + 속도·UQ. sparse는 둘 다 정보한계(corr~0.2).
- **다음(step 3)**: forecast operability 곡선(개구비 vs skill, 물리 BS서 positive) — A 42GB 전사이트 준비됨.
