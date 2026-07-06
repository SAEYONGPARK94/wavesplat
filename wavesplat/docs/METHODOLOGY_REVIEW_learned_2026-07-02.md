# 학습·추론 방법론 전면 검토 — 희소 ship-LiDAR 위상분해 파동장 (2026-07-02)

> 계기: "학습이 결과를 내는 걸 보고 싶다 + 다른 방법론 다 검토". 이 세션의 실증 결과로 방법론 공간을
> **prior 유무**에 따라 정리. 핵심 교훈: **구조를 내는 건 모두 명시적 prior 보유; black-box 회귀는 mean 붕괴.**

## 실증 요약 (이 데이터에서 실제로 나온 숫자)
- 물리 ceiling (RELAX): gen_batch(64성분·200obs·240×200m) **corr 0.17~0.26** (info-starved), 데이터셋 episode
  (5~30성분·수천obs·주기창) **corr 0.7~0.9**. → **task 결정성에 따라 천차만별.**
- black-box/NP 계열(ConvCNP 0.248·wave_np 0.251·spec_np 0.26·d56 0.25·FF/distill/spectral 변형): **전부 mean(0.25).**
- 생성 prior (set-diffusion DPS): **구조 샘플 생성**(field std 0.13, corr 0.24) + posterior UQ. (`results/gifs/DPS_generative_demo.png`)

## I. 물리 solver (per-episode 추론, 학습 없음) — 구조 O
- **RELAX**(dispersion 매칭추구), **F1**(dispersion 기저 ridge-LS), **space-time GP**(파동 공분산 커널).
- dispersion 물리를 명시적으로 담아 희소서도 성분 식별. GP는 forecast positive(B). **현재 작동하는 본체.**
- 한계: 정보한계는 못 넘음(sparse서 corr 0.2).

## II. Amortized 회귀 (black-box / NP) — mean 붕괴 ✗
- transformer(d52)·amortized posterior(d56)·ConvCNP·TNP·wave_np·spec_np·내 FF/distill/spectral.
- **전부 nRMSE~0.25=mean**, 충분히 학습해도(ConvCNP 19분). 원인: under-determined서 pixel-MSE 최적=mean + 구조
  prior 없음 + set→field operator 학습난.
- 표현력 버그(원시 좌표임베딩 spectral bias)는 Fourier feature로 수정되나(overfit 회복), **일반화는 여전히 mean.**
- **결론: 정확도 방법으로 추구 금지.**

## III. 생성 prior — 구조 샘플 + UQ ✓ ("학습이 결과 내는" 답)
- **set-diffusion DPS**(d3): 학습 prior에서 샘플 → 각 샘플이 구조적 파동장, 샘플 spread=UQ. pixel-MSE 최소화 안
  해서 붕괴 안 함. 정확도는 info-limit(corr 0.24)이나 **눈에 보이는 파동 생성.** 이미지-공간 diffusion(wave_diffusion),
  score/flow-matching도 동류.
- **가치: 정확도 아닌 (a) 구조 생성 + (b) 보정 UQ.** 헤드라인 학습 기여 후보.

## IV. Deep-unrolled optimization (LISTA) — 물리내장 학습 solver (조건부)
- ridge-LS/RELAX 반복을 미분가능 net으로 unroll, step-size·λ·preconditioner 학습.
- 이 세션: **obs-fit 학습→과적합(corr 0.05); field-MSE 학습→λ↑→mean.** naive 목적함수는 실패.
- **단 λ 고정(F1식)이면 unrolled=F1=구조.** 학습판은 구조-aware 목적(RELAX distillation, 또는 λ고정+precond학습)
  필요. → 유망하나 목적함수 설계가 관건.

## V. PINN (물리 손실) — 미시도, 유망
- coordinate net + dispersion/파동방정식 PDE 잔차 손실 → 진동해 강제. 희소서도 구조 가능성. **다음 후보.**

## VI. 신경연산자 (FNO/DeepONet/PINO) — 격자 필요
- wave_pino(repo). 희소 obs는 gridding 선행 필요. sparse-native 아님.

## 교차 결론 (방향 함의)
1. **신호가 정보한계**: sparse ship-LiDAR nowcast는 물리 ceiling도 corr~0.2. 어떤 방법도 벽 못 넘음.
2. **prior 유무가 구조를 가름**: 물리기저/파동커널/생성prior/PDE = 구조 O; black-box 회귀 = mean 붕괴.
3. **pixel-MSE는 틀린 목적**(mean 선호). obs-fit / 생성 / 스펙트럼-구조 목적을 써야.
4. **방어가능한 학습 기여** = (a) **생성 prior + 보정 UQ**(diffusion, 구조+정직한 불확실성), (b) **forecast·외삽**
   (물리, B result). **black-box nowcast 정확도 아님.**

## 권장
- 학습 헤드라인: **diffusion 생성 prior(구조+UQ)** 또는 **deep-unroll(구조-aware 목적)**, 물리 forecast와 결합.
- 다음 실험: PINN(V) 1회 시도, DPS를 실측·forecast로 확장, unrolled를 RELAX distillation으로 재학습.

## 부록: "DPS 충분히 학습하면 나아지나?" 검증 (2026-07-02)
- **guidance 스윕**: g0.3→0.1로 BF4_5c corr **0.24→0.45**(≈RELAX 0.49). g≥0.6은 오히려 붕괴(normalized-grad
  guidance overshoot). → 개선 레버는 **추론 guidance 튜닝.**
- **prior 재학습(260ep→600ep, d128l4→d256l8)**: DPS **개선 못 함**(대체로 약간 나빠짐). 즉 prior는 이미 충분.
  | ep | RELAX corr | DPS-old | DPS-new |
  |---|---|---|---|
  | BF4_5c | 0.49 | 0.45 | 0.38 | | BF4_15c | 0.09 | 0.28 | 0.14 | | BF3_5c | 0.89 | 0.27 | 0.17 |
  | BF4_30c | 0.13 | 0.27 | 0.39 | | BF5_15c | 0.25 | 0.13 | 0.17 |
- **DPS vs RELAX episode 의존**: RELAX는 info-rich·저성분(BF3_5c 0.89) 압도, DPS는 고성분(BF4_15c/30c) 우세. 둘 다
  sparse nowcast 정보한계(episode별 corr 0.09~0.89). → **"더 학습"이 아니라 guidance 튜닝이 답; 정보벽은 불변.**
- 스크립트: `tmp/dps_guidance_sweep.py`, `tmp/retrain_prior.py`, `tmp/dps_compare.py`. 체크포인트: `setdiff_prior_long.pt`.

## 축 재정리: 지도(supervised) vs 비지도/생성(unsupervised) — 이번 세션 실증
- **지도 (obs→필드/성분 라벨쌍)**: transformer·ConvCNP·TNP·wave_np·spec_np·KalmanNet·neural_corrector + FF/distill/
  spectral. **전부 mean 붕괴** — 원인은 **라벨+pixel-MSE 목적이 under-determined서 mean을 최적으로 만듦.**
  ⚠️ 단 pixel-MSE일 때만. **구조보존 목적**(성분+순열불변/스펙트럼 loss, adversarial, physics-unroll 고정λ)이면
  지도도 구조 가능 — **미검증(다음 후보).**
- **비지도/생성 (p(파동장) 학습 후 추론시 obs 조건화)**: set-diffusion DPS·wave_diffusion·GP커널·(VAE/flow 미시도).
  **구조 생성 O + UQ.** pixel-MSE 안 해 붕괴 없음. ill-posed 역문제에 자연스럽게 적합.
- **결론**: ill-posed sparse 역문제엔 **비지도 생성**이 구조·UQ를 자연히 제공; **지도-회귀(pixel-MSE)는 붕괴.**
  지도로 구조를 내려면 목적함수를 바꿔야(구조/스펙트럼/adversarial). 정보한계는 어느 쪽도 못 넘음.
- **미검증 후보**: (지도) 성분예측+순열불변/스펙트럼 loss, adversarial; (비지도) VAE/normalizing-flow prior.
