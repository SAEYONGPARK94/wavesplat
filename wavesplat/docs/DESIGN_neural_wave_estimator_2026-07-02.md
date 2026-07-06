# 개선 학습법 설계 — Neural Matching-Pursuit for Dispersion-Constrained Wave Fields

> 목표: RELAX(determined corr ~0.9~0.98)를 **따라잡으면서** 빠르고(amortized) UQ를 주는 학습 추정기.
> off-the-shelf가 아니라 이 세션 경험으로 **우리가 설계.** 현재 최고 학습결과 = plain FF net 0.60 (RELAX 0.98).

## 설계를 제약하는 경험적 교훈 (이번 세션 실증)
1. **파동대역 Fourier features 필수** — raw 선형좌표/aliased FF는 학습 자체를 막음(0.004). 파동 k∈[0.02,0.6] 정합 → 학습됨.
2. **RELAX가 이기는 메커니즘 = dispersion 매니폴드 위 SPARSE 매칭추구** — 소수의 참 성분을 찾아냄. 이 sparse 식별이 0.98의 핵심.
3. **Dense 계수 net 실패**: 344 기저 계수 회귀는 obs-fit→과적합(0.11), field-MSE→느림(0.07@4500). dense 파라미터화가 sparse 구조를 못 잡음.
4. **Direct field 예측(plain FF net)**: 0.60 도달하나 plateau — sparse/dispersion 구조가 없어 RELAX 못 따라감.
5. **목적함수**: pixel-MSE는 under-determined서 mean 붕괴; obs-fit은 과완전기저서 과적합. → 구조·sparsity 필요.

## 핵심 통찰
plain net(0.60)과 RELAX(0.98)의 격차 = **sparse 성분 식별**. dense net·직접필드예측 모두 이걸 못함.
→ **설계는 RELAX의 sparse-매칭추구 알고리즘을 미분가능·학습·amortize하게 만드는 것.**

## 주 설계 (Primary): Deep-Unrolled Neural Matching Pursuit (learned RELAX)
RELAX 알고리즘을 K-스텝 unroll한 미분가능 모듈. 각 스텝:
1. **잔차** r = z_obs − forward(현재 성분들)  (미분가능)
2. **학습된 beamforming/proposal**: 파동대역 FF 인코더가 (obs, r)로부터 다음 성분 (kx,ky)를 제안 —
   RELAX의 beamform-argmax를 amortize. 구현: dispersion 그리드 위 잔차-파워의 **soft-argmax**(미분가능) 또는
   proposal MLP. (dispersion 제약 = 그리드가 ω=√(gk·tanh(kd)) 위에 있음.)
3. **성분 추가 + 진폭 재적합**: 닫힌형 ridge-LS(미분가능)로 모든 성분 진폭 재계산.
K스텝 후 sparse 성분들로 필드 렌더. **end-to-end 학습**(field-MSE on determined; 또는 obs-fit+sparsity).
- **왜 됨**: (a) dispersion 내장(성분이 매니폴드 위), (b) **sparsity 내장**(소수 성분 순차 추가=RELAX), (c) 잔차구동
  반복구조. 학습부(proposal+스텝파라미터)가 비싼 beamform 탐색을 amortize → 빠름+구조+RELAX 추격 가능.
- dense net이 놓친 **sparse 식별**을 구조적으로 강제. novel(문제맞춤).

## 대안 (Simpler): SetConv + dispersion-temporal
transformer가 보간을 느리게 배운 이유 = 보간 inductive bias 부재. ConvCNP의 **SetConv**(RBF scatter-to-grid,
translation-equivariant)를 공간 인코더로 + **dispersion-aware 시간 채널**(obs 시간차를 ω로 위상보정) →
보간은 SetConv가, 시간전파는 dispersion이. transformer보다 빠르게 학습할 것. (단 sparse 식별은 약함.)

## UQ
성분/필드 예측에 **ensemble 또는 dropout-샘플 spread** → 보정 UQ. (DPS의 posterior-spread UQ와 통합 가능.)

## 검증 계획
1. **주설계(unrolled MP)** 구현 → determined(4comp/600obs)서 corr: 목표 plain 0.60 → RELAX 0.9 추격.
2. 되면 → 난이도 sweep(성분·obs) + sparse·실측 확장 + UQ.
3. 안 되면 → 대안(SetConv) + 정직 보고. (성급한 결론 금지 — 격리테스트·하이퍼파라미터 점검 규율 유지.)

## 구현 순서
- Step A: 미분가능 dispersion-grid soft-beamform + closed-form ridge-LS 블록 (`scripts/neural_mp.py`).
- Step B: K-스텝 unroll + 파동대역 FF proposal 인코더.
- Step C: 학습·검증(determined→sweep→실측).

## 설계 라운드 실증 결과 (2026-07-02) — 정직한 negative + 최대수확
시도한 변형과 결과 (determined 4comp/600obs, RELAX ref 0.96~0.98):
| 방법 | 목적함수 | 결과 |
|---|---|---|
| plain wave-band-FF net (직접필드) | field-MSE | **corr ~0.60 (최선 학습결과)** |
| physics-basis 계수 | obs-fit | 과적합 corr 0.11 (obs 맞지만 필드 null-space) |
| physics-basis 계수 | field-MSE | 느림 corr 0.07 |
| LISTA (soft-threshold sparse) | field-MSE | over-sparsify 붕괴 (계수→0, corr NaN) |
| LISTA | obs-fit | over-sparsify 붕괴 (계수→1) |

- **learnable soft-threshold θ가 커져 계수를 다 죽이는 병리** — sparse 학습 불안정. dense는 과적합/mean.
- **GIF(`results/gifs/net_vs_RELAX_ep{1-3}.gif`)**: RELAX r≈0.96 vs 학습net r≈0.33~0.6 — 물리가 명확히 우세,
  net은 지배구조만 blurry하게.
- **최대 수확 = FF-scale 버그 수정**(0.004→0.6, 학습을 가능케 함). 구조적 개선(RELAX 0.9+ 추격)은 **open problem**.
- **실용 결론**: 정확도=물리(RELAX/GP), 학습=속도·UQ·구조생성(DPS)·forecast 보조.
- **다음 안정화 후보**: threshold 고정(비학습) LISTA, ConvCNP-SetConv(보간 bias), 물리-unroll 안정화, 또는
  학습을 "물리 초기화/정칙화 보조"로 재포지셔닝(learned-to-optimize).

## ★ 성공: Neural-Guided RELAX (net 성분제안 + physics refine) — RELAX 품질 달성
핵심 진단: F1/grid-계수는 ~0.3 cap(고정 grid), RELAX 0.98은 **off-grid 연속 성분 식별** 덕. → net이 **연속 (kx,ky)를
예측**(grid계수 아님)해야. 
- **component-distill net**: obs→MOUT 연속 (kx,ky), 참 성분에 Chamfer loss(진폭가중). raw net corr ~0.39
  (rough 식별, 파동 위상은 kx,ky에 민감해 raw론 부족).
- **하이브리드 net+refine**(net 제안 → refine_params off-grid 정밀화 → LS+렌더): **corr 0.97~1.00 ≈ RELAX**
  (determined 5episode, `tmp/hybrid_eval.py`, ckpt `results/direction3/component_distill.pt`).
- **옳은 설계 3요소**: (1)파동대역 FF(버그수정) (2)연속 성분 예측(grid 0.3 cap 회피) (3)net제안+물리refine.
  앞선 실패(필드distill·grid계수·LISTA)는 각각 이 중 하나가 빠짐.
- **답: RELAX 학습 가능** — net(식별)+물리(refine). net이 RELAX의 iterative beamform 탐색을 forward 한 번으로 amortize.
- **캐비엇**: determined만; sparse·복잡해·실측 확장 필요; 속도 아직 RELAX와 동급(711 vs 532ms, refine 병목 → seed 좋으니
  iters 줄여 가속 가능). GIF 비교 가능.

## ★★ 확장 결과: Neural-Guided RELAX가 RELAX를 능가 (comp×obs 일반화, 2026-07-02)
혼합학습 net(comp4~12, obs300~600, MOUT16) + physics refine vs RELAX, comp4~30 × obs600~150 (3seed):
- **15조합 중 10곳서 net+refine ≥ RELAX.** sparse·고성분서 압도: 8c/150 0.90 vs 0.67, 12c/150 0.67 vs 0.47,
  20c/600 0.73 vs 0.59, 30c/150 0.31 vs 0.02(RELAX 사실상 실패). OOD(comp20/30, 학습밖)도 일반화.
- RELAX 우세: 쉬운 저성분/조밀(4c/600, 8c/300)서 둘 다 ~0.95+.
- **메커니즘**: 학습 net이 전 성분을 한 번에 제안 → RELAX greedy beamform이 막히는 sparse/복잡서 더 안정.
  physics refine이 정밀화. → "RELAX 학습"을 넘어 **어려운 레짐서 RELAX 개선.**
- `results/neural_guided_relax_grid.json`, `results/gifs/neural_guided_relax_grid.png`.
- **캐비엇**: pool 균일-obs(실제 전방cone 아님), 절대 corr 고성분·sparse서 낮음(정보한계, net이 덜 실패), 속도 RELAX 동급.
- **다음(핵심 검증)**: 실제 전방-cone obs(44점) + 실측 데이터서 우위 유지 확인 → 유지되면 논문("학습이 물리 solver를 개선").

## sim2real 검증 (2026-07-02) — 직접 전이 실패, realistic 학습 필요
이미-학습 net(합성 clean JONSWAP)을 실측 점에 직접 (재학습 없이, 좌표 shift):
- **BS_2011**: net+refine −0.29~0.28 (raw-net 음수) vs RELAX 0.60~0.70. **실패.**
- **YS_2017**: 조밀 net+refine ~0.1 vs RELAX 0.75~0.83; sparse선 둘 다 실패(정보한계).
- **원인(sim2real gap)**: (1) 실측 broadband+스테레오 노이즈 vs 학습 clean 4~12성분, (2) 개구비/스케일 상이
  (BS 14m=~1파장, net은 수파장 도메인 학습 → FF 유효분해능 불일치).
- **물리(RELAX)는 전이됨**(무학습, 에피소드 적응) → 실측 robust 방법은 여전히 RELAX.
- **개선안**: net을 realistic 분포로 학습 — broadband(64+성분)+스테레오노이즈+전방cone 희소기하+다양한 패치스케일/
  좌표정규화; 또는 실측 소량 fine-tune. 이게 sim2real 좁히는 길.
- **종합**: 학습 접근은 합성서 RELAX 능가(원리 검증)하나 **실측 배치엔 realistic 학습분포 필수.** 현재 실측=RELAX(물리).
