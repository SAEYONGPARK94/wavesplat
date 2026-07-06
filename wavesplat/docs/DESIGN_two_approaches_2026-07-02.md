# 두 학습 접근 검토 — 파수 라벨 없이 실측 점으로 (2026-07-02)

> 배경: 실측은 LiDAR 점(x,y,z)만, 파수 라벨 없음. 지도학습(참 파수 타깃)은 sim 한정→sim2real 실패.
> 두 대안을 검토(현재 component-distill net은 보존).

## 접근 1: 포인트만 직접 자기지도(self-supervised) 학습 (새 설계)
**핵심**: 파수/필드 라벨 없이 **점만으로** 학습 → 실측 직접 적용 가능. 여러 손실 방향:

**1a. Held-out 관측 예측 (context→target)** ★가장 정직
- 한 스캔의 점을 context/target 분할 → net(context)→target z 예측. 라벨=관측점 자체(자기지도).
- 실측 점에 바로 학습. 과적합 방지(unseen 점 예측). NP/conditional-field 계열.
- ⚠️ under-determined면 held-out도 mean 최적 → 정보한계 그대로(sparse서). dispersion/sparsity prior 필수.

**1b. Obs-fit + dispersion basis + sparsity** (RELAX식 자기지도)
- net→sparse dispersion 계수, 손실=obs 재현. sparsity(soft-threshold/L1)로 과적합 억제.
- 앞서 dense obs-fit 과적합·LISTA over-sparsify 실패 → **sparsity 안정화가 관건**(고정 threshold, top-k, entmax).

**1c. PINN 잔차**: 필드가 dispersion PDE 만족 + obs 적합. 라벨 불필요. 진동해 강제.

- **학습 여러 방향 검토**: 1a(held-out) / 1b(sparse obs-fit) / 1c(PINN) 각각 sim+실측서 corr·과적합·안정성 비교.

## 접근 2: RELAX를 네트워크 앞에 (points + RELAX-추정 파수 입력)
**핵심**: RELAX(점)→추정 파수(kx,ky,A) [라벨 아님, 점에서 추정]. net 입력 = (점, RELAX-추정). 
- 파수 라벨 문제 우회: RELAX가 train·test 모두 점에서 파수 추정 제공.
- net이 배우는 것(옵션): (i) RELAX 추정 **보정**(정확도↑), (ii) **UQ**(RELAX 추정의 불확실성), (iii) **forecast**
  개선(RELAX 성분을 비선형 전파), (iv) **가속**(RELAX iterative refine를 net 한 방으로).
- 손실: sim=GT필드 / 실측=held-out obs(자기지도). RELAX 추정이 입력이라 실측서도 작동.
- **가치**: RELAX 위 학습된 corrector/UQ/forecaster. RELAX 강점(라벨-free 파수) + 학습 강점(UQ/보정/속도).
- 리스크: RELAX가 이미 좋으면 보정 여지 작음 → UQ·forecast·속도가 주 가치.

## 검토 계획 (각각)
- **1**: 1a(held-out) 먼저 프로토타입 — sim determined서 되는지 + 실측서 RELAX 대비. 되면 1b/1c 비교.
- **2**: RELAX-front + net(UQ head 또는 corrector) — RELAX 추정 입력받아 (a) 필드 보정 (b) per-점 σ(UQ) 학습.
  sim서 보정/UQ 유효성, 실측서 UQ 보정도.
- 공통: 실측(BS/YS) 라벨-free 평가(held-out obs corr) + RELAX 기준.

## 검토 결과 (2026-07-02)
**접근 1a (자기지도 held-out, black-box)**: field corr ~0.1 (held-out-rmse 0.11<std 0.30이나 필드는 smoothing으로
만족). **under-constrained — dispersion 구조 없이 점만으론 안 됨.** 1b(dispersion basis+sparsity)가 필요하나
sparsity 학습 불안정(앞서 over-fit/over-sparsify). → 접근 1은 "구조 내장 + sparsity 안정화"가 미해결 과제.
**접근 2 (RELAX-front)**: RELAX가 라벨-free로 점→파수 추정 제공(train·test 공통, 실측서도) → net이 그 위에서 작동.
- **가장 유망**: RELAX가 이미 정확(0.98)하니 net 가치 = **(a) 보정UQ**(RELAX엔 UQ 없음, net이 per-pixel σ 학습→
  실측서 "언제 못 믿는지" flag) **(b) forecast 개선 (c) 어려운레짐 보정 (d) 속도**.
- **실측 적용성 ★**: 입력이 RELAX-추정(점에서, 라벨불필요)이라 sim2real gap 우회 — UQ/보정은 sim학습이 실측
  전이 쉬움(필드 자체가 아니라 오차패턴/불확실성 학습).
**권장**: 접근 2(RELAX-front + UQ head)가 실측-적용 가능성 최고. 접근 1은 self-supervised sparsity 연구과제.
</EOF
