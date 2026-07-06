# 창의적 학습 — 교차분야 선행연구 검토 (본문 정독, 2026-07-02)

> 목적: discriminative(obs→적합) 학습이 RELAX 못 넘음 확인 후, **타 분야 선행연구**에서 창의적 학습 아이디어.
> 사용자 지시: 정보한계(B2)는 추정 완성 후에나(지금은 변명으로 보임). 학습을 실제로 성공시킬 창의적 방법 우선.

## 핵심 발견 — 전파간섭계(천문학)가 직접 로드맵
우리 문제(희소 측정→필드)와 **수학적 동형**. 그들의 고전 baseline = **CLEAN**(greedy deconvolution) = **우리 RELAX와 동형**.
그 분야는 **CLEAN → 생성(generative) prior**로 넘어감:
- **RI-GAN 2507.21270**(본문 정독): 조건부 GAN, 32 posterior 샘플 병렬 → **CLEAN 대비 +13.5 dB SNR**, **빠른 UQ**(mean±std,
  오차상관 r≈0.69). 단 GT쌍 지도학습(sim2real 주의). measurement operator를 net에 내장→분포이동 부분강건.
- **Score-based priors 2311.18012 / DDRM 2601.15844 / VIC-DDPM**: score prior + 측정 likelihood → posterior sampling,
  CLEAN 능가 + UQ. **CLEAN/proximal은 UQ 없음**(우리 RELAX도 없음) = 명확한 차별.
- **Score-based PDE-constrained solver 2408.16626**(본문 정독): **★prior를 "샘플"에서 denoising score matching으로 학습
  (GT 해쌍 불필요)**, 물리는 likelihood(soft)로, reverse-SDE+Langevin으로 posterior sampling. 희소 측정→prior가
  미관측 영역 복원. (분포이동은 미논의=우리가 다룰 gap.)

## 의료영상(MRI/CT) — 같은 패턴
self-supervised 측정-only 학습(Noise2X, recorrupted-to-recorrupted, TomoSelfDEQ, Sparse2Inverse), score-based prior
posterior sampling, **equivariant test-time training**(2411.05771) — 파랑의 병진등변성을 자기지도 신호로 쓸 여지.

## 해양 generative — 존재하나 우리 문제엔 미적용
sea-ice diffusion surrogate(2024), ocean state generative DA(2025), ocean interior generative(2504.15308).
**→ 위상분해 파랑을 희소 LiDAR서 generative prior로 복원한 연구는 없음**(Ehlers=지도 FNO, PINO=operator, 둘 다
점추정·UQ없음·discriminative). **generative-prior×wave-LiDAR = 미개척.**

## ★ 창의적 방향 (도출)
**Score-based 확산 prior(파랑 필드/스펙트럼) + 물리(dispersion)·obs likelihood → posterior sampling(DPS)**
- **왜 RELAX 넘을 수 있나**: RELAX(greedy 적합)는 희소서 정보-제한(미관측 못 복원). **prior가 학습된 파랑 구조로
  gap을 채움** → 희소/고성분(내 A1·LISTA가 진 바로 그 레짐)서 posterior mean이 RELAX 점추정보다 참에 가까움.
  (천문서 generative가 CLEAN 이긴 메커니즘과 동일.)
- **라벨 문제 해결**: prior는 **필드 샘플**(시뮬레이터 생성)로 score-matching 학습 — obs 라벨 불필요. 실측선 obs
  likelihood(물리)만 적용.
- **UQ 내장**: posterior 샘플 → mean(점추정)+std(보정 UQ). B3를 추정기 안에 흡수.
- **discriminative와 근본적으로 다름**(내가 시도한 5가지 전부 discriminative) = 진짜 창의적 신규 시도.

## 정직한 캐비엇
- 정보한계는 못 넘음(없는 정보 창조 불가) — 단 **MMSE-최적 정규화 + UQ**로 희소서 RELAX 점추정 개선 가능.
  dense/determined선 RELAX 이미 near-perfect라 이점 작음(희소가 목표 레짐).
- 분포이동(sim prior vs 실측 broadband) = 실측 적용 gap. prior는 필드 구조라 discriminative보다 전이 유리하나
  검증 필요. 필요시 broadband prior + 실측 self-sup 보정.
- 리스크: 구현 복잡(score net + DPS 샘플러), 튜닝. 하지만 machinery는 검증됨(천문·의료).

## 제안
**DPS 프로토타입**: (1) 파랑 필드/스펙트럼 score prior 학습(sim 샘플) (2) dispersion+obs likelihood DPS 샘플러
(3) 희소 레짐(8/300,12/300,실측)서 posterior mean이 RELAX 넘고 UQ 보정되는지. 넘으면 = 창의적 학습 성공 +
UQ 신규 + (그 후에야) 정보한계 B2 논의 정당.

## DPS 프로토타입 결과 + 진단 (2026-07-02)
구현 완료: 2채널(η,η_t) diffusion prior(64×64, 2.11M) + FFT 2분기 dispersion 전파 forward(검증 corr 0.989) +
DPS 샘플러(DDIM+likelihood guidance). config 8/300, zeta 25, deterministic.
| seed | DPS field | obs-fit | RELAX |
|---|---|---|---|
| 1 | 0.836 | 0.996 | 0.872 |  ← RELAX 근접
| 2 | 0.485 | 0.991 | 0.937 |
| 3 | 0.086 | 0.980 | 0.986 |
- **핵심 진단**: DPS가 **obs는 거의 완벽 적합(0.98~0.996)하나 필드 corr 들쭉(0.09~0.84)**. = **obs-fit 만족돼도
  필드 틀림** → 300 희소 obs가 2×4096 필드 DOF를 과소결정, **약한 field-prior가 gap을 틀리게 채움**(hallucination).
- RELAX는 소수(8) dispersion 성분만 적합 → 강하게 제약 → 이김. **정보한계 아님**(RELAX가 같은 obs로 복원).
- **결론**: field-diffusion-prior는 표현이 과다유연 + prior 약함. 창의적 방향은 맞고 **메커니즘 작동(seed1 RELAX급)**
  하나 robust하지 않음.
- **명확한 다음**: prior를 **필드가 아니라 sparse 스펙트럼/성분 집합**(RELAX 구조와 정합)에 두기 →
  set-diffusion prior(자산 `setdiff_prior_long.pt` 보유). 또는 훨씬 강한 field prior. 성분-prior가 과소결정 해소.
