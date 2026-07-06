# 세션 체크포인트 — 창의적 학습 (2026-07-02, 내일 재개용)

## 오늘의 arc (한 줄씩)
1. "파수 라벨 없이 실측 점으로 학습" — 사용자 지적: 실측은 LiDAR 점만, 파수 라벨 없음.
2. **discriminative 학습 5가지 모두 RELAX 못 넘음** (연속회귀 정체·1-shot grid 0.49·A1 refine 0.69·v2 0.69·
   unrolled ISTA 0.74 vs RELAX 0.89). 데이터/학습량 아님, 방법 한계. supervised는 sim2real. → `LEARNING_LEVERS`.
3. **피벗(B2 정보한계/B4 operability/B3 UQ) 기각** — 사용자: 정보한계는 "추정 못한 변명"으로 보임, 추정 완성 후에나.
   **학습을 창의적으로 성공시키기 우선.**
4. **교차분야 본문 정독** — 천문 전파간섭계가 CLEAN(=RELAX)→**생성 prior**로 넘어가 능가+UQ. → `CREATIVE_LEARNING_LITREVIEW`.
5. **DPS 구현·진단** — 아래.

## 지금 위치 (DPS 생성 prior)
- **구현 완료·작동**: 물리 forward operator(2채널 η+η_t, ij축, 2분기 dispersion 전파) **검증 corr 0.989**.
  2채널 diffusion prior(64×64, 2.11M, `results/direction3/diffusion_prior.pt`) + DDIM posterior 샘플러.
- **결과 (8/300)**: DPS field 0.84/0.49/0.09, **obs-fit 0.98~0.996**, RELAX 0.87/0.94/0.99.
- **진단**: obs-fit 항상 완벽하나 필드 들쭉 = **300 희소obs가 2×4096 field DOF 과소결정 → 약한 field-prior가
  틀리게 채움**. 정보한계 아님(RELAX 복원). seed1 RELAX급 = 메커니즘 작동.

## ★ 내일 첫 작업 (진단이 직접 지시)
**prior를 field가 아니라 sparse 성분집합/스펙트럼에 두기** (RELAX와 같은 구조 → 과소결정 해소):
1. **set-diffusion 성분 prior**: 파랑을 {(kx,ky,A,φ)} 성분집합으로 표현, 그 분포를 diffusion 학습.
   자산 `results/direction3/setdiff_prior_long.pt` 재검토(이전 direction3 작업, [[direction3_gridfree_prior]]).
2. **DPS-on-components**: 성분 prior score + obs likelihood(dispersion 렌더) → posterior sampling.
   성분 수가 적어(≈8) 300 obs로 over-determined → RELAX급 제약 + 생성 UQ.
3. 8/300·12/300·6/600서 posterior mean이 RELAX 넘고 UQ 보정되는지. 넘으면 = **창의적 학습 성공 + UQ 신규**.
4. 대안(성분 prior 약하면): field prior 훨씬 강화(장학습/큰 net) 또는 성분 prior + refine 하이브리드.

## 파일/자산
- 스크립트: `tmp/diffusion_prior.py`(2ch prior 학습), `tmp/dps_sampler.py`(DPS eval, forward operator 포함)
- 체크포인트: `results/direction3/diffusion_prior.pt`(2채널), `setdiff_prior_long.pt`(성분 prior 후보)
- 문서: `CREATIVE_LEARNING_LITREVIEW`·`LEARNING_LEVERS`·`IMPROVEMENT_DIRECTIONS`·`DESIGN_two_approaches`(전부 2026-07-02)
- 기준: RELAX(추정 본체) 8/300≈0.90, 6/600≈0.90~0.96, 12/300≈0.75~0.82

## 상시 원칙 (오늘 확립)
- 충분히 학습 후 판단(수렴 확인) — [[feedback_train_enough_before_judging]]
- novelty 주장 전 문헌검증 — PINO/Ehlers 선점 주의
- 정보한계(B2)는 추정 완성 후에만 (변명으로 보이지 않게)
