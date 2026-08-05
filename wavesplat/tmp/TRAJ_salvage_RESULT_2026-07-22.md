# ② 궤적 살리기 = FAILED (Step1b 확정 · 기제 규명) — 2026-07-22

> fable 소유 트랙(설계+실행). 스크립트 `tmp/traj_salvage_diag.py`, 원자료 `results/direction1/traj_salvage_diag.json`,
> 로그 `tmp/traj_salvage.log`. (fable transcript 정리돼 자체 md 미작성 — 본 문서가 기록.) n=24.

## 판정 = FAILED — 궤적 우위 회복 안 됨
회복 가설(더 나은 이동-obs 파수추정이면 회복)을 진단 3단계로 시험:

**A. 파수-ID 진단** (RELAX vs 참, median): |dk|/kp = stationary 0.244 / cross 0.219 / greedy 0.265 / CEM 0.254 /
learned 0.236 — **정책간 비슷.** 단 이동정책이 **spurious 성분 더 검출**(greedy/CEM n_est=10 vs 참 8, stationary 6).

**B. Near/far 분해** (재구성 오차): **오라클 파수면 CEM/learned가 stationary를 far-field서 이김**(CEM far 0.028 vs
stationary 0.048) = **개구 이득 진짜 실재.** 그러나 **RELAX 추정 파수면 이동 전부 열위**(near CEM 0.179·learned 0.223
vs stationary 0.115; far도 전부 열위). ⟹ 퍼진 obs가 재구성 품질(국소밀도·spurious성분)을 악화시켜 개구이득 상쇄.

**C. 회복 시도** (near-obs 제한 재구성): 전 정책 Δ(near) 음수·CI 0제외 (cross −0.14 / greedy −0.15 / CEM −0.065 /
learned −0.11, 모두 "no"). ⟹ **근처만 재구성해도 이동정책이 stationary 못 이김.**

## ⟹ 기제 (규명)
**개구 이득은 오라클-파수에서만 실현**(CEM/learned가 far-field서 stationary 이김). **realistic(RELAX 추정) 파수면
퍼진 obs가 (a)낮은 국소밀도 (b)spurious 성분으로 재구성을 악화 → 개구 이득을 상쇄·초과.** 파수-ID 오차 크기는
비슷하나 재구성 결과가 다름. ⟹ **궤적/센싱-설계 축은 realistic 추정서 죽음 (B-2·M-3와 같은 oracle-washout).**

## 이 라운드 종합 (⓪①②)
- ⓪ CRB 단위버그 = false alarm(경험검증). ① under-det 갭 = 파수 ID로 국소화·학습 헤드룸 +0.45 실재·oracle-의존
  아님(유일 생존축)·realizability 미해결. ② 궤적 = FAILED(oracle-washout 확정).
- **세션 아크: 학습 positive들(B-2·궤적) oracle-washout, under-det 파수-ID만 생존. spine A 견고 재확인.**
- 다음 = 사용자 결정(파수-ID CRB로 ① realizability 시험 / spine A 집필). [[project_methods_landscape_2026-07-22]].
