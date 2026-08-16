> ⛔**2026-08-13 감사 — 이 판정은 철회됨.** "Platform motion DOES unlock reconstruction the
> stationary limit forbids" 는 **실측 관측기하 위의 field-CRB 로 반증됐다**:
> **정지 0.00986 vs cross 0.01615 (0/12) · along 0.02631 (0/12)** ⟹ **이동이 정보를 줄인다.**
> 게다가 이 배치의 "오라클이면 이동이 이긴다" 는 **이중 오라클**(궤적을 진실 성분으로 최적화)
> 이고 승리가 **전부 먼바다**다(근거리 ≤20 m 는 CEM 12/24, p=1.000) = `SUPPORT-TRADEOFF`.
>
> ⛔**또한 이 배치(`scripts/d74_closed_loop_trajectory.py`)는 출하 시스템의 증거가 아니다** —
> **LiDAR 파이프라인을 호출하지 않는다**(`load_episode` **0회**; 관측은 `:69-80` 의
> 플랫폼 주변 상자 균일난수 `L_FOOT = 20 m` × `M_STEP = 30`점 + 해석식 `Σ A·cos` + N(0, σ)),
> 바다도 `synth_sea` 로 합성한다. 추정기도 출하 `SequentialUpdater` 가 아니라
> `wavesplat.model.tnse`(`:34`)다. 정지 팔의 개구는 **19.97×19.96 m** 인 반면
> 출하 정지 팔은 **149–203 m** 다.
>
> ⚠**2026-08-13 3라운드 정정** — 처음 붙었던 *"`wavesim` 임포트 0회"* 와 *"`results/direction1/**`
> 계열"* 이라는 **일괄 적용은 거짓**이다(R3 §B-2): (i) 이 스크립트는 `:33` 에서 `d1_crb` 를 임포트하고
> `d1_crb:39` 가 `wavesim.data_writer` 를 임포트한다 — 다만 **호출은 하지 않으므로 "LiDAR 0회 호출"
> 은 참이다**; (ii) **같은 디렉터리의 `REPORT.md`·`REVIEW_LOG.md` 는 실제 에피소드 LiDAR 점군을 쓴다**
> ⟹ 그 두 파일에는 이 문단을 적용하지 말 것.
> 근거: `tmp/AUDIT_recheck_2026-08-13/{A3_citation_bans.md, R3_regression.md §B-2}`, `reference_citation_bans.md`
>
> 본문 수치는 이력 보존을 위해 남긴다 — **출하 성능으로 인용하지 말 것.**

# VERDICT — closed-loop informative-trajectory control (2026-06-30)
# ⛔ RETRACTED 2026-08-13

Closed-loop platform-heading control to minimize field-reconstruction CRB; learned
non-myopic policy (imitating CEM) vs baselines. Froude-scaled (full 2-8 m/s ↔ tank 0.4-1.6).

## Field-CRB by policy (headline speed 8.0 m/s, lower=better)
- stationary      : 0.575 ± 0.055
- straight-along  : 3.992 ± 0.575
- straight-cross  : 0.896 ± 0.110
- greedy          : 0.424 ± 0.063
- CEM             : 0.132 ± 0.024
- learned         : 0.186 ± 0.033

## Achieved reconstruction nRMSE (RELAX, oracle wavenumbers)
- stationary      : 0.054
- straight-along  : 0.087
- straight-cross  : 0.039
- greedy          : 0.050
- CEM             : 0.019
- learned         : 0.039

## Honest readout
- (A) achievability: trajectory control vs stationary — cross 0.896 vs stationary 0.575 (≈/✗).
- (B) non-myopic headroom: CEM 0.132 vs greedy 0.424 (CEM better → learning has a target).
- (C) learned: 0.186 vs CEM 0.132 / greedy 0.424; inference 7.8 ms/trajectory (matches/approaches planner fast).
- surfing check: straight-along 3.992 vs straight-cross 0.896 (along should be worse).

## ⭐ Corrected honest interpretation (the auto (A) line above is misleading)
The naive intuition (straight cross-track helps) is WRONG for FIELD reconstruction over a path:
straight-cross CRB (0.896) ≈/worse than stationary (0.575) — a dwelling sensor accumulates more info
at one place; naive motion spreads it thin; surfing (3.99) is catastrophic. **The benefit needs
INTELLIGENT trajectory control:**
- ⛔**(A) RETRACTED (2026-08-13 감사)** — "INTELLIGENT control beats stationary … **Platform motion
  DOES unlock reconstruction the stationary limit forbids** — but only when the trajectory is
  optimized." 실측 관측기하 field-CRB 에서는 **정지 0.00986 < cross 0.01615 (0/12)** 로 부호가
  반대이고, 이 배치의 이동 우위는 **이중 오라클 + 먼바다 전용**이다.
  원문 보존: ~~CEM field-CRB 0.132 vs stationary 0.575 (~4×) @8 m/s; achieved nRMSE CEM 0.019 vs
  stationary 0.054 (~3×). Platform motion DOES unlock reconstruction the stationary limit
  forbids — but only when the trajectory is optimized.~~
  근거: `reference_citation_bans.md`
- (B) non-myopic headroom is LARGE: CEM 0.132 vs greedy 0.424 (~3×). Myopic greedy underperforms
  (even loses to naive straight-cross on achieved nRMSE) → big learnable non-myopic structure.
- (C) the LEARNED policy wins where it must: 0.186 field-CRB (beats greedy 0.424 at every speed,
  approaches CEM 0.132) at **7.8 ms/trajectory** vs CEM's hundreds of CRB evals. Honest gap: it does
  not FULLY match CEM on achieved nRMSE (0.039 vs 0.019) — headroom remains.

**Bottom line (learning-central, winning, theory-grounded):** the platform-motion lever (d72) is real
but only an INTELLIGENT, NON-MYOPIC controller unlocks it — naive motion fails, myopic greedy
underperforms. A fast learned policy realizes the benefit (beats greedy, approaches the planner,
7.8 ms). CRB theory = reward/engine; learned policy = contribution. The doctoral positive:
*learned informative-trajectory control for synthetic-aperture phase-resolved wave estimation.*

## Honest caveats
CRB uses true components; achievability uses oracle wavenumbers (isolates the APERTURE effect, not
full deployable wavenumber ID). Linear sum-of-cosines sea, simulation only, 16 episodes. Next: full
deployable ID, HOS nonlinear seas, real/tank validation.