# 전역 RNG 결합 전수조사 (임무 C, 2026-08-28) — 런 0회 · 코드 수정 0

발단: `M4` 에서 `set_seed(42)` 뒤 `torch.randn(1)` **한 draw** 추가로
`BS11c` 0.6453 → **3.4116 (+429%)** · `BS11a` 3.1027 → 1.6077 (−48%).
성장 가지 미발화 셀 {AA15a, BS13c} = `G-15` 통과 2/12 — **12/12 대응**.
⟹ 같은 시드·같은 원자라도 그 지점까지 소모된 전역 draw 수가 다르면 결과가 갈린다.

## 1. 결합 지점 목록 (`wavesplat/**`, 배포 경로 중심)

### A. 추정치에 **직접 착지**하는 전역 draw (값이 결과에 들어간다)
| 지점 | 내용 | 발화 |
|---|---|---|
| `optimizer/sequential_update.py:1617` | `torch.rand(n_actual)*2π` 성장 위상 | **LIVE** — 학습 경로 혼잡 8/87=9.2%(08-19 실측); 이번 조사에서 adaptive 이벤트 B-19 22/48 · B-21 110/192 · B-22 144/144 셀 |
| `optimizer/sequential_update.py:1603-1606` | `np.random.randn/uniform` 성장 pad | beamform 이 n_add 미달일 때만 — 드묾 |
| `optimizer/initializer.py:718` | v2_konly 초기 φ `torch.rand(n)*2π` | analytic/v2_konly 런 **100%** — CV 53% 의 문서화된 원천. 동류: `:266`(v1) `:497`(v2 빈관측 폴백) `:636`(oracle) `:703`(relax_konly) |
| `optimizer/coarse_to_fine.py:138-177` | 전역 `np.random` kx/ky/φ + `torch.rand` φ(:177) | v2_konly staging 성장 (08-19 실측 14/14) |
| `baselines/etkf_grid.py:70,108,148,206` · `letkf_grid.py:88,120,165,268` · `etkf_wang2025.py:77,367` | 전역 `np.random` 앙상블 초기화·inflation·관측 서브샘플 | 기준선 팔 100% (np 스트림) |
| `baselines/oracle_baseline.py:67` | `torch.rand` φ | 오라클 팔 한정 |
| `optimizer/learned_init.py:379-381` | 폴백 `torch.randn` | n==0 축퇴 시만 ("never expected") |

### B. **소모만** 하는 draw (값은 `copy_` 로 덮이나 스트림 위치를 민다)
- `model/spectral_gaussian.py:57-64` — 생성자 `randn` **4n**(무분산 5n) 원소 소모. **모든 모델 빌드마다.**
  ⟹ 팔 간 원자 수 Δn 이 있으면 성장 시점의 스트림 위치가 4·Δn 만큼 어긋난다.
- `model/spectral_gaussian.py:186-196` — `add_gaussians` 폴백 randn (성장은 4값 전부 전달하므로 미발화).

### C. 전역 스트림을 **재시드/복원**하는 지점 (이력 절단·회계)
- ★`scripts/d3_beamform_cond_diffusion.py:160` — `cd.sample` 이 **`torch.manual_seed(diffusion_seed)` 로 전역을 재시드**
  후 n·M·3 + T스텝 `randn_like` 소모. 학습 **인라인** 경로 100% 발화.
  ⟹ **인라인 팔과 캐시(`learned_atoms`) 팔의 스트림 이력이 구조적으로 다르다.**
- `run.py:117-122 set_seed` — `run_estimation`(`run.py:212`)이 **매 런 호출** ✅(방벽).
- `sequential_update.py:475-477` — neural corrector 로드 주변 스냅샷/복원
  (저장소가 이 결함 부류를 **한 곳에만** 수리해 둔 증거).
- `sequential_update.py:1829-1840` — `MedianEnsembleUpdater` 의 `_rng_snapshot/_rng_restore/_seed_globals`
  (전역 스트림 회계에 명시적으로 의존하는 설계 — 멤버 0 이 "ambient RNG stream" 을 진다).

### D. 분리된(안전) RNG ✅
`bank_mmse.py:138 default_rng(seed)`(**실측 배포 경로**) · `learned_init.py:144 default_rng` ·
`sequential_update.py:1171 default_rng(0)` · `initializer.py:238,487` 등의 np `RandomState(seed)`.
⚠단 **`seed` 인자가 np 만 시드한다** — 같은 함수들의 torch 쪽(`torch.rand/randn`)은 전부 전역이다
(`initializer.py:243-246,494-497`, `learned_init.py:379`). "seed 를 받으니 결정적" 은 반쪽 참.

## 2. ⛔독스트링 판정 — `learned_init.py:129-130` 은 **거짓** (원자에 참, 추정치에 거짓)
> *"Selection/dedup happens later in ``pool_and_select`` so the inline and cached paths are byte-identical."*

기제: 인라인은 `cd.sample`(d3:160)이 전역 torch 를 `diffusion_seed`(기본 = est seed)로 재시드+소모하고,
캐시는 `set_seed` 이력을 그대로 잇는다 ⟹ **성장이 발화하는 순간** `seq:1617` 의 위상이 갈린다.
원자(kx,ky,energy)는 동일, 추정치는 상이 — M4 실측 12/12 대응.
⛔코드 수리는 하지 않았다 — 배포 파일이고 **별도 사전등록 사안**.

## 3. 위험 배치 판별 규칙 (EXPOSED ⇔ E1 ∧ E2)
- **E1 (착지 draw 발화)**: 성장 발화(`n_gaussians ≠ auto_n_selected`) OR analytic/v2_konly/oracle 팔(초기 φ 100% 착지) OR np pad 발화.
- **E2 (팔 간 소모 차)**: 다음 중 하나 —
  (a) 러너가 `run_estimation` 을 거치지 않고 직접 생성 + **런별 재시드 부재**,
  (b) 팔 간 원자 수 **Δn ≠ 0** (빌드 4·Δn 소모 차),
  (c) **인라인 vs 캐시** 혼합 (sampler 전역 재시드),
  (d) 팔별 하네스 코드가 전역 draw 소모.
- 스크리닝(전량 감사 없이): ① 러너 grep — 직접 생성 ∧ `set_seed` 0회 → E2a 후보
  ② 결과 행 `n_gaussians`/`auto_n_selected`(또는 동급) 팔 간 대조 → E2b
  ③ `propose_calls`/`atoms_source` → E2c.
- ✅완화 인정: `run_estimation:212` 매 런 재시드 · `MedianEnsembleUpdater` 스냅샷 회계 ·
  부호검정의 다(多)셀·다시드 구조(셀 단위 잡음 방향이 무작위).

## 4. 상위 3개 확인 결과 (확인한 범위만 주장)
1. **B-19** (`B19_KGRID_SPARSE_2026-08-25`, 판정 14/16): 팔 간 **Δn = 0/48 셀**.
   adaptive 이벤트 22/48 발화하나 세 팔이 동조(같은 n 경로) ⟹ **비노출**(n 채널 기준).
2. **B-21** (`B21_MEDIATION_2026-08-27`, FITSWAP<KGRID1 48/48): WS 4팔 **Δn = 0/192 셀** ⟹ **비노출**(동일 기준).
3. **B-22** (`B22_PRIOR_ABLATION_2026-08-27`, PRIOR-EARNS): ★**노출 144/144 셀, 두 채널 동시** —
   (E2b) Δn 중앙 9 · 최대 20 ∧ adaptive 144/144 발화,
   (E2c) `SHIP`·`ANA-SHIP` 인라인(`propose_calls=1`) vs `LEARN*` 캐시(0) — 스트림 이력이 팔 계열별로 다르다.
   그리고 ANA 계열 `ensemble_k=1` — 초기 φ draw 가 **무완화로 착지**.
   ⚠**판정 무효 아님**: Δn 은 처치(사전 교체) 그 자체이고, 48쌍×3시드 부호검정이 셀 잡음을 흡수하며,
   위상 잡음의 방향은 셀별 무작위다. 단 ⛔**셀 단위 크기 인용 금지** · ⛔**SHIP vs LEARN 쌍의
   "추정치 등가" 주장 금지**(원자 등가까지만 — §2와 같은 사유). 기존 `ANA`/`S2` % 금지에 **RNG 기제 사유가 추가**된다.
- 부수 발견 ①: **B-16 은 행에서 확인 불가** — `final_n` 이 432행 전열 **null**
  (`run_b16.py:51` 이 `r.get('final_n')` 을 적는데 `run_estimation` 결과에 그 키가 없다)
  = ★**M2 `G-13` 과 같은 "참조 키 부재 → 항등 null" 부류의 세 번째 사례**(M2·M4 에 이어).
  B-19/B-21 이 같은 처치 계열에서 깨끗하나 **B-16 자체는 미확인**으로 남긴다.
- 부수 발견 ②: `COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py` — `set_seed` **0회** + 직접 생성
  = **E2a 노출** (BS11 +429% 시연의 서식지). 반면 `B-7`/`B-8` 은 `run_b7.py:57-58`/`run_b8.py:58-59` 가
  **매 런 `set_seed(G.SEED)` 직후 생성** ⟹ E2a 비노출.
- 부수 발견 ③: 캐시(`learned_atoms`) 사용 배치 목록 = W4_SOLVER(08-19) · B-22(08-27) ·
  learned_only/heldout40/confirm2/unified_guard(07-26~27) · M4(현행) — E2c 심사 대상 모집단.

## 5. 처방 (⛔별도 사전등록 사안 — 이 조사에서는 적기만 한다)
성장 위상(`seq:1617`)과 초기 φ(initializer 계열)를 **`self.seed` 파생 전용 `torch.Generator`** 로 옮기면
초기화·성장이 **(원자, 시드)의 순수함수**가 되어 draw-소모 이력 의존이 사라진다.
⚠이 변경은 성장이 발화한 **모든 기록 결과의 비트를 바꾼다** ⟹
① 별도 사전등록 없이 배포 파일을 고치지 마라 ② 고칠 때는 값이 아니라 **판정에서 재라**
(B-2 수심 교훈: 전후 구성을 나란히 재집계해 verdict 을 diff) ③ `d3:160` 의 전역 `manual_seed` 도
같은 사전등록에서 `Generator` 로 옮겨야 인라인/캐시 등가가 회복된다.

— 조사 산출물. 실행한 추정 런 0회 · 수정한 배포 코드 0줄. 동결 결과 파일만 읽었다.
