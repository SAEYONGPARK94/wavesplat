# W3_SOLVER — 사전등록 (2026-08-15)

**배치 ID** `W3_SOLVER_2026-08-15` · **작성 시각은 `FROZEN_md5.txt` 참조.**

> ⚠**동결의 성격 = self-attested freeze.** 이 저장소에는 외부 타임스탬프 권위가 없다.
> `FROZEN_md5.txt` 는 *내가* 이 파일의 md5 와 동결 시각·git HEAD 를 스스로 적은 것이고,
> 그 이상의 증명력은 없다. ⛔**"preregistered" 라는 단어를 쓰지 말고 "self-attested freeze" 로 쓸 것**
> (M2 감사 2026-08-15: `tmp/*/PREREG.md` 50개 중 `FROZEN_md5.txt` 는 7개뿐).
> 이 배치는 동결 직후 커밋한다. 어떤 **판정 런보다도 먼저** 동결됐다(스모크 포함).

---

## 0. 한 문장

> **오라클 `k` 를 준 팔에서 λ 를 0 으로 내리면 혼잡 레짐의 오차가 줄어드는가** — 그리고 부수적으로,
> **이미 계산해 놓은 위상을 버리는 배포 코드 한 줄**과 **옵티마이저를 완전히 제거한 닫힌형 상한**이
> 각각 어디에 있는가.

---

## 1. 동기와 선행 측정 (전부 열람 확인)

1. **`ORA-K` 계열 4연속 널** — K6 (p=0.79) · S1 (8/16) · S3 (8/16) · S4 (6/16).
   ★**네 번 모두 λ=1e-3 에서 측정됐고, `ORA-K` × λ=0 셀은 저장소에 존재하지 않는다.**
2. **S5** (2026-08-08): 정규화항이 범인이며 그 **94% 가 `lambda_l1`**. 진실에서 출발시킨 팔이
   λ=1e-3 에서 16/16 나빠지고 λ=0 이면 회복.
3. **M4 의 반대 증거** (2026-08-15): 출하 λ=1e-4 에서도 오라클 팔이 **6/6 나쁘고(9.5배)**,
   λ=0 에서도 30성분 잔여가 15성분의 **3.2배** ⟹ **성분 수와 함께 자라는 잔여 벽이 실재**.
   ⟹ λ 가설과 잔여-벽 가설이 **둘 다 살아 있다.** 이 배치는 그 둘을 가른다.
4. **미실행 수리** — `RX2_GROW_FITPHI` 는 환경변수 게이트이고 **기본값 OFF**
   (`initializer.py:56` 문서, 게이트 함수 `_rx2_gate`). 배포 경로는
   `coarse_to_fine.py:177` 의 `torch.rand(n_new)*2π`. 저장소 주석 자신이
   *"`_linear_projection` 이 **이미 적합한 위상**을 출하 코드가 **버린다**"* 고 적었다.
   러너 8개의 셀 dict 에 실려 있으나 **결과 행에 0회 등장**. Rx-2 측정치 −0.0337(4/4).
5. **닫힌형 상한** — `w·cos(θ+φ) = a·cosθ + b·sinθ` ⟹ `k` 고정 시 `(a,b)` 에 **선형**
   (`spectral_gaussian.py:118-124` 의 forward 와 동일한 식). M1 이 실제 희소 LiDAR 관측 위에서
   `lstsq` 한 번으로 전역 ρ=0.999964 를 얻었다(**PREREG 없는 탐색적 측정**). 이 배치가 그것을
   **저장소 채점기 위에서** 다시 잰다.

### 1.1 ★설계 중 발견한 전제 결함 3건 (판정 전에 명시)

- **F-1 `BASE_CFG` 에 `lambda_l1` 이 없다.** frozen `tmp/rx2_init_2026-08-05/run_rx2b.py:BASE_CFG`
  는 λ 를 명시하지 않고 `run.py:DEFAULT_CONFIG` 폴백에 맡긴다. 그 폴백이 2026-08-11 에
  **1e-3 → 1e-4 로 배포**됐다(커밋 `40e22f6`). ⟹ *과거 4연속 널이 "λ=1e-3 조건"이었다는 것은
  그 배치들이 배포 이전에 돌았기 때문에만 참*이고, **오늘 같은 러너를 그대로 돌리면 λ=1e-4 가 된다.**
  ⟹ **이 배치의 모든 팔은 `lambda_l1` 을 명시**한다(1e-3 / 1e-4 / 0.0). (S7·S8 무효화와 같은 함정.)
- **F-2 배포 팔의 `FITPHI` 개입 지점이 조건부다.** `sequential_update._initialize_learned` 는
  *"the coarse-to-fine N-staging is bypassed"* 라고 스스로 적었다 ⟹ `init_method='auto'` 가
  `learned` 로 라우팅되면 **`coarse_to_fine._beamform_residuals`(FITPHI 사이트 ①)는 0회 호출**된다.
  살아 있는 것은 `sequential_update._adaptive_prune_grow`(사이트 ②)뿐이고, 그마저 성장 트리거
  (`grow_metric ≥ 0.05`)가 걸려야 한다. ⟹ ★**"처치 팔이 해당 코드를 0회 호출" 전례를 피하려면
  원자 단위 카운터가 필수**이고, 사이트 ①이 확실히 도는 `v2_konly` 짝(`KONLY` / `KONLY-FITPHI`)을
  **함께** 돌린다. (Rx-2 의 −0.0337 도 `v2_konly` 위에서 측정된 값이다.)
- **F-3 PRIMARY 는 "4연속 널이 λ 인공물인가"를 직접 묻지 않는다.** 지정된 PRIMARY 는
  `ORA-K-NOREG < ORA-K@L3` = *"λ 가 오라클 팔을 해친다"* 이다. 4연속 널은 **오라클 팔 vs 배포 팔**
  대비였으므로, 그 문장을 회복하려면 `ORA-K-NOREG < SHIP` 이 필요하다 ⟹ **`S-2` 로 사전등록**한다.
  PRIMARY 는 지정대로 유지하고, ⛔**PRIMARY 통과만으로 "4연속 널은 λ 인공물이었다"고 쓰지 않는다.**

---

## 2. 팔 정의 (동결)

공통 바탕 = frozen `tmp/rx2_init_2026-08-05/run_rx2b.py:BASE_CFG`
(`use_dispersion=True, use_confidence=True, adaptive_n=True, tracking_window=3,
prune_threshold=0.10, max_n=30`), **추적기는 출하 설정 그대로 ON**(`tracking_mode='w_phi'`,
`tracking_lr=0.005`, `tracking_steps=10`).

| # | 팔 | `init_method` | `n_gaussians` | `max_iters_per_stage` | **`lambda_l1`** | `RX2_GROW_FITPHI` | 역할 |
|---|---|---|---|---|---|---|---|
| 1 | **`ORA-K@L3`** | `oracle` | `[n_gt]` | `[900]` | **1e-3** | 0 | **PRIMARY 대조** (4연속 널의 λ) |
| 2 | **`ORA-K-NOREG`** | `oracle` | `[n_gt]` | `[900]` | **0.0** | 0 | **PRIMARY 처치** |
| 3 | `ORA-K@L4` | `oracle` | `[n_gt]` | `[900]` | **1e-4** | 0 | SECONDARY (현 출하 λ) |
| 4 | `ORA-K@L3-DUP` | = #1 과 **바이트 동일** | | | 1e-3 | 0 | 워커 누출 프로브 (혼잡 전용) |
| 5 | `SHIP` | `auto` (+`analytic_ensemble_k=3`) | `[3,8,15]` | `[200,200,500]` | **1e-4** | 0 | 배포 기준선 |
| 6 | `SHIP-FITPHI` | `auto` (+`analytic_ensemble_k=3`) | `[3,8,15]` | `[200,200,500]` | **1e-4** | **1** | SECONDARY (배포 위 수리) |
| 7 | `KONLY` | `v2_konly` | `[3,8,15]` | `[200,200,500]` | **1e-4** | 0 | 기제 대조 (사이트 ① 확실히 작동) |
| 8 | `KONLY-FITPHI` | `v2_konly` | `[3,8,15]` | `[200,200,500]` | **1e-4** | **1** | SECONDARY (기제 위 수리) |
| 9 | **`ORA-K-CVX`** | — (추정기 **미실행**) | `n_gt` 원자 | **경사하강 0 스텝** | **0.0** (무벌점 `lstsq`) | n/a | **상한** |
| 10 | `ZERO` | — | 0 원자 | 0 | n/a | n/a | 퇴화 필요조건 |

`loss_thresholds` 는 `[None]`(오라클 팔) / `[0.1,0.05,None]`(사다리 팔) — frozen 값 그대로.

### 2.1 `ORA-K-CVX` 정의 (동결)

1. 에피소드의 **GT 성분**에서 `(kx, ky)` 를 그대로 가져온다(`wave_components` 열 2·3, `ORA-K` 와 동일).
2. `ω_i = √(g·|k_i|·tanh(|k_i|·d))` — **분산관계로 유도**(출하 모델 `use_dispersion=True` 와 동일).
   ⚠GT 의 `ω`(열 1)는 **주 정의에 쓰지 않는다.** 별도 진단 `cvx_gtomega_*` 로만 병기한다.
3. **추정기가 보는 것과 같은 관측 집합** = `EpisodeDataLoader.get_init_window(tp_factor=2.0)` 의
   모든 프레임의 `(x_obs, y_obs, z_obs)` 를 쌓는다. ⛔**GT 격자로 적합하면 무효.**
4. `Φ = [cos θ | sin θ]`, `θ_i = kx_i·x + ky_i·y − ω_i·t` (2N 열) 위에서
   `np.linalg.lstsq(Φ, z, rcond=None)` **한 번**. 가중 없음(무가중 최소제곱), 벌점 없음.
5. `w_i = √(a_i²+b_i²)`, `φ_i = atan2(−b_i, a_i)` — `_linear_projection` 과 같은 변환.
6. **추적 없음.** 이 정적 해를 그대로 각 추적 프레임 시각 `t` 에서 GT 격자에 평가하고,
   **`compute_all_metrics` 로 다른 팔과 완전히 같은 채점**을 한다.
   ⟹ ⚠**CVX 는 추적기 이득을 못 받는다**(우리에게 유리한 방향이 아니다). 명시한다.
7. 결정론적이다(RNG 0회) ⟹ `est_seed=null`, 복제 1회.

### 2.2 조작 검사 (manipulation check) — 이 배치의 핵심 계측

**(a) λ 조작 검사.** 매 행에 `effective_config` 를 **객체에서 되읽는다**(의도 dict 가 아니라
`make_updater()` 가 만든 `SequentialUpdater` / `MedianEnsembleUpdater` 객체의 속성).
`effective_config['lambda_l1']` 가 팔의 의도 λ 와 다르면 그 행은 **위반**으로 표시되고
게이트 G-3 가 판정을 막는다.

**(b) FITPHI 조작 검사 — 원자 단위 카운터.**
`initializer._linear_projection` 과 `coarse_to_fine._linear_projection` 을 감싸 반환 위상을
float32 로 기록하고, `SpectralGaussianModel.add_gaussians` 를 감싸 **새로 추가된 각 원자의
`init_phi` 가 그 기록에 있는지**를 본다.

| 필드 | 뜻 |
|---|---|
| `fitphi_atoms` | **적합된 위상으로 심어진 원자 수** |
| `randphi_atoms` | 난수 위상으로 심어진 원자 수 |
| `fitphi_site_hits` | `_rx2_gate('RX2_GROW_FITPHI')` 호출을 **호출자 파일:행별로** 센 dict |
| `rx2_fitphi_env` | 워커 안에서 실측한 `os.environ['RX2_GROW_FITPHI']` |

**사전 규칙**: `*-FITPHI` 팔에서 `fitphi_atoms == 0` 인 행은 **개입 부재**다.
어떤 팔의 개입 부재 비율이 **≥50%** 이면 그 팔이 참여하는 SECONDARY 는 판정 없이
**`UNDECIDED-NO-INTERVENTION`** 으로 인쇄된다(게이트 G-4). OFF 팔은 `fitphi_atoms == 0` 이
**전 행에서** 성립해야 한다(아니면 누출 = G-4 실패).

---

## 3. 표본 · 시드 · 독립 단위

### 3.1 에피소드 (총 48, 신규 생성 0)

| 층 | 코퍼스 | 구성 | n |
|---|---|---|---|
| **혼잡 (PRIMARY)** | `data_indep_crowded_2026-08-08` | BF4/BF5 × 15/30comp × seed 20–23 | 16 |
| **혼잡 (PRIMARY)** | `data_s9_crowded_2026-08-11` | BF4/BF5 × 15/30comp × seed 48–51 | 16 |
| 희소 (SECONDARY) | `data_indep_sparse_2026-08-08` | BF4/BF5 × 2/5comp × seed 20–23 | 16 |

- 혼잡 **n = 32** (두 코퍼스 16+16). 앞 절반은 4연속 널이 측정된 바로 그 코퍼스(S3·S4·S5),
  뒷 절반은 그 이후 생성된 신선 코퍼스다. `corpus` 는 매 행에 기록하고 **16/16 분해를 서술로 병기**한다
  (PRIMARY 판정은 **풀링된 32** 위에서 한다 — 사전 동결).
- 파일 확장자 `.h5`, 접미사 `uniform_ra_rf_ds`. **새 `.h5` 생성 0회** (게이트 G-9).

### 3.2 시드

- **추정기 시드 = `1592, 1593, 1594`** (median-of-3). 원장 잔여 `[1592–1818]` → 소모 후 `[1595–1818]`.
- **충돌 검사(수행함, 2026-08-15)**: `tmp/**/*.jsonl` 전체에서 `seed|est_seed|filter_seed` 필드를
  훑어 1500–1900 구간의 실사용 최대값이 **1591**(S15)임을 확인 ⟹ **충돌 0**. 게이트 G-9 가 재검사한다.
- `ORA-K-CVX` · `ZERO` 는 **결정론적**이므로 시드 축이 없다(`est_seed=null`, 복제 1).

### 3.3 ★독립 단위 = **에피소드** (가짜복제 금지)

- 팔·에피소드마다 **3 시드의 중앙값**을 먼저 취하고, **그 다음** 에피소드 단위로 부호검정한다.
- ⛔**episode × seed 를 표본으로 세지 않는다.** S4 가 `48/48` 로 보고했다가 **오라클 두 팔의
  시드 편차가 정확히 0** 이어서 3배 가짜복제로 정정된 사건(정직한 값 `16/16`)이 이 규칙의 이유다.
- 오라클 팔에서도 `(w, φ)` 초기값은 `torch.rand` 로 시드 의존이므로 편차가 0 이 아닐 수 있다.
  **그래도 n 은 에피소드 수다.** 게이트 G-6 이 팔별 시드 퍼짐을 인쇄하고, 퍼짐이 정확히 0 인 팔은
  `deterministic=True` 로 표시된다(그 경우 median-of-3 은 **가짜복제**이며 n 에 기여하지 않는다).

---

## 4. 가설 · 문턱 · 검정력 (결과 전 동결)

### 4.1 ★PRIMARY (**하나뿐**)

> **P-1** — 혼잡 층에서 `nRMSE(ORA-K-NOREG) < nRMSE(ORA-K@L3)`.

- 통계량: **부호검정**(에피소드별 중앙값 차의 부호), **단측**(방향 = 음, 사전 지정).
- n = **32**, 문턱 = **≥ 23/32**.
  - 정확 이항 단측 꼬리: `P(X≥23 | n=32, p=0.5) = 0.010030` ≤ α=0.025 ✅
  - 바로 아래(`X≥22`)는 `0.025048` > 0.025 ⟹ **22/32 는 통과가 아니다**(하나 차이 사건 사전 고지).
- **사전 검정력**(부호검정, 팔당 승률 p):
  | p | 0.75 | **0.80** | 0.85 | 0.90 |
  |---|---|---|---|---|
  | power | 0.737 | **0.910** | 0.984 | 0.999 |
  - **MDE**: power 0.80 ⟹ p ≥ **0.766**; power 0.90 ⟹ p ≥ **0.796**.
  - 선행 근거로 기대되는 p 는 S5 의 λ 효과(오라클 팔 16/16)에서 0.85–0.95 대 ⟹ **검정력 충분**.
- **사망 조건**: 23/32 미달 ⟹ 판정 = **`NOT-A-LAMBDA-ARTIFACT`**
  (*"λ 를 1e-3 에서 0 으로 내려도 오라클 팔은 구해지지 않는다"*). 이 경우 M4 의 **잔여-벽** 가설이
  남고, `ORA-K-CVX` 값이 벽의 위치를 지정한다.

### 4.2 SECONDARY (전부 **SECONDARY 라고 라벨**하고, 다중성 보정은 Holm-4 로 명시)

| ID | 대비 | 층 | n | 문턱 | 질문 |
|---|---|---|---|---|---|
| **S-1** | `ORA-K-NOREG < ORA-K@L4` | 혼잡 | 32 | 23/32 | 출하 λ 도 오라클 팔을 해치는가 (M4 의 "6/6 나쁨" 재시험) |
| **S-2** | `ORA-K-NOREG < SHIP` | 혼잡 | 32 | 23/32 | ★**4연속 널이 λ 인공물이었는가** (F-3) |
| **S-3** | `SHIP-FITPHI < SHIP` | 혼잡 | 32 | 23/32 | 배포 위에서 위상 수리가 버는가 |
| **S-4** | `KONLY-FITPHI < KONLY` | 혼잡 | 32 | 23/32 | 기제 위에서 위상 수리가 버는가 (Rx-2 −0.0337 재시험) |

- Holm-4: p 값을 오름차순으로 α/4, α/3, α/2, α 와 비교(α=0.025 단측). **부호검정 정확 p 를 쓴다.**
- 희소 층(n=16, 문턱 **13/16**, 정확 α=0.01064, MDE power .80 ⟹ p≥0.853)에서 같은 4개 대비를
  **서술적으로** 병기한다 — 희소는 **확증 층이 아니다**.
- **`ORA-K-CVX` 는 가설검정 대상이 아니다.** **상한 서술**이다: 혼잡·희소 각각의
  중앙값 `nRMSE` 와, 각 팔이 그 상한까지 남긴 격차(`arm − CVX`)를 표로 인쇄한다.

### 4.3 퇴화 필요조건 (discipline 10)

> 어떤 팔이든 해당 층에서 **`ZERO` 를 이기는 에피소드가 ≤ 50%** 이면 그 팔은 **`DEGENERATE`** 로
> 표시되고, 그 팔이 들어간 어떤 우월성 문장도 쓰지 않는다(수치는 그대로 인쇄).

---

## 5. 지표 · 집계 · 크기 3규약

- **주 지표** = `avg_nRMSE` = `run_estimation` 의 `frame_results` 전 프레임 `nRMSE` 산술평균.
  `frame_results` 는 `get_tracking_frames(tp_factor=2.0)` 만 담으므로 **이미 `t > 2·Tp` 로 잘려 있다.**
  `ORA-K-CVX` · `ZERO` 도 **같은 프레임 집합·같은 `compute_all_metrics`** 로 채점한다.
- **집계 순서(동결)**: ① 팔×에피소드에서 시드 3개의 **중앙값** → ② 에피소드 단위 부호검정.
  ⛔평균-후-변환 금지, ⛔프레임 단위 풀링 금지.
- ★**크기 3규약 의무 병기** (셋 다 안 쓰면 크기 인용 금지):
  1. **A** 절대 중앙값 차: `median_ep( x_treat − x_ctrl )` [nRMSE 단위]
  2. **B** 에피소드별 상대차의 중앙값: `median_ep( (x_treat − x_ctrl) / x_ctrl )` [%]
  3. **C** 풀링 상대차: `(mean_ep x_treat − mean_ep x_ctrl) / mean_ep x_ctrl` [%]
- **셸 분해 의무** (`SUPPORT-TRADEOFF` 7회 재현): 전역 승리는 반드시 셸로 분해해 병기한다.
  셸 경계는 `run.py` 가 이미 프레임마다 계산하는 값을 그대로 쓰되 **키에 `obs:` 접두어**를 붙인다:
  `obs:0_5, obs:5_10, obs:10_20, obs:20_30, obs:30_50, obs:50_75, obs:75_100, obs:100_150, obs:150_inf`
  (관측점까지의 최소거리 기준). **모든 팔**(`ORA-K-CVX`·`ZERO` 포함)이 9키 전부를 기록한다.
- 부가 기록(서술용): `decorr_mean`, `shrink_mean`(프레임별 직교분해), `obs_resid_rms`,
  `elapsed_ms`(⛔**지연 인용 금지** — 이 상자의 시계는 별도 재측정 대상).

---

## 6. ★필수 기록 필드 (DCC — 없으면 판정 인쇄 없이 종료)

**모든 팔 공통** (`REQUIRED_COMMON`):
`batch, arm, episode, corpus, stratum, bf, ncomp, ep_seed, est_seed, n_gt,
avg_nRMSE, zero_nRMSE, n_frames_scored, shell_rmse(obs: 9키), zero_shell(obs: 9키),
shell_frac(obs: 9키), lambda_l1_intent, effective_config, code_md5, code_files_n,
git_head, git_dirty_n, wall_s, deterministic`

**추정기 팔 추가** (`REQUIRED_EST` — #1,2,3,4,5,6,7,8):
`init_method_intent, init_method_resolved, lambda_l1_effective, rx2_fitphi_env,
fitphi_atoms, randphi_atoms, fitphi_site_hits, N_final, decorr_mean, shrink_mean,
obs_resid_rms_mean, elapsed_ms_mean`

**`ORA-K-CVX` 추가**: `cvx_n_obs, cvx_cond, cvx_rank, cvx_window_resid_rms,
cvx_truth_window_resid_rms, cvx_gtomega_nrmse, cvx_omega_source`

> **DCC 규칙**: `analyze_w3.py` 는 위 목록의 키가 **한 팔이라도 한 행에서 빠지면**
> 어떤 사망조건·판정도 인쇄하지 않고 `SCHEMA-FAIL` 로 종료한다.
> (S13/HOSGATE 가 기준선 셸 미기록으로 두 번 죽은 규율.)

---

## 7. 게이트 (판정 전에 전부 통과해야 한다)

| ID | 내용 | 실패 시 |
|---|---|---|
| **G-1** | 표본 48 에피소드 파일 존재·이름 파싱·층 배정 일치 | 중단 |
| **G-2** | DCC 스키마 검사 (§6) — 전 행 | `SCHEMA-FAIL`, 판정 인쇄 없음 |
| **G-3** | ★λ 조작 검사: `effective_config['lambda_l1']`(객체 되읽기) == 팔 의도 λ, 위반 0 | 중단 |
| **G-4** | ★FITPHI 조작 검사: ON 팔 `fitphi_atoms>0` 비율 ≥50%; OFF 팔 전 행 `fitphi_atoms==0` | 해당 SECONDARY → `UNDECIDED-NO-INTERVENTION` |
| **G-5** | 누출 프로브: `ORA-K@L3-DUP` 이 `ORA-K@L3` 과 **\|Δ\|=0** (혼잡 32×3) | 중단 |
| **G-6** | 팔별 시드 퍼짐 인쇄; 퍼짐 0 인 팔은 `deterministic=True` 로 표시 | 표시만(중단 아님) |
| **G-7** | `code_md5` 가 전 행에서 **단일 digest** | 중단 |
| **G-8** | ★`ZERO` 가 팔·시드에 무관하게 동일(퍼짐 0) | 중단 |
| **G-9** | 시드 원장 충돌 0 · 신규 `.h5` 0 · 소모 시드 == {1592,1593,1594} | 중단 |
| **G-10** | 에러 행 0 (`error` 키 존재 행 0) | 중단 |
| **G-11** | `ORA-K-CVX` 자기검증: `cvx_rank == 2·n_gt` 이고 `cvx_window_resid_rms` 가 진실의 창 잔차 이하 | 표시만 |

---

## 8. 판정표 (사전 동결 — 결과를 보고 고르지 않는다)

| P-1 | S-2 | 판정 이름 | 뜻 |
|---|---|---|---|
| 통과 | 통과 | **`LAMBDA-ARTIFACT`** | λ 가 오라클 팔을 죽였고, 고치면 배포 팔도 이긴다 ⟹ ⛔"k 축 소진" 인용 재검토 |
| 통과 | 미달 | **`LAMBDA-HURTS-ONLY`** | λ 는 해롭지만 그것만으론 4연속 널을 못 설명한다 |
| 미달 | — | **`NOT-A-LAMBDA-ARTIFACT`** | 4연속 널은 λ 인공물이 아니다 ⟹ M4 의 잔여-벽 가설이 남는다 |

`ORA-K-CVX` 는 위 어느 칸에서도 **해석의 자**로 병기한다:
- `CVX ≈ 0` 이면 ⟹ 정보도 조건수도 아니고 **솔버**가 벽이다.
- `CVX` 가 여전히 나쁘면 ⟹ 벽은 **표현/모델차수**이고 위상 서사는 축소된다.

---

## 9. 범위 밖 (이 배치가 **말하지 않는** 것)

1. ⛔**배포 권고 없음.** `ORA-K*` 는 전부 GT 를 쓰는 연구 팔이다. 배포 후보는
   `SHIP-FITPHI` 하나뿐이고, 그것도 S-3 통과 + 퇴화 필요조건 통과 + 셸 분해에서 거래 없음일 때만.
2. ⛔**실측 일반화 없음.** 합성 코퍼스 전용.
3. ⛔**"비볼록성" 이라는 이름을 사지 않는다** — 다중 재시작 팔이 없다. 안전한 이름은
   **"위상 하위공간의 최적화 실패"**.
4. ⛔**지연(ms) 인용 금지.**
5. ⚠**`ORA-K` 는 3요인 묶음**이다(오라클 k + `n_gaussians=[n_gt]` 단일단계 + 900 스텝 한 단계).
   `SHIP` 과의 대비(S-2)는 **묶음 대비**이고 그렇게 라벨한다.
6. ⚠**추적기 ON.** S3/S4 의 `ORA-K-NT` 는 추적기 OFF 셀이었다 ⟹ 이 배치의 `ORA-K@L3` 은
   그 셀의 **바이트 재현이 아니다**(frozen `run_rx2b.CELLS['ORACLE-K']` 의 재현이다).
7. ⚠**`ORA-K-CVX` 는 무가중·무추적**이다. 출하 손실은 confidence 가중을 쓴다 ⟹ 두 방향의
   비대칭을 각각 명시했다(§2.1).

---

## 10. 실행

```bash
cd /home/vexit/wavesplat/wavesplat
WORKERS=8 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full 2>&1 | tee tmp/W3_SOLVER_2026-08-15/run.log
python3 tmp/W3_SOLVER_2026-08-15/analyze_w3.py | tee tmp/W3_SOLVER_2026-08-15/VERDICT.log
```

- `results.jsonl` 은 **append + 재개**(완료 `job_id` 스킵). 스모크는 `smoke_*.jsonl` 로 **분리**되며
  `analyze_w3.py` 는 `results.jsonl` **만** 읽는다.
- 비용 상한이 걸리면 **에피소드를 줄이지 않는다.** 줄일 수 있는 것은 SECONDARY 팔뿐이고
  (순서: `ORA-K@L3-DUP` → `SHIP`/`SHIP-FITPHI` → `KONLY`/`KONLY-FITPHI` → `ORA-K@L4`),
  줄이면 해당 SECONDARY 는 `NOT-RUN` 으로 인쇄된다. **PRIMARY 4팔(#1,#2,#9,#10)은 절대 줄이지 않는다.**

---

## 11. 사전등록 이후 변경

없음. 변경이 필요하면 `AMENDMENT_N.md` 를 새로 만들고 md5 를 `FROZEN_md5.txt` 에 추가하며,
**`results.jsonl` 을 몇 행 읽은 시점인지 명시**한다. PREREG.md 자체는 수정하지 않는다.
