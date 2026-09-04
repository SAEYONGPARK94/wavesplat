# `T15-HOMEREG` 배관 조사 — *"`A-MAPSCALE` 를 우리 코퍼스에 걸면 해치는가"* 를 재기 위한 사전 실측

작성 2026-09-05 · **런 1회**(§5 단가 실측, 메모리 캡 안) · 그 밖은 전부 **읽기·동결자료 재집계**
⛔이 문서는 **사전등록·러너·게이트가 아니다**. 설계자가 쓴다.
⚠**열람 범위 = 주장 범위**. 확인 못 한 것은 「미확인」이라 적었다.

경로는 별도 표기가 없으면 `/home/vexit/wavesplat/wavesplat/` 기준이다.

---

## 0. 한 줄 요약

| 질문 | 답 |
|---|---|
| 자국 코퍼스 | `data/*.h5` **110개**(레거시 생성기) · 그와 **다른** 고정생성기 코퍼스 **128개**(`_ra_rf_ds`, `data_s8_*`/`data_s9_*`)가 헤드라인의 실제 기반 |
| 로딩·채점 | `wavesim/data_writer.py:163 load_episode` → `wavesplat/data_loader.py:41` → `wavesplat/run.py:200 run_estimation` (초기창 `t≤2·Tp`, 채점창 `t>2·Tp`) |
| 처치 도달 | ✅**도달한다.** 라우터 `learned` 전건 · 실측 게이트 **미발화** · `_bmap_from_obs` 런당 **1회** |
| 단가 | **7.81 s/런**(실측, 1스레드, GPU 제안) ⟹ 110ep × 3팔 ≈ **43분 직렬** |
| 앵커 | ★**`B-15` 동결 jsonl 과 오늘 프로브가 `avg_nRMSE` **비트 동일**(0.15821650810539722) — 12일 격차 |

---

## 1. 자국 코퍼스의 실체

### 1.1 `data/` = 110 에피소드 (레거시 생성기)

`ls data/*.h5` = **110** 파일, 각 **약 11.6 MB**(총 ≈ 1.28 GB). ⛔**git 미추적**
(`.gitignore:119` 의 `*.h5` 가 잡는다 — `git check-ignore -v` 로 확인). `wavesplat/data/` 에서
추적되는 것은 `agulhas_wass_2017/` 의 3 파일뿐.

파일명 규약 = `episode_{BF}_{N}comp_seed{S}_{raydrop}.h5`, **110/110 이 이 규약을 지킨다**(예외 0).

| 축 | 분포 |
|---|---|
| 해상상태 | BF2 10 · BF3 30 · **BF4 42** · BF5 20 · BF6 4 · BF7 4 |
| 성분수 | 1comp 32 · 2comp 10 · 5comp 22 · 15comp 22 · **20comp 1** · 30comp 23 |
| raydrop | `uniform` 60 · `slope` 50 |
| 시드 | 대부분 `seed0`–`seed4`, 예외 **2건**: `seed7`(BF4_20comp) · `seed42`(BF4_30comp) |

`(sea_state, Tp, Hs, depth)` 는 완전히 결정적이다 — BF2 (4.0, 0.3) · BF3 (5.5, 0.6) ·
BF4 (7.0, 1.2) · BF5 (9.0, 2.0) · BF6 (10.0, 3.0) · BF7 (12.0, 5.0), **depth 는 110/110 이 50.0**.
LiDAR 구성은 110/110 동일(`channels 152 · height 5.0 · scan_rate 10.0`).

**격자·프레임**: `frames/gt_wave_field` = `(F, 60, 80)`, `gt_grid_x` = 10 → 250 m (80점),
`gt_grid_y` = −100 → +100 m (60점), dt = 0.1 s.
⚠**`F` 는 균일하지 않다** — 301 이 108개, 그리고 **2개가 짧다**:

| 파일 | F | t_end | 초기창 `2·Tp` | **채점 프레임** |
|---|---|---|---|---|
| `episode_BF4_20comp_seed7_uniform.h5` | 26 | 5.0 s | 14.0 s | **0** |
| `episode_BF4_30comp_seed42_uniform.h5` | 101 | 10.0 s | 14.0 s | **0** |

⛔★**이 둘은 `run_estimation` 이 `avg_nRMSE = nan` 을 낸다**(`run.py:498` 의 `if frame_results:`
가 거짓 → `run.py:502` 미실행). **NaN 은 조용히 「패배」가 된다** — T15 게이트는 `T11 G-FINITE`
부류의 유한성 검사를 반드시 넣거나 이 둘을 모집단에서 **사전에** 배제해야 한다.
⚠`episode_BF4_30comp_seed42_uniform.h5` 는 **`run.py:6` 의 사용 예시로 인쇄된 바로 그 파일**이다.

유효 반환점 수는 프레임당 **약 50개**(패딩 배열 폭 65–103, 유효율 0.68–0.74).
110 에피소드의 **채점 프레임 총합 = 17,340**(평균 157.6/에피소드).

⛔**레거시 생성기다.** `config` 속성 집합이 110/110 에서
`{Hs, Tp, depth, gamma, n_components, sea_state}` 뿐이고 **`random_amplitudes` ·
`random_frequencies` · `decorrelate_stream` 키가 없다**. 이것이
`reference_citation_bans.md:167` 이 *"레거시 생성기 산출물 ⟹ 위상/방향 draw 가 시드 0–4 뿐 ⟹
**클러스터 단위 바닥 p = 2⁻⁵ = 0.031**"* 이라 적은 바로 그 결함이다.
⟹ ★**T15 를 이 코퍼스에서만 돌리면 「에피소드 단위 p」와 「draw-클러스터 p」를 병기해야 한다.**

⚠2026-06-28 메모 `data_corrupt_slope_files.md`(*"다수 `_slope.h5` 가 HDF5 서명 없음"*)는
**오늘 실측에서 재현되지 않았다** — 110/110 이 h5py 로 열리고 mtime 이 전부 2026-08-31 06:10 이다
(재생성 또는 복구된 것으로 보인다; **경위는 미확인**).

### 1.2 혼잡/희소 층 규약 — ★그 규약의 코퍼스는 `data/` 가 **아니다**

저장소에서 「혼잡/희소」의 정본 정의는
**`tmp/W15_DEPLOY_VERIFY_2026-08-22/corpus.json`** 이고, `gates_w15.py:9-11` 이 그것을 읽는다.
`tmp/S_ENS_2026-09-01/gates_sens.py:24` 가 그 둘을 그대로 임포트한다.

| 층 | n | 구성 |
|---|---|---|
| `crowded` | 64 | `BF4/BF5 × 15comp/30comp × seed32–63`, 전부 `_uniform_ra_rf_ds` |
| `sparse` | 64 | `BF4/BF5 × 2comp/5comp × seed32–63`, 전부 `_uniform_ra_rf_ds` |

⛔**그 128 파일은 `data/` 에 없다.** 실제 위치는 `data_s8_crowded_2026-08-10/`(seed32–47) ·
`data_s9_crowded_2026-08-11/`(seed48–63) 과 sparse 짝이다. 이름의 `_ra_rf_ds` =
`random_amplitudes` + `random_frequencies` + `decorrelate_stream` = **수정 생성기**이고,
`config` 속성이 21개(위 세 키 + `mesh_verdict`·`lambda_peak` 등)로 레거시와 다르다.
Tp 는 {7.0, 9.0} 두 값뿐.

⟹ ★★**「자국 합성 코퍼스」는 하나가 아니다. T15 는 어느 쪽인지 사전등록에 못박아야 한다.**
- **(a) `data/` 110** — 넓다(BF2–BF7, 1–30comp, uniform+slope). ⛔레거시 draw 결함 · ⛔층 규약 없음
  (직접 정의해야 한다) · ⛔NaN 에피소드 2개.
- **(b) `_ra_rf_ds` 128** — 헤드라인(`B-1`·`S13`·`W15`·`W16`·`S-ENS`)이 사는 곳. 층 규약이 이미 있고
  수정 생성기다. ⛔BF4/BF5 · uniform 만 · Tp 두 값 = **좁다**.

### 1.3 A-MAPSCALE 이 위험해 보이는 이유 — 참 파수 밴드 실측

`config/wave_components` 열 순서는 `[A, ω, kx, ky, φ, Q]`(`data_loader.py:57-64`).
`|k| = hypot(kx, ky)` 를 전수 집계했다(런 0회):

| 코퍼스 | 에피소드 중앙 `|k|` 중앙값 | `|k|max` 중앙/최대 | `|k|>0.3` 있는 에피소드 | `|k|>0.6` 있는 에피소드 |
|---|---|---|---|---|
| `data/` 110 | **0.2515** | 0.6211 / **1.0061** | **95/110** | **62/110** |
| `_ra_rf_ds` crowded 64 | 0.1973 | 0.5550 / 0.7369 | 64/64 | 32/64 |
| `_ra_rf_ds` sparse 64 | 0.1981 | 0.4325 / 0.7360 | 51/64 | 13/64 |
| **제3자(T12 `kp_true`)** | **0.0415** | — | — | — |

해상상태별 `data/`: BF2 kmed **0.7703** · BF3 0.4074 · BF4 0.2515 · BF5 0.1522 · BF6 0.1232 · BF7 0.0856.
(BF2/BF3 의 `1comp` 은 단일 성분이 첨두 주기가 아니라 스펙트럼에서 뽑혀 고파수에 앉는다 —
예: `BF2_1comp_seed0_slope` 는 ω=2.749 ⟹ |k|=0.770.)

⟹ **자국의 참 파수는 제3자보다 4–6배 높다.** 그런데 T12 실측에서
`SHIP` 의 선택 원자 중앙 `|k|` 는 **0.4271**, `A-MAPSCALE` 은 **0.1003** 이다
(`tmp/T12_ISOLATE_2026-09-04/results_t12.jsonl` 재집계).
⟹ ★**처치가 우리 참 밴드(0.15–0.25, 꼬리 1.0)의 아래로 제안을 끌어내리는 방향**이고,
`KLIM 0.6→0.3` 은 **95/110 에피소드에서 참 성분이 맵 밖으로 나간다**는 뜻이다.
⛔단 이것은 **기제 가설이지 측정이 아니다** — 자국에서의 `kmed_sel` 은 **미측정**이다
(오늘 프로브는 그 계수기를 달지 않았다).

⚠**교락 공시 후보**: 배포 사전 `mixed_v1` 의 학습 장면 분포가 `tmp/mixed_retrain_2026-07-26/gen_mixed.py:78-83`
에서 **`ANCHORS = ((0.3,4.0),(0.6,5.5),(1.2,7.0),(2.0,9.0))` = 자국 BF2–BF5 의 (Hs,Tp) 그 자체**이고
(장면 50%가 이 앵커 ±15%), 관측 기하 도너는 같은 파일 `:66-71` 의 `data_finemesh_*` LiDAR 스윕창이다.
그리고 `:58` 이 **`K_LIM_HARD = 1.1  # frozen: stats['K_LIM']`** 을 못박는다 —
즉 `A-MAPSCALE` 의 `K_LIM 1.1→0.55` 는 **학습 시 인코딩과 어긋난 디코딩**이다.
⟹ T15 의 널/음성은 *"자국은 사전의 학습 분포다"* 와 분리되지 않는다. **공시 의무.**

---

## 2. 로딩·채점 경로 (배포 경로)

```
wavesim/data_writer.py:163   load_episode(filepath) -> dict          # h5 전량 읽기
   ↑
wavesplat/data_loader.py:41  EpisodeDataLoader.__init__              # self.data = load_episode(...)
wavesplat/data_loader.py:84  __getitem__ -> FrameData                # 유효 마스크 적용
wavesplat/data_loader.py:99      gt_field = torch.tensor(gt)         # ★항상 붙는다(비영)
wavesplat/data_loader.py:106 get_init_window(tp_factor=2.0)          # t <= 2*Tp
wavesplat/data_loader.py:124 get_tracking_frames(tp_factor=2.0)      # t >  2*Tp
   ↑
wavesplat/run.py:200         run_estimation(h5_path, config, verbose)
```

`run_estimation` 의 요지(파일:줄):

| 줄 | 내용 |
|---|---|
| `run.py:214-217` | `cfg = DEFAULT_CONFIG.copy(); cfg.update(config)` ⟹ ★**호출자 dict 은 변형되지 않는다**(제안 원자 캐시 누출 없음) |
| `run.py:224` | `loader = EpisodeDataLoader(h5_path, device=cfg['device'])` |
| `run.py:225` | `cfg['depth'] = loader.depth` ⚠**`load_episode` 는 `depth` 를 반환하지 않는다** ⟹ `data_loader.py:46` 의 `.get('depth', 50.0)` 폴백이 항상 먹어 **`depth` 는 언제나 50.0**. 자국 h5 는 실제로 50.0 이므로 **오늘은 일치**하지만, ⛔다른 수심의 자료를 넣으면 **조용히 50 으로 돈다**(`B-2` 수심 사건과 같은 부류). 같은 이유로 `data_loader.py:48 raydrop_model` 도 항상 `'unknown'` |
| `run.py:235` | `init_frames = loader.get_init_window(tp_factor=2.0)` |
| `run.py:242` | `updater = make_updater(cfg)` |
| `run.py:243` | `init_result = updater.initialize(init_frames, x_range=(gx[0], gx[-1]), y_range=(gy[0], gy[-1]))` |
| `run.py:256-257` | `has_gt = gt_field is not None and abs(gt).max() > 1e-10` (**마지막 초기창 프레임만** 본다) |
| `run.py:272` | `tracking_frames = loader.get_tracking_frames(tp_factor=2.0)` |
| `run.py:348` | `update_result = updater.update(frame)` |
| `run.py:351` | `pred = updater.predict_field(grid_x, grid_y, frame.t)` |
| `run.py:356` | `frame_metrics = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true, obs_x, obs_y)` |
| `run.py:498-503` | 채점 프레임 평균 → `avg_nRMSE` 등 |

**관측이 만들어지는 곳**: `data_loader.py:85-88` — `lidar_valid_mask[idx]` 로 패딩을 걷어내고
`lidar_points[idx][mask]` 의 `(x, y, z)` 를 `FrameData.x_obs/y_obs/z_obs` 로,
`lidar_confidence` → `confidence`, `lidar_incidence_angles` → `incidence_angles`.
⟹ ★**관측 잡음·차폐·raydrop 은 전부 생성 시점에 h5 안에 굳어 있다**; 추정 경로는 재샘플링하지 않는다.

### 2.1 ★본보기 러너 — 자국 코퍼스에서 학습 팔을 돌린 기존 배치

**1순위 본보기 = `tmp/B15_RAYDROP_2026-08-24/`** — 이것이 **`data/` 110 코퍼스 위에서 배포
학습 팔을 돌린** 배치다.
- `run_b15.py:52-64 run_ship()` — `cfg = dict(DEFAULT_CONFIG); cfg['seed'] = est_seed;
  run_estimation(ep['path'], config=cfg)`. **팔 정의가 빈 dict 이 아니다**(★`gates_b7` 사고의 처방을 지킴).
- `run_b15.py:28-38` — 모집단을 `data/` 에서 **파일 존재로 구성**한다(`os.path.exists`).
- `run_b15.py:41-50 obs_stats()` — 개입 계수기(반환점 수·평균 입사각·`gt_hash`)를 **h5 되읽기**로 만든다.
- `run_b15.py:68-75 run_spec()` — 기준선은 `tmp/S16_BASELINE_2026-08-14/run_s16.py::run_letkf` 를 임포트.
- 산출: `results_b15_{0..5}.jsonl`(6 샤드) · `gates_b15.py` · `analyze_b15.py` · `verdict_b15.json` · `FROZEN_md5.txt`.

**2순위 본보기 = `tmp/S_ENS_2026-09-01/run_sens.py`** — 팔이 노브 하나만 다른 무해성/이득 시험의
최신 형식. `run_sens.py:60-93 run_one()` 이 `gates_t5.assert_shipping_defaults()` 로 **배포 핀을
런 안에서** 검사하고, 행에 `auto_selected`·`ensemble_k`·`code_md5`·`git_head`·`wall_s` 를 싣는다.
⚠단 이 배치의 코퍼스는 §1.2(b) 128개다.

**3순위 = `tmp/T12_ISOLATE_2026-09-04/run_t12.py`** — ★**처치 구현의 정본**.
`Arm.__enter__`(`:87-170`)이 `BN.KLIM` → `BN.kgrid()` 재호출 → `LI._load_cond` 래핑 순으로 걸고
`__exit__`(`:172-176`)이 역순으로 되돌린다. **`read_knobs()`(`:48-67`)가 살아있는 모듈 상태를
되읽어** `G-KNOB`/`G-RESTORE` 가 대조한다.
⟹ ★T15 는 **`Arm` 을 그대로 가져오되 채점만 `run_estimation` 으로 갈아끼우면 된다**
(T12 의 `estimate()` 는 `initialize()` 만 하고 단일 프레임을 채점한다 — **추적 없음**).

---

## 3. 채점 규약

**정답**: `frames/gt_wave_field` = `(F, 60, 80)`, 축은 `frames/gt_grid_x`(80) · `frames/gt_grid_y`(60).
`data_loader.py:79` 가 통째로 읽고 `:99` 가 프레임마다 `FrameData.gt_field` 로 붙인다.

**정규화**: `evaluation/metrics.py:32 nrmse(pred, gt, Hs) = rmse / max(Hs, 1e-6)`,
`Hs = loader.Hs_true`(= `gt_params.attrs['Hs_true']`, 자국은 BF 등급의 명목값).
⛔⚠`metrics.py:33-46` 의 축자 경고: **합성은 `Hs`, 실측 하네스는 `σ` 로 나눈다**(비 3.99–4.06).
⟹ **자국 nRMSE 를 실측 nRMSE 와 직접 비교 금지.**

**셸(거리 분해)은 두 종류이고 다른 양이다** — ⚠접두어를 반드시 적을 것:

| 접두어 | 정의 | 코드 |
|---|---|---|
| `ship:` (근/원) | **격자 x 좌표** 기준 `x < 80 m` = near, `x ≥ 80 m` = far | `metrics.py:98-122 rmse_spatial_split(split_distance=80.0)` → `metrics.py:433 RMSE_near` / `:434 RMSE_far` |
| `obs:` (9고리) | **최근접 관측점까지의 거리** `[0,5,10,20,30,50,75,100,150,inf]` | `run.py:432 _edges` → `run.py:440 shell_rmse` · `:441 shell_frac` |

✅**둘 다 요약에 집계된다**: `run.py:595 avg_RMSE_near`(2026-08-19 수리) ·
`run.py:604 avg_shell_rmse` · `run.py:610 shell_frac`(2026-08-22 수리).
오늘 프로브가 `avg_RMSE_near = 0.1370` 과 9개 셸 키 전부를 되읽어 **두 수리가 살아 있음을 확인**했다.
그 밖에 `run.py:446-453` 이 반경 5–150 m 의 `RMSE_obs_r*` / `RMSE_unobs_r*` 를 프레임마다 채우지만
**요약이 집계하지 않는다**(`frame_results` 를 통째로 들고 있어야 접근 가능).

**제로필드 기준**: ⛔**`run_estimation` 은 `zero_nRMSE` 를 반환하지 않는다** —
`B15/run_b15.py:61` 이 그것을 읽으려 하고 실제로 `null` 이 들어간다(전 SHIP 행 확인).
정본 정의는 `tmp/S16_BASELINE_2026-08-14/run_s16.py:240,266`:
`zero_nRMSE = mean_{t>2·Tp} [ sqrt(mean(gt²)) / Hs ]` — ★**채점창이 `run_estimation` 과 동일**하다.
⟹ T15 가 제로 기준을 쓰려면 이 식을 **h5 에서 직접** 계산하면 되고 추정기 런이 필요 없다.

---

## 4. ★처치가 자국 경로에서도 도달하는가 — **도달한다**

### 4.1 라우팅 (코드)

```
run.py:242  make_updater(cfg)
   └ sequential_update.py:2261  init_method=='auto' && analytic_ensemble_k(3) > 1
        ⟹ MedianEnsembleUpdater  (★래퍼가 만들어진다)
run.py:243  updater.initialize(init_frames, ...)
   └ sequential_update.py:2128  MedianEnsembleUpdater.initialize
        └ :2134  self._primary.initialize(...)               # SequentialUpdater
             ├ :776  if self._init_method_requested == 'auto':
             ├ :795  ... and not _frames_have_gt(frames):    # ← 실측 게이트
             │        ⛔자국은 gt 가 붙으므로 **False** ⟹ 게이트 미진입
             ├ :844  self._auto_real_gate = False
             ├ :845  self.init_method = self._auto_select_init(frames)
             │        └ :1096 thresh = auto_switch_thresh = 0
             │           :1108 propose_atoms(...)            # ★GPU 확산 제안
             │           :1122 cfg['learned_atoms'] = {...}  # 캐시
             │           :1129 chosen = 'learned' if n_sel >= 0  ⟹ **항상 learned**
             └ :850  if self.init_method == 'learned': _initialize_learned(...)
                        └ learned_init.py:346-349 cache hit ⟹ propose_atoms 재호출 없음
        └ :2151  k_eff = self._effective_k('learned', real_gate=False)
                  :2117 learned_ensemble_k = 1 ⟹ **k_eff = 1 ⟹ 앙상블 없음**
```

`_frames_have_gt`(`sequential_update.py:411-428`)의 독스트링이 축자로 적는다:
*"EpisodeDataLoader always attaches a non-zero field."* ⟹ **자국은 구조적으로 합성 경로**다.

### 4.2 실측 (오늘 프로브 1런, `cost_probe.json`)

```
auto_selected     = "learned"     ← 라우터가 학습을 골랐다
auto_real_gate    = false         ← ★실측 게이트 미발화
auto_n_selected   = 14            ← _auto_select_init 이 실제로 돌았다(제안 성공)
n_gaussians       = 14            ← 그 원자가 그대로 모델이 됐다
ensemble_k        = 1             ← 래퍼는 있으나 멤버 1개(배포 기본값 그대로)
varpro_fired      = true          ← 배포 VARPRO 발화(1.354 s / init 1.409 s)
init_iters        = 0             ← learned_init_iters=0 (08-22 배포) 확인
```

`auto_n_selected` 는 `sequential_update.py:1130` 에서만 설정되고, 그 줄에 닿으려면
`:1108 propose_atoms` 가 성공해야 한다(실패 시 `:1117` 가 `None` 을 넣고 `v2_konly` 로 떨어진다).
`propose_atoms` 는 `learned_init.py:151` 에서 **`Bmap = cd._bmap_from_obs(Ps, zf)`** 를 부른다.
⟹ ✅**`_bmap_from_obs` 는 자국 경로에서 런당 정확히 1회 불린다**(제안은 `:1122` 에 캐시되고
`learned_init.py:346-349` 가 그것을 재사용하므로 두 번 돌지 않는다).

### 4.3 처치 훅이 실제로 그 호출에 걸리는가 — ✅걸린다

| 훅 | 경유 | 확인 |
|---|---|---|
| `d3_beamform_net.KLIM 0.6→0.3` + `KX_G,KY_G,OM_G = kgrid()` | `learned_init.py:151` → `d3_beamform_cond_diffusion.py:174 _bmap_from_obs` → `:176 beamform` | `d3_beamform_net.py:46-57 beamform` 이 **모듈 전역** `KX_G/KY_G/OM_G/NK` 를 읽는다 ⟹ 재대입이 먹는다. ⛔`KLIM` 자체는 `beamform` 이 읽지 않는다 — **`kgrid()` 재호출이 필수**(T12 `run_t12.py:97-101` 이 그렇게 한다) |
| `stats['K_LIM'] 1.1→0.55` | `LI._load_cond` 래핑 | `learned_init.py:135` 이 **모듈 전역 이름** `_load_cond` 로 부른다 ⟹ `LI._load_cond = wrapped` 가 먹는다. T12 `run_t12.py:102-109` 가 `dict(stats)` 사본을 고쳐 **`_CACHE` 원본을 안 건드린다**. 소비처 = `d3_beamform_cond_diffusion.py:89,102,117,121`(정규화/역정규화) |

⛔**`d3_beamform_cond_diffusion.py:33` 은 `from d3_beamform_net import beamform, KLIM, ...` = bare-name
임포트**다. `cd.KLIM` 은 별도 바인딩이지만 **cd 안에서 한 번도 쓰이지 않는다**(전수 grep: `:33` 뿐).
⟹ `BN.KLIM` 만 고쳐도 부작용은 없고, 실효는 `kgrid()` 재호출에서 나온다.

### 4.4 ⚠T15 러너가 조심할 것

1. ⛔**`run_estimation` 은 `arm_provenance()` 를 요약에 넣지 않는다**(전수 grep: `run.py` 에 0회).
   요약이 주는 팔 정체는 `run.py:520 auto_selected` · `:521 auto_n_selected` · `:524 auto_real_gate` ·
   `:529 ensemble_k` **넷뿐**이다. `CLAUDE.md` 의 ★규율(*"`upd.arm_provenance()` 를 결과 행에 넣어라"*)
   을 지키려면 **`run_estimation` 을 쓰지 않거나**, 그 넷으로 대체함을 사전등록에 적어야 한다.
2. ⚠배포 기본값에서 `arm_provenance()` 는 `ensemble_k_requested = 3`(analytic) 인데
   `ensemble_k = 1` 이라 **`ensemble_collapsed = True`** 가 나온다. 이는 **자국 학습 경로의 정상값**이고
   결함이 아니다(`sequential_update.py:2090-2126 _effective_k` 가 노브를 route 로 다시 읽는다).
   게이트가 이것을 위반으로 잡지 않게 할 것.
3. ⚠`Arm` 은 **프로세스 전역 모듈 상태**를 고친다. `run_estimation` 을 팔마다 컨텍스트 안에서 부르고
   `G-RESTORE` 로 되돌림을 검사할 것(T12 `read_knobs` 재사용).
4. ⛔`LI._CACHE`(`learned_init.py:66`, 채움 `:119`)는 `(ckpt, device)` 키 캐시다. `_load_cond` 래퍼는 원본을 호출해
   캐시를 공유하고 **stats 사본만** 고치므로 안전하지만, 캐시를 직접 비우는 코드를 넣지 말 것.

---

## 5. 비용 — **실측**

★**런 1회**, `systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0` 안에서,
`OMP/MKL/OPENBLAS/NUMEXPR/VECLIB = 1`, `torch.set_num_threads(1)`.
스크립트 = `tmp/T15_HOMEREG_2026-09-05/cost_probe.py` · 출력 = `cost_probe.json`.

```
에피소드   data/episode_BF4_30comp_seed0_uniform.h5   (301 프레임, Tp 7.0, 채점 160 프레임)
팔         SHIP = dict(DEFAULT_CONFIG)   (init_method='auto', 무처치)
벽시계     7.814 s          ← ★단가
  init      1.409 s   (그중 VARPRO 1.354 s)
  추적      avg_inference_ms = 5.93 ms/frame  ⟹ 160 프레임 ≈ 0.95 s
  나머지    ≈ 5.5 s  = h5 로드 + **프레임당 채점**(셸 거리 O(4800×50) + RMSE_obs/unobs 16회)
프로세스   real 9.52 s (파이썬·torch 임포트 ≈ 1.7 s 포함)
장치       proposal 은 GPU (cuda=True, RTX 4080 SUPER) · 나머지 CPU 1스레드
```

⟹ **채점이 추정보다 5.8배 비싸다.** 단가는 채점 프레임 수에 거의 선형이다:
`≈ 0.4 s(로드) + 1.4 s(init) + 0.040 s × n_track`.

**110 에피소드 × 3팔 = 330 런:**

| 방식 | 추정 총 시간 |
|---|---|
| 직렬 1프로세스 (임포트 1회) | 110 × 3 × 7.8 s ≈ **2,570 s ≈ 43 분** |
| 채점프레임 기반 재계산 (Σ n_track = 17,340/팔) | 3 × (110×1.8 + 0.040×17,340) ≈ **2,680 s ≈ 45 분** |
| 4샤드 병렬 | ≈ **11–15 분** (⚠경합으로 단가가 오른다 — 아래) |

⚠**교차 참조**: `B-15`(같은 `data/` 코퍼스, 같은 배포 팔) 의 `SHIP-NOW` **wall 중앙 12.9 s**
(n=121). 그 배치는 **6 샤드 동시**로 돌았고 `SPEC25`(중앙 61.3 s)와 코어를 나눴다.
⟹ 오늘의 7.81 s 는 **무경합 하한**, 12.9 s 는 **6샤드 경합 상한**으로 읽는 것이 안전하다.
보수적으로 **110×3 ≈ 45–70 분**을 잡아라.

⚠**미측정**: (a) 층별·성분수별 단가 분산 — 1런밖에 안 쟀다. (b) **GPU 없이** 돌 때의 단가
(확산 제안이 CPU 로 내려간다). (c) `A-MAPSCALE` 팔의 단가(맵 크기 `NK=64` 불변이므로 동일할 것으로
보이나 **측정 안 함**).

---

## 6. 앵커로 쓸 수 있는 동결값

### 6.1 ★★1순위 — `B-15`: **오늘 프로브와 비트 동일**

`tmp/B15_RAYDROP_2026-08-24/results_b15_*.jsonl`, 2026-08-24 동결.
- 코퍼스 = **`data/` 40 에피소드**(BF4/BF5 × 15comp/30comp × seed0–4 × uniform/slope).
- 팔 = `SHIP-NOW` = `dict(DEFAULT_CONFIG)`, **추정기 시드 42·43·44** (에피소드당 3행, 120행).
- 배포 상수가 행에 있다: `varpro_steps 400 · learned_init_iters 0 · auto_switch_thresh 0 · lambda_l1 1e-4`
  — **오늘 `DEFAULT_CONFIG` 와 전건 일치**(프로브가 되읽어 확인).

★**비트 재현**:

| | `avg_nRMSE` |
|---|---|
| `B-15` 동결 행 (2026-08-24, `episode_BF4_30comp_seed0_uniform`, `est_seed=42`) | `0.15821650810539722` |
| **오늘 프로브** (2026-09-05, 같은 에피소드, `seed=42`) | `0.15821650810539722` |

⟹ ★**12일·다른 세션·다른 프로세스에서 비트 동일.** T15 의 `G-ANCHOR` 는
*"부호 재현"* 이 아니라 **비트 동일**을 요구할 수 있다(40 에피소드 × 시드 42 = **40짝** 사용 가능).
⚠단 `B-15` 는 `data/` 40개만 덮는다 — 110 중 나머지 70 에는 앵커가 없다.

시드 42–44 중앙(median-of-3) 요약:

| 슬라이스 | n | 중앙 `avg_nRMSE` | 범위 |
|---|---|---|---|
| uniform | 20 | **0.1514** | 0.0364 – 0.3083 |
| slope | 20 | **0.0937** | 0.0309 – 0.2185 |
| BF4 15comp | 10 | 0.0731 | |
| BF4 30comp | 10 | 0.1521 | |
| BF5 15comp | 10 | 0.0691 | |
| BF5 30comp | 10 | 0.1682 | |

`verdict_b15.json` 라벨 = `HEADLINE-SURVIVES-CORRECT-SENSOR`(slope 17/20, p=0.001288, 게이트 6/6).
✅**인용금지 목록에 `B-15` 항목 없음**(전수 grep). ⚠단 그 판정문 자체가 아니라 **`SHIP-NOW` 절대값**을
앵커로 쓰는 것이므로 라벨을 인용할 필요도 없다.

### 6.2 2순위 — `W-16`: `_ra_rf_ds` 128 코퍼스의 배포 팔 절대값

`tmp/W16_REBASELINE_2026-08-22/results.jsonl` (384행 = 128ep × 시드 1629/1630/1631, 전건 `SHIP`).
`auto_selected` **384/384 `learned`**, `ensemble_k` **384/384 = 1** ⟹ 오늘 프로브와 같은 팔.

| 층 | n_ep | median-of-3 의 중앙 | 범위 | 런 wall 중앙 |
|---|---|---|---|---|
| crowded | 64 | **0.1275** | 0.0235 – 0.2362 | 14.9 s |
| sparse | 64 | **0.0202** | 0.0013 – 0.1141 | 12.2 s |

### 6.3 ⛔앵커로 **쓰지 말 것** (인용금지 확인 완료)

- ⛔**`B-1` 의 `62/64 · −38.58% · p=1.128e-16`** (`tmp/B1_BASELINE_2026-08-22/verdict_b1.json`) —
  판정 자체는 살아 있으나 (i) **`SPEC25` 대비 상대 크기**이지 절대 앵커가 아니고
  (ii) `reference_citation_bans.md:426,599,724` 가 **`B-11`·`B-12`·`B-16`·`B-19`·`B-20b` 크기와의 직접
  비교를 금지**하며 (iii) `B1/verdict_b1._honesty` 축자 *"gates/analyze 는 결과 뒤에 작성됐다"*.
  (iv) `MEMORY.md` 의 *"p=2.3e-16"* 은 동결 파일의 `1.128e-16` 과 **다르다** — 원장 쪽이 낡았다.
- ⛔**`0.2229 / 0.3189 / 13.376 ms / 132.639 ms`**(`results/b1_full_comparison.json`) — `CLAUDE.md` 가
  프레임 불일치·기준선 선택·프로세스 간 지연 셋을 사유로 금지.
- ⛔**희소 `−93%` / `W-16` 의 `−91.89%`**(`reference_citation_bans.md:1123-1125`).
- ⛔**`85/100 p=4.8e-13`** 및 그 코퍼스(=`data/` 100개)에서 나온 p 값 일체
  (`reference_citation_bans.md:167`) — ★**바로 이 `data/` 코퍼스의 레거시 draw 결함이 사유**다.

---

## 7. T15 설계자에게 남기는 결정 사항 (⛔내가 정하지 않았다)

1. **코퍼스**: `data/` 110(넓지만 레거시 draw) vs `_ra_rf_ds` 128(층 규약·수정 생성기, 좁음) —
   **또는 둘 다**. §1.2 참조.
2. **NaN 두 건**: `BF4_20comp_seed7_uniform` · `BF4_30comp_seed42_uniform` 을 모집단에서 사전 배제할지,
   유한성 게이트로 잡을지.
3. **층 규약**: `data/` 에는 혼잡/희소 정의가 **없다**. `ncomp ≥ 15` = 혼잡, `≤ 5` = 희소 로 잡으면
   혼잡 46(15/20/30comp) · 희소 64(1/2/5comp) 가 되고 110 을 정확히 나눈다. ⚠단 `20comp` 는 **단 1개**이고 그것이 §1.1 의 NaN 에피소드다. **결과 전에 동결할 것.**
4. **채점 접두어**: `ship:`(near/far, `x<80`) 인지 `obs:`(9고리) 인지 — `S13` 이 여기서 이탈했다.
5. **arm_provenance**: `run_estimation` 경로를 쓰면 4키로 대체됨을 사전등록에 명시.
6. **교락 공시**: 자국 = 사전의 학습 분포(§1.3) · GPU 제안 · 시드 축(오늘 프로브는 42 하나).

---

## 부록 — 오늘 실행한 것

- 런 **1회**: `cost_probe.py data/episode_BF4_30comp_seed0_uniform.h5` (메모리 캡 8G, §5).
- 그 밖은 전부 h5 속성/성분 읽기와 동결 jsonl 재집계(추정기 런 0회).
- ⛔`T11_*`·`T12_*`·`T13_*`·`T14_*` 는 **읽기만** 했다(수정 0).
- ⛔패키지·`scripts/` 수정 0.
