# P-3 — `B-1`·`B-12` 코퍼스의 재현 명령 확정 (런 0회)

**작성 2026-08-26 · 임무 R2 · 새 에피소드 생성 0회 · 새 추정 0회 · 배치 산출물 수정 0회**

외부 심사 처방 P-3 의 판정 기준은 *"독자가 그 코퍼스를 **명령 한 줄로 재현**할 수 있는가"* 다.
아래는 **전부 코드/자료에서 실측**했다. 심사위원이 제시한 네 수치(`raydrop='uniform'` ·
`M` 기대값 50 · `sigma_range` 2 cm 상수 · 채점 격자 `x∈[10,250] m`)는 **네 개 다 맞다**.
그러나 **셋이 불완전**하고 **하나는 재현을 막는 누락**이 있다 — §6 에 명시했다.

---

## 0. 열람 범위 (이 문서의 주장 범위)

**코드**: `wavesim/pipeline.py` · `wavesim/config.py` · `wavesim/noise_model.py` ·
`wavesim/raydrop_uniform.py` · `wavesim/lidar_gpu.py` · `wavesplat/run.py` ·
`wavesplat/data_loader.py` · `wavesplat/evaluation/metrics.py`
**생성기**: `tmp/S8_deploy_confirm_2026-08-10/gen_s8.py` · `tmp/S9_lr_x_l1_2026-08-11/gen_s9.py` ·
`tmp/HOSGATE_2026-08-13/gen_hos.py` · `tmp/B12_HOS_EXTEND_2026-08-23/gen_b12.py`
**게이트/러너/판정**: `gates_b1.py`·`verdict_b1.json` · `gates_b11.py` · `gates_b12.py`·
`run_b12.py`·`analyze_b12.py`·`verdict_b12.json` · `gates_w16.py`·`run_w16.py`·`corpus.json` ·
`run_s16.py`(채점 창 부분) · `verdict_b13.json`
**자료**: `B-1` 혼잡 **64 파일 전건** · `B-12` HOS **48 파일 전건** + LIN 짝 **48 파일 전건**
(HDF5 `/config`·`/raydrop_model`·`/frames` 를 열어 직접 셌다) ·
`results.jsonl` 3개(W16 384행 · S16 384행 · B12 288행).

⛔**열지 않은 것**: `B-11b`(희소) · `HOSGATE` 희소 32 파일 · 실측 경로 · 논문 `.tex` 본문.

---

## 1. `B-1` 코퍼스

`B-1` 은 **런 0회 재집계**다(`gates_b1.py:13-16`). 두 팔의 원자료는 다른 배치에서 왔다:
`SHIP` = `tmp/W16_REBASELINE_2026-08-22/results.jsonl`(384행) ·
`SPEC25` = `tmp/S16_BASELINE_2026-08-14/results.jsonl`(384행).
**에피소드 목록의 단일출처는 `tmp/W16_REBASELINE_2026-08-22/corpus.json`** 이고
`gates_w16.py:13` 이 그것을 읽는다. 헤드라인이 인용하는 것은 그중 **혼잡 층 64개**다.

| 항목 | 실측값 | 근거 |
|---|---|---|
| 에피소드 수 | **64** (혼잡 층만; 희소 64 는 인용 금지) | `verdict_b1.json` `crowded.n=64`; `corpus.json` |
| 층 정의 | 혼잡 = `n_components ∈ {15, 30}` | `corpus.json` `crowded` 키 |
| 설계 | `BF4/BF5 × {15,30}comp × 16 시드` = 64 | 실측 census: BF4 32 · BF5 32 |
| 시드 | **32–39** 와 **48–55** (각 8개, 총 16) | `gen_s8.py:41` `SEEDS=range(32,40)` · `gen_s9.py:46` `CROWDED_SEEDS=range(48,56)` |
| 디스크 위치 | `data_s8_crowded_2026-08-10/`(32) + `data_s9_crowded_2026-08-11/`(32) | 파일 존재 실측 64/64 |
| 참값 종류 | **선형** (Gerstner 성분합, `cfg.hos.enabled=False`) | 두 생성기 모두 `cfg.hos` 미설정 = `config.py` 기본 `False` |
| `raydrop_model` | **`uniform`** | `gen_s8.py:109`·`gen_s9.py:115`; **h5 `/raydrop_model.model_type` 64/64 = `'uniform'`** |
| `n_pts` (목표 반환점) | **50** | h5 `/raydrop_model.n_pts` 64/64 = `50` |
| 실측 `E[M]` (프레임당 반환점) | **코퍼스 평균 49.98** (에피소드 평균 최소 48.81 · 최대 50.47) | `frames/n_returns_raw` 전건 집계 |
| `sigma_range` | **0.02 m 상수** | `pipeline.py:263` 이 인자 없이 호출 ⟹ `noise_model.py:15` 기본값 |
| `sigma_angle_deg` | **0.05°** ⚠심사위원 미언급 | 같은 호출 ⟹ `noise_model.py:16` |
| `duration` / `dt` | **30.0 s / 0.1 s** → **301 프레임** | h5 `frames/timestamps` 64/64 동일 |
| `depth` | **50.0 m** | h5 `/config.depth` 64/64 |
| `direction_deg` / spread | **0.0° / 30.0°** | 세 생성기 모두 미전달 ⟹ `pipeline.py:77-78` 기본값; h5 `/config.direction_spread_deg`=30.0 64/64 |
| 레이캐스트 메시 | **200 × 167** (기본 40×30 을 덮어씀) | `gen_s8.py:40`·`gen_s9.py:43` `NX, NY = 200, 167` |
| 메시 적정성 | **ADEQUATE 64/64** | h5 `/config.mesh_verdict` |
| 채점 격자 | **x∈[10,250] m ×80 · y∈[−100,100] m ×60** (dx 3.038 · dy 3.390) | `config.py:45-48`; h5 `frames/gt_grid_{x,y}` 64/64 동일 |
| 채점 창 | `t > 2·Tp` ⟹ **BF4 160/301 프레임**(t≥14.1 s) · **BF5 120/301**(t≥18.1 s) | `data_loader.py:124-131`; `run.py:263`; `run_s16.py:255` |
| 추정기 팔 | 2026-08-22 배포 기본값 — **행에서 되읽음**: `varpro_steps=400` 384/384 · `init_iters=0` 384/384 · `auto_selected='learned'` 384/384 | `results.jsonl` 전수 |
| 시드 (추정기) | **1629, 1630, 1631**, 에피소드당 median-of-3 | `gates_w16.py:19` |
| 판정 | `SHIP-BEATS-BEST-CROWDED` · **62/64 · −38.58% · 단측 p=1.128e-16**(양측 2.26e-16) | `verdict_b1.json`; **독립 재집계로 자릿수까지 재현**(§5) |

---

## 2. `B-12` 코퍼스

`B-12` 는 `B-11`(n=16)의 **확장**이고, 생성 파라미터를 손으로 옮기지 않고
`gen_hos` 를 **그대로 import** 한다(`gen_b12.py:28`). 바뀌는 것은 **시드 목록 하나**다
(`gen_b12.py:30`). 판정 규칙은 `gates_b11` 을 상속한다(`gates_b12.py:45-46`).

| 항목 | 실측값 | 근거 |
|---|---|---|
| 에피소드 수 | **48** (혼잡 HOS 만) | `gates_b12.py:51` `N_EP=48`; 파일 존재 실측 48/48 |
| 층 정의 | 혼잡 = `n_components ≥ 15` | `run_b12.py:crowded_episodes()` |
| 설계 | `BF4/BF5 × {15,30}comp × 12 시드` = 48 | 실측 census: BF4 24 · BF5 24 |
| 시드 | 기존 **1581–1584** + 신규 **1633–1640** | `gates_b12.py:48-50`; `gen_hos.py:45`; `gen_b12.py:30` |
| 디스크 위치 | `tmp/HOSGATE_2026-08-13/data_hos/` (LIN 짝은 `../data_lin/`) | `gates_b11.py` `HOS=` 경로 |
| 참값 종류 | ★**HOS 비선형 M=3** (`cfg.hos.enabled=True`) | `gen_hos.py:41` `HOS_M=3`; `gen_hos.py:143-145`; `gen_hos.py:167` `arm=='HOS'` |
| HOS 격자 | `Nx×Ny = 192×128` · `Lx×Ly = 600×400 m` · 내부 `dt = 0.02 s` | `gen_hos.py:41` |
| 짝 설계 | 같은 (해상상태, 성분수, 시드)에서 **같은 스냅 성분**으로 LIN/HOS 두 팔. LiDAR·잡음·레이드롭·메시·격자·duration·dt·depth 전부 동일, **파면 시간전개 방식 하나만 다르다** | `gen_hos.py:8-10`; 몽키패치 3종 `:63-123` |
| `raydrop_model` | **`uniform`** | `gen_hos.py:166`; h5 48/48 = `'uniform'` |
| `n_pts` | **50** | `gen_hos.py:43` `N_PTS=50`; h5 48/48 |
| 실측 `E[M]` | **코퍼스 평균 50.16** (에피소드 평균 최소 **49.29** · 최대 **51.09**) | `frames/n_returns_raw` 전건 |
| `sigma_range` / `sigma_angle_deg` | **0.02 m / 0.05°** (B-1 과 동일 경로) | `pipeline.py:263` → `noise_model.py:15-16` |
| `duration` / `dt` / `depth` | **30.0 s / 0.1 s / 50.0 m** → 301 프레임 | `gen_hos.py:43`; h5 48/48 |
| `direction_deg` / spread | **0.0° / 30.0°** | `gen_hos.py:164-170` 미전달; h5 48/48 |
| 레이캐스트 메시 | **200 × 167** (B-1 과 동일) | `gen_hos.py:42` |
| 메시 적정성 | ⚠**ADEQUATE 31 · DEGRADED 17** | h5 `/config.mesh_verdict` 48/48 |
| 채점 격자 | **B-1 과 비트 동일** (x∈[10,250]×80 · y∈[−100,100]×60) | h5 `frames/gt_grid_{x,y}` 48/48 |
| 채점 창 | `t > 2·Tp` (B-1 과 동일 규칙) | `data_loader.py:129` |
| 추정기 팔 | `SHIP-NOW` = 배포 기본값 — **행에서 되읽음**: `varpro_steps=400` 144/144 · `learned_init_iters=0` 144/144 (게이트 `G-6`) | `results_b12.jsonl`; `gates_b12.py:73-75` |
| 기준선 팔 | `SPEC25` — **되읽음**: `loc_radius=25.0` · `n_ensemble=600` · `background='jonswap'` 144/144 (게이트 `G-5`) | `gates_b12.py:69-72` |
| 시드 | 추정기 `EST_SEEDS=(42,43,44)` · 필터 `FILTER_SEEDS=(42,43,44)`, median-of-3 | `gates_b11.py`; 행에서 `filter_seed` 48/48/48 |
| 판정 | `HEADLINE-SURVIVES (CONFIRMATORY, n=48)` · **43/48 · −40.19% · 단측 p=6.84e-09** · 음성대조 `SPEC25<ZERO` **47/48 p=1.7e-13** | `verdict_b12.json`; **독립 재집계로 재현**(§5) |

---

## 3. 관측 모형 — 코드에서 도출한 단일 서술

`B-1` 과 `B-12` 는 **같은 관측 모형**을 쓴다(위 두 표의 해당 행이 전건 일치).
파이프라인 순서와 근거:

1. **광선 생성** — `pipeline.py:162-173` 이 `config.py:16-27 LidarCfg` 기본값으로
   `LidarConfig` 를 만든다. 세 생성기 중 **어느 것도 `cfg.lidar` 를 덮어쓰지 않는다** ⟹
   height **5.0 m** · channels **152** · scan_rate **10 Hz** · FOV_h **120°** ·
   FOV_v **−25°…0°** · range **1.5–250 m** · n_h_beams **500** · λ **1550 nm**.
   센서 원점은 `(0, 0)` 고정 — `pipeline.py:179` `generate_rays(..., ship_x=0.0, ship_y=0.0)`.
2. **레이캐스트** — `pipeline.py:207-217`. 세 생성기 모두 `use_gpu_raycast=True`.
   `lidar_gpu.py:29` 가 CUDA 없으면 CPU 로 자동 폴백하며, 파일 독스트링이 스스로
   *"hit-for-hit, **not** bit-for-bit"*(좌표가 ~1e-13 m 이동)라 적는다 ⟹ §6-(e).
3. **레이드롭(`uniform`)** — `pipeline.py:236-241` → `raydrop_uniform.apply_dropout`.
   빔마다 독립 베르누이, 확률 `P_i = η·R_fresnel(θ_i)·cos θ_i /(d_i/125 m)²`,
   `η` 는 `raydrop_uniform.py:137` 에서 `target_n_pts / ΣP_raw` 로 정해진다
   ⟹ ★**`E[M] = 50` 은 하드 쿼터가 아니라 기대값**이고, 실제 반환수는 이항분포로 요동한다
   (`raydrop_uniform.py:107-111`). **실측 코퍼스 평균 49.98(B-1) · 50.16(B-12)** 로 확인.
4. **센서 잡음** — `pipeline.py:263` `add_noise(hit_pts[final_idx], ray_dirs[final_idx], rng=rng)`.
   **`sigma_range` 를 넘기지 않는다** ⟹ `noise_model.py:15-16` 기본값
   **`sigma_range = 0.02 m`(상수, 거리 무관) · `sigma_angle_deg = 0.05°`**.
   잡음은 **레이드롭 이후 생존점에만** 걸린다.
   ⚠거리 잡음은 **광선 방향으로** 더해지므로(`noise_model.py:40-41`) 연직 오차는
   `0.02·|d_z|` 다 — 표본 에피소드에서 **중앙 4.9 mm · p95 7.9 mm**(입사각 중앙 **75.4°**,
   경사거리 중앙 **20.3 m**). ⟹ *"2 cm"* 를 **연직 정밀도로 읽으면 4배 과대**다.

---

## 4. 채점 격자·채점 창·지표

- **격자** — `run.py:242-243` 이 `loader.gt_grid_x / gt_grid_y` 를 **h5 에 저장된 그대로** 쓴다.
  저장된 값은 `config.py:44-48 GtGridCfg` 기본값 = **x∈[10, 250] m (80점, dx=240/79=3.0380 m) ×
  y∈[−100, 100] m (60점, dy=200/59=3.3898 m)** = **4,800 격자점**. 112 파일 전건 동일.
- **채점 창** — `data_loader.py:124-131 get_tracking_frames(tp_factor=2.0)` = `t > 2·Tp_true`.
  `run.py:226·263` 이 초기화/추적 양쪽에 같은 인자를 쓰고, 기준선 `run_s16.py:255` 도
  `sel = t > 2.0*Tp_true` 로 **같은 규칙**이다. 실측: **BF4(Tp=7 s) 160/301 · BF5(Tp=9 s) 120/301**.
  `verdict_b13.json` 이 표본 1개에서 두 팔의 창 일치(160 프레임 · 시작 14.1 s)를 보충 도출했다.
- **지표** — `metrics.py:32-50` `nRMSE = RMSE / Hs`(**공칭** Hs: BF4 1.2 m · BF5 2.0 m),
  마스크 없이 **격자 전체**. ⚠`metrics.py:35-48` 이 경고하듯 실측 하네스는 σ 로 나눈다
  ⟹ 합성·실측 nRMSE 를 나란히 놓지 마라.
- ⚠**관측의 공간 지지집합이 채점 영역보다 훨씬 좁다** (실측, 이 문서에서 처음 집계):
  반환점 x 의 에피소드별 중앙값의 중앙값이 **16.3 m**, p95 가 **39 m** 인데 채점은 **250 m** 까지 간다
  (B-1 64개 · B-12 48개 둘 다 동일). ⟹ 헤드라인은 **대부분 외삽 영역에서 채점된 수**다.

---

## 5. 독립 재집계 (배치 산출물 무수정)

`verdict_*.json` 을 신뢰하지 않고 `results.jsonl` 에서 **처음부터 다시** 집계했다
(에피소드당 시드 median → 짝 부호검정). 분석기를 재실행하지 **않았다** — `verdict_*.json` 의
mtime 을 건드리면 `tmp/_audit/check_batch_artifacts.py` 의 검사 대상이 흔들린다.

```
B-1  crowded n=64  SHIP<SPEC25 62/64  rel_median -38.5796%  p_one=1.1281e-16  p_two=2.26e-16
B-12 HOS     n=48  SHIP<SPEC25 43/48  rel_median -40.1941%  p_one=6.8402e-09
B-12 음성대조     SPEC25<ZERO 47/48                          p_one=1.74e-13
```

⟹ `verdict_b1.json`·`verdict_b12.json` 및 원장의 인용값과 **전부 일치**.

---

## 6. ⚠심사위원 수치와 다른 항목 (명시)

| # | 심사위원 문언 | 실측 판정 | 내용 |
|---|---|---|---|
| (a) | `raydrop_model='uniform'` | ✅**정확** | h5 112/112 = `'uniform'`. ⚠단 그 모형의 입사각 의존성이 논문 제2장 서술과 **부호가 반대**다(`raydrop_uniform.py:20-39`, 심사 `F-7`). `B-15` 가 개입으로 헤드라인 생존을 확인했으나 **문서 불일치 자체는 별개**다. |
| (b) | `M 기대값 50` | ✅**정확하나 불완전** | `n_pts=50` 은 **기대값**이지 쿼터가 아니다(`raydrop_uniform.py:107-111`). 실측 코퍼스 평균 **49.98**(B-1) · **50.16**(B-12), 프레임 단위로는 이항 요동(한 에피소드에서 30–73 관측). ⟹ 범위 문장에 *"기대값"* 이라는 단어가 **반드시** 들어가야 한다. |
| (c) | `sigma_range 2 cm 상수` | ⚠**절반만 맞다** | 상수·거리무관은 맞다. 그러나 **관측 잡음은 두 항**이고 심사위원은 **각도 잡음 `sigma_angle_deg = 0.05°` 를 빠뜨렸다**(`noise_model.py:16`). 또한 2 cm 는 **경사거리** 방향이라 **연직 오차는 중앙 4.9 mm** 다(측정). |
| (d) | `채점 격자 x ∈ [10, 250] m` | ⚠**절반만 맞다** | x 범위는 맞다. 그러나 **y∈[−100, 100] m 과 해상도 80×60(dx 3.04 · dy 3.39 m)이 빠졌다** — 격자를 이 셋 없이 적으면 **재현 불가**다(`config.py:45-48`). |
| (e) | (미언급) | ⛔**재현을 막는 누락 1건** | **`B-1` 코퍼스 64 파일이 git 미추적**이다(`data_s8_crowded_2026-08-10` 0/32 tracked · `data_s9_crowded_2026-08-11` 0/32). `B-12` 도 48 중 **29 만 추적**(신규 시드 1633–1640 의 19 파일 미추적). ⟹ **독자는 파일을 내려받을 수 없고 반드시 재생성해야 한다.** 그리고 `lidar_gpu.py:1-5` 가 GPU/CPU 경로가 *"bit-for-bit 아님"*(~1e-13 m)이라 적었으므로 ★**비트 단위 재현은 하드웨어 의존**이고 보장되는 것은 **hit-for-hit** 이다. |
| (f) | (미언급) | ⚠**`n_pts=50` 의 출처가 폴백이다** | `gen_s8.py:100`·`gen_s9.py:106` 은 `n_pts` 를 원본 h5 에서 읽는 것처럼 보이지만, 실제로 원본 `config/lidar_config` 에는 **`n_pts` 키가 없다**(실측: `{channels, height, scan_rate}` 뿐) ⟹ `... or 50` 리터럴로 떨어진다. **값은 50 이 맞지만 계보는 "원본 상속" 이 아니라 "코드 리터럴"** 이다. |
| (g) | (미언급) | ⚠**`B-1` 의 팔 정의가 빈 dict 다** | `gates_w16.py:19` `ARMS = {'SHIP': {}}` — 원장이 BLOCKER 로 못 박은 *"빈 dict 로 팔을 정의하지 마라"* 위반과 같은 형태다. **다만 이번에는 손상 없음**: 행에서 `varpro_steps=400`·`init_iters=0`·`auto_selected='learned'` 가 **384/384 되읽혀** 배포 구성이었음이 확인된다. ⟹ 논문에는 **되읽은 값**을 적고 `{}` 에 의존하지 마라. |
| (h) | (미언급) | ⚠**두 코퍼스의 메시 적정성이 다르다** | `B-1` 은 ADEQUATE **64/64**, `B-12` 는 DEGRADED **17/48**. 원장이 이미 *"DEGRADED 셀은 사실상 2차 비선형성"* 이라 적었다 ⟹ `B-12` 를 *"M=3 비선형"* 이라 단일 서술하면 **35%가 과대 진술**이다. |

---

## 7. ★재현 명령 (파일 존재 확인 완료)

전제: `cd /home/vexit/wavesplat/wavesplat`.
네 생성기 모두 스스로 `os.chdir(REPO)` 하므로 절대경로로 어디서든 호출 가능하다.
**이미 있는 파일은 건너뛴다**(`gen_s8.py:87-90`·`gen_s9.py:100-102`·`gen_hos.py:160-162`·
`gen_b12.py:45-46`) ⟹ 지금 실행하면 **no-op** 이고 기록을 덮지 않는다.

### `B-1` 코퍼스 (혼잡 64 · 선형 참값)

```bash
python3 tmp/S8_deploy_confirm_2026-08-10/gen_s8.py crowded && \
python3 tmp/S9_lr_x_l1_2026-08-11/gen_s9.py crowded
```

⚠**전제 파일 4개**(추적됨, 존재 확인): `data_finemesh_2026-07-23/episode_{BF4,BF5}_{15,30}comp_seed0_uniform.h5`
— 생성기가 여기서 `duration`·`dt`·`depth` 를 읽는다(`n_pts` 는 §6-(f) 대로 리터럴 50).

### `B-12` 코퍼스 (혼잡 HOS 48 + LIN 짝 48)

```bash
python3 tmp/HOSGATE_2026-08-13/gen_hos.py crowded && \
python3 tmp/B12_HOS_EXTEND_2026-08-23/gen_b12.py
```

앞이 시드 **1581–1584**(16쌍), 뒤가 시드 **1633–1640**(32쌍)을 만들며 **같은 디렉터리**에 넣는다
(`gen_b12.py:32-33` 이 그 선택을 명시적으로 기록했다).

### 판정까지 (에피소드가 준비된 뒤)

```bash
# B-1  — 새 런 0회, 두 배치의 results.jsonl 재집계만 한다
python3 tmp/B1_BASELINE_2026-08-22/analyze_b1.py
# B-12 — 288런(48 에피소드 × 2팔 × 3시드) 후 판정
python3 tmp/B12_HOS_EXTEND_2026-08-23/run_b12.py && \
python3 tmp/B12_HOS_EXTEND_2026-08-23/analyze_b12.py
```

⚠`B-1` 을 **원자료부터** 재현하려면 두 러너가 추가로 필요하다:
`python3 tmp/W16_REBASELINE_2026-08-22/run_w16.py`(`SHIP`, 384런) ·
`python3 tmp/S16_BASELINE_2026-08-14/run_s16.py`(`SPEC25`, 384런).
⛔**두 분석기는 `verdict_*.json` 을 덮어쓴다** — 동결 산출물의 mtime 이 바뀌므로
검증 목적이라면 사본 디렉터리에서 돌려라.

### 판정 기준 (P-3 문언)에 대한 답

> *"독자가 그 코퍼스를 명령 한 줄로 재현할 수 있는가"*

**부분적으로 예.** 각 코퍼스는 **명령 한 줄**(`&&` 로 이은 두 스크립트)로 재생성되고,
관측 모형·격자·창은 위 표대로 코드에 고정돼 있다. 그러나 ⛔**바이트 단위로 같은 파일을
받을 방법은 없다** — 자료가 대부분 git 미추적이고(§6-(e)) GPU/CPU 경로가
비트 동일을 보장하지 않는다. ⟹ 논문에는 *"재생성 가능"* 이라 쓰고
*"파일을 배포한다"* 나 *"비트 동일하게 재현된다"* 라고 **쓰지 마라**.

---

## 8. 논문 범위 문장에 넣을 LaTeX 문단 초안

> 배치 위치 제안: 제5장(합성 실험) 서두의 "코퍼스" 절, `B-1`·`B-12` 를 처음 인용하기 직전.
> ⚠본 초안은 **본 문서에서 실측한 값만** 담았다. 논문에 옮길 때 `\label` 은 기존 규약에 맞춰라.

```latex
\subsection{합성 코퍼스와 관측 모형}
\label{subsec:corpus_obs}

본 논문의 합성 결과는 두 코퍼스에서 나온다. \textbf{선형 코퍼스}(\texttt{B-1})는
해상상태 BF4/BF5, 성분수 $\{15,30\}$, 시드 $32\text{--}39$ 및 $48\text{--}55$ 의
$2\times2\times16=64$ 에피소드로 이루어지며 참값은 선형 중첩(Gerstner 성분합)이다.
\textbf{비선형 코퍼스}(\texttt{B-12})는 같은 해상상태\,$\times$\,성분수 격자에
시드 $1581\text{--}1584$ 와 $1633\text{--}1640$ 을 붙인 $2\times2\times12=48$ 에피소드이며,
참값은 고차 스펙트럼법(HOS) $M=3$ 전개다($192\times128$ 모드, $600\times400$\,m 영역,
내부 시간간격 $0.02$\,s). 후자는 같은 파랑 성분에서 선형 팔과 짝을 이루도록 생성되어
\emph{파면의 시간전개 방식 하나만} 다르다. 다만 48 에피소드 중 17 개는 격자 적정성 판정이
\textsc{degraded} 로, 이 셀들에서 실현된 비선형성은 사실상 2차에 그친다.

두 코퍼스는 \emph{동일한} 관측 모형을 공유한다. 에피소드 길이는 $30$\,s, 시간간격은
$0.1$\,s(301 프레임), 수심은 $50$\,m 이고, 파향 중심은 $0^\circ$, 방향 확산은 $30^\circ$ 다.
LiDAR 는 높이 $5$\,m 에 고정되어 있으며 수평 화각 $120^\circ$, 연직 화각
$-25^\circ$부터 $0^\circ$, 유효 거리 $1.5$--$250$\,m 다. 반환점 소실(raydrop)은
Fresnel 반사율\,$\times$\,$\cos\theta$\,$\times$\,역제곱 감쇠에 비례하는 확률의
독립 Bernoulli 시행으로 모형화되며, 비례상수는 \emph{프레임당 반환점 수의 기대값이 50 이
되도록} 정규화된다. 따라서 50 은 할당량이 아니라 기대값이고 실제 반환수는 프레임마다
요동한다(실측 코퍼스 평균 $49.98$ 및 $50.16$). 생존한 점에는 광선 방향으로
표준편차 $2$\,cm 의 거리 잡음과 표준편차 $0.05^\circ$ 의 각도 잡음이 더해진다. 거리 잡음이
광선 방향으로 걸리므로 연직 방향 오차는 이보다 작아 중앙값 약 $5$\,mm 이다.

채점은 $x\in[10,250]$\,m 을 80 등분, $y\in[-100,100]$\,m 을 60 등분한
$80\times60$ 격자(간격 $3.04$\,m 및 $3.39$\,m) 전체에서 이루어지며, 지표는
공칭 유의파고로 정규화한 $\mathrm{nRMSE}=\mathrm{RMSE}/H_s$ 다. 채점 구간은
초기화 창을 제외한 $t>2T_p$ 이며, 이는 BF4 에서 301 프레임 중 160, BF5 에서 120 프레임에
해당한다. 제안 기법과 기준선은 이 구간과 이 격자를 공유한다.
\emph{관측의 공간적 지지집합은 채점 영역보다 훨씬 좁다}: 반환점의 절반이
$x<16.3$\,m 안에 놓이고 95\,\%가 $x<39$\,m 안에 놓이는 반면 채점은 $250$\,m 까지 이어진다.
따라서 보고되는 오차의 대부분은 외삽 영역에서 측정된 것이다.

두 코퍼스는 저장소의 생성 스크립트로 재생성할 수 있다. 선형 코퍼스는
\texttt{gen\_s8.py crowded} 와 \texttt{gen\_s9.py crowded}, 비선형 코퍼스는
\texttt{gen\_hos.py crowded} 와 \texttt{gen\_b12.py} 로 만들어진다. 다만
에피소드 파일 자체는 저장소에 포함되어 있지 않으며, 광선-삼각형 교차 계산이
GPU 와 CPU 에서 동일한 삼각형을 맞히되 좌표가 $10^{-13}$\,m 수준에서 갈리므로,
보장되는 것은 바이트 단위 동일성이 아니라 재생성 가능성이다.
```

