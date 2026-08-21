# AUDIT-B — 코드·배포·재현성 감사

**감사관**: AUDIT-B · **날짜**: 2026-08-12 · **작업 디렉터리**: `/home/vexit/wavesplat/wavesplat`
**원칙**: 코드를 수정하지 않았다. 발견만 보고한다. 열람 범위 = 주장 범위.

---

## 0. 감사 범위 선언

### 열었고 **줄 단위로 읽은** 것 (주장의 근거)
- `wavesplat/run.py` **전문(636줄)**
- `wavesplat/optimizer/sequential_update.py` **전문(1589줄)**
- `configs/estimation.yaml`, `configs/default.yaml`, `configs/server.yaml` **전문**
- `scripts/realdata_analysis.py:1-140`, `scripts/d3_sa_gonogo.py:40-120`, `scripts/d3_sa_multirecord.py:1-100`
- `scripts/d74_closed_loop_trajectory.py:24-95`(관측 생성부), `tmp/traj_step1b_relax.py:1-95`
- `wavesim/pipeline.py:54-60,98-100,142,150-200,255-260,307-345`, `wavesim/lidar_gpu.py:1-60`,
  `wavesim/wave_generator.py:95-100,130-140,330-360`, `wavesim/mesh_adequacy.py:1-40,112-132`
- `generate_dataset.py` **전문**
- `tmp/S9_lr_x_l1_2026-08-11/{run_s9.py:1-130, gate.py:20-145}`, `tmp/S7_.../run_s7.py:70-90`,
  `tmp/S8_.../run_s8.py:60-85`, `tmp/rx2_init_2026-08-05/run_rx2b.py`(grep 기반 구조),
  `tmp/B1_precondition_debate_2026-08-12/{EVIDENCE.md, make_evidence.py:84-96}`,
  `scripts/d72_platform_aperture.py:48-92`, `tmp/probe_lj_range_quality.py:1-80`

### **실행해서 측정**한 것 (grep 단정 아님)
- LJ_2018 Grid2 `.nc` 를 직접 열어 mask/클램프/통계 재계산 (96 프레임)
- `SequentialUpdater(R.CELLS['AUTO']['cfg']).lambda_l1` 을 **오늘의 코드로 실제 인스턴스화**
- `wavesim.config.load_config` 로 yaml 키 도달 여부 전수 확인
- `git log --numstat` (⚠ 최초 pathspec 이 틀려 한 번 오판했고, repo root 기준으로 재측정함)
- 저장소 전체 `episode_*.h5` 1107개의 `_dir` 태그 census

### **보지 않은 것** — "문제없음"이라 말하지 않는다
- `tmp/` 118개 디렉터리 중 **약 100개**(`S5_window_loss_2026-08-08` 이전 + flat `tmp/*.py`)
- `results/direction3/` 518개 json 중 13개 파일군만 표본 확인
- `paper/`, `thesis/`, `docs/`, `hydrofoil/`, `dcspecnet/`, `external/`, `ref_paper/`
- `wavesplat/model/*.py` 의 수치적 정확성(순전파 수식 자체) — 변경 이력만 봤다
- `wavesplat/baselines/*`, `wavesplat/evaluation/metrics.py` 내부 구현
- 3개 하위 에이전트가 낸 목록 중 **내가 직접 재확인하지 않은 항목**은 아래에서 `[미재확인]` 으로 표시

---

## BLOCKER (10건)

---

### 🔴 BLOCKER-1 — S7·S8 을 **오늘 재실행하면 기준선 = 처치**, 효과가 조용히 0 이 된다

**[파일:줄]**
- `tmp/S7_confirm_n32_2026-08-10/run_s7.py:78-79` — `R.CELLS.setdefault('AUTO', ...)`, 주석 `# 실제 출하 기본값... EXTRA 없음 = 아무것도 덮지 않는다`
- `tmp/S8_deploy_confirm_2026-08-10/run_s8.py:79` — `R.CELLS.setdefault('AUTO', ...)   # 출하 기본값, EXTRA 없음`
- `wavesplat/optimizer/sequential_update.py:173` — `self.lambda_l1 = config.get('lambda_l1', LAMBDA_L1_DEFAULT)`
- `wavesplat/optimizer/sequential_update.py:40` — `LAMBDA_L1_DEFAULT = 1e-4`

**[코드가 하는 일]** 두 배치의 기준선 팔 `AUTO` 는 `lambda_l1` 을 **일부러 명시하지 않고** 추정기 폴백에 맡긴다.
배치가 돌던 08-10 에 그 폴백은 `1e-3` 이었으므로 `AUTO`(1e-3) ≠ `AUTO-L1-1e4`(1e-4), 대조가 성립했다.
**08-11 배포로 폴백이 `1e-4` 가 되었다.**

**[실행 증명]**
```
S7/S8 AUTO cell cfg = {'init_method': 'auto', 'analytic_ensemble_k': 3}
  -> lambda_l1 the run WOULD use TODAY : 0.0001
  -> AUTO-L1-1e4 (treatment)          : 0.0001
  IDENTICAL? True
```

**[우리가 믿던 것]** "S7 은 λ 단독 널을 쟀다", "S8 은 혼잡 −13.5% 를 쟀다" — 재현 가능한 배치라고 믿었다.

**[영향]** S7·S8 의 **재현·확장·재실행이 전부 무효**. 짝 Δ 가 구조적으로 0 이 되고, 그 결과는
"λ 는 효과 없음"으로 읽힌다. ⛔이미 기록된 08-10/08-11 결과 자체는 당시 코드로 측정됐으므로
유효하지만, **"다시 돌려서 확인했다"는 문장은 앞으로 쓸 수 없다.**
S9 에서만 `run_s9.py:78-96` 으로 수리됐고 **S7·S8 은 미수리**이며, 두 파일 모두 `FROZEN_md5.txt` 에
**포함돼 있지 않아** 봉인을 깨지 않고는 고칠 수도 없다. `[일부 미재확인: FROZEN_md5 목록은 에이전트 보고]`

---

### 🔴 BLOCKER-2 — 이 부류 결함의 **탐지기(누출 프로브)가 원리적으로 못 잡는다**

**[파일:줄]** `tmp/S8_.../run_s8.py:74` `'AUTO-DUP2': {}` · `tmp/S9_.../run_s9.py:84` `'AUTO-DUP'`

**[코드가 하는 일]** `AUTO-DUP` 은 `AUTO` 의 바이트 복제이고, 게이트는 둘의 **비트 동일성**을 본다.
기준선이 조용히 처치값으로 미끄러질 때 **복제본도 같이 미끄러지므로 비트 동일성은 계속 성립**한다.

**[우리가 믿던 것]** "누출 프로브가 셀 오염을 잡는다."

**[영향]** BLOCKER-1 이 배치 도중 발생해도 **초록불로 통과**한다. 실제로
`tmp/S9_.../G0_selftest_on_S8.txt` 의 `워커 누출 프로브: 64 짝 중 비트 불일치 0개 ✅` 가 그 상태다.
저자 자신이 `run_s9.py:76-77,94` 에 이 한계를 적어 두었으나 **프로브는 교체되지 않았다.**

---

### 🔴 BLOCKER-3 — 사전 게이트가 **의도값을 인쇄**하고, 자기시험은 **자기 정답을 하드코딩**한다

**[파일:줄]**
- `tmp/S9_lr_x_l1_2026-08-11/gate.py:116-120` (G-6)
```python
extra = RUN.S5.S4.EXTRA.get(c, {})
print(f'  {c:16s} lambda_l1={extra.get("lambda_l1", RUN.L1_SHIP):.0e} ...')
```
- `tmp/S9_lr_x_l1_2026-08-11/gate.py:48-63` (G-0)

**[코드가 하는 일]**
(a) **G-6**: 런이 실제로 쓴 λ 가 아니라 `EXTRA` 딕셔너리(=실험자의 **의도**)를 인쇄한다.
키가 없으면 `RUN.L1_SHIP`(= `run_s9.py:68` 의 하드코딩 `1e-3`)로 대체하는데,
**옵티마이저의 실제 폴백은 `LAMBDA_L1_DEFAULT` = 1e-4** 다.
⟹ 키 없는 셀에서 게이트는 `lambda_l1=1e-03` 을 **초록으로 인쇄하고 런은 1e-04 를 쓴다.**
(b) **G-0 "채점기 자기시험"**: 채점기를 **호출하지 않는다.** 정답 7개를 리터럴로 들고
디스크에 이미 있는 `G0_selftest_on_S8.txt` 안에서 그 문자열을 `in` 으로 찾는다.
그 파일은 같은 저자가 같은 시각에 만든 전사본이고 `FROZEN_md5.txt` 에 함께 봉인돼 있다.

**[우리가 믿던 것]** "G-0 7/7 통과", "G-6 셀 cfg 검증" = 측정된 보증.

**[영향]** 두 게이트 모두 **측정이 아니라 자기 선언**이다. 08-11 E1 이 잡은 결함이
`run_s9.py` 에서만 수리되고 **게이트 자신은 그대로**다. S7·S8 게이트에는 셀 cfg 검사 자체가 없다.

---

### 🔴 BLOCKER-4 — `mask_Z` 를 믿는 실측 로더가 **3곳 남아 있다** (LJ Grid2 는 25.8%가 상수)

**[파일:줄]**
- `scripts/realdata_analysis.py:71-72` — `mask = _unpack(f['mask_Z'], np.s_[:]); Z[:, mask == 0] = np.nan` ← **공용 로더**
- `scripts/d3_sa_gonogo.py:70-71` — `mask = _unpack(...); self.valid = mask > 0` ← **`WassRecord` 기반 클래스**
- `tmp/realspread_2026-08-05/measure_records.py:84` — `mask = f['mask_Z'][:] == 1`

**[내가 실제로 파일을 열어 측정한 값]** (`LJ_2018/Surfaces_20180103_093938_Grid2_short.nc`, 96프레임)
```
mask_Z 고유값               : [1]        ← 격자 전체가 "유효"
mask_Z 유효 비율            : 1.0000     ← 100%
시간축 std < 1e-6 인 화소   : 0.2581     ← 25.8% 가 상수
그 상수값                   : -13.9391 m (77,720 화소)
NaN(_FillValue) 화소        : 0.0000     ← 마스크가 거짓말한 게 아니라, 마스크가 재는 것이 NaN 뿐
```
`mask_Z` 의 `long_name` 은 문자 그대로 `"mask of grid point timeseries without any NaN"` 이다.
**죽은 화소는 NaN 이 아니라 상수 클램프이므로 마스크의 정의상 걸리지 않는다.**

**[클램프가 통계에 미치는 크기 — 측정]**
| 지표 | 클램프 포함(현재 코드) | 클램프 제거(robust) |
|---|---|---|
| 시간평균면 공간 std | **6.231 m** | **3.271 m** |
| 4σ(η) | **27.205 m** | **18.210 m** |

**[우리가 믿던 것]** "오늘 LJ Grid2 결함을 발견했다."
⛔**정정: 이미 2026-07-04 에 발견돼 있었다.** `scripts/d3_sa_multirecord.py:23-25` 에
`LJ_2018 Grid2 has dead pixels NOT flagged by mask_Z (constant fill -13.94, temporal std = 0)
-> robust valid mask = mask_Z & (temporal std > 0.1 median) & (finite fraction > 0.9)` 이라 적혀 있고
`StridedRecord(robust=True)`(`:101-123`)로 **한 파일에만** 구현돼 있다. **공용 로더로 전파되지 않았다.**

**[추가 결함 — 오늘 만든 프로브 자신]** `tmp/probe_lj_range_quality.py:52-56`
```python
m = _unpack(f['mask_Z'], np.s_[0]) if 'mask_Z' in f else None
valid = np.isfinite(Z).all(0)
if m is not None: valid &= np.isfinite(m)
```
`np.isfinite(m)` 은 mask 가 전부 1 이므로 **항상 True = 완전 무연산**이다.
⟹ 이 프로브의 헤드라인 **"Grid2 시간평균면 공간 std = 6.2 m ⟹ 6 m 그릇은 바다가 아니라 스테레오 복원"** 은
**25.8% 클램프에 오염돼 있다.** 클램프를 빼면 **3.27 m**.
결론의 방향(3.27 m 도 바다가 아니다)은 살지만 **크기는 1.9배 부풀려져 있다. "6.2 m" 인용 금지.**

**[영향]** 실측 스테레오 특성표(`reference_bergamasco_stereo_dataset`), `realspread` 방향확산 측정,
`d3_sa_gonogo` 를 상속한 모든 실측 실험, 오늘의 LJ 범위-품질 판정.
⚠**단, `d3_sa_multirecord.py` 를 통과한 결과는 robust mask 가 적용돼 있어 영향 없다.**

**[⚠수치 불일치 고지]** B3 는 `31.47%` 를 적었고 나는 `25.81%` 를 쟀다.
나는 17,999 프레임 중 96개를 균등 샘플했다. 프레임 수·판정식이 다르면 값이 달라진다.
**어느 쪽이 맞는지 나는 확정하지 않았다.** 결함의 존재와 방향만 확정한다.

---

### 🔴 BLOCKER-5 — 저장소에 **"WaveSplat" 이라 불리는 서로 다른 추정기가 최소 4개** (직접 확인분)

| # | 경로 | 알고리즘 | 관측원 | "WaveSplat" 이라 부르는 증거 |
|---|---|---|---|---|
| 1 | `wavesplat/run.py:126` → `sequential_update.py:141` | 스펙트럴 가우시안 N개 + 분산관계 하드제약 + Adam coarse-to-fine + adaptive N + `w_phi` 추적 | **실제 레이캐스트 LiDAR** (`data_loader.py`, `lidar_points`) | `run.py:3` `"Main inference script for WaveSplat wave field estimation."` |
| 2 | `wavesplat/optimizer/bank_mmse.py:121 fit_bank_mmse` | **성분 추정 없음.** 파수 bank 위 선형 MMSE, 폐형해, 경사 없음 | 실측(GT-free) — 출하 경로의 real-domain 게이트가 라우팅 | 출하 패키지 내부 ⟹ 모든 실측 "WaveSplat" 숫자가 여기서 나온다 (`run.py:48-57`) |
| 3 | `scripts/d3_sa_nomp.py:499 mmse_bank_estimate` | #2 의 연구 원본 (별개 구현) | **해석식/HOS 면을 극좌표 부채꼴로 균일 샘플** (`d3_compare_gifs.py:105-108` `rr=rng.uniform(20,100)`, `aa=rng.uniform(-π/3,π/3)`) | `d3_compare_gifs.py:231` `print("  WaveSplat (analytic prior)...")`, `:260` `(('ws','WaveSplat'),...)` |
| 4 | `d18_relax_estimator.relax_step` + `wavesplat/model/tnse.py` `fit_amplitudes_phases`/`render_wave_field` | 탐욕 beamform peak-pick + ridge LS, **추적 없음**, 단일창 스냅샷 | **20×20 m 상자 균일난수 + 해석식 z** (BLOCKER-6) | `wavesplat/optimizer/relax_estimator.py:251` `"RELAX spectral initialization for WaveSplat."`; 브리핑에서 WaveSplat 으로 인용 |

**[실행 증명 — #3 이 디스크에 남기는 리터럴 키]**
```
results/direction3/compare_metrics.json
  /linear/WaveSplat   /hos/WaveSplat   /real/WaveSplat
```
이 `WaveSplat` 값은 **출하 추정기가 만든 것이 아니다.**

`[미재확인 3개 — 하위 에이전트 보고]` `d3_gpu_mmse.py:88 mmse_gpu`(torch 포팅),
`d3_stream_seqmmse.py:107 StreamMMSE`(스트리밍 RLS), `d3_sa_nomp.py:171 nomp_estimate`(NOMP).
포함하면 **7개**. 나는 4개를 직접 확인했다.

**[우리가 믿던 것]** "WaveSplat 결과"는 한 추정기의 성능이다.

**[영향]** 서로 다른 관측모델·서로 다른 알고리즘의 숫자가 같은 라벨로 표·논문·브리핑에 섞여 들어간다.
**저장소가 이미 이 결론에 도달해 있다** — `tmp/B1_precondition_debate_2026-08-12/DEBATE_P2_OE.md:129-130`.

---

### 🔴 BLOCKER-6 — `direction1` 계열: **LiDAR 0회, 20×20 m 균일난수, 해석식 z** — 그런데 오늘 결정의 근거다

**[파일:줄]** `scripts/d74_closed_loop_trajectory.py:40,67-85`
```python
L_FOOT = 20.0          # per-step LiDAR footprint patch size [m]   ← 주석만 "LiDAR"
...
ox = px + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)     # 20x20 m 상자 균일난수
oy = py + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
psi_w = (sea['kx']*ox + sea['ky']*oy - sea['om']*t + sea['phi'])
z = (sea['A'] * np.cos(psi_w)).sum(1) + rng.normal(0, SIGMA, M_STEP)   # 해석식 + N(0,0.02)
```

**[코드가 하는 일]** 메시 없음 · FOV 없음 · 입사각 없음 · raydrop 없음 · range 게이트 없음 · 차폐 없음.
`sea` 는 `make_sea`(:48-56)의 **8성분·kmag U(0.05,0.12)·spread 22°** 이지 JONSWAP/BF 해상상태가 아니다.
`import` 확인: `d74`, `traj_step1b_relax.py`, `traj_step1_netmmse.py`, `d72_platform_aperture.py`
**어느 것도 `wavesim` 을 임포트하지 않는다.**

**[우리가 믿던 것]** 이 결과들이 "우리 시스템"의 궤적/개구 성능이라고 읽어 왔다.

**[영향]** `results/direction1/*` 전부 → `EVIDENCE.md` → **오늘의 B-1 페르소나 토론** → **미결 결정 2건**.
특히 EVIDENCE.md §1 표의 `stationary 20x20` 은 센서의 성질이 아니라 **`L_FOOT=20.0` 이라는 상수**다.
같은 파일 §5 는 실제 LiDAR 에피소드에서 **정지 배가 30 s 에 173×197 m 를 훑는다**고 적는다.
**같은 문서 안에서 두 관측모델이 화해 없이 나란히 놓여 있다.**

---

### 🔴 BLOCKER-7 — 선체 운동이 **파이프라인에 구현돼 있지 않다** (개입 지점 부재)

**[파일:줄]**
- `wavesim/pipeline.py:142` — `origins, ray_dirs = generate_rays(lidar_cfg, ship_x=0.0, ship_y=0.0)` ← **하드코딩**
- `configs/default.yaml:66-71` — 이동선 시나리오 S11–S15 가 **전부 주석 처리**,
  주석: `# Moving ship (requires ship motion in pipeline — Phase 2)`

**[코드가 하는 일]** `run_episode` 전체에 `ship_speed` / `heading` / `velocity` / `motion` 파라미터가 **없다.**
`grep` 결과 `pipeline.py` 에서 선체 위치 관련 토큰은 142행 한 줄뿐이다.

**[우리가 믿던 것]** 플랫폼 운동/합성개구 축은 "재면 되는" 축이다.

**[영향]** ★**오늘 대기 중인 PRO 배치(신선 128ep·1584런, 플랫폼 운동/조우기하 축)에는
출하 생성기 상의 개입 지점이 없다.** 08-11 P1 을 런 0회로 죽인 것과 **동일한 실패 유형**
(`동기는 있으나 개입 지점 4건 부재`). 저장소의 모든 "이동 플랫폼" 결과는
BLOCKER-6 의 상자 대체물로 측정된 것이지 LiDAR 파이프라인이 아니다.
⟹ **배치 전에 `pipeline.py` 에 선체 운동을 먼저 구현해야 한다. 이것은 코드 작업이지 실험이 아니다.**

---

### 🔴 BLOCKER-8 — 정확도 배치 전체가 **following seas**(파가 센서에서 **멀어지는** 반-인과 기하)

**[파일:줄]**
- `wavesim/wave_generator.py:138` — `direction_deg: float = 0.0` (함수 시그니처 기본값)
- `wavesim/wave_generator.py:98` — `phase = c.kx*x + c.ky*y - c.omega*t + c.phi` ⟹ 위상면은 **+k̂** 로 진행
- `wavesim/wave_generator.py:332-334,357-358` — `dir_rad = deg2rad(direction_deg)` → `kx=k·cos, ky=k·sin`
- `wavesim/pipeline.py:142` — 배는 원점, heading 0(= +x), `configs/default.yaml:30` GT 격자 `x∈[10,250]` = 전방
- `wavesim/pipeline.py:257` — `dtag = '' if abs(direction_deg) < 1e-9 else f"_dir{int(direction_deg)}"`

**[측정]**
```
episode_*.h5 총계          : 1107
그중 _dir180 태그          :   90
그 외 dir 태그             :    0
⟹ 1017 / 1107 = 91.9% 가 direction_deg = 0
S7/S8/S9/S11 러너·생성기의 'direction' 토큰 : 0건 (전부 시그니처 기본값 0 상속)
```

**[코드가 하는 일]** `direction_deg=0` ⟹ 파는 **+x 로 진행** = 배가 보는 방향과 **같은 방향** =
전방 부채꼴에서 관측한 물이 **배에서 멀어진다.** 저장소가 스스로 이렇게 적는다 —
`scripts/d3_h6_v2_e2e.py:15` `"direction_deg = 0 (waves travel +x, AWAY from the sensor)
-- the anti-causal configuration for encounter"`.

**[우리가 믿던 것]** 정확도 배치와 제어 배치가 같은 우주를 잰다.

**[영향]** ★**완전한 이분화**: 제어·E2E·hydrofoil 실험은 **전부 180**
(`d3_e2e_stream_ff.py:103`, `d3_d17_ctrl_closedloop.py:197`, `d3_e2e_velff.py:48`,
`d3_e2e_ctrl_tune.py:53,86`, `d3_d18_ctrl_tune.py:314,349,458`, `hydrofoil/*` 시나리오표 전체 —
**3건은 내가 직접 확인**, 나머지는 에이전트 census).
정확도 배치(S1–S11 전부)는 **전부 0**.
⟹ **"정확도 이득이 제어 이득으로 이어진다"는 문장은 두 다른 기하를 가로질러 인용하는 것이다.**
S2 가 잰 `α_wave` 혼잡 17–22% 도 이 교락 위에 있다.

---

### 🔴 BLOCKER-9 — `init_method` 폴백이 **세 곳에서 서로 다르다**

**[실행 증명]**
```
run.DEFAULT_CONFIG         : auto        (wavesplat/run.py:43)
load_yaml_config 폴백      : v2          (wavesplat/run.py:99)
SequentialUpdater 폴백     : v2_konly    (sequential_update.py:181)
```

**[코드가 하는 일]** `run_estimation` 을 거치면 `auto` 를 받지만,
(a) yaml 에서 `init_method` 키가 빠지면 → **`v2`**
(b) `SequentialUpdater`/`make_updater` 를 **직접 생성**하면 → **`v2_konly`**, 그리고
`make_updater` 는 `init_method=='auto'` 일 때만 `MedianEnsembleUpdater` 를 준다
(`sequential_update.py:1585`) ⟹ **median-of-3 앙상블도 조용히 꺼진다.**

**[우리가 믿던 것]** "출하 팔"을 돌리고 있다.

**[영향]** 이것이 08-08 S1 이 잡은 `출하 팔은 beamform 0회 호출` 결함의 **일반형**이다.
`SequentialUpdater`/`make_updater` 를 직접 구동하는 실험 드라이버(에이전트 집계 **약 49개** `[미재확인]`)는
`init_method` 를 명시하지 않는 한 **출하 팔이 아니다.**
✅**규율 사례**: `run_rx2b.py:179` 는 `assert 'init_method' in cfg  # W9 discipline: never implicit` 로
이 문제를 막고 있다. **이 assert 를 다른 드라이버로 확산시켜야 한다.**

---

### 🔴 BLOCKER-10 — 오늘 결정의 근거인 `EVIDENCE.md` 에 **손으로 적은 숫자**가 "기계 생성" 배너 아래 들어 있다

**[파일:줄]** `tmp/B1_precondition_debate_2026-08-12/make_evidence.py`
- `:2` 독스트링 `"원자료에서 생성 ... 손으로 옮긴 숫자 없음"`, `:22` 그 문장을 출력물에 **인쇄**
- `:90-95` — 실제로는 **리터럴 문자열**:
```python
print("- 에피소드 = 30.0 s, 301 프레임. 프레임당 유효 반사점 중앙 50/71.")
print("- **정지 배도 30 s 동안 173x197 m 를 훑는다** ... 피복 4,188 m².")
print("- following 11.3 = 19,024 m²(4.5x) · crossing 11.3 = 12,396 m²(3.0x) · ... 10,068 m²(2.4x)")
```
§5 에서 **실제로 계산되는 것은 `c_p` 뿐**(`:84-88`, 8개 h5 중앙값).

**[완화 요인 — 공정하게 기록]** `:92-93` 에 자기 정정이 **이미 인쇄**돼 있다:
`⛔위 두 값은 토론이 정정했다 — n=1 이고(실제 149–203 m), extent 이지 개구가 아니다
(유효개구 39.5×48.3 m). 피복은 cell=2.0 m 손잡이에 의존(cell 1→5 m 에서 3.7배 변동).`
⟹ **숨긴 것이 아니라 배너와 본문이 서로 모순**이다. 그러나 배너를 믿고 §5 를 인용하면 틀린다.

**[추가 — §3 은 n=1]** `scripts/d72_platform_aperture.py:50`
`def realistic_sea(speed, heading_deg, ..., seed=1)` 이고 `:88` 은
`dl = [realistic_sea(v, head) for v in speeds]` — **seed 를 넘기지 않는다.**
⟹ 7개 속도가 **동일한 난수 실현 하나**를 공유한다. 분산 추정치 0, 속도 스윕이 단일 시드와 완전 교락.
그런데 EVIDENCE.md §3 은 `**verdict**: CONFIRMED: ... (≥2× sharper)` 로 적혀 있다.
⛔**"CRB 2배 붕괴"를 n=1 이라는 고지 없이 인용 금지.**

**[영향]** B-1 토론 5인 → 미결 결정 2건.

---

## WARN

### ⚠W-1 — `configs/estimation.yaml` 의 **7개 키가 코드에 전혀 도달하지 않는다**
`wavesplat/run.py:81-123 load_yaml_config` 가 읽지 않는 키:

| yaml 키 | 코드의 실제 값 | 결과 |
|---|---|---|
| `optimizer.max_steepness: 0.15` | `sequential_update.py:208` 함수 기본값 `0.15` | **무연산** (값이 우연히 같아 무해) |
| `init.tp_factor: 2.0` | `run.py:161,198` 에 `tp_factor=2.0` **하드코딩** | 무연산 |
| `init.grid_nx: 32` / `grid_ny: 24` | `initializer.py:218,668` 시그니처 기본값 32/24 | 무연산 |
| `eval.converge_threshold: 0.05` | `run.py:204` `converge_threshold = 0.05` **하드코딩** | 무연산 |
| `eval.split_distance: 80.0` | `metrics.py:99` 기본값 80.0 | 무연산 |
| `eval.unobserved_min_dist: 10.0` | 이 이름은 `wavesplat/` 어디에도 없음(`metrics.py:128 min_dist=10.0`) | 무연산 |

★특히 `max_steepness` 는 `sequential_update.py:204-207` 에
`"Was documented above and set in configs/estimation.yaml, but never read ... (2026-07-24)"`
라고 **수리했다고 적혀 있으나, yaml→코드 배선은 지금도 없다. 반쪽 수리다.**
⟹ **"estimation.yaml 한 줄만 바꾸면 된다"는 문장은 이 7개 키에 대해서는 거짓이다.**

### ⚠W-2 — `model.depth` 는 **항상 덮어씌워진다**
`run.py:142` 에서 읽고 `run.py:151` `cfg['depth'] = loader.depth` 로 **무조건 교체**. yaml 값은 무의미.

### ⚠W-3 — S8 게이트 두 곳이 **낡은 `0.001` 을 하드코딩**하고 주석이 거짓이 됐다
- `sequential_update.py:426` (`S8_L1_TRACK_ONLY`) — `self.lambda_l1 = 0.001`
- `sequential_update.py:965-967` (`S8_L1_PHASE1_ONLY`) — `self.lambda_l1 = 0.001`,
  주석 `"pinned to the SHIPPED default (configs/estimation.yaml:12, lambda_l1=0.001)"`
  → **estimation.yaml:12 는 지금 `0.0001` 이다.**
⟹ 이 게이트를 오늘 켜면 "출하 기본값에 고정"한다고 말하면서 **출하 아닌 값에 고정**한다.
**상수 이중기재 결함의 4번째 재발이며 현재 파일에 살아 있다.**
(동류: `run_s9.py:67-68` `TRACK_LR ... (estimation.yaml:16)` / `L1_SHIP = 1e-3 ... (estimation.yaml:12)`
— 두 줄 인용이 모두 현재 파일과 불일치하고, `L1_SHIP` 은 더 이상 "출하 기본값"이 아니다.)

### ⚠W-4 — 메시 부적합 경고가 **문서화된 호출에서 침묵**한다
- `configs/default.yaml:25-26` — `mesh: nx 40 / ny 30`
- `wavesim/mesh_adequacy.py:8-12` — `"the shipped default (mesh 40x30 ...) renders the BF2
  1-component sea at 1.2 points per wavelength -- below Nyquist. Every conclusion drawn from
  those episodes measured the simulator, not the sea"`
- `wavesim/pipeline.py:56` — `strict_mesh: bool = False` ⟹ sub-Nyquist 여도 **예외 안 남**
- `mesh_adequacy.py:131` — `if verbose: print(...)`; `generate_dataset.py` 는 `verbose=args.verbose`(기본 False)

⟹ CLAUDE.md 가 안내하는 `python generate_dataset.py --config configs/default.yaml` 은
**알려진 결함 메시로, 경고 한 줄 없이** 데이터를 만든다. (`configs/server.yaml:25-26` 은 80×60 로 개선)

### ⚠W-5 — 메모리의 자기 정정 자체가 과잉이다: **모든 생성 드라이버가 GPU 인 것은 아니다**
`wavesim/lidar_gpu.py:19 import torch` 는 **맞다**(확인). 그러나
- `wavesim/pipeline.py:57` — `use_gpu_raycast: bool = False` ← **기본값 CPU**
- `generate_dataset.py` — `run_episode(...)` 호출에 `use_gpu_raycast` **없음** ⟹ CPU
  (`tmp/regen_run_finemesh.py:10-13` 이 이 사실을 명시적으로 적고 있다: `"silently falls back to
  the ~14x slower CPU path"`)
- `tmp/heldout40_2026-07-26/gen_heldout40.py:122` — `use_gpu_raycast=False` ← **명시적 CPU**
✅ S7/S8/S9/S1/indep 생성기는 전부 `True` (확인).
✅ GPU 경로는 `lidar_gpu.py:1-4` 가 `"hit-for-hit identical"`(좌표 ~1e-13 m 차)이라 주장하고
`wavesim/tests/test_lidar_gpu.py` 가 존재한다 — **테스트 통과 여부는 내가 실행하지 않았다.**

### ⚠W-6 — **출하 기본값을 고정하는 테스트가 하나도 없다**
`wavesplat/tests/*.py` 전체에서 `lambda_l1` 은 항상 **명시적으로 전달**되고(`0.0` 등),
`init_method` 도 항상 명시된다. `DEFAULT_CONFIG['lambda_l1'] == LAMBDA_L1_DEFAULT == yaml 값`
을 검사하는 테스트가 **없다.**
⟹ 오늘 만든 "단일 출처"(`sequential_update.py:40`, 커밋 `5f0846c`, **오늘 날짜**)에는 회귀 방어가 없다.
S9 사전등록이 약속한 `단일출처+자기시험` 중 **자기시험이 미구현**.

### ⚠W-7 — `configs/default.yaml:50` 주석이 실제와 불일치
`# 15 scenarios × 5 seeds × 2 raydrop models = 150 episodes` 인데 활성 시나리오는 **10개 = 100** 이다
(CLAUDE.md 의 "100 total" 과 일치). 무해하지만 낡았다.

### ⚠W-8 — 고아 결과 파일 (하위 에이전트 보고, `[미재확인]`)
생성기를 찾지 못한 것: `results/forecast_experiment.json`, `results/operability_curve.json`,
`results/realdata_LJ2018_Grid1.json`, `results/neural_guided_relax_grid.json`,
`results/gp_disp_pilot.json`(이름 불일치), `results/tetnp_stage1_sanity.json`(이름 불일치),
`results/direction3/posterior_eval_v{1,2}.json`(수동 리네임),
`tmp/P1_real_path_2026-08-11/regress_synth_{before,after}.json`.
그리고 **오늘 인용된 3개 디렉터리가 git 미추적**:
`tmp/B1_precondition_debate_2026-08-12/`, `tmp/B3_regime_position_2026-08-12/`,
`tmp/S11_null_corrected_2026-08-12/`.

### ⚠W-9 — `EVIDENCE.md` 의 생성기는 **사후에 재구성**된 것
mtime: `EVIDENCE.md` 13:29 → `make_evidence.py` 13:58 (**29분 뒤**).
`make_evidence.py:4-6` 이 스스로 인정한다(원본은 heredoc 으로 돌리고 버렸다).
⟹ 생성기가 저장돼 있다는 점은 ✅ 이지만, **그 파일이 그 출력을 만든 파일은 아니다.**

### ⚠W-10 — 봉인(`FROZEN_md5.txt`)에 **러너가 빠져 있다** `[미재확인]`
S7/S8/S9 모두 `run_s*.py` 가 봉인 목록에 없고, S9 는 `results.jsonl`(15:35) 이후
`run_s9.py`(17:43)가 수정됐다. 봉인은 이 간극을 탐지할 수 없다.

---

## UNVERIFIED (열지 않았거나, 열었으나 판정 불가)

1. **`추정기 변경 0줄` 논쟁(CONTESTED)** — 아래로 **부분 해소**했으나 완전 해소는 아니다.
   ```
   git log --since=2026-07-01 --numstat -- wavesplat/wavesplat/    (repo root 기준)
     sequential_update.py  +909 / -60      run.py          +291 / -21
     learned_init.py       +412 /  -6      initializer.py  +242 / -10
     bank_mmse.py          +222 /  -0      coarse_to_fine.py +94 / -18
     metrics.py            +116 /  -1      tests(4개)      +729 /  -3
     model/  (순전파)        변경 없음
     ─ 합계 +3015 / -119, 그중 sequential_update+run.py 추가분 = 주석 400 / 코드 710
   ```
   ⟹ **"0줄"은 거짓**이다. 다만 **순전파 모델(`model/`)은 정말 0줄**이고,
   바뀐 것은 대부분 **기본값·라우팅·앙상블·계측·연구 게이트**다.
   출하 기본값은 최소 7회 바뀌었다: `TRACK_PHASE_PREDICT` True→False(07-25 `a1a1d4e`),
   `init_method` v2_konly→auto(07-27 `e4209ee`), `analytic_ensemble_k`=3 신설(동), `auto_guard`
   True→False(동), `real_estimator`=bank_mmse 신설(07-29 `5a9841a`), `real_bank_ensemble_k`=3(동),
   `lambda_l1` 1e-3→1e-4(08-11/08-12 `5f0846c`).
   ⟹ **두 페르소나가 서로 다른 대상을 두고 옳았다.** 어느 쪽 정의가 결정에 맞는지는 사용자 판정 사항.
   ⚠**내 오류 자백**: 최초 측정에서 pathspec 을 한 단계 틀리게 줘서 `model/` 을 "변경 없음"이라
   **잘못 확인**할 뻔했다. repo root 기준으로 재측정해 정정했다.

2. `wavesim/tests/test_lidar_gpu.py` 의 hit-for-hit 동등성 **실행하지 않았다.**
   ⟹ CPU 생성(`generate_dataset.py`, `gen_heldout40.py`)과 GPU 생성 데이터의 **동등성 미확인.**

3. B3 의 `31.47%` vs 내 `25.81%` — **어느 쪽이 맞는지 확정 못 했다** (프레임 샘플·판정식 차이).

4. `tmp/` 약 100개 디렉터리, `results/direction3/` 518개 중 대부분, `paper/`·`thesis/`·`docs/`·
   `hydrofoil/`·`dcspecnet/` — **열지 않았다.**

5. "WaveSplat 추정기 7개" 중 3개(`mmse_gpu`, `StreamMMSE`, `nomp_estimate`)는 **내가 직접 안 봤다.**

6. `direction_deg` census 의 그룹별 세부 목록(약 80개 파일)은 하위 에이전트 결과이고,
   나는 **4개 지점 + 전체 파일 태그 census + 부호 규약**만 직접 재확인했다.

7. S7/S8/S9 `FROZEN_md5.txt` 내용·mtime 은 **하위 에이전트 보고**이며 내가 직접 열지 않았다.

---

## 요약 — 지금 무엇을 멈춰야 하는가

1. ⛔**S7·S8 "재실행 확인" 금지** — 오늘 돌리면 기준선 = 처치다 (BLOCKER-1, 실행 증명).
2. ⛔**"6.2 m 그릇" 인용 금지** — 클램프 제거 시 3.27 m (BLOCKER-4, 측정).
3. ⛔**`results/direction1/*` 와 `results/direction3/*` 의 "WaveSplat" 을 출하 성능으로 인용 금지**
   (BLOCKER-5·6).
4. ⛔**"CRB 2배 붕괴"를 n=1 고지 없이 인용 금지** (BLOCKER-10).
5. ⛔**정확도 결과와 제어 결과를 가로질러 인용 금지** — 기하가 다르다 (BLOCKER-8).
6. ★**PRO 배치를 승인하기 전에 `wavesim/pipeline.py` 에 선체 운동을 먼저 구현해야 한다** —
   지금은 개입 지점이 없다 (BLOCKER-7).
