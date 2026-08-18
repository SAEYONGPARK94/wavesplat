# W1 — 배포 경로 정찰 (런 0회, 정적 추적 + 기존 결과 재집계만)

작성 2026-08-19 · 범위 = 아래 "열람 목록"에 있는 파일만. 추정기 실행 **0회**.
표기 `[M]` = MEASURED(코드 줄 또는 저장된 결과행으로 확인) · `[I]` = INFERRED(추론).
경로는 전부 `/home/vexit/wavesplat/wavesplat/` 기준.

**열람 목록**: `wavesplat/run.py` · `wavesplat/optimizer/{sequential_update,coarse_to_fine,initializer,learned_init}.py` ·
`wavesplat/model/{spectral_gaussian,tnse}.py` · `configs/estimation.yaml` ·
`tmp/W3_SOLVER_2026-08-15/{run_w3.py,results.jsonl,verdict_w3.json}` ·
`tmp/S17_WHITE25_2026-08-18/{recon/T1B_corpus.md,AMENDMENT_2.md,results.jsonl}`.

---

## 0. 결론 먼저

### 0.1 삽입 지점 (배포 혼잡 경로에서 **실제로 실행되는** 자리)

| 후보 | 삽입 지점 | 실행 보장 | 패키지 수정 |
|---|---|---|---|
| **post-lasso 재적합** | `SequentialUpdater.initialize` **직후**(= `sequential_update.py:608`·`:662` 의 `_build_tracking_optimizer()` 다음). 몽키패치는 `SequentialUpdater.initialize`(`:405`)를 감싸 원본 반환 후 `(a,b)` 닫힌형 재적합 → `w.copy_()`/`phi.copy_()` | **런당 정확히 1회, 두 라우트 전부** `[M]` | **불필요** |
| **VARPRO** | `SequentialUpdater._initialize_learned`(`:617-675`) 전체 교체 = learned 라우트. v2_konly 도 덮으려면 `_initialize_inner`(`:434-615`) 의 `:561-591` 루프까지 | learned = 혼잡 **86.5~90.3%** `[M]` | **불필요**(메서드 몽키패치) |
| **`merge_atoms`** | 호출 지점은 **이미 존재** — `learned_init.py:367-369`. `config['learned_merge_frac']` 만 넣으면 켜진다 | learned 라우트 한정(= 혼잡 86.5~90.3%), v2_konly 행은 **비트 동일 널** `[M]` | **불필요** (배치에서 `run_estimation(..., config={'learned_merge_frac': x})`) |

### 0.2 `merge_atoms` — 기록이 맞다, 단 "미노출"의 의미를 좁혀야 한다

- ✅ **구현돼 있다** — `learned_init.py:190-235`. 단일연결 응집, 반경 `r = merge_frac·kp`, 에너지가중 중심 + 에너지 합. `[M]`
- ✅ **GT-free** — 입력은 제안 원자 `(kx,ky,energy)` 뿐. `[M]`
- ✅ **끄면 비트 동일** — `:205` `if merge_frac is None or merge_frac <= 0 or n <= 1: return kx, ky, en` (입력 배열 그대로 반환) + `:368` `if merge_frac:` 이 호출 자체를 건너뛴다. `[M]`
- ✅ **기본값 `None`** — `:367` `config.get('learned_merge_frac')`. 어떤 기본값 dict 에도 이 키가 없다. `[M]`
- ⚠ **"`configs/`·`run.py` 에 미노출"은 참이나, "도달 불가"는 거짓** `[M]`:
  - `configs/estimation.yaml` 에 `learned_*` 키 **0개**(`grep learned configs/` → 주석 2줄뿐).
  - `run.py` 에 `learned_merge_frac` 문자열 **0회**, CLI 플래그 없음.
  - `load_yaml_config`(`run.py:81-123`)는 **화이트리스트**라 yaml 에 키를 적어도 **버려진다**.
  - **그러나** `run_estimation(h5, config=…)` → `cfg.update(config)`(`run.py:141`) → `make_updater(cfg)`(`:168`) → `SequentialUpdater.__init__` 이 `self.config = config` 보관(`sequential_update.py:171`) → `_initialize_learned` 가 `self.config` 를 그대로 전달(`:632`) → `learned_init_params(config=…)` → `:367` 이 읽는다. **⟹ 배치 하네스는 패키지 무수정으로 켤 수 있다.**
- **노출하려면 정확히 4곳**(W4 에는 불필요, 배포 결정용):
  1. `run.py:23-70 DEFAULT_CONFIG` 에 `'learned_merge_frac': None,` 추가
  2. `run.py:91-105`(optimizer 블록)에 `if 'learned_merge_frac' in raw['optimizer']: cfg['learned_merge_frac'] = float(...)` 추가
  3. `configs/estimation.yaml` optimizer 블록에 키 + 주석
  4. (선택) `run.py` 에 `--learned_merge_frac` argparse + `:598` 부근 오버라이드
- ⚠**라우팅과 교락 없음** `[M]`: 라우터는 `_auto_select_init`(`sequential_update.py:724-729`)에서 `pool_and_select` 만 부르고 **merge 를 부르지 않는다**. merge 는 라우팅 후 `learned_init_params` 안에서만 일어난다 ⟹ `n_selected` 도 `route` 도 안 바뀐다 = 깨끗한 단일 요인.
- ⚠**분리 문턱 재확인** `[M]`: `pool_and_select` 의 `dedup_dk` 기본 `0.03`(`learned_init.py:341`), 해석적 `MIN_DK = 0.15`(`initializer.py:63`). merge 는 **선택된 뒤** 다시 뭉치므로 두 문턱이 직렬로 걸린다.

---

## 1. 배포 혼잡 콜 그래프 (`init_method='auto'`)

```
run.main / run.run_estimation                       run.py:126
 └ cfg = DEFAULT_CONFIG.copy(); cfg.update(config)  run.py:139-141
    · init_method='auto'  (run.py:43)   · lambda_l1=LAMBDA_L1_DEFAULT=1e-4 (run.py:30 ← sequential_update.py:40)
    · analytic_ensemble_k=3 (run.py:47) · tracking_mode='w_phi' (run.py:33) · tracking_steps=10 (run.py:32)
 └ init_frames = loader.get_init_window(tp_factor=2.0)      run.py:161
 └ updater = make_updater(cfg)                              run.py:168
     └ auto + k>1 ⟹ MedianEnsembleUpdater                   sequential_update.py:1586-1588
 └ updater.initialize(init_frames, …)                       run.py:169   ← ★Phase 1
 └ for frame in loader.get_tracking_frames(2.0):            run.py:198,246
      updater.update(frame)                                 run.py:274   ← ★Phase 2 (프레임당)
      updater.predict_field(grid_x, grid_y, frame.t)        run.py:277
```

### 1.1 라우터 분기 — `_auto_select_init`

```
MedianEnsembleUpdater.initialize                 sequential_update.py:1487
 └ self._primary.initialize(frames,…)            :1493
    └ SequentialUpdater.initialize               :405   (S8 λ 게이트 래퍼, 기본 무동작)
       └ _initialize_inner                       :434
          ├ if init_method=='auto':              :443
          │   ├ REAL-DOMAIN GATE (gt_field 부재) :460-508  → bank_mmse   ← 합성 코퍼스에선 미발동
          │   └ else: init_method = _auto_select_init(frames)  :511
          │        ├ propose_atoms(…)  (확산 제안, GPU)   learned_init.py:121  ← 원자 캐시 :723
          │        ├ pool_and_select(energy_thresh .03, dedup_dk .03, max_n 30)  :724-728
          │        └ n_sel >= auto_switch_thresh(8) ? 'learned' : 'v2_konly'     :730
          │           (guard 는 2026-07-27 RETIRED, 기본 False :754)
          ├ if init_method=='learned': return _initialize_learned(...)   :516-517
          └ (아니면 아래 v2_konly 코스투파인)
 └ 앙상블: route != 'v2_konly' ⟹ ensemble_k=1 (멤버 1개)   :1498-1506
    route == 'v2_konly' ⟹ 멤버 3개(시드 s, s+1000, s+2000)  :1511-1524
```

**라우터 실측** `[M]`
- W3 `SHIP` 144행: **learned 130 / v2_konly 14** = 90.3% learned (`tmp/W3_SOLVER_2026-08-15/results.jsonl` 의 `init_method_resolved` 재집계).
- S17 이 참조한 S13+S13b 수입 `SHIP` 384행: **learned 332 / v2_konly 52** = 86.5% learned
  (`tmp/S17_WHITE25_2026-08-18/recon/T1B_corpus.md:100`, `AMENDMENT_2.md:21`).
- ⟹ **혼잡의 86~90%가 `_initialize_learned`**, 나머지 10~14%만 코스투파인. 두 라우트 모두 W4 가 덮어야 한다.

### 1.2 learned 라우트 (배포 혼잡의 주경로)

```
_initialize_learned                              sequential_update.py:617
 ├ learned_init_params(frames, depth, config, seed)         :631   → learned_init.py:324
 │   ├ 원자: config['learned_atoms'] 캐시(라우터가 :723 에 넣음) 또는 propose_atoms
 │   ├ pool_and_select(e_th .03, dedup_dk .03, max_n 30)     learned_init.py:355
 │   ├ [옵션] peak_anchor_filter — learned_peak_anchor 기본 False (no-op)   :360-364
 │   ├ [옵션] merge_atoms       — learned_merge_frac 기본 None  (no-op)     :367-369   ★후보
 │   └ (w, φ): learned_konly 기본 False ⟹ ls_fit_wphi        :386  → tnse.fit_amplitudes_phases(ridge 0.5)
 ├ model = _build_model(n)  ; kx/ky/w/phi copy_              :634-639
 ├ _build_optimizer()                                        :640   (Adam, 아래 §2)
 ├ ★ Adam 500 iters (learned_init_iters 기본 500)            :643-651
 ├ _adaptive_prune_grow(frames,…)                            :660   (adaptive_n 기본 True)
 └ _build_tracking_optimizer()                               :662   ← ★후크 후보(정확히 여기와 :608 두 곳뿐)
```
⚠**독스트링이 스스로 코스투파인 우회를 적고 있다** — `:620-626` *"the coarse-to-fine N-staging is bypassed"*. `[M]`

### 1.3 v2_konly 라우트 (혼잡의 10~14% + 앙상블 3멤버)

```
_initialize_inner (계속)                          sequential_update.py:519-608
 ├ CoarseToFineSchedule(stages [3,8,15], thr [.1,.05,None], iters [200,200,500])  :525-531
 ├ initialize_from_frames(init_method='v2_konly')            :544  → initializer.py:665,705
 │   └ initialize_from_frames_v2 (빔포밍 + 그리디 차감)       initializer.py:453-588
 │      · KONLY_RANDOM_WPHI ⟹ **적합된 (w,φ) 버리고** w=0.01, φ~U(0,2π)  initializer.py:710-718
 ├ ★ Adam max_total = sum(max_iters) = 900 iters             :561-591
 │   └ schedule.should_advance ⟹ advance() → _beamform_residuals → add_gaussians
 │        coarse_to_fine.py:100-190, :192-272  (배포 사다리에서 15원자 중 12개를 여기서 배치)
 ├ _adaptive_prune_grow                                       :605
 └ _build_tracking_optimizer()                                :608   ← ★후크 후보
```

### 1.4 Phase 2 (양 라우트 공통)

```
SequentialUpdater.update(frame)                   sequential_update.py:911
 ├ TRACK_PHASE_PREDICT=False(:83) ⟹ φ += ω·Δt **실행 안 됨**            :938
 ├ _frame_buffer 최근 tracking_window=3 프레임 유지                       :948-950
 └ ★ for step in range(tracking_steps=10): opt.step()                    :970-974
      loss = _compute_loss(self._frame_buffer, reg=True)                 :972
   clamp_steepness(0.15)                                                  :979
```

### 1.5 `(w, φ)` 와 `(kx, ky)` 가 실제로 갱신되는 함수 — 확정 목록 `[M]`

| 파라미터 | 갱신 지점 | 라우트 | 비고 |
|---|---|---|---|
| `w, φ` | `sequential_update.py:649` `self.optimizer.step()` | learned | Adam lr **0.01**, 500 iters |
| `w, φ` | `sequential_update.py:565` `self.optimizer.step()` | v2_konly | Adam lr **0.01**, ≤900 iters |
| `w, φ` | `sequential_update.py:1208` `self.optimizer.step()` | 양쪽(성장 시) | Adam lr 0.01, **50 iters** |
| `w, φ` | `sequential_update.py:974` `opt.step()` | 양쪽 | Adam lr **0.005**, 프레임당 10 스텝 × 120~220 프레임 |
| `w` (clamp) | `spectral_gaussian.py:219-230` `clamp_steepness` | 양쪽 | 매 스텝, `ε=|k||w| ≤ 0.15` 비선형 사영 |
| `kx, ky` | `sequential_update.py:565` / `:649` / `:1208` | 양쪽 | Adam lr = `lr·0.5·init_k_lr_scale` = **0.005** (`:262-265`) |
| `kx, ky` | **추적 중 갱신 없음** | 양쪽 | `tracking_mode='w_phi'` ⟹ `requires_grad_(False)` (`:287-288`) |
| `w, φ` 설치(비경사) | `initializer.py:717-718`(v2_konly 초기), `learned_init.py:386`(learned LS), `coarse_to_fine.py:181-190`·`sequential_update.py:1194-1200`(신규 원자) | — | 학습 라우트의 초기 `(w,φ)` 는 **이미 LS 적합**(ridge 0.5) |

---

## 2. 최적화가 실제로 일어나는 지점 — 전수

| # | 위치 | 라우트 | 옵티마이저 | lr | 반복 | 손실 | 실행 여부 |
|---|---|---|---|---|---|---|---|
| O1 | `:645-651` | learned | Adam | w,φ 0.01 / k 0.005 | 500 | `_compute_loss(init_frames)` | 항상 `[M]` |
| O2 | `:561-591` | v2_konly | Adam | 동일 | ≤900 | `_compute_loss(init_frames)` | 항상 `[M]` |
| O3 | `:1204-1209` | 양쪽 | Adam | 동일 | 50 | `_compute_loss(init_frames)` | **혼잡 17/96 · 희소 6/48 에서만** `[M]` |
| O4 | `:970-974` | 양쪽 | Adam | 0.005 | 10/프레임 | `_compute_loss(buffer[≤3])` | 항상 `[M]` |
| O5 | `learned_init.py:386` | learned | 닫힌형 ridge LS | — | 0 | `‖Φc−z‖²+0.5‖c‖²` | 항상(초기값) `[M]` |
| O6 | `initializer.py:417` `_linear_projection` | v2_konly | 닫힌형 사영 | — | 0 | — | 값이 `:717-718` 에서 **버려짐** `[M]` |

**O3 의 실행률은 W3 결과행에서 원자 카운터로 실측된다** `[M]`:
`tmp/W3_SOLVER_2026-08-15/results.jsonl` 의 `fitphi_atoms + randphi_atoms > 0` 인 `SHIP` 행 =
**혼잡 17/96, 희소 6/48**(추가된 원자 총 430 / 207). `add_gaussians` 호출은 이 두 곳뿐이고
learned 라우트는 staging 이 없으므로 **learned 행의 추가 원자는 전부 O3 = `_adaptive_prune_grow` 의 성장 분기**다.
⟹ ⚠**CLAUDE.md 의 "빔포밍·`MIN_DK`·성장 분기·난수 위상이 배포 혼잡에서 실행되지 않는다"는 문자 그대로는 과대**다.
정확한 문장은 **"혼잡 런의 82%(79/96)에서 실행되지 않고, 18%(17/96)에서는 실행된다"**.
(W3 의 `SHIP-FITPHI` 개입 17/96 = 바로 이 숫자이고, `UNDECIDED-NO-INTERVENTION` 은 "성장이 드물다"의 다른 이름이다.)

### 2.1 손실 함수 — `_compute_loss` (`:313-369`) `[M]`

```
L = Σ_frames Σ_i c_i (ẑ_i − z_i)² / n_obs            :329-342   (가중합을 **비가중** n_obs 로 나눔)
  + lambda_l1 · Σ|w_i|                                :350-358   (λ=1e-4, 단위가 [m] ⟹ [m²] 데이터항과 이종)
  + lambda_smooth · Σ (w_sorted[j+1]−w_sorted[j])²    :361-367   (λ=1e-4, |k| 오름차순 정렬)
```
- `c_i = frame.confidence` (`use_confidence=True` 기본, `run.py:62`).
- `SCALE_INVARIANT_L1 = False`(`:63`), `SCALE_INVARIANT_GROW = False`(`:54`), `RX1_*` 게이트 전부 off ⟹ 위가 배포형.
- 추적에서도 `reg=True`(`:957` `track_reg = not _rx1_gate(...)` = True) ⟹ **λ_l1 이 Phase 2 에서도 걸린다**.

### 2.2 파라미터화 — 극좌표 `(w, φ)`, 그러나 **`(a,b)` 에 정확히 선형** `[M]`

`spectral_gaussian.py:118-127`:
```
phase = kx·x + ky·y − ω·t + φ            :121-124
z     = Σ w_i cos(phase_i)                :127
```
θ ≡ kx·x + ky·y − ω·t 로 두면
`w cos(θ+φ) = (w cos φ)·cos θ + (−w sin φ)·sin θ = a·cos θ + b·sin θ`, 즉 **`a = w cos φ`, `b = −w sin φ`**.

**선형성이 깨질 수 있는 세 경로 — 배포에서 전부 꺼져 있다** `[M]`:
- `use_stokes2`: `_build_model`(`:239-241`)이 인자를 넘기지 않음 ⟹ 생성자 기본 `False`(`spectral_gaussian.py:45`).
- `use_neural_corrector`: `_build_model` 이 명시적으로 `False`(`:241`), `neural_corrector_path` 기본 None.
- `use_dispersion=True` ⟹ ω 가 `(kx,ky)` 의 함수, `(a,b)` 와 무관.
⟹ **배포 순방향 모델은 `(a,b)` 에 대해 정확히 선형이다.** 이것이 W4 의 근거로 성립한다.
⚠단 `clamp_steepness`(`:219-230`)는 `w` 에 대한 **비선형 사영**이라 닫힌형 해 뒤에 다시 걸린다.

### 2.3 ★발견 1 — `amplitude_refit` 의 `(a,b) → φ` **부호 결함** `[M]`

- `sequential_update.py:1330-1335`: `coeffs = lstsq([cosθ|sinθ], z)`; `a=coeffs[0::2]`, `b=coeffs[1::2]`;
  `w=√(a²+b²)`; **`phi = np.arctan2(b, a)`**.
- `model/tnse.py:214-216`(같은 규약, learned 라우트가 실제로 쓰는 것): **`phi = np.arctan2(-b, a)`**.
- 대수 확인(추정기 미실행, 순수 항등식 검산):
  `a=0.7, b=−0.4` 일 때 `atan2(b,a)` → `max|err| = 7.671e-01`, `atan2(-b,a)` → `3.331e-16`.
  ⟹ **`sequential_update.py:1335` 가 틀렸고 `tnse.py:216` 이 맞다.**
- 파급: `amplitude_refit` 은 `residual_after` 를 `A @ coeffs`(올바른 선형결합)로 보고하므로
  **보고 잔차는 개선되는데 설치된 모델은 위상 켤레**가 된다.
- 배포 영향 **없음** `[M]`: 호출은 `run.py:398 cfg.get('amplitude_refit', False)` 뿐이고,
  이 키는 `DEFAULT_CONFIG`·`configs/estimation.yaml` 어디에도 없다 ⟹ 항상 False.
- ★**W4 에 직결**: post-lasso 재적합을 짤 때 이 줄을 복사하면 **똑같이 틀린다.** `tnse.py:216` 규약을 쓸 것.

### 2.4 ★발견 2 — `_ls_w_update` 는 **호출자 0개(dead code)** `[M]`

`sequential_update.py:1215-1257` 가 `w = argmin‖Aw−z‖²`(φ 고정, 벌점 없음) 닫힌형을 이미 구현하지만,
저장소 전체 grep 결과 호출 지점이 **0곳**(패키지·스크립트·테스트·tmp 전부). 즉 **"post-lasso w 재적합"은 46일 넘게 코드에 있었는데 한 번도 실행된 적이 없다.**
⚠단 이건 `w` 만 풀고 `φ` 를 고정하므로 W4 가 원하는 `(a,b)` 동시 해가 **아니다**. 재사용하지 말고 새로 쓸 것.

---

## 3. 삽입 지점 — 후보별 상세

### 3.1 post-lasso 재적합 (L1 로 지지집합 선택 → 벌점 0 으로 `(a,b)` 닫힌형 재적합)

**"lasso 로 고른 지지집합"은 이미 존재한다** — `_adaptive_prune_grow` 의 에너지 프루닝
(`:997-1057`, `w²` 상위 90% 유지 + 잔차 30% 악화 시 롤백). 이게 지지집합 선택 단계다. `[M]`

**삽입 지점 (권장, 패키지 무수정)**
```python
# 배치 디렉터리에서:
import wavesplat.optimizer.sequential_update as SU
_orig_init = SU.SequentialUpdater.initialize          # sequential_update.py:405

def initialize_postlasso(self, frames, x_range=(10.,250.), y_range=(-100.,100.), verbose=False):
    r = _orig_init(self, frames, x_range, y_range, verbose)   # 프루닝·성장·트래킹옵티마 구축까지 끝난 상태
    if self.model is not None and getattr(self, '_bank', None) is None:
        _refit_ab_closed_form(self, frames)                    # ← λ=0, 가중 LS
    return r
SU.SequentialUpdater.initialize = initialize_postlasso
```
**왜 여기가 맞는가** `[M]`
- `initialize` 는 **두 라우트 모두**의 유일한 Phase-1 진입점(`:405`, `_initialize_inner`/`_initialize_learned` 를 모두 감싼다).
- `MedianEnsembleUpdater` 는 `self._primary.initialize`(`:1493`)와 멤버 `m.initialize`(`:1518`)를 부르므로,
  **클래스 메서드 패치 하나로 3멤버 전부** 덮인다.
- 이 시점에서 `remove_gaussians`/`add_gaussians`(둘 다 `nn.Parameter` 를 **교체**한다,
  `spectral_gaussian.py:191-194`,`:210-213`)는 이미 끝났다 ⟹ 텐서 객체 재교체 위험 없음.
- `w.copy_()`/`phi.copy_()` 로 **제자리 갱신**하면 `_build_tracking_optimizer`(`:662`/`:608`)가 이미 잡아둔
  Adam 파라미터 참조가 그대로 유효 ⟹ 옵티마이저 재구축 불필요.

**재적합 구현 요건 (배포 목적함수와 정렬)**
1. 설계행렬 `Φ = [cos θ | sin θ]`, `θ = kx·x + ky·y − ω·t`, `ω` 는 `model._omega()`(`spectral_gaussian.py:97-103`)에서 읽을 것 — 재계산하면 분산관계 상수가 갈릴 수 있다.
2. **가중** LS: 행을 `√c_i` 로 스케일. 배포 데이터항이 `Σ c_i r_i²`(`:333`)이기 때문.
   ⚠W3 의 `ORA-K-CVX` 는 `weighting: 'none'`(`run_w3.py:479`)이었다 — 상한과 직접 비교하려면 이 축을 기록할 것.
3. 창(window) = `init_frames` (`_compute_loss` 가 Phase 1 에서 보는 것과 동일).
4. `(a,b) → (w,φ)`: **`w=√(a²+b²)`, `φ=atan2(−b, a)`** (§2.3).
5. 마지막에 `self.model.clamp_steepness(self.max_steepness)` 를 부를지 **사전등록할 것**(배포 불변식 유지 vs 닫힌형 해 보존).

**⚠설계 위험(반드시 사전등록)**: Phase 1 만 고치면 그 뒤 **추적 10스텝 × 120~220 프레임**이
λ 포함 목적함수로 계속 돈다(O4). W3 의 ×95.4 상한은 `tracking: 'none'`(`run_w3.py:479`)에서 나온 값이라
**Phase-1 전용 개입으로는 도달할 수 없다.** ⟹ W4 는 (a) Phase-1 전용 팔, (b) 매 프레임 재적합 팔을
분리해서 재야 한다. (b) 의 삽입 지점은 `SequentialUpdater.update`(`:911`) 래핑, `:974` 루프 이후.

### 3.2 VARPRO (선형 `(a,b)` 소거 후 `(kx,ky)` 만 비선형 최적화)

**삽입 지점**: `SequentialUpdater._initialize_learned`(`:617-675`) **전체 교체**.
```python
_orig_learned = SU.SequentialUpdater._initialize_learned
def _initialize_learned_varpro(self, frames, x_range=..., y_range=..., verbose=False):
    kx, ky, w, phi, n = learned_init_params(frames, self.depth, self.config, seed=self.seed)  # 원본 :631 과 동일
    kx, ky = _varpro_solve(kx, ky, frames, self.depth)       # J(k) = ‖P⊥_Φ(k) z‖² 를 k 에 대해서만 최소화
    a, b   = _linear_solve(kx, ky, frames)                   # 소거된 선형 파라미터 복원
    ... 모델 설치 ...
    if self.adaptive_n: self._adaptive_prune_grow(frames, x_range, y_range, verbose)  # 원본 :660 유지
    self._build_tracking_optimizer()                                                   # 원본 :662 유지
```
- learned 라우트만 덮으면 **혼잡의 86.5~90.3%** 를 덮는다 `[M]`. 나머지 v2_konly 를 덮으려면
  `_initialize_inner`(`:434`)의 `:561-591` 루프도 교체해야 하고, 그쪽은 **코스투파인 staging 과 얽혀 있어**
  훨씬 침습적이다. ⟹ **W4-VARPRO 는 learned 한정으로 사전등록하고, v2_konly 행은 "미개입"으로 따로 집계**할 것.
  (그러면 배포 팔의 10~14% 는 구조적으로 널이다 — S13 의 "부분 개입" 실패를 반복하지 않으려면 **사전에 적어야 한다**.)
- ⚠`_adaptive_prune_grow` 를 남기면 성장 분기의 **Adam 50 iters(O3)** 가 다시 들어온다(혼잡 17/96 `[M]`).
  VARPRO 팔에서 이걸 남길지/끌지도 사전등록 대상.
- ⚠`_compute_loss` 의 λ_l1·λ_smooth 는 VARPRO 목적함수에 넣을 수 없다(‖w‖₁ = Σ√(a²+b²) = **group lasso**, 닫힌형 소거 불가).
  ⟹ VARPRO 팔은 자동으로 λ=0 팔이 된다. **W3 가 λ 축을 이미 닫았으므로**(`ORA-K-NOREG` 30/32 이지만 `S-1` 18/32 널 = 배포 λ=1e-4 ≈ λ=0)
  **VARPRO 의 이득은 λ 로 설명될 수 없다** — 이건 W4 에 유리한 사전 논증이다.

### 3.3 `merge_atoms`

**삽입 지점: 없다 — 이미 배선돼 있다.** `learned_init.py:367-369`.
```python
config = dict(BASE, learned_merge_frac=0.15)   # 배치에서 이것만
```
- 켜지는 조건: `init_method` 가 `learned` 로 해결된 런에서만(`_initialize_learned:631` → `learned_init_params`).
- ⚠**v2_konly 로 라우트된 행(혼잡 10~14%)에서는 처치 팔과 대조 팔이 비트 동일**이다 `[M]`.
  ⟹ 짝 설계에서 그 셀들은 Δ≡0 이므로 **부호검정의 분모에서 빼거나, "묶임(tie)"으로 사전 선언**해야 한다.
  (이걸 안 적으면 검정력이 조용히 10~14% 새어나간다.)
- `merge_frac` 후보 눈금: 반경이 `r = merge_frac·kp`. `dedup_dk=0.03` 이 이미 걸러낸 뒤이므로
  `merge_frac` 은 `dedup_dk/kp` 보다 커야 no-op 이 아니다 — **런 전에 카운터로 `n_in→n_out` 을 확인**할 것(§4.3).

---

## 4. 개입 카운터 설계 (원자 단위, 런 전 확인용)

**본보기**: W3 의 `install_probe()`(`tmp/W3_SOLVER_2026-08-15/run_w3.py:181-253`) —
게이트 **읽기**를 세지 않고 **실제로 설치된 원자의 위상값**을 셌다. 그래서 `SHIP-FITPHI` 개입이
17/96 뿐임을 결과 전에 잡을 수 있었다. 같은 원칙: **"플래그가 켜졌나"가 아니라 "모델 파라미터가 바뀌었나"를 센다.**

### 4.1 post-lasso 카운터
| 카운터 | 심는 곳 | 기록 |
|---|---|---|
| `pl_calls` | 래핑한 `initialize` 안, 재적합 함수 진입점 | 런당 기대 **1** (앙상블 v2_konly 행은 **3**) |
| `pl_atoms` | 재적합 직전 `self.model.n_gaussians` | N_final 과 일치해야 함 |
| `pl_dw_max`, `pl_dphi_max` | `copy_` 직전 `max|w_new−w_old|`, 위상은 `max|wrap(φ_new−φ_old)|` | **0 이면 개입 없음** = 즉시 사전등록 분기 |
| `pl_resid_before/after` | 같은 창의 가중 잔차 RMS | 재적합이 목적함수를 **반드시** 낮춰야 함(λ=0 LS) — 안 낮아지면 구현 결함 |
| `pl_clamp_hits` | `clamp_steepness` 가 실제로 스케일한 성분 수 | 닫힌형 해가 사영에 잘리는 정도 |

**대조 팔에서도 같은 프로브를 켜고 `pl_calls == 0` 을 기록**할 것(= 누출 없음의 증거).

### 4.2 VARPRO 카운터
| 카운터 | 심는 곳 | 기록 |
|---|---|---|
| `vp_entries` | 교체한 `_initialize_learned` 진입 | learned 행에서 1, v2_konly 행에서 **0**(구조적 미개입 — 사전등록된 널) |
| `vp_grad_steps` | Adam `step()` 호출 수 (원본 `:649` 를 감싸 카운트) | VARPRO 팔에서 **0** 이어야 함 |
| `vp_k_shift` | `‖(kx,ky)_out − (kx,ky)_in‖` 중앙값 | 0 이면 VARPRO 가 초기값에서 안 움직인 것 |
| `o3_grow_atoms` | `SpectralGaussianModel.add_gaussians` 래핑(W3 방식) | 성장 분기 재유입량. 혼잡 기준선 **17/96** `[M]` 과 대조 |
| `o4_track_steps` | `update` 안 `opt.step()` 카운트 | Phase-1 전용 개입이 추적에 희석되는 규모의 분모 |

### 4.3 `merge_atoms` 카운터 (★런 전에 반드시)
| 카운터 | 심는 곳 | 기록 |
|---|---|---|
| `mg_calls` | `learned_init.merge_atoms` 래핑 | 처치 팔 learned 행에서 1, 대조 팔 **0** |
| `mg_n_in`, `mg_n_out` | 같은 래퍼의 입출력 길이 | **`n_in == n_out` 이면 개입 0** ⟹ `merge_frac` 이 너무 작다는 즉각 신호 |
| `mg_route` | `updater._auto_selected`(`sequential_update.py:766`) | learned/v2_konly 층화 — 널 셀을 미리 분리 |
| `mg_n_selected` | `updater._auto_n_selected`(`:731`) | 라우터 신호가 merge 로 **안 바뀌는지** 확인(바뀌면 구현 오류) |

⚠**스모크 게이트 제안**: 혼잡 4셀 × 3시드로 `mg_n_in − mg_n_out` 분포를 먼저 찍고,
**중앙값 감소가 0 이면 그 `merge_frac` 으로 본런을 돌리지 말 것.** (W3 가 `SHIP-FITPHI` 에서
바로 이 확인을 못 해 96런을 `UNDECIDED-NO-INTERVENTION` 으로 태웠다.)

### 4.4 앙상블 배수 주의 `[M]`
프로세스 단위 카운터는 v2_konly 라우트에서 **3배**로 센다(멤버 3개, `:1511-1524`).
learned 라우트는 `ensemble_k=1`(`:1498`). ⟹ 카운터를 `route` 와 함께 기록하고,
"런당 기대값"을 route 별로 다르게 사전등록할 것.

---

## 5. 패키지 수정 최소화 판정

### **판정: 세 후보 전부 `wavesplat/**` 무수정으로 가능하다.** `[M]`

| 후보 | 방법 | 근거 |
|---|---|---|
| post-lasso | `SU.SequentialUpdater.initialize` 메서드 몽키패치 + `finally` 복원 | `MedianEnsembleUpdater` 가 `self._primary.initialize`(`:1493`)/`m.initialize`(`:1518`)를 부르므로 클래스 패치가 전 멤버에 전파 |
| VARPRO | `SU.SequentialUpdater._initialize_learned` 메서드 몽키패치 | `_initialize_inner:517` 이 `self._initialize_learned(...)` 로 **인스턴스 조회** ⟹ 패치 도달 |
| merge_atoms | **패치조차 불필요** — `config` 키만 | `run.py:141` → `:168` → `sequential_update.py:171` → `:632` → `learned_init.py:367` 정적 추적 완결 |
| 카운터 | `learned_init.merge_atoms`, `SpectralGaussianModel.add_gaussians`, `torch.optim.Adam.step` 래핑 | W3 `install_probe`(`run_w3.py:181-253`)가 동일 3종을 이미 검증 |
| 객체 되읽기 | `wavesplat.run.make_updater` 래핑 | W3 `install_updater_capture`(`run_w3.py:260-276`) 재사용 — S13b 교훈(의도표 금지) |

**서브클래스 대신 메서드 패치를 권하는 이유**: `MedianEnsembleUpdater.__init__:1460` 과 `:1517` 이
`SequentialUpdater(config)` 를 **모듈 전역 이름으로** 생성한다. 서브클래스를 쓰려면
`SU.SequentialUpdater = MySubclass` 로 전역을 바꿔야 하는데, 그러면 `isinstance` 검사·테스트가
같이 흔들린다. 메서드 패치는 부작용면이 좁다. (S16/S17 이 `letkf_spec.py` **서브클래스**로 패키지 무수정을
달성한 것은 **기준선 팔**이라 배포 클래스를 안 건드려도 됐기 때문 — W4 는 배포 팔 자체를 건드리므로 상황이 다르다.)

**멀티프로세싱 주의**: W3 는 워커 프로세스 안에서 패치를 걸고 `finally` 로 되돌린다
(`run_w3.py:8-12` 의 원칙, `:246-253` 의 `undo`). W4 도 **워커 안에서** 패치할 것 —
부모에서 걸면 `fork` 시점에 따라 팔이 새어나간다.

---

## 6. W4 설계에 직접 영향을 주는 경고 (요약)

1. **Phase-1 전용 개입은 ×95 상한에 도달할 수 없다.** `ORA-K-CVX` 는 `tracking: 'none'`(`run_w3.py:479`)
   + 초기창 1회 적합으로 전 추적프레임을 렌더한다(`run_w3.py:415-419, 449-459`). 배포 팔은 추적에서
   **10스텝 × 120~220 프레임**(O4)을 더 돈다. 두 팔(Phase-1 전용 / 매프레임)을 나눠 재야 한다. `[M]`
2. **v2_konly 행(혼잡 10~14%)은 learned 전용 처치에서 구조적 널**이다. 사전등록에 명시. `[M]`
3. **성장 분기(O3)는 혼잡 17/96 에서 실제로 돈다** — "안 돈다"가 아니다. VARPRO/post-lasso 뒤에
   남겨두면 Adam 50 iters 가 재유입된다. `[M]`
4. **`amplitude_refit:1335` 의 부호를 베끼지 마라** — 올바른 규약은 `tnse.py:216`. `[M]`
5. **`_ls_w_update`(`:1215`)를 재사용하지 마라** — `φ` 고정이라 `(a,b)` 동시해가 아니고, 호출자 0개라 검증된 적도 없다. `[M]`
6. **재적합은 가중(`c_i`) LS 여야 배포 목적함수와 정렬**된다(`:331-334`). W3 CVX 는 비가중이었다. `[M]`
7. **`effective_config` 는 객체 되읽기로** (`run_w3.py:286-300` 의 `_readback`) — 의도 dict 금지(S13b 교훈).

---

## 7. `[I]` 로 남는 것 (정적으로 확정하지 못한 것)

- `_varpro_solve` 의 수렴/비용은 **미측정**. VARPRO 가 Adam 900 iters 보다 빠를지 느릴지 정적으로 알 수 없다.
- post-lasso 재적합이 nRMSE 를 **개선할지**는 미지 — 확정된 것은 "훈련창 가중 잔차를 반드시 낮춘다"(LS 성질)뿐이고,
  **일반화(추적창 nRMSE)는 별개 축**이다. S4/S5 가 정확히 이 구분에서 갈렸다.
- `merge_frac` 의 유효 눈금. `dedup_dk=0.03` 과 `kp` 의 비에 의존하므로 **스모크 전에는 모른다**.
- 혼잡 라우터 비율의 신선 코퍼스 일반화. 두 실측(90.3%, 86.5%)은 **서로 다른 코퍼스**에서 나왔다.
