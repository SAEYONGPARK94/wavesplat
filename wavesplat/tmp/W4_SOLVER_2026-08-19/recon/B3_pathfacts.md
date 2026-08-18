# B3 — 배포 경로 사실 독립 재현 (추정기 런 0회)

**작성** 2026-08-19 · **에이전트** B3 · **런** 0회 (읽기 + 원자료 재집계 + numpy 항등식 검산만)
**표기** `[M]` = 코드/원자료에서 직접 측정 · `[A]` = 논증

**열람 범위(= 주장 범위)**
`wavesplat/run.py` · `wavesplat/optimizer/{sequential_update,learned_init,coarse_to_fine,initializer}.py` ·
`wavesplat/model/{tnse,spectral_gaussian}.py` · `wavesplat/tests/test_bank_mmse_port.py` ·
`configs/estimation.yaml` · `tmp/W3_SOLVER_2026-08-15/{results.jsonl,run_w3.py}` (1,212행) ·
저장소 전체 `grep`(amplitude_refit / learned_* / atan2) + `git log -S`(6패턴).
⛔**다른 에이전트 산출물(`recon/W1_path.md`·`W2_upper_bound.md`·`PREREG.md`)은 독립성 보존을 위해 근거로 쓰지 않았다** —
`amplitude_refit` grep 결과에 파일명이 걸린 것 외에 내용을 인용하지 않았다.

---

## 0. 요약 표

| # | 검증 대상 | 판정 |
|---|---|---|
| 1 | 라우팅 87/96, 26/32 안정 · 5/32 뒤집힘 | ✅ **전건 재현** `[M]` |
| 2 | 성장 17/96, 학습경로 9/130 | ✅ **전건 재현** `[M]` ⟹ ⛔**CLAUDE.md·MEMORY.md 서술이 틀렸다** |
| 3 | `learned_init_iters=500`, 순수 config 노브 | ✅ **성립** (단 **학습 경로 전용**) `[M]` |
| 4 | `learned_merge_frac=None`, 미노출, 도달가능, 끄면 비트동일·GT-free | ✅ **전건 성립** `[M]` |
| 5 | `amplitude_refit` 부호 결함 | ✅ **결함 확정** · ⛔**과거 오염 없음**(호출자 0곳, git 이력 0건) `[M]` |
| 6 | `_initialize_learned` 의 코스투파인 우회 범위 | ✅ **확정** — 우회는 `coarse_to_fine.py` **뿐** `[M]` |

---

## 1. 라우팅 — 87/96 · 26/32 · 5/32 `[M]` ✅

원자료 `tmp/W3_SOLVER_2026-08-15/results.jsonl`, 필드 `init_method_resolved`
(= `run_w3.py:258 install_updater_capture` 의 **객체 되읽기**, 의도표 아님).

```
SHIP 총 144행 = crowded 96 + sparse 48
SHIP crowded 96행 : learned 87 · v2_konly 9          ← 90.6% 재현
에피소드 32개 × 추정기 시드 3개(1592/1593/1594)
  전-learned    26/32
  전-v2_konly    1/32
  시드에 따라 뒤집힘 5/32
SHIP sparse  48행 : learned 43 · v2_konly 5 (89.6%)
```
⟹ **PREREG 가 A족을 `init_method='learned'` 로 고정한 근거는 원자료에서 그대로 재현된다.**
⚠추가 관찰 `[M]`: 뒤집힘 5개 + 전-v2_konly 1개 = **에피소드 6/32 가 라우터 임계 근방**이다.
라우터를 열어두면 A족 96셀 중 **약 9셀이 다른 알고리즘(v2_konly)** 을 돌게 되고,
아래 §2 에서 보듯 그 9셀은 **성장 분기 발화율이 100%** 라 다른 코드 경로까지 섞인다.
⟹ 고정은 편의가 아니라 **교락 제거**다.

---

## 2. ★성장 분기 — 17/96 이 맞고, 저장소 문서가 틀렸다 `[M]`

### 2.1 재현
`run_w3.py:230-244` 가 `SpectralGaussianModel.add_gaussians` 를 원자 카운터로 감싼다
(`add_calls`), `:202-210` 이 `initializer._linear_projection` 과 `coarse_to_fine._linear_projection`
**양쪽**을 감싼다(`lp_calls`), `:219-227` 이 `_rx2_gate('RX2_GROW_FITPHI')` 의 **호출 프레임 줄번호**를 기록한다.

```
SHIP crowded 96행
  add_calls > 0 : 17/96   ← 17/96 재현 ✅
    ├ route=learned  :  8/87  (9.2%)
    └ route=v2_konly :  9/9   (100%)
SHIP(양 층) route=learned 130행 : add>0 = 9  → 9/130 = 6.9% 재현 ✅
   (crowded 8/87 + sparse 1/43)
```

### 2.2 ★★★결정적 증거 — 어느 사이트가 돌았는가
`_linear_projection` 은 저장소에 **세 호출지점**뿐이다 `[M]`:
`initializer.py:536`(= `initialize_from_frames_v2`, 학습 경로에서 **미호출**) ·
`coarse_to_fine.py:241` · `sequential_update.py:1142`.
학습 경로는 `initialize_from_frames` 를 **한 번도 부르지 않으므로**(§6),
**학습 경로에서 `lp_calls>0` 은 `sequential_update.py:1142` 외에 원인이 없다.**

```
SHIP crowded, route=learned (87행)
  lp_calls   : 0×79 · 3×5 · 4×2 · 5×1
  add_calls  : 0×79 · 1×8
  randphi    : 0×79 · 3×5 · 4×2 · 5×1     (= lp_calls 와 동일)
SHIP crowded, route=v2_konly (9행)
  lp_calls   : 45~63   add_calls : 6~9    (초기화 v2 + 성장 혼합)
```
`RX2_GROW_FITPHI` 게이트 **줄번호별 히트**(`fitphi_site_hits`):
```
SHIP·SHIP-FITPHI·SHIP-DUP, route=learned  271행 → coarse_to_fine.py:173  0 행
                                                 sequential_update.py:1188  20 행
                          route=v2_konly   29행 → coarse_to_fine.py:173   29 행
                                                 sequential_update.py:1188  14 행
```

### 2.3 판정
- ✅ **맞는 절반**: 학습 경로에서 `coarse_to_fine.py` 의 성장·빔포밍 사이트는 **271행 전부에서 0회**.
- ⛔ **틀린 절반**: `sequential_update._adaptive_prune_grow` 안에 **같은 기능의 두 번째 구현이 있고**
  (`:1104-1200`, 조건절 `self.init_method in ('v2','v2_konly','learned','oracle')` 에 **`'learned'` 가 포함**),
  이것이 **`SHIP` 혼잡 96런 중 17런에서 실제로 실행됐다**. 그 안에서 실행된 것:
  `_beamform_coarse`(`:1119`) · `_beamform_refine`(`:1139`) · `_linear_projection`(`:1142`) ·
  `MIN_DK`(`:1115-1116`) · `torch.rand` 난수 위상(`:1193`) · `RX2_GROW_FITPHI` 게이트(`:1188`) ·
  `add_gaussians`(`:1194`).
  **`SHIP-FITPHI` 팔이 `fitphi_atoms>0` 을 20행에서 찍었다는 것은 그 게이트가 실제로 원자를 바꿨다는 뜻**이다.

### 2.4 ★제안하는 대체 문장 (파일은 수정하지 않음)
`CLAUDE.md` §측정 규율 및 `MEMORY.md` 2026-08-15 항목의
> "빔포밍 · `MIN_DK` · 성장 분기 · 난수 위상 · `RX2_GROW_FITPHI` 는 배포 혼잡에서 실행되지 않는다."

를 다음으로 교체할 것:
> **`_initialize_learned` 가 우회하는 것은 `coarse_to_fine.py` 뿐이다** (`CoarseToFineSchedule` 미생성 ·
> `initialize_from_frames` 미호출 · `coarse_to_fine.py:181 add_gaussians` 와 `:241 _linear_projection`
> **0회**, W3 학습경로 271행 전건). **그러나 동일 기능의 복제본이 `sequential_update._adaptive_prune_grow`
> (`:1104-1200`)에 있고 조건절이 `'learned'` 를 포함하므로, 빔포밍 · `MIN_DK` · 난수 위상 ·
> `RX2_GROW_FITPHI` · `add_gaussians` 는 배포 혼잡에서 실행된다 — `SHIP` 혼잡 17/96런
> (학습경로 8/87 = 9.2%, `v2_konly` 경로 9/9 = 100%).**
> ⟹ 이 축들은 "죽은 코드"가 아니라 **"발화율 9%의 저-검정력 축"** 이다. 개입을 설계하면
> **개입이 실제로 착지한 런만 세는 원자 카운터가 필수**다(W3 `SHIP-FITPHI` 가 96셀 중 17셀에서만 개입 ⟹
> `UNDECIDED-NO-INTERVENTION`).

⚠**S1 의 "출하 팔은 beamform 0회 호출" 은 여전히 참일 수 있다** — 그것은 **초기화 시점**의
`initialize_from_frames_v2` 빔포밍이고, 여기서 문제 삼는 것은 **성장 시점**의 빔포밍이다.
두 문장이 하나로 뭉개진 것이 이번 오류의 기제로 보인다 `[A]`.

---

## 3. `learned_init_iters` — `L-ITER0` 팔 성립 `[M]` ✅ (단서 1개)

- **기본값 500**: `sequential_update.py:643` `max_iters = int(self.config.get('learned_init_iters', 500))`. **유일 정의**.
- **순수 config 노브 · 화이트리스트 통과**:
  `run.py:139-141` 이 `cfg = DEFAULT_CONFIG.copy(); cfg.update(config)` 로 **키를 거르지 않는다**.
  `make_updater(cfg)`(`:168` → `sequential_update.py:1586-1589`) → `SequentialUpdater(config)` →
  `:171 self.config = config` 로 **원본 dict 이 그대로 보존**된다.
  `MedianEnsembleUpdater` 경로도 `:1460 SequentialUpdater(config)` + `:1513 cfg = dict(self.config)` 로 전달한다.
  ⟹ `run_estimation(h5, config={'learned_init_iters': 0})` 로 **도달한다**.
- **미노출 확인** `[M]`: `configs/estimation.yaml` 0회 · `run.py` DEFAULT_CONFIG 0회 · argparse 0회.
  `load_yaml_config`(`run.py:81-123`)는 **명시 화이트리스트**라 `--config` YAML 로는 **도달 불가**.
  ⟹ **Python API 전용**. (배치 러너는 API 를 쓰므로 문제 없음.)
- **`=0` 안전성** `[M]`: `:645 for iteration in range(0)` → 루프 미실행 → `loss_history=[]`.
  `:668` 은 `if loss_history else float('nan')` 로 가드돼 있고 `if verbose:` 안이다.
  `:656-662`(`_last_t` · `_adaptive_prune_grow` · `_build_tracking_optimizer`)는 그대로 실행. **예외 없음.**

⚠**단서 2개 — PREREG 에 적을 것** `[M]`:
1. **`learned_init_iters` 는 학습 경로에서만 읽힌다.** `v2_konly` 경로는 `max_iters_per_stage` 를 쓴다.
   라우터를 열어두면 A족 96셀 중 약 9셀에서 **처치가 no-op** 이 된다 ⟹ §1 의 고정이 이 팔의 **전제**다.
2. ★**`=0` 은 잔차를 키워 성장 분기 발화율을 올린다** `[A]`. `:1087`
   `if grow_metric < self.grow_residual_threshold: return` 이 잔차 RMS 로 게이트되므로, 조인트 핏을
   0스텝으로 두면 §2 의 빔포밍 성장 경로가 **9.2%보다 훨씬 자주** 켜질 수 있다.
   ⟹ **`L-ITER0` 은 "조인트 핏 제거" 단일 요인이 아니라 "조인트 핏 제거 × 성장 발화율 증가" 2요인**이 될 위험.
   **처방**: `add_calls`/`lp_calls` 를 팔별로 기록하고, 필요하면 `adaptive_n=False` 동반 팔을 둘 것.

---

## 4. `learned_merge_frac` — 전건 성립 `[M]` ✅

| 주장 | 판정 | 근거 |
|---|---|---|
| 기본값 `None` (no-op) | ✅ | `learned_init.py:367 merge_frac = config.get('learned_merge_frac')`, `:368 if merge_frac:` |
| 끄면 **비트 동일** | ✅ | `:368` 이 falsy(`None`·`0`·`0.0`)면 `merge_atoms` **미호출** → `kx,ky,en` 배열이 손대지 않은 채 통과. `merge_atoms` 자체도 `:205-206` 에서 조기 반환. **RNG 소비 0** |
| **GT-free** | ✅ | 입력이 `pool_and_select(propose_atoms(frames,…))` 의 `(kx,ky,energy)` 뿐(`:355`). `merge_atoms`(`:190-232`) 는 `argmax(en)` 로 `kp` 를 잡고 `r=merge_frac·kp` 로 단일연결 병합 후 **에너지 가중 중심**을 취한다. GT·`Hs`·`gt_field` 참조 **0회** |
| `configs/`·`run.py` **미노출** | ✅ | `grep -rn "merge_frac" configs/ wavesplat/run.py` → **0 hit** |
| `run_estimation(config=…)` **도달 가능** | ✅ | §3 과 동일 사슬 + `sequential_update.py:632 learned_init_params(frames, self.depth, self.config, seed=self.seed)` |

⚠**범위** `[M]`: **학습 경로 전용**(`:632` 는 `_initialize_learned` 안). `v2_konly` 경로에서는 무시된다 — §3 단서 1 과 동일.
⚠`merge_atoms` 는 `pool_and_select` **이후**에 돈다. 즉 이미 `learned_dedup_dk=0.03`
(`:341`)로 걸러진 원자에 **추가로** kp-비례 반경 병합을 건다. 두 문턱이 **직렬**이라는 점을 팔 설계에 반영할 것.

---

## 5. ★`amplitude_refit` 부호 결함 — 확정, 그러나 **과거 오염 0건** `[M]`

### 5.1 항등식 검산 (numpy, 추정기 미기동)
전방모델 `spectral_gaussian.py:6,127`: `z = Σ wᵢ·cos(θᵢ + φᵢ)`, `θ = kx·x + ky·y − ω·t`.
전개하면 `w·cos(θ+φ) = (w cosφ)·cosθ − (w sinφ)·sinθ`.
설계행렬이 `[cosθ, sinθ]`·계수 `[a,b]` 이면 `a = w cosφ`, **`b = −w sinφ`**
⟹ `w = √(a²+b²)`, **`φ = atan2(−b, a)`**.

수치 검산 (N=6, M=4000 관측, GT 를 전방모델로 **정확히** 생성, `rng seed 0`):
```
LS 기저 잔차 ‖A·c − z‖    = 1.57e-15
φ = arctan2( b, a) 재구성 잔차 = 1.867   ← sequential_update.py:1335
φ = arctan2(−b, a) 재구성 잔차 = 1.41e-15 ← model/tnse.py:216
참고: z 자체의 rms          = 1.141
w 복원 오차 = 6.7e-16 · φ 복원 오차 = 2.7e-15 (올바른 규약)
```
⟹ ✅ **주장 확정.** ★**틀린 규약의 잔차(1.867)가 `z` 의 rms(1.141)보다 크다** —
즉 부호를 틀리면 **아무것도 예측하지 않는 것보다 나쁘다.**

### 5.2 파급 — 보고된 `residual_after` 는 실제로 실현되지 않는다 `[M]`
`sequential_update.py:1355` 가 `residual_after = ‖A @ coeffs − z‖` 로 **선형해 그대로** 보고하는데,
`:1359-1362` 는 모델에 `(w_refit, arctan2(b,a))` 를 심는다. 두 양이 **다른 함수**다
⟹ 이 함수가 켜지면 **"잔차가 1e-15 로 떨어졌다"고 로그를 찍으면서 모델은 망가진다.**

### 5.3 저장소 규약 감사 `[M]`
`arctan2/atan2` 전 사이트 (테스트 제외):
```
model/tnse.py:216            atan2(−b, a)  ✅
optimizer/initializer.py:448 atan2(−b, a)  ✅
optimizer/relax_estimator.py:68  atan2(−b, a)  ✅
optimizer/cs_initializer.py:160  atan2(−c_sin, c_cos)  ✅
optimizer/deep_relax.py:232      atan2(−b, a)  ✅
optimizer/sequential_update.py:1335  atan2( b, a)  ⛔  ← 6곳 중 유일한 이탈
   (독스트링 :1268 도 같은 오류를 문서화하고 있다 — 함께 고칠 것)
```
✅**배포 학습 경로의 위상 초기화는 안전하다**: `learned_init.py:386 ls_fit_wphi` → `:312 fit_amplitudes_phases`
= `tnse.py:216` (올바른 규약). 그 독스트링(`learned_init.py:309-310`)이
*"its convention matches SpectralGaussianModel exactly"* 라 적은 것도 **참**이다.

### 5.4 ★★과거 결과 오염 여부 — **없음** `[M]`
- **유일한 실행 게이트**: `run.py:398 if cfg.get('amplitude_refit', False):` → `:402 updater.amplitude_refit(...)`.
  argparse 플래그 **없음**(`run.py:540-609` 전수) · `configs/estimation.yaml` **0회** · `DEFAULT_CONFIG` **0회**.
- **현 작업트리 전수 grep**: 이 키를 **True 로 설정하는 곳 0곳**. `amplitude_refit` 문자열이 있는 곳은
  `run.py` · `optimizer/sequential_update.py` · `tests/test_bank_mmse_port.py` · `tmp/rx2_init_2026-08-05/snap/`(스냅샷)
  · 문서 6건뿐.
- **git 이력 `-S` 6패턴**(`'amplitude_refit': True` / `amplitude_refit=True` / `"amplitude_refit": true` /
  `amplitude_refit'] = True` / `--amplitude_refit`) → **전부 0 커밋**.
  최초 도입 커밋 `874419a`(2026-04-29) 의 메시지 자체가 *"amplitude_refit: … 구현 (기본 off)"*.
- **결과 JSON**: `grep -rln '"refit"' --include=*.json` → **0 파일**.
- **테스트 커버리지 0**: `tests/test_bank_mmse_port.py:146 assert upd.amplitude_refit([new]) == {}` 은
  **bank 경로**라 `sequential_update.py:1283 return {}` 에서 조기 반환 ⟹ **`:1335` 는 한 번도 실행되지 않는다.**

⟹ ✅**결함은 실재하나 기록된 어떤 결과도 오염시키지 않았다.**
⚠**부재증명의 한계** `[A]`: 커밋되지 않은 REPL/일회성 세션이 `config={'amplitude_refit':True}` 를
넘겼을 가능성은 배제할 수 없다. 다만 그 흔적(결과 JSON 의 `refit` 키)이 **0건**이다.

### 5.5 처방
1. `:1335` → `phi_refit = np.arctan2(-b, a)` · `:1268` 독스트링 동시 수정. **1글자 + 1줄.**
2. ★**회귀 테스트를 붙여라** — 지금 이 줄의 커버리지는 0이고,
   저장소는 이미 *"테스트가 배포 기본값을 전혀 지키지 않는다"* 로 같은 결함 부류에 값을 치렀다.
   테스트는 §5.1 의 **항등식**(전방모델로 생성 → refit → 재구성 잔차 < 1e-10)이면 충분하다.
3. ⛔**W4 의 어떤 팔도 `:1335` 의 규약을 베끼지 마라.** 올바른 출처는 `model/tnse.py:216`.

---

## 6. `_initialize_learned` 의 우회 범위 — 정적 확정 `[M]`

`sequential_update.py:515-517`
```
if self.init_method == 'learned':
    return self._initialize_learned(frames, x_range, y_range, verbose)
```
**`return` 이므로 `:519` 이후 `_initialize_inner` 본문은 전혀 실행되지 않는다.**

### 6.1 실행되지 **않는** 것 (학습 경로) `[M]`
| 코드 | 위치 | 확인 |
|---|---|---|
| `CoarseToFineSchedule` 생성 | `sequential_update.py:525-531` | 정적 도달 불가 |
| `schedule.loss_scale` / `SCALE_INVARIANT_ADVANCE` | `:536-540` | 〃 |
| `initialize_from_frames` | `:544-547` | 〃 ⟹ `initialize_from_frames_v2` 의 **초기화 빔포밍**(`initializer.py:536`) 0회 |
| `schedule.should_advance` / `schedule.advance` | `:578`, `:582` | 〃 |
| `coarse_to_fine._place_new_components` → `_beamform_residuals`(`:192`) → `add_gaussians`(`:181`) | `coarse_to_fine.py:135-190` | 〃 |
| `GROW_RANDOM_PHI`(`coarse_to_fine.py:18,177`) · `coarse_to_fine.py:173` FITPHI 게이트 | 〃 | **원자료 확인: 학습경로 271행 중 히트 0행** |

### 6.2 그대로 실행되는 것 `[M]`
| 코드 | 위치 |
|---|---|
| `learned_init_params`(제안 · `pool_and_select` · `merge_atoms` 훅 · `ls_fit_wphi`) | `:631-632` → `learned_init.py:324-392` |
| 전체 N 조인트 핏 (`_compute_loss` + Adam + `clamp_steepness`) | `:643-651` |
| ★**`_adaptive_prune_grow`** — 프루닝 + **빔포밍 성장 전체** | `:659-660` → `:990-1213` |
| `_build_tracking_optimizer` + Phase 2 추적기 | `:662` |

### 6.3 정확한 한 문장 `[M]`
> **`_initialize_learned` 는 `coarse_to_fine.py` 를 통째로 우회하지만 `_adaptive_prune_grow` 는 우회하지 않는다.
> 그리고 `_adaptive_prune_grow` 안에는 `coarse_to_fine._beamform_residuals` 와 기능적으로 같은
> 두 번째 빔포밍 성장 구현이 있고(`sequential_update.py:1104-1200`), 그 조건절이 `'learned'` 를 포함한다.**

⟹ 축이 죽은 이유는 **"실행되지 않아서"** 가 아니라, **동일 기능이 두 파일에 중복 구현돼 있어
한쪽만 계측하면 발화를 놓치기 때문**이다 `[A]`. 이는 저장소가 이미 겪은
`lambda_l1` 3중 기재 · `n`-상수 3회 재발과 **같은 결함 부류(중복 정의)** 다.
**처방**: 성장 사이트를 단일출처화하거나, 최소한 두 사이트 모두에 카운터를 걸 것
(W3 `install_probe` 가 이미 그렇게 설계돼 있고, 그래서 이 사실이 드러났다).

---

## 7. 최종 판정

| 항목 | 재현 | 배치 설계에 대한 함의 |
|---|---|---|
| 1 라우팅 87/96 · 26/32 · 5/32 | ✅ **완전 일치** | A족 `learned` 고정은 **필수**(교락 제거) |
| 2 성장 17/96 · 9/130 | ✅ **완전 일치** | SFW 제외 근거는 **유효**하되 사유를 "0회"→**"발화율 9.2%"** 로 고칠 것. ⛔**저장소 문서 2건 수정 필요** |
| 3 `learned_init_iters` | ✅ 500 · 순수 노브 · 도달 가능 · `=0` 안전 | `L-ITER0` **성립**. 단 ⚠**성장 발화율 상승 교락** 계측 필요 |
| 4 `learned_merge_frac` | ✅ `None`·미노출·도달가능·비트동일·GT-free | 팔 **성립** |
| 5 `amplitude_refit` 부호 | ✅ 결함 확정 (틀린 잔차 1.87 > z rms 1.14) | ⛔**과거 오염 0건** — 최우선 산출 아님. **1줄 수정 + 회귀 테스트**로 종결 |
| 6 우회 범위 | ✅ 확정 | 우회는 **`coarse_to_fine.py` 뿐** |
