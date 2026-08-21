# A4 — `CLAUDE.md` 주장 vs 실제 코드 전건 대조 (2026-08-13)

**감사 범위 (열람 범위 = 주장 범위).** 실제로 연 파일만 근거로 삼는다.

열람/실행한 것:
- `/home/vexit/wavesplat/CLAUDE.md` (전문), `git diff` (CLAUDE.md · `scripts/realdata_analysis.py`)
- `wavesplat/configs/{estimation,default,server}.yaml` (전문)
- `wavesplat/wavesplat/run.py` (1–130, 330–400, 490–610), `optimizer/sequential_update.py` (1–220, 440–530, 950–1045),
  `optimizer/{initializer,coarse_to_fine}.py` (모듈 상수만), `model/dispersion.py`, `evaluation/metrics.py` (95–215)
- 배치 러너: `tmp/rx2_init_2026-08-05/run_rx2b.py`, `tmp/S4…/run_s4.py`, `tmp/S5…/run_s5.py`, `tmp/S6…/run_s6.py`,
  `tmp/S7…/run_s7.py`, `tmp/S8…/run_s8.py`, `tmp/S9…/{run_s9.py,s9_constants.py}`, `tmp/S11…/run_s11.py`,
  `tmp/S13…/run_s13.py`, `tmp/BLANKSLATE…/{bs1_shipping_default.py,bs3_ship_rings.py}`,
  `tmp/DEBATE_blankslate…/r2_*.py`, `tmp/B3…/b3_regime.py`
- 실행: `pytest --collect-only` · `SequentialUpdater({}).lambda_l1` · `load_yaml_config` vs `DEFAULT_CONFIG` diff ·
  문서화된 `python3 -m wavesplat.run …` 커맨드 · `git show 40e22f6 / 5f0846c`
- 파일시스템 카운트 (`ls`/`find`)

**안 본 것 (주장하지 않음):** `wavesim/` 구현체, `paper/*.tex` · `thesis/*.tex` 본문, `scripts/` 413개 중
Key Commands 5개 + direction1 생산자 5개 외, `tmp/` 121개 배치 중 위 목록 외, `results/*.json` 내용.

---

## 0. 요약

| 등급 | 건수 |
|---|---|
| BLOCKER | 1 |
| MAJOR | 8 |
| MINOR | 10 |
| **일치 확인 (MEASURED)** | **24** |

---

## 1. BLOCKER

### B-1. `Key Commands` 첫 줄이 실행되지 않는다

```
CLAUDE.md:31   python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n
```

**실행 결과 (MEASURED, `cd wavesplat && ` 후 실행):**
```
run.py: error: argument --adaptive_n: expected one argument
```

두 개의 독립 결함:

1. `run.py:537` — `parser.add_argument('--episode', type=str, required=True, …)`.
   **positional 인자가 없다.** 에피소드 경로를 그냥 붙이면 파싱 실패.
2. `run.py:565` — `parser.add_argument('--adaptive_n', type=str, default=None, …)`,
   `run.py:603` — `config['adaptive_n'] = args.adaptive_n == 'true'`.
   **값을 받는 옵션**이지 bare flag 가 아니다.

`configs/estimation.yaml:2` 의 사용례(`python -m wavesplat.run --episode data/episode.h5 --config configs/estimation.yaml`)
가 올바른 형태다 — CLAUDE.md 만 틀렸다.

**제안 수정 (`CLAUDE.md:30-31`):**
```
# Run estimation  (repo root = /home/vexit/wavesplat, 아래는 wavesplat/ 안에서 실행)
python3 -m wavesplat.run --episode data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n true
```

---

## 2. MAJOR — ⚠기준선 λ 오염 (임무 2)

### M-1. 기제는 `SequentialUpdater` 폴백이 아니라 `DEFAULT_CONFIG` 병합이다

```python
run.py:139-141
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
```

배치 러너가 넘기는 config 에 `lambda_l1` 키가 **없으면** `DEFAULT_CONFIG['lambda_l1']`
(= `LAMBDA_L1_DEFAULT` = **1e-4**) 이 들어간다. `sequential_update.py:173` 의 `.get()` 폴백은
`run_estimation` 을 우회할 때만 닿는다. 두 경로 모두 현재 **1e-4** 다.

**실행 확인 (MEASURED):**
```
LAMBDA_L1_DEFAULT                = 0.0001
SequentialUpdater({}).lambda_l1  = 0.0001
DEFAULT_CONFIG['lambda_l1']      = 0.0001
```

### M-2. 오염의 뿌리 = frozen `run_rx2b.BASE_CFG` — 20개 러너가 이걸 공유한다

```python
tmp/rx2_init_2026-08-05/run_rx2b.py:61-64
BASE_CFG = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
                adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30,
                n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                max_iters_per_stage=[200, 200, 500])
```
`lambda_l1` **없음**. `lr` · `tracking_lr` · `tracking_mode` · `lambda_smooth` · `analytic_ensemble_k` ·
`real_estimator` · `grow_residual_threshold` 도 **없음** — 전부 `DEFAULT_CONFIG` 에서 온다.

`run_rx2b.py:108` — `'AUTO': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3))` → λ 없음.
`tmp/S4…/run_s4.py:43-44` — `'AUTO-NT': {'tracking_lr': 0.0}`, `'AUTO-NT-DUP': {'tracking_lr': 0.0}` → λ 없음.

**`run_rx2b` 를 import 하는 러너 20개 전부 `lambda_l1` 문자열이 0회 등장** (grep 실측):
`S1/run_s1.py` · `S2/run_s2.py` · `S3/run_s3.py` · `S4/run_s4.py` · `K6/run_k6.py` ·
`Rx4/run_rx4.py` · `Rx5/run_rx5.py` · `Rx6/run_rx6.py` · `Rx7/run_rx7.py` ·
`EB/run_eb.py` · `EC/run_ec.py` · `ED/run_ed.py` · `B3_rx3/run_b3.py` (외 파생 분석기).

### M-3. ★CLAUDE.md 는 오염 범위를 **과소보고**한다 — S5·S6 이 빠져 있다

CLAUDE.md:63 은 "S7·S8 러너가 현재 이 상태" 라고만 쓴다. **S5 와 S6 도 λ-처치 배치이고 동일 결함이다.**

| 배치 | 처치 | 기준선 팔이 λ 명시? | 배포 후 재실행 판정 |
|---|---|---|---|
| **S5** `S5_window_loss_2026-08-08` | `lambda_l1: 0.0` (`run_s5.py:30,36,37`) | ⛔**아니오** — `AUTO-NT`/`AUTO-NT-DUP` 은 `S4.EXTRA` 상속(λ 없음) | ⚠**오염**. 대조가 1e-3↔0 → 1e-4↔0 으로 이동 |
| **S6** `S6_l1_sweep_2026-08-08` | λ 스윕 3e-4/1e-4/0 (`run_s6.py:26,29`) | ⛔**아니오** — `run_s6.py:31` `AUTO`, `AUTO-NT`(상속) 둘 다 λ 없음 | ⛔**치명적**. `AUTO-NT-L1-1e4` 와 `AUTO-NT` 가 **같은 셀**이 된다 |
| **S7** `S7_confirm_n32_2026-08-10` | `lambda_l1: 1e-4` | ⛔**아니오** — `run_s7.py:78` `"EXTRA 없음 = 아무것도 덮지 않는다"`; `AUTO-NT`/`AUTO-NT-DUP` 상속 | ⛔**치명적**. `AUTO-L1-1e4` − `AUTO` = 0 |
| **S8** `S8_deploy_confirm_2026-08-10` | `lambda_l1: 1e-4` | ⛔**아니오** — `run_s8.py:79` `AUTO`, `:73 AUTO-SEED-NULL {}`, `:74 AUTO-DUP2 {}`, `:72 AUTO-TRK-QTR` | ⛔**치명적**. 처치·널·복제 전부 붕괴 |
| **S9** `S9_lr_x_l1_2026-08-11` | `L1_TREAT=1e-4` vs `L1_SHIP=1e-3` | ✅**예** — `run_s9.py:78-96`, **`AUTO` 포함 전 셀 명시** (E1 수리) | ✅**깨끗** |
| **S10** `S10_selector_2026-08-12` | (러너 없음 — `.md` 만) | n/a | ✅ 해당 없음 |
| **S11** `S11_null_corrected_2026-08-12` | λ 고정 1e-4 | ✅**예** — `run_s11.py:57-59` 3 셀 전부 명시 | ✅**깨끗** (단 기준선은 S9 저장분 재사용) |
| **S13** `S13_oos_2026-08-12` | `SHIP` vs `SHIP_L1E3` | ✅**예** — `run_s13.py:53-63` `ship_config() = dict(DEFAULT_CONFIG)` 명시 전개 + G-1 게이트(`:109-116`)가 실효값 대조 | ✅**깨끗** |
| **BLANKSLATE** `BLANKSLATE_2026-08-12` | 출하 기본값 자체 | ✅ `bs1_shipping_default.py:36 'shipping_default': {}` = **의도적으로** DEFAULT_CONFIG 사용; `bs3_ship_rings.py:92-97` 3 셀 λ 명시 | ✅**깨끗** |
| **DEBATE_blankslate** | 출하 기본값 | ✅ `r2_auto_rest70.py:32` λ 를 **기록**, `r2_p1_support_probe.py:190-191` 명시 | ✅**깨끗** |
| **AUDIT_full** `AUDIT_full_2026-08-12` | (추정기 호출 **0회** — 순수 재분석) | n/a | ✅ 해당 없음 |

**⟹ CLAUDE.md:63 의 "S7·S8" 은 "S5·S6·S7·S8" 이어야 한다.**

**제안 수정 (`CLAUDE.md:62-63`):**
```
  ⚠**기본값을 바꿀 때는 기준선 팔이 λ 를 명시하는지 반드시 확인** — 명시 안 한 배치 러너는
  기준선이 처치값을 받는다. 뿌리는 `tmp/rx2_init_2026-08-05/run_rx2b.py:61 BASE_CFG` 에
  λ 가 없다는 것이고, 이를 상속하는 러너 20개가 전부 그 상태다. λ-처치 배치인
  **S5·S6·S7·S8 은 재실행·확장 무효**(S9·S11·S13·BLANKSLATE 는 명시돼 있어 유효).
```

### M-4. `results.jsonl` 의 `cfg` 는 **의도**이지 **실효 설정**이 아니다 — 사후 감사가 불가능

S7·S8 `results.jsonl` 첫 행 실측:
```json
"cfg": {"init_method":"auto","use_dispersion":true,"use_confidence":true,"adaptive_n":true,
        "tracking_window":3,"prune_threshold":0.1,"max_n":30,"n_gaussians":[3,8,15],
        "loss_thresholds":[0.1,0.05,null],"max_iters_per_stage":[200,200,500],
        "analytic_ensemble_k":3}
```
**`lambda_l1` 키가 없다.** 저장된 어느 행으로도 그 런의 λ 를 확인할 수 없다.
S13 만이 `run_s13.py:181-182` 에서 `effective_config` 를 통째로 기록한다 — 이 방식이 표준이어야 한다.

### M-5. `init_method` 에 **같은 함정이 세 개의 서로 다른 값으로** 살아 있다 (미수정)

| 위치 | 폴백값 | 출하 기본값과 일치? |
|---|---|---|
| `run.py:43` `DEFAULT_CONFIG['init_method']` | `'auto'` | ✅ |
| `run.py:99` `load_yaml_config` `.get('init_method', 'v2')` | **`'v2'`** | ⛔ |
| `sequential_update.py:181` `config.get('init_method', 'v2_konly')` | **`'v2_konly'`** | ⛔ |

CLAUDE.md:59 는 `⛔"v2_konly" 는 낡음` 이라고 못 박는데, **코드의 폴백은 여전히 `v2_konly`(와 `v2`)** 다.
`SequentialUpdater(cfg)` 를 직접 만들면(= `run_estimation` 우회) 조용히 은퇴한 초기화기가 붙는다.
λ 에 적용된 단일출처 수리가 이 키에는 적용되지 않았다.

**제안 수정 (`sequential_update.py:181`):** 모듈 상수 `INIT_METHOD_DEFAULT = 'auto'` 를 만들고
`run.py:43` 과 `run.py:99` 가 그것을 import.

### M-6. `load_yaml_config` 는 λ 한 키만 단일출처화됐고 나머지 6개는 손으로 쓴 사본

```python
run.py:92-99
  cfg['lr']             = raw['optimizer'].get('lr', 0.01)
  cfg['lambda_l1']      = raw['optimizer'].get('lambda_l1', DEFAULT_CONFIG['lambda_l1'])   # ← 수리됨
  cfg['lambda_smooth']  = raw['optimizer'].get('lambda_smooth', 0.0001)
  cfg['tracking_steps'] = raw['optimizer'].get('tracking_steps', 10)
  cfg['tracking_mode']  = raw['optimizer'].get('tracking_mode', 'w_phi')
  cfg['tracking_lr']    = raw['optimizer'].get('tracking_lr', 0.005)
  cfg['init_method']    = raw['optimizer'].get('init_method', 'v2')                        # ← 낡음
```
`run.py:93` 주석은 `"fallback must track DEFAULT_CONFIG, not a third hand-written copy (S9 lesson)"`
라고 **스스로 규칙을 선언해 놓고 바로 다음 줄부터 6번 어긴다.**

### M-7. CLAUDE.md 의 "세 곳에 중복 기재" 는 **사실이 아니다** (문서가 옛 상태를 서술)

```
CLAUDE.md:60-61
- `lambda_l1: 1e-4` — 2026-08-11 배포(커밋 `40e22f6`). 세 곳에 중복 기재:
  `configs/estimation.yaml:12` · `wavesplat/run.py:30` · `optimizer/sequential_update.py:40`.
```

세 줄번호는 **전부 정확**하다(아래 §5 확인). 그러나 실제 코드는:
```python
run.py:19    from wavesplat.optimizer.sequential_update import (..., LAMBDA_L1_DEFAULT)
run.py:30    'lambda_l1': LAMBDA_L1_DEFAULT,           # ← 리터럴이 아니라 import
sequential_update.py:40   LAMBDA_L1_DEFAULT = 1e-4     # ← 유일 정의 (SINGLE SOURCE)
configs/estimation.yaml:12  lambda_l1: 0.0001          # ← 미러
```
`git show 5f0846c` 확인: `-'lambda_l1': 0.0001` → `+'lambda_l1': LAMBDA_L1_DEFAULT`.
⟹ **리터럴은 두 곳(파이썬 1 + yaml 1)** 이고 파이썬 쪽은 **단일출처**다.
문서가 "세 곳을 동기화하라"는 **이미 폐기된 유지보수 지시**를 남기고 있다.

`configs/estimation.yaml:18-19` 도 같은 낡은 지시를 담고 있다:
```
# NOTE: this constant is ALSO written in wavesplat/run.py
# (DEFAULT_CONFIG and the yaml-loader fallback). Keep in sync.
```
→ 둘 다 이제 유도값이다.

**제안 수정 (`CLAUDE.md:60-61`):**
```
- `lambda_l1: 1e-4` — 2026-08-11 배포(커밋 `40e22f6`), 2026-08-11 단일출처화(커밋 `5f0846c`).
  **유일 정의 = `optimizer/sequential_update.py:40 LAMBDA_L1_DEFAULT`**;
  `wavesplat/run.py:30` 은 그것을 import 하고 `configs/estimation.yaml:12` 이 미러한다.
```

### M-8. 인용금지 "~2x" 가 코드 주석 **2곳에 살아 있다** — 그리고 수리 주석이 틀린 줄을 가리킨다

```
run.py:53-56
#   CORRECTION 2026-08-11 (P1 audit): this said "~2x better".  Re-derived ratios
#   are 1.723/1.392/1.460 -- median 1.46x … Do not cite "2x".
#   (Same correction applied at sequential_update.py:459.)
```
실측:
- `sequential_update.py:459` 는 `if frames and self.config.get('real_domain_gate', True) \` — **주석이 아니다.**
  실제 정정은 `sequential_update.py:478` 근방에 있다.
- ⛔**`sequential_update.py:197`** 는 아직 `"~2x better on the frozen real windows"` 그대로.
- ⛔**`configs/estimation.yaml:34`** 도 `"~2x better on the frozen real windows;"` 그대로.

MEMORY 는 `~2배` 를 인용금지로 등재했다(2026-08-12 정정). **4개 site 중 2개만 고쳐졌다.**

---

## 3. MAJOR/MINOR — 상수 단일출처 위반 (임무 3)

### C-1 (MAJOR). `SequentialUpdater.__init__` docstring 이 실제 기본값과 어긋난다
```
sequential_update.py:159   lambda_l1: L1 sparsity weight (default 0.001)     ⛔ 실제 = 1e-4 (:40,:173)
sequential_update.py:166   tracking_mode: … (default 'phi_only')             ⛔ 실제 = 'w_phi' (:179)
```
같은 파일 안에서 40줄 떨어져 모순된다. λ 는 특히 위험 — 감사자가 이 docstring 을 근거로 삼으면 배포 전 값을 읽는다.

### C-2 (MAJOR). 분석기에 배포 이전 λ 가 하드코딩
```
tmp/S6_l1_sweep_2026-08-08/analyze_s6.py:115
    lam[r['cell']].add(c.get('lambda_l1', 0.001))
```
`cfg` 에 λ 가 없는 셀(= §M-4 의 모든 셀)을 **1e-3 으로 라벨링**한다.
배포 후에는 이 라벨이 거짓이고, 그 값으로 `§4.2 lambda_l1 확인` 게이트가 통과한다 = **게이트가 스스로 속는다.**

### C-3 (MINOR). 셸 경계가 두 번 구현돼 있고 한 쪽은 죽어 있다
- 생산: `run.py:358` `_edges = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]` — 인라인 최소거리 루프.
- 라이브러리: `evaluation/metrics.py:162 rmse_shell` · `:190 shell_fraction` — **동일 로직**인데
  저장소 전체에서 호출자 **0건** (grep 실측).
- 소비: 셸 키 리스트가 tmp 분석기 **12개 이상**에 각각 하드코딩
  (`AUDIT_full/audit_c_recompute.py:25` · `skeptic_p4.py:10` · `S8/analyze_s8.py:31` ·
   `S9/{analyze_s9.py:29,mde_prespec_s9.py:34}` · `ED/{run_ed.py:629,analyze_ed.py:16,audit_con_ed.py:7}` ·
   `BLANKSLATE/bs3_recompute.py:390` · `DEBATE_blankslate/r2_p1_verdict.py:15` …).

### C-4 (MINOR). `configs/estimation.yaml` 이 로더가 **읽지 않는** 키 7개를 문서화한다
`load_yaml_config`(`run.py:81-123`)가 처리하는 섹션은 `model/optimizer/physics/adaptive/tracking/schedule/device/seed` 뿐이다.

| yaml 키 | 줄 | 로더가 읽는가 | 비고 |
|---|---|---|---|
| `optimizer.max_steepness: 0.15` | :38 | ⛔ | `sequential_update.py:204-208` 주석은 "2026-07-24 수리" 라 하나 **yaml→cfg 경로는 여전히 끊겨 있다** |
| `init.tp_factor: 2.0` | :59 | ⛔ | `run.py:161` 이 `tp_factor=2.0` 하드코딩 |
| `init.grid_nx / grid_ny` | :60-61 | ⛔ | 패키지 내 동명 인자는 `model/{wave_np,spec_np}.py` 의 무관한 것 |
| `eval.converge_threshold: 0.05` | :69 | ⛔ | `run.py:204` 이 `converge_threshold = 0.05` 하드코딩 |
| `eval.split_distance: 80.0` | :70 | ⛔ | `metrics.py:99` 이 기본인자 80.0 하드코딩 |
| `eval.unobserved_min_dist: 10.0` | :71 | ⛔ | **패키지 어디에도 이 이름이 없다** |

### C-5 (MINOR). `s9_constants.py` 는 훌륭하나 **S9 전용**이다
`tmp/S9…/s9_constants.py` 는 n-의존 상수를 유도하고 import 시 자기시험(`primary_k(32)==23`, `mde_sign80_c(32,23)≈0.7234`)을
돌린다 — 실행 확인 OK. 그러나 `tmp/S13…/analyze_s13.py` 는 이를 import 하지 않는다(별도 문턱 41/64 사용).
처방("상수를 한 곳에서 유도")이 **후속 배치로 전파되지 않았다.**

### C-6 (MINOR). 데이터 생성 상수 중복·낡음
- `configs/default.yaml:48` `# 15 scenarios × 5 seeds × 2 raydrop models = 150 episodes` —
  실제 활성 시나리오는 **10개**(S11–S15 는 주석처리, `:63-68`) ⟹ 10×5×2 = **100**.
- `sea_states` BF2–BF5 블록이 `default.yaml:44-48` 과 `server.yaml:45-49` 에 **글자 그대로 중복**.
- `sequential_update.py:1004` 주석 `# e.g., keep 95%` — 출하 `prune_threshold: 0.10` 이므로 **90%**
  (`estimation.yaml:48` 주석은 "keep top 90%" 로 맞다).

---

## 4. MINOR — 문서 카운트·트리 (임무 1)

| CLAUDE.md 주장 | 줄 | 실측 | 판정 |
|---|---|---|---|
| `wavesim/` Wave simulation + LiDAR | :10 | 존재 | ✅ |
| `wavesplat/` … **(196 tests total)** | :11 | `pytest --collect-only` → **196 tests collected** | ✅ |
| `model/` spectral_gaussian, dispersion, confidence | :12 | 3개 파일 전부 존재 | ✅ |
| `optimizer/` initializer, coarse_to_fine, sequential_update | :13 | 전부 존재 | ✅ |
| `baselines/etkf_grid.py` · `evaluation/metrics.py` · `run.py` | :14-16 | 전부 존재 | ✅ |
| `scripts/` **413** experiment/paper scripts | :17 | `*.py` **413** (전체 파일 416) | ✅ |
| `paper/` … `wavesplat.tex` + `figures/` (**5 png**) | :18 | `wavesplat.tex` 2,796줄 존재; `paper/figures/*.png` = **5** | ✅ |
| `thesis/thesis_main.tex` | :19 | 존재, **3,992줄** (MEMORY 와 일치) | ✅ |
| `results/` **96** JSON (+ direction1/2/3) | :20 | 최상위 `*.json` **96**; `direction1/2/3` 존재(재귀 624) | ✅ |
| `configs/` estimation/server/default.yaml | :21 | 정확히 이 3개 | ✅ |
| `data/` HDF5 episodes (**110**) | :22 | `data/*.h5` **110** (표준 100 + BF6/BF7 8 + BF4_20comp_seed7·BF4_30comp_seed42) | ✅ |
| `data/real_bergamasco/` 실측 **13 레코드** | :22 | 사이트 **6**개(AA_2014·AA_2015·BS_2011·BS_2013·LJ_2018·YS_2017), 파일 합 **13** (1+2+4+3+2+1) | ✅ |
| `tmp/` 배치별 사전등록·결과 (**S1–S11**, B-3 등) | :23 | 배치 디렉터리 **121**개, **S13 이 이미 존재** | ⚠**낡음** → `S1–S13` |
| `docs/PRESENTATION_SUMMARY.md` (**~600줄**) | :24 | **562줄** | ⚠ 근사치로는 허용 범위, 정확히는 562 |
| `requirements.txt` | :25 | 존재 | ✅ |
| **트리 누락** | :9-25 | 저장소 루트(`/home/vexit/wavesplat`)에 **`scripts/`·`tmp/`·`external/`·`ref_paper/`·`graphify-out/`** 가 따로 있고, `wavesplat/` 안에 `Falcon_lidar/`·`dcspecnet/`·`hydrofoil/`·`data_*` 14개 디렉터리가 문서화 안 됨 | ⚠ |

### 4-2. `Key Commands` 나머지 (B-1 외 전부 통과)

| 커맨드 | 판정 |
|---|---|
| `python3 -m pytest wavesim/tests/ wavesplat/tests/ -v` | ✅ 196 collected |
| `scripts/generate_paper_tables.py` | ✅ 존재 + `py_compile` OK |
| `scripts/generate_paper_figures.py` | ✅ 존재 + compile OK |
| `scripts/run_p1_comparison.py` (Main **12-episode** comparison) | ✅ 존재 + compile OK; `:5` docstring "12 representative episodes", `:23 EPISODES` 리스트 확인 |
| `scripts/run_gpr_comparison.py` | ✅ 존재 + compile OK |
| `scripts/run_raydrop_comparison.py` | ✅ 존재 + compile OK |

### 4-3. `Important Design Decisions` 줄번호 전건 대조 — **전부 정확**

| 주장 | 실측 |
|---|---|
| `tracking_mode: w_phi` @ `estimation.yaml:22` | ✅ `  tracking_mode: w_phi         # 'phi_only', 'w_phi', or 'all'` |
| `init_method: auto` @ `run.py:43` | ✅ `    'init_method': 'auto',` |
| `init_method: auto` @ `estimation.yaml:24` | ✅ `  init_method: auto            # 'auto' (GT-free router …` |
| `lambda_l1` @ `configs/estimation.yaml:12` | ✅ `  lambda_l1: 0.0001` |
| `lambda_l1` @ `wavesplat/run.py:30` | ✅ `    'lambda_l1': LAMBDA_L1_DEFAULT,` (리터럴 아님 — §M-7) |
| `lambda_l1` @ `optimizer/sequential_update.py:40` | ✅ `LAMBDA_L1_DEFAULT = 1e-4` |
| 배포 커밋 `40e22f6` | ✅ `git show` 확인: yaml `0.001→0.0001`, run.py `0.001→0.0001`, 로더 폴백 `0.001→DEFAULT_CONFIG[...]` |
| `ω = √(g·|k|·tanh(|k|·d))` | ✅ `model/dispersion.py:33` `torch.sqrt(G * k_mag * torch.tanh(kd) + 1e-12)`, `G=9.81` |
| Adaptive N: energy-based pruning + residual-triggered growth | ✅ `sequential_update.py:988-1005` (에너지 정렬 프루닝) + `:1040` 성장/롤백 |
| `⛔"v2_konly" 는 낡음` | ⚠**문서는 옳으나 코드 폴백 2곳이 아직 v2_konly/v2** (§M-5) |

`Adaptive N` 한 줄은 이번 diff 에서 `with safety rollback` 을 **삭제**했는데, 그 가드는 코드에 살아 있다:
```
sequential_update.py:1040    if rms_before > 1e-12 and rms_after > 1.3 * rms_before:
```
→ 30% 롤백 가드 = MEASURED 존재. 문서에서 지운 것은 **과소기술**. (MINOR)

### 4-4. `측정 규율` 절 검증

| 주장 | 실측 |
|---|---|
| "WaveSplat 이라 불리는 추정기 **최소 4개**" | ✅ `SequentialUpdater` (`sequential_update.py:141`) · `bank_mmse` 라우트 (`sequential_update._initialize_bank`, `test_deploy_auto_v1_m3.py:224`) · `mmse_bank_estimate` (`scripts/d3_sa_nomp.py:499`) · `relax_step` (`scripts/d18_relax_estimator.py:69`) + `model/tnse.py`. **4개 전부 실재** |
| "`results/direction1/*` 계열은 LiDAR 를 0회 호출" | ✅ 생산자 5개(`d1_crb.py`·`d1_predictable_zone.py`·`d72_platform_aperture.py`·`d73_tank_scaled_aperture.py`·`d74_closed_loop_trajectory.py`) 전부 `SequentialUpdater`/`run_estimation`/`make_updater` **0회**, LiDAR 시뮬레이터 호출 **0회**. ⚠단 `d1_crb.py:154` 는 에피소드 h5 의 **기록된** `lidar_points`/`lidar_valid_mask` 를 읽는다 ⟹ 정확한 문장은 "출하 추정기를 0회 호출한다" |
| "`SUPPORT-TRADEOFF` 가 7회 재현됐다" | 코드로 검증 불가(측정 이력 주장). MEMORY 08-12 항목과 일치. **ARGUED** |

### 4-5. `Current State` 절

| 주장 | 판정 |
|---|---|
| "**196 tests** (…로 실측)" | ✅ 재현됨 |
| ⛔"WS 0.223 / ETKF 0.319 / 13.4 ms / 132.6 ms" 인용 금지 | ✅ 이 파일 자체가 인용을 하지 않음. 별도 감사(A3) 소관 |
| ⛔"All phases complete" 금지 | ✅ 본문에 해당 문구 없음 |
| 권위는 `MEMORY.md` 와 `tmp/<배치>/RESULT.md` | ✅ 해당 경로 실재 |

---

## 5. 미커밋 변경 검토 (임무 4)

### 5-1. `CLAUDE.md` (+52/−24)

diff 는 2026-04 시점 서술을 현재 측정으로 교체한다. **삭제된 항목은 전부 삭제가 옳다** (검증):

| 삭제된 주장 | 검증 |
|---|---|
| `wavesim/ (39 tests)` · `wavesplat/ (18 tests, 57 total)` | ⛔ 실측 196 ⟹ 삭제 정당 |
| `scripts/ 11` · `results/ 7 JSON` · `data/ 100` | ⛔ 실측 413 / 96 / 110 ⟹ 정당 |
| `paper/ + 9 figures` | `figures/` 에 pdf 9 + png 5 ⟹ 새 문구("5 png")가 더 정확 |
| `All phases complete` / `57 tests all passing` | ⛔ 반증됨 ⟹ 정당 |
| `WS 0.223 / ETKF 0.319 / 13.4ms / 132.6ms` | 인용금지 등재 ⟹ 정당 |
| `init_method: v2_konly` | ⛔ 2026-07-26 flip ⟹ 정당 |

**추가된 주장 중 정정 필요:** §B-1(Key Commands, 이 diff 에서 손대지 않아 낡은 채 남음),
§M-3(S5·S6 누락), §M-7("세 곳에 중복 기재"), §4-1(S1–S11 → S1–S13, ~600줄 → 562줄), §4-3(safety rollback 삭제).

**회귀 없음** — 삭제된 것 중 되살려야 할 사실은 발견되지 않았다.

### 5-2. `wavesplat/scripts/realdata_analysis.py` (+32/−9) — **정당하고 회귀 없음**

변경 요지: `load_surface_nc(path, tmax_frames=None)` → `(path, tmax_frames=None, frames=None)`.

| 검사 | 결과 |
|---|---|
| **동기의 사실성** | docstring "the 11 GB LJ Grid2 record asks for 20.2 GiB and dies" ⟹ MEMORY B-3 의 `"WSL 을 죽인 20.2 GiB 로더 결함이 이 배치(수정 완료)"` 와 **일치** |
| **레거시 경로 보존** | `frames is None` → `sel = np.arange(min(T_all, tmax_frames) …)` 이고 `_unpack(zd, np.s_[:T])` 로 **평 슬라이스 유지** ⟹ 기존 결과 비트 동일 |
| **호출자 파손** | 저장소 전체 호출자 5곳(`tmp/e46_real_bimodality.py:146` · `tmp/e59_real_anchor.py:175` · `tmp/e59v2_real_anchor.py:184` · `scripts/realdata_analysis.py:154` · `tmp/B3…/b3_regime.py:183`) **전부 키워드 인자** ⟹ 새 3번째 파라미터로 인한 위치인자 오해석 **없음** |
| **CLI 파손** | `:248 --frames` → `:253 tmax_frames=args.frames` 로 **변경 없음**. 새 `frames=` 는 라이브러리 전용 |
| **★fps 버그 수정의 필요성** | 구: `fps = 1/median(diff(t))`. `frames` 로 **성기게** 뽑으면 `t` 가 stride 되어 fps 가 stride 배만큼 **거짓**이 된다. 신: `dt = median(diff(t_all))` = **원 cadence** ⟹ 올바름. `b3_regime.py:201` 이 `meta['fps']` 를 진단값으로 기록하므로 이 수정이 없었으면 B-3 의 fps 가 전부 틀렸다 |
| **경계 조건** | `sel.size == 0` 이면 `if sel.size and …` 가드로 IndexError 회피, `T=0`, `t` 빈 배열 — 크래시 없음. 범위 밖은 `raise IndexError` 로 명시 실패 |
| **h5py fancy index** | `sel = np.unique(...)` 로 **증가순 보장** ⟹ h5py 요구 충족. `_unpack:33-34` 의 `out[raw == fill]` 은 (T,n0,n1) 에서 정상 동작 |
| **메타 정합** | `shape[0] = sel.size`(적재분), `frames_total = T_all`(원본), `frames_idx = sel` ⟹ 세 값이 서로 구분돼 기록됨 — 좋음 |

⚠**남은 미세 흠 (MINOR, 사전 존재):** 모듈 docstring `:10` 의 사용례가 `[--frames 50]` 인데 실제 기본값은
`:248` 에서 80 이다. 이번 diff 와 무관.

**판정: 변경은 측정 결과(B-3)와 일치하고, 의도치 않은 회귀는 발견되지 않았다.**

---

## 6. 일치 확인 목록 (MEASURED, 24건)

1. `tracking_mode: w_phi` @ `estimation.yaml:22`
2. `init_method: auto` @ `run.py:43`
3. `init_method: auto` @ `estimation.yaml:24`
4. `lambda_l1` @ `estimation.yaml:12`
5. `lambda_l1` @ `run.py:30`
6. `lambda_l1` @ `sequential_update.py:40`
7. `SequentialUpdater({}).lambda_l1 == 1e-4` (실행)
8. `DEFAULT_CONFIG['lambda_l1'] == 1e-4` (실행)
9. 배포 커밋 `40e22f6` 의 λ 변경 (git show)
10. `DEFAULT_CONFIG` ↔ `estimation.yaml` 공유 20키 **불일치 0건** (실행 diff)
11. 196 tests (pytest collect)
12. scripts 413
13. results 최상위 JSON 96
14. data h5 110
15. real_bergamasco 13 레코드 / 6 사이트
16. paper/figures png 5
17. `thesis/thesis_main.tex` 존재 (3,992줄)
18. `docs/PRESENTATION_SUMMARY.md` 존재
19. configs 3종 정확
20. 분산관계 `ω=√(g|k|tanh(kd))` @ `dispersion.py:33`
21. Adaptive N = 에너지 프루닝 + 잔차 성장 @ `sequential_update.py:988,1040`
22. 추정기 4종 전부 실재
23. `results/direction1` 생산자 5/5 가 출하 추정기·LiDAR 시뮬 **0회 호출**
24. `TRACK_PHASE_PREDICT = False` @ `sequential_update.py:83` (07-26 flip 과 일치)

---

## 7. 우선순위 권고

1. **B-1** — Key Commands 한 줄 수정 (1분, 문서만).
2. **M-3** — `CLAUDE.md:63` 에 S5·S6 추가 + 뿌리(`run_rx2b.py:61`) 명시.
3. **M-5** — `sequential_update.py:181` · `run.py:99` 의 `init_method` 폴백을 단일출처화.
   λ 와 정확히 같은 사고가 이 키에서 **아직 대기 중**이다.
4. **M-8** — `sequential_update.py:197` · `estimation.yaml:34` 의 `~2x` 제거,
   `run.py:56` 의 잘못된 줄 참조(459→478) 수정.
5. **M-4** — 신규 배치는 S13 방식(`effective_config` 통째 기록)을 표준으로.
6. **M-7** — CLAUDE.md 의 "세 곳에 중복 기재" 를 "단일출처 + yaml 미러" 로 교체
   (지금 문구는 **이미 고쳐진 결함을 고치라고 지시**한다).
