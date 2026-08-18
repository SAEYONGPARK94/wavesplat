# C1 — `L-ITER0` · `L-MERGE` 팔 구현 + 개입 카운터 (런 0회)

**산출**: `tmp/W4_SOLVER_2026-08-19/arms_config.py` (`md5 47cb1cd1af976cc487660e4b6d9cb511`)
**dry-run 로그**: `recon/C1_dryrun.log` — 26/26 `PASS`, `DRYRUN OK`
**git HEAD**: `cb4b4fb` · 임포트 경로 파일 md5 (더러운 트리 방어, CLAUDE.md 지시):

| 파일 | md5 |
|---|---|
| `wavesplat/optimizer/sequential_update.py` | `63e6362b7c2f0e66e1a4ea3dd38ed075` |
| `wavesplat/optimizer/learned_init.py` | `e6a46f8f249566afd035cca88dd23939` |
| `wavesplat/run.py` | `e546e834a414571dfd7d80248d043331` |
| `wavesplat/model/spectral_gaussian.py` | `3e4a6fe1b34b84313a33cba443cf7bfe` |

**추정기 런 0회.** 에피소드 로드 0회, LiDAR 0회, 필드 채점 0회. `wavesplat/**`·`configs/` 무수정
(`git status` 상 두 트리 모두 변경 없음). 아래 `MEASURED` 는 전부 `python3 arms_config.py` 의
정적/모의 검증에서 나온 것이고, 실제 에피소드 위에서의 값은 하나도 없다.

---

## 1. ★★★요구 3 판정 — `L-ITER0` 은 **엄밀히 단일요인이 아니다** (2요인)

`PREREG §2` 는 `L-ITER0` 를 **"순수 config 노브 / 단일"** 이라 라벨한다. **그 라벨은 좁게 틀렸다.**

`learned_init_iters` 가 실제로 쓰이는 곳은 **단 한 줄**이다 —
`sequential_update.py:643` `max_iters = int(self.config.get('learned_init_iters', 500))`,
그리고 그 값이 지배하는 것은 `:645` `for iteration in range(max_iters):` 하나뿐이다.
그 루프 본문은 `:646-656` 이고 **문장이 넷**이다:

| 줄 | 문장 | `iters=0` 일 때 |
|---|---|---|
| `:646-649` | `zero_grad` / `_compute_loss` / `backward` / **`optimizer.step()`** | 꺼짐 — **의도된 요인(하강)** |
| `:650` | **`self.model.clamp_steepness(self.max_steepness)`** | 꺼짐 — ⚠**두 번째 요인** |
| `:651` | `loss_history.append(...)` | 기록만 (아래 §1.3) |
| `:652-654` | `verbose` 인쇄 | 무해 |

### 1.1 ⚠두 번째 요인 = **Phase-1 물리 제약 투영이 500회 → 0회**
`clamp_steepness`(`model/spectral_gaussian.py:219-230`)는 `|k|·|w| > max_steepness(0.15)` 인
원자의 진폭을 **그 자리에서 축소한다**. `L-SHIP` 은 이것을 500번 적용하고 `L-ITER0` 은 **0번** 적용한다.

- **Phase 1 안에서 다른 어떤 지점도 이것을 재적용하지 않는다.** `_initialize_learned` 안의
  `clamp_steepness` 호출은 `:650` **하나뿐**이고, 나머지 두 호출부는
  `_adaptive_prune_grow` 의 성장 미세조정 루프(`:1209`)와 추적 스텝(`:979`)이다.
- 따라서 `L-ITER0` 은 **클램프되지 않은 `w`** 를 그대로 `_adaptive_prune_grow` 에 넘긴다.
  그 프루닝 규칙은 `:998` `energy = self.model.w ** 2` 의 **에너지 비율**이므로,
  과대 진폭 원자 하나가 프루닝 판정을 바꿀 수 있다.
- 초기 `w` 는 `learned_init.py:386 ls_fit_wphi`(ridge 0.5)의 **LS 적합 절댓값**이다.
  희소 LiDAR 위 저조건 원자에서 큰 진폭이 나올 수 있고, 제안 원자의 `|k|` 는
  빔포밍 격자 범위상 최대 4.0 rad/m 까지 가므로 `|w| > 0.0375` 만 되어도 문턱을 넘는다.
  **⟹ 원리적으로 무해하지 않다.** (`INFERRED` — 실제 코퍼스에서 발생하는지는 **안 재봤다**.)

### 1.2 ⟹ 처방 (런을 태우지 않고 판정을 지키는 방법)
카운터 `clamp_calls_p1` · `clamp_hits_p1` 을 넣었다. **`clamp_hits_p1` 은 호출이 아니라
"실제로 축소가 일어났는가" 를 센다**(`arms_config.py` `clamp()` 이 원본 호출 전에 초과 원자
존재 여부를 직접 계산한다). 판정 규칙 제안:

- `L-SHIP` 전 행에서 **`clamp_hits_p1 == 0`** ⟹ 두 번째 요인이 **한 번도 작동하지 않았다**
  ⟹ `PREREG §2` 의 "단일" 라벨이 **효과 수준에서 참**. 그대로 진행.
- `L-SHIP` 어느 행에서든 **`clamp_hits_p1 > 0`** ⟹ `P-2` 는 **"500스텝 하강 + Phase-1 스티프니스
  투영" 묶음의 효과**다. `PREREG §1.1` 의 세 라벨(`OPTIMIZER-HARMS`/`-INERT`/`-EARNS`)을
  **"하강이"** 로 읽으면 안 되고 **"Phase-1 루프 전체가"** 로 읽어야 한다. RESULT 서술에
  반드시 적을 것. (이것이 `AMENDMENT` 대상인지는 사용자·감사 판단.)

추가로 **수동 관측** `steepness_max_post_p1`(프루닝 진입 시점 = P1 루프 종료 직후,
모델을 **변형하지 않고** `max |k|·|w|` 를 읽는다)을 행에 넣었다. `L-ITER0` 에서 이 값이
0.15 를 넘는지가 두 번째 요인의 **직접 증거**다.

### 1.3 ⛔함께 꺼지지 **않는** 것 (오해 방지, 전부 `MEASURED` 정적 추적)
- **수렴 판정·코스투파인 staging·스케줄**: `_initialize_learned` 에는 **애초에 없다.**
  `schedule` / `should_advance` / `advance` 는 해석 경로 `_initialize`(`:558-592`)에만 있고,
  독스트링(`:621-626`)이 스스로 *"the coarse-to-fine N-staging is bypassed"* 라 적었다.
  ⟹ **`learned_init_iters=0` 이 staging 을 끄는 일은 없다 — 이미 꺼져 있다.**
- **프루닝/성장 트리거**: `:658-660` 은 루프 **밖**이다. `self._last_t` 설정도,
  `if self.adaptive_n: self._adaptive_prune_grow(...)` 도, `_build_tracking_optimizer()` 도
  **`max_iters` 와 무관하게 그대로 실행된다.**
  (프루닝의 *결과* 가 달라지는 것은 매개된 효과이지 별도 요인이 아니다.)
- **옵티마이저 구성**: `_build_optimizer()` 는 `:640`, 루프 **앞**에서 무조건 실행.
- **크래시 위험 없음**: `loss_history` 가 비어도 `:671` 은 `if loss_history else nan` 로 가드되고,
  `run.py:461-462` 는 `n_iters`·`elapsed_s` 만 읽는다(둘 다 존재).

### 1.4 ⚠부수 발견 — `init_iters` 는 **측정이 아니라 의도표다**
`sequential_update.py:673` 이 돌려주는 `'n_iters': max_iters` 는 **config 값의 반향**이고,
`run.py:461` 이 그것을 `init_iters` 로 결과에 인쇄한다. **실제 실행 스텝 수가 아니다.**
⟹ ⛔**G-2 를 `init_iters` 로 검사하면 W3 의 실패를 그대로 반복한다.** 카운터가 이것을 대체한다.

---

## 2. 요구 4 — `learned_merge_frac=0.5` 도달 경로 (화이트리스트 **0개**)

| # | 지점 | 코드 | 통과 사유 |
|---|---|---|---|
| 1 | `run.py:139-141` | `cfg = DEFAULT_CONFIG.copy()` → `cfg.update(config)` | **키 필터 없음** — 임의 키가 살아남는다 |
| 2 | `run.py:168` → `sequential_update.py:1586-1589` `make_updater` | `init_method != 'auto'` ⟹ **plain `SequentialUpdater`** | `MedianEnsembleUpdater` 우회 |
| 3 | `sequential_update.py:171` | `self.config = config` | **참조 저장, 복사·필터 없음** |
| 4 | `sequential_update.py:445` / `:516-517` | `init_method=='learned'` ⟹ auto·real-gate 블록 **전부 건너뜀** → `_initialize_learned` | G-1 이 구성적으로 100% |
| 5 | `sequential_update.py:626-628` | `learned_init_params(frames, self.depth, **self.config**, seed=self.seed)` | 같은 dict 객체 전달 |
| 6 | `learned_init.py:367-369` | `merge_frac = config.get('learned_merge_frac')` → `if merge_frac:` → `merge_atoms(kx, ky, en, float(merge_frac))` | `0.5` 는 truthy |

**화이트리스트에 걸리는 경로 — 있는 것과 없는 것**
- ⛔**있음**: `run.py:585-609` 의 **CLI argparse 는 고정 키 목록**이라
  `--learned_merge_frac` 같은 인자가 **없다**. ⟹ **CLI 로는 이 팔을 만들 수 없다.**
  반드시 `run_estimation(h5, config=<dict>)` 로 주입해야 한다. 러너 작성자는 이 점을 지킬 것.
- ⛔**있음**: `run.py:~94` 의 YAML 로더도 특정 키만 옮긴다 ⟹ `configs/` 경유 불가(그리고 이 배치는
  `configs/` 를 건드리지 않는다).
- ✅**없음**: dict 주입 경로(1→6)에는 **어떤 단계에도 키 화이트리스트가 없다.**

**함정 두 개 (`INFERRED`, 러너가 지킬 것)**
1. `if merge_frac:` 는 **falsy 검사**다 — `0.0`·`""`·`None` 은 조용히 no-op. `0.5` 만 쓸 것.
2. `learned_atoms` 캐시가 config 에 미리 들어 있으면 확산 제안을 건너뛰지만 `pool_and_select` →
   `merge_atoms` 는 **그대로 탄다**. 우리는 `init_method='learned'` 고정이라
   `_auto_select_init`(캐시를 심는 유일한 지점)이 **실행되지 않으므로** 캐시는 비어 있다.

**dry-run 결과 (`MEASURED`)**: `L-MERGE` 에서 `cfg:learned_merge_frac == 0.5` 가
**살아있는 updater 객체 되읽기**로 확인됐고, `merge_atoms` **피호출자가 실제로 받은 값**이
`merge_frac_seen == [0.5]`, 실제 제거량이 `5 → 2 (removed 3)` 였다.

---

## 3. 카운터 설계 — **플래그가 아니라 사건을 센다**

W3 의 `install_probe`(`tmp/W3_SOLVER_2026-08-15/run_w3.py:181-253`) 방식을 따라, 훅은 전부
**일을 하는 문장 그 자체**에 건다.

| 카운터 | 훅 | 세는 사건 |
|---|---|---|
| `adam_steps_phase1` | `torch.optim.Adam.step` + phase 스택 | `:645-653` 루프에서 **실제 실행된** 스텝 |
| `adam_steps_grow` | 동일 훅, `_adaptive_prune_grow` 안에서 | `:1202-1207` 의 50스텝 미세조정 |
| `adam_steps_track` / `_other` | 동일 훅 | 추적 / **미귀속(=경보)** |
| `merge_removed` | `learned_init.merge_atoms` | `len(kx_in) − len(kx_out)` |
| `merge_frac_seen` | 동일 | 피호출자가 **받은** 값 |
| `n_atoms_pooled` / `_after_merge` | `learned_init.pool_and_select` / 위 | D-3 원자 감소량 |
| `clamp_calls_p1` / `clamp_hits_p1` | `SpectralGaussianModel.clamp_steepness` | §1.2 의 2요인 탐침 |
| `learned_init_calls` | `SequentialUpdater._initialize_learned` | G-1: 학습 경로 진입 실측 |
| `steepness_max_post_p1` / `_post_init` | 위 두 래퍼 | **수동 관측**(변형 없음) |

### ★★★설계상 가장 중요한 한 가지 — `grow` 를 `phase1` 에서 분리한 이유
`_adaptive_prune_grow` 는 성장 분기에서 **`self.optimizer.step()` 을 50회 돈다**
(`sequential_update.py:1202-1207`). 이 함수는 `_initialize_learned` **안에서**(`:660`) 호출된다.
⟹ **"`_initialize_learned` 동안의 Adam 스텝" 을 그냥 세면 `L-ITER0` 이 `adam_steps_phase1 == 50`
을 찍고 `G-2`("정확히 0")가 **엉뚱한 이유로 중단**된다.** `PREREG §0.2` 가 `add_gaussians` 를
`SHIP` 혼잡 **17/96** 에서 관측했으므로 이것은 가상의 위험이 아니다.
phase 스택(가장 안쪽이 이김)이 이 오귀속을 막고, dry-run `grow nests inside p1 (G-2 safe)` 가
**중첩 의미론을 실측 확인**했다.

부수 효과로 `adam_steps_other > 0` 은 **"우리가 모르는 곳에서 최적화가 돌았다"** 는 경보가 되고,
`check_gates()` 가 이것을 실패로 처리한다.

---

## 4. dry-run 검증 — 무엇을 실행했고 무엇을 실행하지 **않았나**

`python3 arms_config.py` (`recon/C1_dryrun.log`, 26/26 `PASS`). 실행한 것:

1. **팔 델타가 정확히 한 키** — `L-ITER0` = `{learned_init_iters}`, `L-MERGE` = `{learned_merge_frac}`.
2. **주입이 살아있는 객체에 도달** — `run.py` 와 **같은 방식**(`DEFAULT_CONFIG.copy()` + `update`)으로
   config 를 만들고 `make_updater` 를 실제로 호출한 뒤 **객체 되읽기**.
   ⚠**의도표 금지 규율(S13b)** 을 지키기 위해 `config_is_input_dict` 를 함께 기록한다 —
   세 팔 모두 **`False`**(즉 우리가 넘긴 dict 이 아니라 `run.py` 가 만든 dict 을 읽었다).
   속성이 없는 키(`learned_*`)는 `cfg:` 접두어로 **살아있는 `updater.config`** 에서 읽는다.
3. **merge 경로 end-to-end** — `learned_atoms` 캐시(확산 우회) + `frames=[]`
   (`_gather_init_obs` 가 `None` 반환 → `ls_fit_wphi` 가 0 반환, `learned_init.py:89-90`)로
   **에피소드 I/O·GPU·추정기 스텝 0** 인 채 `pool_and_select → merge_atoms` 를 실제로 통과시켰다.
4. **Adam 훅이 진짜 torch 심볼에서 발화** — 실제 `torch.optim.Adam` 인스턴스로 귀속 (2,1,1,1) 확인.
5. **래퍼가 실제 메서드에 설치됐는지** 심볼 확인 + 중첩 의미론.
6. **클램프 카운터가 호출과 적중을 구분** — 실제 `SpectralGaussianModel` 로 확인.
7. **`undo()` 가 6개 심볼 전부 복원.**

⛔**실행하지 않은 것 = 주장하지 않는 것**: 실제 에피소드에서 `adam_steps_phase1` 이 `L-SHIP` 500 /
`L-ITER0` 0 이 되는지, `clamp_hits_p1` 이 실제로 0 인지, `merge_removed` 의 코퍼스 중앙값이 0 보다
큰지는 **전부 미측정**이다. §1.2 의 2요인 판정 역시 **코드 추적(`MEASURED`)** 이지
**효과 측정(`미측정`)** 이 아니다.

---

## 5. 러너에게 넘기는 계약

```python
from arms_config import (arm_config, install_counters, install_tracking_phase,
                         install_updater_capture, counters_reset,
                         counters_snapshot, effective_config, check_gates)

_, undo_c   = install_counters()          # 프로세스당 1회
undo_t      = install_tracking_phase()
undo_u      = install_updater_capture()

for ep, arm, est_seed in plan:            # 행마다
    counters_reset()                      # ★필수 — 안 하면 카운터가 누적된다
    cfg = arm_config(arm, est_seed=est_seed)
    res = run_estimation(ep, config=cfg)  # ⛔CLI 금지 (§2: argparse 에 키가 없다)
    row = dict(arm=arm, **counters_snapshot(),
               effective_config=effective_config(input_cfg=cfg),
               **check_gates(arm))
```

- `ARM_DELTAS` 에 `L-SHIP-DUP`(G-5 복제 프로브)를 함께 정의해 뒀다 — `L-SHIP` 과 델타 0.
- `L-VP` 와 `ZERO`·`ORA-K-CVX` 는 **이 파일의 범위 밖**이다(다른 담당).
- `check_gates()` 는 **행 단위** 판정이다. `G-4` 의 *중앙값 > 0* 은 집계 단계에서 러너가 볼 것.
- `base_config()` 는 `lambda_l1` 을 `LAMBDA_L1_DEFAULT` 에서 **import** 한다(재타이핑 금지).
  `analytic_ensemble_k=1` 은 여기서 **무효**지만(`init_method != 'auto'` 라 `make_updater:1586` 이
  읽지 않는다) 되읽기 오독 방지를 위해 명시했다.
