# C3 — `run_w4.py` 러너 골격 + §G 게이트 (추정기 런 0회)

**작성**: 2026-08-19 · **에이전트 C3(구현)** · **에피소드 로드 0회 · 추정기 런 0회**
**산출**: `tmp/W4_SOLVER_2026-08-19/run_w4.py` (`py_compile` 통과) · `gates.log` (실행 결과) ·
`code_fingerprint.json` (게이트가 씀)

---

## 0. 실행 결과 (실제로 돌렸다 — 인용 가능)

```
$ python3 tmp/W4_SOLVER_2026-08-19/run_w4.py gates    # exit=0
GATES(run-free) PASS
```
전문은 `gates.log`. 통과한 항목(런 0회로 판정 가능한 것만):

| 게이트 | 결과 | 값 |
|---|---|---|
| G-1 표본 | PASS | n=64 · ncomp `[15,30]` · 층 `['crowded']` · 중복 0 · 결측 0 |
| G-1b 라우터 고정 | PASS | A족 5팔 전부 `init_method='learned'` |
| G-1c 배포 기본값 | PASS | 8개 전건 일치(`lambda_l1=1e-4`·`tracking_mode=w_phi`·`tracking_lr=0.005`·`max_n=30`·`adaptive_n`·`use_dispersion`·`learned_konly=False`·`init_k_lr_scale=1.0`) |
| G-2/3/4 **단일요인** | PASS | `L-ITER0` Δ=`['learned_init_iters']` · `L-MERGE` Δ=`['learned_merge_frac']` · `L-VP` Δ=`(none)`+varpro 설치 · `L-SHIP-DUP` Δ=`(none)` |
| G-5 복제 프로브 | PASS | `L-SHIP-DUP` 설정 == `L-SHIP` 설정, 시드 1595 (수치 \|Δ\|=0 은 행이 있어야 판정) |
| G-6 필드계약 | PASS | 팔종별 3계약, 공통집합과 서로소 |
| G-7 위상 항등식 | PASS | `atan2(-b,a)` max\|Δ\|=**4.441e-15** · ⛔`sequential_update.py:1335` 의 `atan2(+b,a)` 는 **6.071e+00** · `varpro.ab_to_wphi` = **4.441e-15** |
| G-8 코드 지문 | PASS | `a678de3be555cc4bbc8477d30a474bc5` · 91파일 · 재계산 안정 · git `cb4b4fb2` dirty 62 |
| G-8b W3 재사용 | PASS | `run_w3.py` md5 `b221ec29…` == W3 `FROZEN_md5.txt` 기록값 |
| G-8c 셸 규약 | PASS | `run_w4.SHELL_KEYS == run_w3.SHELL_KEYS`, 전부 `obs:` |
| G-9 워치독 | PASS | `watchdog.PLAN` == 러너 계획 · N_EP 64/64 · TOTAL **960/960** |
| 시드 원장 | PASS | 소비 `[1595,1596,1597]`, 기존 최대 1594, 충돌 0 |

⛔**아직 판정되지 않은 것(정직하게 PENDING 으로 인쇄된다)**: `G-2`·`G-3`·`G-4`·`G-5` 의 **수치부**와
`G-6` DCC 실측은 **행 0개**라 `PENDING (not PASS)` 로 찍힌다. 통과라고 쓰지 마라.

**추가 dry-run 검증**(에피소드 0회, 패치/복원만):
`varpro` → `arms_config` 순서로 설치하면 `arms_config.initialize_learned` 가 **바깥**,
`varpro._initialize_learned__L_VP` 가 **안쪽**임을 클로저에서 직접 확인했고, LIFO 해제 후
`_initialize_learned` · `Adam.step` · `SequentialUpdater.update` · `run.make_updater` 4개 전부
원본으로 복귀함을 확인했다. **DCC 시뮬레이션**(각 팔이 실제로 쓰게 될 키 집합을 코드 경로에서
합성해 `missing_fields()` 통과) 결과 7팔 전부 `missing = NONE`.

---

## 1. ★필수 필드 결정 — 산출 가능성을 코드에서 확인했다

`PREREG §5` 의 필수 필드 목록을 **한 줄씩 코드에 대조**했다. S17 은 산출 불가능한 필드
(`background_diag`)를 전 팔 필수로 걸어 **판정 팔이 아닌 팔에서 배치가 자살**했다. 그래서 규칙은
**"그 팔이 산출할 수 있을 때만 필수"** 이고, 아래 면제는 전부 **구조적 불가능**이다.

### 1.1 산출 경로 (전건 확인됨)

| PREREG §5 필드 | 산출 경로 | 팔 |
|---|---|---|
| `episode·arm·stratum·ncomp·seed·wall_s` | 러너가 붙임 | 전 팔 |
| `avg_nRMSE·zero_nRMSE·shell_rmse·zero_shell` | est=`frame_results`, zero/cvx=`run_w3` | 전 팔 |
| `n_match` | est=`len(frame_results)`, zero/cvx=`n_frames_scored` | 전 팔 |
| `n_all` | `h5py['frames/timestamps'].shape[0]` (=301) | 전 팔 |
| `est_seed` | 잡 정의 | est(비null) |
| `route` | `arms_config.effective_config()['init_method']` = **객체 되읽기** | est |
| `adam_steps_phase1` | `arms_config` 의 `torch.optim.Adam.step` 훅 | est |
| `vp_steps` | `varpro.stats()['vp_steps']` | 전 팔(비-VP 는 0) |
| `merge_removed` | `arms_config` 의 `merge_atoms` 훅 (`n_in − n_out`) | est |
| `n_atoms_final` | est=`run_estimation()['n_gaussians']`, cvx=`n_gt` | est·cvx |
| `window_resid_rms` | est=`run_estimation()['window_resid']` (**`S5_WINDOW=1`**), cvx=`cvx_window_resid_rms`, zero=`rms(z_obs)` | 전 팔 |
| `effective_config` | est=**객체 되읽기**, zero/cvx=W3 provenance dict | 전 팔 |
| `git_head·code_md5` | 러너 | 전 팔 |

### 1.2 ★산출 불가로 판정해 **필수에서 뺀 것** (임의로 넣지 않았다)

**(A) `ZERO`·`ORA-K-CVX` 에서 구조적으로 불가능 — 팔종별 계약으로 분리**

| 뺀 필드 | 사유(구조적) |
|---|---|
| `route`·`route_source` | 두 팔에 라우터가 **없다**. `ZERO` 는 항등 0 필드, `ORA-K-CVX` 는 `numpy.linalg.lstsq` 한 번. |
| `adam_steps_phase1`(+`_grow`/`_track`/`_other`) | **옵티마이저가 없다**. `ORA-K-CVX` 는 정의상 `grad_steps=0`. |
| `merge_removed`·`merge_calls` | 학습 제안 원자 풀이 **없다** — `learned_init.merge_atoms` 는 호출 자체가 없다. |
| `counters`·`gates_pass`·`intent_config`·`lambda_l1_*`·`init_method_intent`·`window_n_obs` | 추정기 설정/카운터 자체가 없다. |
| `n_atoms_final` (**`ZERO` 만 추가 면제**) | `ZERO` 는 원자가 **0개도 아니라 개념이 없다**. `ORA-K-CVX` 는 GT 원자수를 쓰므로 필수 유지. |
| `est_seed` (**null 허용**, 부재 아님) | 두 팔은 결정적이다. 행에는 **존재하되 `null`** — "시드가 없다"가 그 팔의 참이지 결측이 아니다. |

**(B) 이름을 바꾸거나 매핑한 것**

- `PREREG §5` 의 `n_match, n_all` 은 W3 러너의 `n_frames_scored` 와 같은 양이 아니다
  (`n_frames_scored` 는 채점 프레임 **수** 하나뿐). 그래서 `n_match=n_frames_scored`,
  `n_all=h5 프레임 총수` 로 **둘 다** 기록한다. `results/b1_full_comparison.json` 이
  100/100 에피소드에서 두 팔을 **다른 프레임 수로 채점**했는데도 검출되지 않은 이유가
  `n_all` 부재였다 ⟹ **G-6b(프레임 동수) 게이트를 새로 넣었다.**
- `effective_config(객체 되읽기)` 는 **est 팔에만** 성립한다. `ZERO`·`ORA-K-CVX` 는
  되읽을 객체가 없어 W3 의 provenance dict(`{'solver': 'numpy.linalg.lstsq', 'grad_steps': 0, …}`)를
  그대로 쓴다. **이것을 "객체 되읽기"라고 부르지 마라.**

**(C) 뺄 뻔했다가 살린 것 — `window_resid_rms`**

기본 실행에서는 `run_estimation()['window_resid']` 가 **NaN** 이다(`run.py` 가 `S5_WINDOW=1`
게이트 뒤에 둔다). 그대로면 `D-1`(솔버 이득이 필드로 전이되는가)이 **측정 불가**다. 확인 결과
그 블록은 `torch.no_grad()` 안의 순수 읽기이고 **난수를 소비하지 않는다** ⟹ **모든 est 팔에서
균일하게 `S5_WINDOW=1`** 로 켜고, 그것이 아무것도 흔들지 않았다는 증거를 **G-5(`|Δ|=0`)** 가 준다.
⚠단 이 때문에 W4 의 est 행은 **W3 의 `SHIP` 행과 비트 비교가 불가능**하다(W3 는 `S5_WINDOW='0'`
으로 강제했다). W4 의 비트 앵커는 `ZERO`/`ORA-K-CVX` 쪽에만 있다.

---

## 2. 구조

```
run_w4.py
├─ 표본        episodes()            data_s8_crowded + data_s9_crowded, ncomp∈{15,30} → 64
├─ 팔 표       ARMS                  kind / cfg_from / vp / seeds / factor
├─ 설정        arm_cfg(arm, seed)    DEFAULT_CONFIG ∪ arms_config.arm_config(...)   ← run.py:139-141 과 같은 병합
├─ 계약        REQUIRED_COMMON / _EST / _CVX / _ZERO  +  missing_fields()
├─ 지문        code_fingerprint()    고정 파일집합(91) · preload() 로 단일화
├─ 팔 실행     run_est / run_zero / run_cvx
├─ 잡          build_jobs / run_one  (job_id = arm|episode|seed)
├─ 게이트      gates()  +  posthoc(rows)
└─ main        gates | smoke | run   (mp spawn Pool, WORKERS=8)
```

### 2.1 팔 · 표본 · 잡 수 (960)

| 팔 | 종류 | 설정 출처 | 시드 | 잡 |
|---|---|---|---|---|
| `ZERO` | zero | — | 1 (결정적) | 64 |
| `L-SHIP` | est | `arms_config['L-SHIP']` | 1595/1596/1597 | 192 |
| `L-SHIP-DUP` | est | `arms_config['L-SHIP-DUP']` | 1595 | 64 |
| `L-VP` | est | `arms_config['L-SHIP']` + **varpro 설치** | 1595/1596/1597 | 192 |
| `L-ITER0` | est | `arms_config['L-ITER0']` | 1595/1596/1597 | 192 |
| `L-MERGE` | est | `arms_config['L-MERGE']` | 1595/1596/1597 | 192 |
| `ORA-K-CVX` | cvx | — | 1 (결정적) | 64 |

⚠**희소는 돌리지 않는다** — `PREREG §1` 이 혼잡 층 한정이고 §7-5 가 희소 주장을 금지한다.
표본 구성 자체에 희소 디렉터리가 없다(부재 ≠ 증거).
**팔 순서** = `ZERO → L-SHIP → L-SHIP-DUP → L-VP → L-ITER0 → L-MERGE → ORA-K-CVX`:
IUT 대조(`ZERO`)와 PRIMARY 두 팔이 먼저 끝나고 **결정성 프로브가 세 번째**라, 중간에 끊겨도
"판정할 자격이 있는가"를 먼저 안다.

### 2.2 재사용 — `ZERO` · `ORA-K-CVX` 는 **복제하지 않았다**

`import run_w3` 로 **`run_zero` / `run_cvx` 를 그대로 호출**한다. 스코어링 헬퍼
(`_episode_static` · `_zero_and_shape` · `_mean_shells` · `_shell_masks`)도 같은 모듈에서 온다.
따라서 정의 차이가 **원리적으로 생길 수 없다**. 증거는 두 개:
- **G-8b**: `run_w3.py` md5 = `b221ec29198ea096bb3e81041cbca18a` = W3 자신의 `FROZEN_md5.txt` 기록값.
- **G-8c**: `run_w4.SHELL_KEYS == run_w3.SHELL_KEYS` (이 파일의 사본을 믿지 않고 **모듈에서 재도출**).
- 추가로 `smoke` 는 W3 도 채점한 에피소드(`BF5_30comp_seed51…`)에서 `ZERO`·`ORA-K-CVX` 의
  **저장값 대비 `|Δ|` 를 인쇄**한다 = 살아있는 비트 앵커. (⛔이건 `gates` 가 아니라 `smoke` 다 —
  에피소드를 로드하므로 "런 0회" 지시에 걸린다.)

### 2.3 ★패치 순서가 하중을 받는다 (`L-VP`)

`varpro` **먼저**, `arms_config` 카운터 **나중**에 설치한다.
- 그러면 `arms_config.initialize_learned`(`_Phase('p1')`)가 **바깥**, `varpro._patched` 가 **안쪽**이라
  VARPRO 의 **400 Adam 스텝이 `p1` 위상에 귀속**된다.
- 반대로 설치하면 그 400스텝이 `adam_steps_other` 로 떨어지고, `arms_config.check_gates` 가
  *"unattributed Adam steps"* 로 **위반이 아닌 팔을 위반으로 찍는다**.
- 해제는 `finally` 안에서 **엄격 LIFO**(capture → track → counters → `varpro.uninstall()`).
  dry-run 으로 4개 패치 전부 원본 복귀 확인.

⚠**내가 처음 짠 자체 카운터는 버렸다.** "`_build_tracking_optimizer` 시점에 스냅샷" 방식은
`_adaptive_prune_grow` 의 **50스텝 미세조정을 phase1 에 합산**해서 `L-ITER0` 의
`adam_steps_phase1` 을 **0 이 아니라 50** 으로 만들고 **G-2 를 엉뚱한 이유로 떨어뜨린다.**
`arms_config` 가 이걸 `adam_steps_grow` 로 분리해 뒀고, 그쪽이 옳다. 정의는 하나만 존재한다.

### 2.4 재개 가능

`job_id = "arm|episode|seed"`. `results.jsonl` 에서 **`error` 없는 행의 job_id 를 모아 건너뛴다**
(에러 행은 재시도 대상). `smoke` 도 `smoke_results.jsonl` 로 동일하게 재개된다.
`run` 은 시작 전에 `gates(verbose=False)` 를 다시 돌리고 실패하면 **거부**한다(§5 판정 전순서).

---

## 3. §G 게이트 목록 — 어디서 판정되는가

| | 무엇 | 런 0회 판정부 (`gates`) | 행 판정부 (`posthoc`) | 실패 시 |
|---|---|---|---|---|
| **G-1** | 라우터 100% learned | A족 5팔 `init_method='learned'` (구성적) | 전 est 행 `route=='learned'` **및** `learned_init_calls==1` | 중단 |
| **G-1b/c** | 표본·배포 기본값 | n=64 · 8개 기본값 일치 | — | 중단 |
| **G-2** | `adam_steps_phase1` | 단일요인 Δ=`learned_init_iters` | `L-ITER0` 전건 0 · `L-SHIP` 전건 >0 (+`adam_steps_grow` 별도 인쇄) | 중단 |
| **G-2b** | 미귀속 Adam | — | `adam_steps_other>0` 행 수 | 조사 |
| **G-3** | `vp_steps` | `L-VP` 만 vp=400 | `L-VP`=`{400}`, 그 외 `{0}` | 중단 |
| **G-4** | `merge_removed` | 단일요인 Δ=`learned_merge_frac` | `L-MERGE` 중앙 >0 | **`S-1` 만** `UNDECIDED-NO-INTERVENTION` |
| **G-5** | 복제 프로브 | `L-SHIP-DUP` 설정 == `L-SHIP` 설정, 시드 일치 | **`|Δ|=0`** 짝 전건 | 중단(비결정성) |
| **G-6** | 필드계약 DCC | 팔종별 3계약 정의·서로소 | 전 행 `missing_fields()==[]` | 판정 인쇄 없이 종료 |
| **G-6b** | ★프레임 동수 | — | 에피소드별 `n_match` 가 팔 간 동일 | 중단 |
| **G-6c** | ★제로필드 동일 | — | 에피소드별 `zero_nRMSE` 분산 ≤1e-12 | 중단 |
| **G-7** | 위상 규약 항등식 | 난수 4096점, `atan2(-b,a)` 및 `varpro.ab_to_wphi` ≤1e-12 | — | 중단 |
| **G-8** | 코드 지문 1종 | 고정 파일집합·재계산 안정 | 전 행 `code_md5` 1종 | 중단 |
| **G-8b/c** | W3 재사용 무결성 | `run_w3.py` md5 · 셸 규약 재도출 | — | 중단 |
| **G-9** | 진행 감시 | **`watchdog.py` 와 계획 일치만 확인** | — | 조사 |

⛔**G-9 를 다시 만들지 않았다.** `watchdog.py` 는 이미 있고, 프로세스 테이블을 보지 않고
행수+mtime 만 본다(W3 의 자기매칭 8시간 손실 대응). `gates` 는 `watchdog.PLAN`/`N_EP`/`TOTAL` 이
러너 계획(960)과 **같은 수인지만** 검사한다 — 두 파일이 조용히 어긋나는 것을 잡는 용도.

**G-6b·G-6c 는 PREREG 에 없는 내가 추가한 게이트**다. 근거는 `results/b1_full_comparison.json`
사망 사유(프레임 불일치가 100/100 에서 검출 안 됨)와 W3 의 제로필드 동일성 점검. 둘 다
**PREREG 를 넓히지 않는다**(가설·문턱·방향 불변, 자료 무결성 검사만 추가).

---

## 4. 병렬 안전성

- **구조**: `multiprocessing.get_context('spawn').Pool(WORKERS)` + `imap_unordered(chunksize=1)`.
  `WORKERS` 환경변수, 기본 **8**(W2 가 8워커를 언급).
- **결정성이 워커 수에 의존하지 않는 이유** (전부 확인함):
  1. 잡은 **(팔, 에피소드, 시드) 독립 3중쌍**이고 공유 상태가 없다. 추정기 시드는 잡마다
     **명시 전달**(`arm_config(..., est_seed=)` 가 `seed` 와 `learned_diffusion_seed` 를 함께 고정).
  2. `code_md5` 는 **파일 내용의 순수 함수** — 워커 수·임포트 순서와 무관(고정 파일집합, S17 A9).
  3. 워커마다 `OMP/OPENBLAS/MKL_NUM_THREADS=1` — BLAS 스레드 수가 부동소수 축약 순서를 바꾸지 못한다.
  4. `spawn` 은 워커마다 **새 인터프리터** ⟹ 부모의 몽키패치가 상속되지 않는다
     (`recon/W1_path.md §5` 의 fork 누출 경고 대응). 게다가 잡 안의 모든 패치는 `finally` 에서 해제된다.
  5. 워커 수가 바꾸는 것은 **`results.jsonl` 의 줄 순서**뿐이다. 분석은 `job_id` 로 조인한다.
- **실증 근거**: 같은 학습(GPU 디퓨전) 경로를 8워커로 돌린 W3 에서 `SHIP` vs `SHIP-DUP`
  12쌍의 **max\|Δ\| = 0.0** 이었다(내가 W3 `results.jsonl` 에서 재집계). W4 의 `L-SHIP-DUP` 은
  같은 검사를 **64쌍**으로 확대한 것이다.
- ⚠**남는 위험**: 8워커가 동시에 GPU 디퓨전 제안을 친다. W3 에서 전례가 있으나 W3
  `AMENDMENT_1` 이 **일시적 CUDA OOM**을 기록했다 ⟹ 에러 행은 재개 시 자동 재시도되며,
  `watchdog.py` 가 `err` 를 별도로 센다.
- ⚠**벽시계는 경합 오염된다**(S17 NOTES: 다른 배치가 11.7배 부풀렸다). `smoke` 의 비용 추정은
  **min 기반**이고, `D-2`(팔별 벽시계)를 인용할 때 경합 여부를 함께 적어야 한다.

---

## 5. 셸 규약 — 이 러너는 **일관되게 `obs:`** 를 쓴다 (문서화 의무)

- 9셸 키: `obs:0_5 · obs:5_10 · obs:10_20 · obs:20_30 · obs:30_50 · obs:50_75 · obs:75_100 ·
  obs:100_150 · obs:150_inf`. 경계는 `wavesplat/run.py:358` 과 동일.
- ⚠**저장소는 여기서 일관되지 않다**: S13 의 `SHIP` 행은 **무접두어**, 기준선 행은 `obs:` 다.
  `obs:`(관측점까지 최단거리)와 `ship:`(배까지 거리)는 **같은 표본에서 1.8–4.1배 갈린다**.
- **본 러너는 SHIP 계열 행에서도 `obs:` 로 쓴다.** est 팔의 `frame_results['shell_rmse']` 는
  무접두어 bare 키로 오므로 `BARE_KEYS → SHELL_KEYS` 로 **명시 재매핑**한다(`run_est`).
  ⟹ **분석 코드는 접두어를 절대 벗기지 마라.** `S-3` 는 `obs:` 셸이다(PREREG §3 과 일치).

---

## 6. 다른 에이전트 모듈과의 경계

| 모듈 | 소유 | 이 러너의 의존 |
|---|---|---|
| `arms_config.py` | 다른 에이전트 | **하드 의존**(est 팔). `arm_config` · `install_counters` · `install_tracking_phase` · `install_updater_capture` · `effective_config` · `counters_snapshot` · `check_gates`. 없으면 est 팔이 에러 행이 되고 `gates` 가 FAIL 로 알린다. `L-VP` 는 `ARM_DELTAS` 에 없으므로 러너가 `cfg_from='L-SHIP'` 으로 매핑한다(설정은 동일, 요인은 설치). |
| `varpro.py` | 다른 에이전트 | **`L-VP` 한정 의존**. `install(steps=)` · `uninstall()` · `reset_stats()` · `stats()` · `ab_to_wphi`. 없으면 **L-VP 잡만** 에러 행. |
| `watchdog.py` | 다른 에이전트 | 의존 없음 — 계획 수치 일치만 검사(G-9). |
| `tmp/W3_SOLVER_2026-08-15/run_w3.py` | **읽기 전용** | `run_zero` · `run_cvx` · 스코어링 헬퍼를 import. md5 앵커 G-8b. |

## 7. 스코프 · 아직 하지 않은 것

- ⛔`smoke` 와 `run` 은 **실행하지 않았다**(지시: 추정기 런 0회). 따라서 `G-2/3/4/5/6` 의 수치부,
  W3 비트 앵커, 비용 추정은 **전부 미측정**이다. `gates` 만 실행했고 그 결과만 위에 적었다.
- 러너는 **판정을 인쇄하지 않는다.** Holm-2·IUT·med3 집계는 `analyze_w4.py`(별도)의 몫이고,
  그쪽은 `posthoc()`/`missing_fields()` 를 재사용해 **G-6 실패 시 판정을 거부**해야 한다.
- `PREREG §6` 의 med3 는 **에피소드 수준**(시드 median 먼저, 그 다음 짝) — 러너는 원자료만 쓰고
  집계하지 않는다.
