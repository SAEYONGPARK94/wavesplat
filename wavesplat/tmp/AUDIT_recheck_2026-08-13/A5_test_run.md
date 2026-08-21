# A5 — 저장소가 실제로 돌아가는가 (실행 감사)

- 날짜: 2026-08-13
- HEAD: `5f0846cc435bc62ff31af6beff72a6a629bc50a4`
- 감사자: A5 (실행 전용). **코드 수정 0건.** 이 파일 1개만 생성.
- **열람·실행 범위**: pytest 4개 테스트 디렉터리 · `Key Commands` 5개 스크립트 + `wavesplat.run`
  진입점 · 환경 3종 · `tmp/S13_oos_2026-08-12/` (run_s13.py 읽기 + gates 실행 + jsonl 재집계).
  **범위 밖**: 413개 스크립트 중 5개 외 전부, S13 이외 배치, 실제 추정 런(무거움).

---

## 0. 한 줄 결론

**패키지 코드는 건강하다 (196/196 통과, 실패 0, 스킵 0).**
깨진 것은 **CLAUDE.md 의 문서화된 명령들**이다 — `Key Commands` 5줄 중 **6줄 전부가 문서대로는
실행 불가**(경로 1건 + CLI 문법 2건). 코드 결함이 아니라 **문서 결함**이다.

---

## 1. 테스트 스위트 (MEASURED)

### 1.1 실제 경로 확인

CLAUDE.md 는 `wavesim/tests/ wavesplat/tests/` 라고 적었는데, 저장소 루트가
`/home/vexit/wavesplat` 이고 패키지가 그 안의 `wavesplat/` 이므로 **루트에서는 그 경로가 없다**.
실제 경로:

```
./wavesplat/wavesim/tests
./wavesplat/wavesplat/tests
./wavesplat/dcspecnet/tests      <- 문서화된 명령이 포함하지 않음
./wavesplat/hydrofoil/tests      <- 문서화된 명령이 포함하지 않음
```

### 1.2 실행 결과

```
$ cd /home/vexit/wavesplat
$ python3 -m pytest wavesplat/wavesim/tests/ wavesplat/wavesplat/tests/ -q -rs --durations=15

........................................................................ [ 36%]
........................................................................ [ 73%]
....................................................                     [100%]
196 passed in 61.61s (0:01:01)
EXIT=0
```

**`196 passed, 0 failed, 0 skipped, 0 errors in 61.61s`**

- **CLAUDE.md 의 "196 tests" 주장은 실측과 정확히 일치**(수집 단계에서도
  `196 tests collected in 1.70s`). 이 수치는 살아 있다.
- `-rs` 로 스킵 사유를 강제 출력했으나 **스킵 절 자체가 출력되지 않았다 = 스킵 0건**.
  조용히 사라진 테스트는 없다.
- 실패 0건이므로 원인 진단 대상 없음.

### 1.3 "스킵 0" 은 이 기계에서만 참이다 (조건부 스킵 3파일)

스킵이 0인 이유는 **이 기계가 GPU 와 데이터를 둘 다 갖췄기 때문**이다. 조건이 빠지면
아래가 조용히 사라진다:

| 파일 | 게이트 | 조건 불충족 시 |
|---|---|---|
| `wavesim/tests/test_lidar_gpu.py:19` | `pytest.mark.skipif(not torch.cuda.is_available())` | **3 tests** 전부 스킵 |
| `wavesplat/tests/test_integration.py:22,72,108` | `pytest.skip("No HDF5 episodes available")` 외 2 | 데이터 없으면 스킵 |
| `wavesplat/tests/test_deploy_auto_v1_m3.py:38,47` | `pytest.skip("No HDF5 episodes available")` 외 1 | 데이터 없으면 스킵 |

`test_integration.py` + `test_deploy_auto_v1_m3.py` = **22 tests 수집**. 즉 데이터 없는 기계에서는
최대 22개, GPU 없는 기계에서는 3개가 "통과"로 보이지 않고 **사라진다**.
⟹ **CI 나 다른 기계에서 "196 passed" 를 봤다면 그건 같은 196 이 아닐 수 있다.**
데이터(`data/*.h5` 110개)와 GPU 가 실측 확인된 이 기계에서만 196 = 196 이다.

### 1.4 문서화된 명령이 안 도는 41 tests

```
$ python3 -m pytest wavesplat/dcspecnet/tests/ wavesplat/hydrofoil/tests/ -q -rs
41 passed in 3.48s
```

**전량 통과.** 다만 CLAUDE.md `Key Commands` 의 pytest 줄은 이 둘을 포함하지 않으므로,
문서를 따르면 **41개가 상시 미실행**이다. 저장소 전체 실측 = **237 tests, 237 passed**.

---

## 2. 진입점 스모크 테스트

### 2.1 스크립트 존재 + 문법 (MEASURED)

`Key Commands` 는 `scripts/...` 라고 적었다. **저장소 루트의 `scripts/` 에는 5개 전부 없다**
(루트 `scripts/` 는 `generate_ppt.py`, `restructure_pptx_20min.py` 2개뿐 — 다른 디렉터리다).
실제 위치는 `wavesplat/scripts/` (413개 `.py` — CLAUDE.md 의 "413 experiment/paper scripts" 와 일치).

`cd /home/vexit/wavesplat/wavesplat` 기준:

| 스크립트 | 존재 | `python3 -m py_compile` |
|---|---|---|
| `scripts/generate_paper_tables.py` | EXISTS | **COMPILE-OK** |
| `scripts/generate_paper_figures.py` | EXISTS | **COMPILE-OK** |
| `scripts/run_p1_comparison.py` | EXISTS | **COMPILE-OK** |
| `scripts/run_gpr_comparison.py` | EXISTS | **COMPILE-OK** |
| `scripts/run_raydrop_comparison.py` | EXISTS | **COMPILE-OK** |

**존재하지 않거나 컴파일 실패하는 것: 0건.** (단 cwd 가 저장소 루트면 5개 전부 `MISSING`.)

### 2.2 패키지 임포트 (MEASURED)

```
$ cd /home/vexit/wavesplat/wavesplat && python3 -c "import wavesplat.run"
IMPORT-OK wavesplat.run
$ python3 -m wavesplat.run --help
usage: run.py [-h] --episode EPISODE [--config CONFIG] ... [--adaptive_n {true,false}] ...
```

임포트·CLI 파서 모두 살아 있다.

### 2.3 ⛔ 문서화된 실행 명령이 **실패한다** (MEASURED)

CLAUDE.md `Key Commands`:
```bash
python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n
```

이 명령은 **두 개의 독립적인 오류**로 실패한다:

```
$ python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n
run.py: error: argument --adaptive_n: expected one argument
EXIT=2

$ python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n true
run.py: error: the following arguments are required: --episode
EXIT=2
```

1. `--adaptive_n` 은 `store_true` 가 아니라 `choices={true,false}` 로 **값을 요구**한다.
2. `--episode` 는 **required 플래그**이고, 위치인자 경로는 받지 않는다.

**입력 파일 자체는 존재한다** (`data/episode_BF4_5comp_seed0_uniform.h5`, 11,640,563 bytes)
⟹ 데이터 문제가 아니라 **문서의 CLI 문법이 코드와 어긋난 것**.
올바른 형태는 `python3 -m wavesplat.run --episode data/... --adaptive_n true`.

---

## 3. 환경 사실 (MEASURED)

```
Python   3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]
numpy    2.5.0
torch    2.12.1+cu126     (torch.version.cuda = 12.6)
scipy    1.18.0
cuda_available = True
GPU      NVIDIA GeForce RTX 4080 SUPER
```

메모리의 서버 사양(WSL2 + RTX 4080 Super)과 일치. GPU 경로가 실제로 켜져 있으므로
`test_lidar_gpu.py` 3건이 스킵 없이 돌았다.

---

## 4. S13 재현 가능성 (읽기 + 게이트만, 배치 미실행)

대상: `wavesplat/tmp/S13_oos_2026-08-12/run_s13.py` (230줄).

### 4.1 판정: **재현 조건 충족 — 단 재실행 전 `results.jsonl` 을 치워야 한다**

| 조건 | 판정 | 근거 |
|---|---|---|
| 입력 데이터 존재 | **충족** | `data_s9_sparse_2026-08-11` 64 h5 · `data_s8_crowded_2026-08-10` 32 · `data_s9_crowded_2026-08-11` 32 = **sparse 64 + crowded 64** |
| 하드코딩 절대경로 | **없음** | `run_s13.py:24-27` — `HERE`/`ROOT` 를 `__file__` 에서 유도 후 `os.chdir(ROOT)`. 이식 가능 |
| 외부 의존 파일 | **충족** | G-2 가 `tmp/mixed_retrain_2026-07-26/gen_mixed.py` 를 임포트 — 존재(25,318 bytes) |
| 시드 고정 (LETKF) | **충족** | `run_s13.py:130` `np.random.seed(42)` |
| 시드 고정 (WS) | **충족** | `run_ws` 는 직접 시드를 안 걸지만 `DEFAULT_CONFIG['seed'] == 42` 이고 `run.py:144-147` 이 `run_estimation` 진입 시 `set_seed()` 호출 ⟹ `run.py:74-78` 에서 `torch.manual_seed` + `np.random.seed` + `torch.cuda.manual_seed_all` |
| 코드 버전 일치 | **충족(강)** | 704행 **전부** `git_head=5f0846cc` = **현재 HEAD 와 동일**. 커밋 이동 없음 |
| 사전등록 게이트 | **오늘 PASS** | 아래 |

### 4.2 게이트 재실행 (오늘, MEASURED)

```
$ python3 tmp/S13_oos_2026-08-12/run_s13.py gates
G-3 sample: sparse=64 crowded=64 suffixes={'uniform_ra_rf_ds'} dup=False   G-3 PASS
G-2 leakage: donor seeds=[0,2,3,4] eval seeds=32-63  seed overlap=[]       G-2 PASS
G-1 effective-config: HEAD=5f0846cc lambda_l1=0.0001 init_method=auto      G-1 PASS
GATES PASS -- cleared to run
EXIT=0
```

세 게이트 전부 오늘도 통과. 누출 없음(도너 seed 0,2,3,4 vs 평가 seed 32–63).

### 4.3 결과 파일 무결성 (MEASURED)

```
rows 704   errors 0   unique (episode,arm) pairs 704   dups 0
per-arm: SHIP 128, LETKF 128, LETKF_PLUS 128, SHIP_NOCONF 128, SHIP_NODISP 128, SHIP_L1E3 64
all rows have avg_nRMSE: True
```

MEMORY.md 의 "704런, 에러 0" 과 **정확히 일치**.

### 4.4 헤드라인 독립 재도출 (MEASURED)

`analyze_s13.py` 를 **스크래치패드로 복사해** 실행(저장소의 `verdict_primary.json` 을 덮어쓰지
않기 위해). 원자료 `results.jsonl` 에서 재집계한 값:

```
Holm-2 (PRIMARY 2개)
  crowded  56/64  p=2.781e-10 -> p_adj=2.781e-10   ZERO필요조건=통과   ⟹ PASS
  sparse   63/64  p=3.524e-18 -> p_adj=7.047e-18   ZERO필요조건=통과   ⟹ PASS
  SHIP < ZERO: 64/64  (문턱 41/64)  단측 p=5.421e-20
[S-2e] 사전노출 24 제외:  crowded 44/52 (p=2.02e-07) · sparse 51/52 (p=1.177e-14)
[셀별] ncomp=15 29/32 med(rel)=-41.0% · ncomp=30 27/32 med(rel)=-19.8%
[라우터] crowded learned 61 / v2_konly 3 · sparse learned 51 / v2_konly 13
```

**MEMORY.md RESUME 의 수치(56/64 · 63/64 · 64/64 · 41/64 문턱 · 44/52 · 51/52 · 29/32 · 27/32)와
전건 일치.** S13 은 원자료에서 재도출된다.

### 4.5 ⚠ 재실행 시 주의 2건

1. **`main_run` 은 재개형(resumable)이다** (`run_s13.py:188-199`). `results.jsonl` 이 제자리에
   있으면 704개가 전부 `done` 으로 잡혀 **`remaining=0` 으로 아무것도 안 돈다**. 진짜 재현을
   하려면 `results.jsonl` 을 먼저 옮겨야 한다. 그냥 돌리면 "재현 성공"처럼 보이는 무런(no-op).
2. **GPU 비트 재현은 보장되지 않는다.** 시드는 걸리지만
   `torch.use_deterministic_algorithms` / `cudnn.deterministic` 은 어디에도 설정돼 있지 않다
   (`run.py`·`optimizer/*`·`model/*` grep 결과 `manual_seed` 계열만 존재). 부호·순위 판정은
   견고하겠지만 **소수점 하위 자리의 드리프트는 가능**하다.

### 4.6 `ship_config()` 의 구조적 위험 (ARGUED)

`run_s13.py:53-55` 의 `ship_config()` 는 `DEFAULT_CONFIG` 를 **런타임에** 읽는다. 따라서
기본값이 미래에 바뀌면 `SHIP` 팔의 의미가 조용히 바뀐다. **지금은 안전**하다 — 704행이 전부
현재 HEAD 와 같은 커밋에서 나왔고, 행마다 `effective_config` 를 통째로 기록하기 때문이다
(`run_s13.py:181-183`). 이건 결함이 아니라 **완화가 이미 걸린 설계**로 읽는 게 맞다.
(MEMORY.md 가 기록한 S7·S8 의 "기준선이 처치값을 받는" 함정을 S13 은 명시 기록으로 피했다.)

---

## 5. 부수 확인 (문서 주장 대조)

| CLAUDE.md 주장 | 실측 | 판정 |
|---|---|---|
| "196 tests" | 196 collected / 196 passed | **일치** |
| "413 experiment/paper scripts" | `wavesplat/scripts/*.py` = 413 | **일치** |
| "HDF5 episodes (110)" | `data/*.h5` = 110 | **일치** |
| "real_bergamasco 실측 13 레코드" | `data/real_bergamasco/**/*.nc` = 13 (6 사이트) | **일치** |
| `lambda_l1: 1e-4` 배포 | `configs/estimation.yaml:12` = 0.0001 · `sequential_update.py:40` `LAMBDA_L1_DEFAULT=1e-4` · `run.py:30` 임포트 · `SequentialUpdater({}).lambda_l1 == 0.0001` | **일치** |
| `init_method: auto` | `DEFAULT_CONFIG['init_method'] == 'auto'` | **일치** |
| "세 곳에 중복 기재" (lambda_l1) | `run.py:30` 은 **중복이 아니라 임포트**(`LAMBDA_L1_DEFAULT`). 진짜 중복은 yaml 1곳 | **문서가 낡음(경미)** |

---

## 6. 발견 목록

### [MAJOR-1] `Key Commands` 의 추정 실행 명령이 실패한다
- 문서 주장: `python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n`
- 실측: `EXIT=2`, 오류 2건 (`--adaptive_n: expected one argument`; `required: --episode`)
- 증거: §2.3. 입력 h5 는 존재하므로 **문서-코드 불일치**.

### [MAJOR-2] `Key Commands` 의 모든 경로가 잘못된 cwd 기준
- 문서 주장: `scripts/...`, `wavesim/tests/ wavesplat/tests/` (저장소 루트 기준으로 읽힘)
- 실측: 루트에서 5개 스크립트 전부 `MISSING`, 테스트 경로 부존재. `cd wavesplat/` 후에만 동작
- 증거: §1.1, §2.1. CLAUDE.md 자신의 Repository Layout 은 `wavesplat/` 를 "Root package
  directory" 로 그려놨는데 명령줄은 그 안에 있다고 전제한다 — 문서 내부 불일치.

### [MINOR-1] 문서화된 pytest 명령이 41 tests 를 상시 누락
- 실측: `dcspecnet/tests` + `hydrofoil/tests` = 41 passed, 문서 명령에 미포함. 전체 237.
- 증거: §1.4.

### [MINOR-2] "196 passed" 는 기계 의존 — 최대 25건이 조용히 스킵될 수 있다
- 실측: GPU 게이트 3건(`test_lidar_gpu.py:19`), 데이터 게이트 파일 22건 수집.
  이 기계는 둘 다 충족 ⟹ 스킵 0. 다른 기계에서는 같은 "196" 이 아닐 수 있다.
- 증거: §1.3.

### [MINOR-3] S13 재실행은 `results.jsonl` 을 치우지 않으면 무런(no-op)
- 실측: 704/704 done ⟹ `remaining=0`. 증거: §4.5-1.

### [MINOR-4] S13 은 GPU 결정성 플래그가 없다
- 실측: `use_deterministic_algorithms`/`cudnn.deterministic` grep 0건. 증거: §4.5-2.

### [MINOR-5] CLAUDE.md 의 lambda_l1 "세 곳 중복" 서술이 낡음
- 실측: `run.py:30` 은 임포트(단일 출처). 증거: §5.

**BLOCKER: 없음.** 패키지 코드·테스트·S13 원자료는 전부 건강하다.

---

## 7. 이 감사가 하지 않은 것 (주장 범위 제한)

- 실제 추정 런을 **한 번도 돌리지 않았다** (S13 배치·`run_p1_comparison.py` 등 무거운 것 전부).
  따라서 "스크립트가 끝까지 돈다" 는 **주장하지 않는다** — 확인한 것은 존재·문법·임포트까지다.
- `wavesplat/scripts/` 413개 중 **5개만** 검사했다.
- S13 외 배치(`S11`, `BLANKSLATE`, `B1`, …)의 재현성은 **미검사**.
- `verdict_primary.json` 재도출은 원자료 `results.jsonl` 재집계이지 **런 재현이 아니다**.
