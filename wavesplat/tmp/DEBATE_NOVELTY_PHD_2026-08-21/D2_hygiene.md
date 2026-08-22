# D2 — 코드 위생 수리 (2026-08-21)

**에이전트**: D2 (실무, 파일 수정 허용) · **추정 실험 런 0회** · **커밋 0회**
**브랜치**: `learn-boundwave-retrain` · **작업 시작 HEAD**: `7c328c1e8cb9619b2db9f402c6bed8276dafa90a`
**작업 시작 시 추적 파일 트리**: `git status --porcelain --untracked-files=no` = **빈 출력 (clean)**
(미추적 `*.h5` 데이터는 다수 있으나 이 작업과 무관하며 건드리지 않았다.)

---

## 판정 요약

| # | 항목 | 판정 | 한 줄 |
|---|---|---|---|
| 1 | `init_method` dead default 단일출처화 | **DONE (부분) + 정정 1건** | ⛔**`sequential_update.py` 쪽은 dead 가 아니다** — 지시서·CLAUDE.md 의 전제를 반증했다. 따라서 **병합하지 않고** 두 상수를 각각 명명·문서화·테스트로 고정. `run.py` yaml 폴백 `'v2'` 는 dead 확인 후 단일출처화. |
| 2 | λ=0.001 하드코딩 2곳 | **DONE** | 값 불변. `LAMBDA_L1_PRE_S9 = 1e-3` 로 상수화 + **주석이 거짓이 된 것을 정정**. |
| 3 | 생성기 파일명에 선속 미포함 | **DONE** | `run_episode(ship_speed_mps=0.0)` 신설 → 파일명 태그 + HDF5 attr. 0 일 때 기존 이름 **바이트 동일**. |
| — | `wavesim/config.py:76 ship_speed_kts` (dead config) | **NOT-DONE** | 아래 §5. 고치면 생성 동작이 바뀐다. |

**테스트: 211 passed → 217 passed** (신규 6건, 실패·스킵 0). 회귀 0.

```
 wavesplat/wavesim/data_writer.py                   | 11 +++-
 wavesplat/wavesim/pipeline.py                      | 50 ++++++++++++++++--
 wavesplat/wavesim/tests/test_pipeline.py           | 53 +++++++++++++++++++
 wavesplat/wavesplat/optimizer/sequential_update.py | 59 ++++++++++++++++++---
 wavesplat/wavesplat/run.py                         | 17 ++++--
 .../wavesplat/tests/test_shipping_defaults.py      | 60 +++++++++++++++++++++-
 6 files changed, 235 insertions(+), 15 deletions(-)
```

---

## §0 열람 범위 = 주장 범위

**연 파일 (전부 직접 열람)**
`tmp/DEBATE_NOVELTY_PHD_2026-08-21/BRIEF.md` · `wavesplat/optimizer/sequential_update.py` (해당 구간) ·
`wavesplat/run.py` · `wavesplat/tests/test_shipping_defaults.py` · `wavesplat/tests/test_pipeline.py`(=`wavesim/tests/`) ·
`configs/estimation.yaml` · `wavesim/pipeline.py` · `wavesim/data_writer.py` · `wavesim/config.py` (60–95) ·
`tmp/C_ENCOUNTER_2026-08-20/{gen_c.py, moving_sensor.py, AMENDMENT_1.md}`

**전수 grep 으로만 본 것 (열람 아님, 판정 근거로는 grep 결과만 사용)**
`tmp/**/*.py` 131개 러너의 `init_method` 등장 여부 · `scripts/*.py` 의 `SequentialUpdater(` 호출부 9개 ·
저장소 전체 `*.yaml` 의 `optimizer:` 블록 · `save_episode(` 호출부.

**안 연 것**: `MEMORY.md`·`reference_citation_bans.md` 본문(이 문서는 수치 주장을 하지 않는다) · 실험 결과 JSON · 논문/학위논문.

---

## §1 항목 1 — `init_method` 단일출처화 ★내 임무 중 가장 중요한 산출은 **정정**이다

### 1-a ⛔지시서와 CLAUDE.md 의 전제를 반증했다 (`MEASURED`)

지시서: *"⚠둘 다 실행경로 도달 불가(dead default)로 실측됐다"*.
CLAUDE.md: *"둘 다 실행경로 도달 불가(dead default)로 실측됐다 (V3 2026-08-13)"*.

**두 갈래를 따로 재확인했고, 결론이 갈렸다.**

| 대상 | 재확인 결과 | 근거 |
|---|---|---|
| `run.py` yaml 폴백 `'v2'` | ✅**dead 확인** | 저장소 전체에서 `optimizer:` 블록을 가진 yaml 은 **`configs/estimation.yaml` 단 1개**이고 그 파일은 `init_method: auto` 를 **명시**한다. `load_yaml_config` 호출부도 `run.py:608` 하나뿐. |
| `sequential_update.py` 클래스 폴백 `'v2_konly'` | ⛔**dead 가 아니다 (LIVE)** | `config` 에 `init_method` 키가 **없는 채로** `SequentialUpdater`/`make_updater` 를 만들고 **실제로 돌리는 호출부가 존재**한다. |

**LIVE 호출부 (grep 실측, `init_method` 문자열이 파일 전체에 0회인 것만 셈)**
- `wavesplat/tests/test_phase_convention.py:125` — **저장소에 배포된 테스트**. `SequentialUpdater({'depth': DEPTH})`
- `tmp/G1_waveprior_power/` 4개 (`run_frozen_forecast.py:32,36` · `run_heldout_init.py:28` · `fork_tracker_decisive.py:27` · `err_tracker.py:23,36`)
- `tmp/tracker_ampdecay_2026-07-25/` 2개 (`fix_preview.py:81` · `diag_tracker.py:114`)
- `tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py:107` — `make_updater(cfg)`, **2026-08-19 실측 데이터 경로**
- `tmp/DEBATE_blankslate_2026-08-12/r2_auto_rest70.py:32` — `SequentialUpdater({}).lambda_l1`

직접 실행 확인(추정 런 아님, 객체 생성 후 속성 조회):
```
SequentialUpdater({}).init_method = 'v2_konly'
make_updater({})                  -> SequentialUpdater  (MedianEnsembleUpdater 아님)
```

### 1-b ★그래서 "두 곳이 하나의 상수를 import" 는 **틀린 처방**이다

지시서의 수리 방향(둘 다 배포 기본값을 import)을 그대로 실행하면 **동작이 바뀐다**:

1. 위 9개 호출부가 조용히 `'auto'` 로 재라우팅된다 ⟹ 기록된 배치가 재실행 시 무효.
2. ★더 나쁜 것: `make_updater({})` 는 `config.get('init_method') == 'auto'` 가 아니므로 **평범한 `SequentialUpdater`** 를 돌려준다. 거기에 `init_method='auto'` 가 들어가면 **라우팅은 되는데 median-of-k 앙상블은 없는 제3의 거동** — 배포된 적도, 측정된 적도 없는 팔이 생긴다.

⟹ **두 상수는 중복이 아니라 서로 다른 양이다.** `LAMBDA_L1_DEFAULT` 와는 사정이 다르다(그건 진짜 같은 값의 3중 기재였다).

### 1-c 실제로 한 것

`wavesplat/optimizer/sequential_update.py` — `LAMBDA_L1_DEFAULT` 옆에 두 상수를 **명명하고 왜 다른지를 코드에 적었다**:
- `INIT_METHOD_DEFAULT = 'auto'` — 배포 기본값. **단일출처**.
- `INIT_METHOD_CLASS_FALLBACK = 'v2_konly'` — 클래스 폴백. **의도적으로 다름**. 위 LIVE 호출부 목록과 "병합하면 안 되는 이유"를 주석에 기재.
- `self.init_method = config.get('init_method', INIT_METHOD_CLASS_FALLBACK)` (값 불변)

`wavesplat/run.py`:
- `INIT_METHOD_DEFAULT` 를 import → `DEFAULT_CONFIG['init_method']` 가 그것을 참조(`lambda_l1` 패턴 동일).
- yaml 폴백 `'v2'` → `DEFAULT_CONFIG['init_method']`.
  ★**`'v2'` 는 배포 기본값도 클래스 폴백도 아닌 제3의 값이었다.** 그리고 `run_estimation` 은 `DEFAULT_CONFIG.copy()` 후 `update(config)` 이므로 **yaml 이 이긴다** ⟹ `optimizer:` 는 있는데 `init_method` 가 없는 yaml 을 `--config` 로 주면 **배포 라우터가 조용히 `v2` 로 강등됐을 것**이다. 저장소 안에서는 도달 불가라 **과거 오염 0건**이지만, 잠재 결함이었다.

### 1-d 동작 불변 확인 (`MEASURED`)

| 양 | 수리 전 | 수리 후 |
|---|---|---|
| `DEFAULT_CONFIG['init_method']` | `'auto'` | `'auto'` |
| `load_yaml_config('configs/estimation.yaml')['init_method']` | `'auto'` | `'auto'` |
| `SequentialUpdater({}).init_method` | `'v2_konly'` | `'v2_konly'` |
| `SequentialUpdater({}).lambda_l1` | `0.0001` | `0.0001` |
| `make_updater(DEFAULT_CONFIG)` | `MedianEnsembleUpdater` | `MedianEnsembleUpdater` |

---

## §2 항목 2 — λ=0.001 하드코딩 2곳

**위치 (수리 전 행번호)**: `sequential_update.py:427` (`S8_L1_TRACK_ONLY` 게이트) · `:968` (`S8_L1_PHASE1_ONLY` 게이트).

★**단순한 "씨앗"이 아니라 이미 거짓이 된 주석이었다** (`MEASURED`):
- `:427` 주석 — *"PHASE 1 is pinned to the **SHIPPED default**"*
- `:963` 주석 — *"pinned to the SHIPPED default (**configs/estimation.yaml:12, lambda_l1=0.001**)"*

2026-08-10 S8 실행 시점엔 참이었고, **2026-08-11 S9 배포로 그 다음 날 거짓이 됐다**. 아무도 그 줄을 편집하지 않았는데 리터럴의 **뜻만 조용히 바뀌었다** — CLAUDE.md 가 이미 지적한 그대로다. 현재 `configs/estimation.yaml` 의 그 키는 `0.0001` 이므로 **주석이 가리키는 파일·행·값이 셋 다 틀렸다**.

**한 것**: `LAMBDA_L1_PRE_S9 = 1e-3` 상수 신설(값 동일, `1e-3 == 0.001`) → 두 게이트가 그것을 쓴다. 주석은 *"S8 재현을 위한 **구 기본값** 이며 ⛔`LAMBDA_L1_DEFAULT` 로 올리지 말 것"* 으로 정정. **값 불변 ⟹ S8 배치 재현성 무손상.**

---

## §3 항목 3 — 생성기 파일명에 선속

**근본 원인** (`MEASURED`): `wavesim/pipeline.py` 의 `fname` 조립이 `dtag`·`ptag`·`mtag`·`rtag`·`ftag`·`stag` 로 **모든 요인을 이름에 넣는데 선속만 없었다**. 그럴 수밖에 없던 이유는 **`run_episode()` 에 선속 인자가 아예 없었기 때문**이다(선속은 `tmp/C_ENCOUNTER_2026-08-20/moving_sensor.py` 가 레이캐스트를 몽키패치해 바깥에서 넣는다).
C 배치 `AMENDMENT_1.md` A-3: *"`U=0, dir 0` 과 `U=11.3, dir 0` 이 같은 파일명을 쓰고 뒤가 앞을 덮어썼다(스모크에서 실제로 발생, 파일 3개 중 2개만 남았다)"* → 우회는 **셀당 별도 디렉터리 + `G-0b` 파일 수 검사**였다.

**한 것**
- `wavesim/pipeline.py` 에 모듈 수준 `speed_tag(ship_speed_mps) -> str`: `0.0 → ''`, `11.3 → '_u11p3'`, `-5.0 → '_um5'`. `.`→`p`, `-`→`m` 로 **셸 안전한 단일 토큰** 유지.
- `run_episode(..., ship_speed_mps: float = 0.0)` 신설 → `fname` 에 `{dtag}{utag}{ptag}…` 로 삽입.
- `save_episode(..., ship_speed_mps=None)` → `/config` attr `ship_speed_mps` 기록. **0 일 때는 `None` 을 넘겨 attr 자체를 생략** — 기존 파일들과 attr 집합까지 동일하게 유지(`data_writer` 의 *"None means caller did not say"* 규약 준수).

**하위호환 (`MEASURED`, 테스트로 고정)**: 기본값 `0.0` 에서 파일명은 `episode_BF4_5comp_seed0_uniform.h5` 로 **수리 전과 바이트 동일**하고 attr 도 추가되지 않는다 ⟹ **디스크의 에피소드 110+개 중 이름이 바뀌거나 무효가 되는 것 0개.**

⚠★**정직한 한계 — 이건 라벨이지 노브가 아니다.** `run_episode` 는 여전히 센서를 움직이지 않는다(원점 고정). `ship_speed_mps` 는 호출자가 "내가 밖에서 이 속도로 움직였다"고 **선언**하는 값이고, 검증되지 않는다. 이 위험을 독스트링에 ⛔로 명시했고 `wavesim/config.py:76 ship_speed_kts`(아무도 안 읽는 진짜 dead config)와 대조시켰다. **거짓 노브가 될 소지가 있으므로 다음 사용자가 반드시 읽어야 한다.**

**회귀 테스트 2건 추가** (`wavesim/tests/test_pipeline.py`)
- `test_speed_tag_is_empty_only_when_stationary` — 런 0회 순수 단위. 0 에서 빈 문자열, 서로 다른 5개 속도가 **충돌 0**, 태그에 `.`/`-`/공백 없음.
- `test_pipeline_filename_and_attr_carry_ship_speed` — 같은 `(state,N,seed,dir)` 로 정지·이동 2회 생성해 **파일명이 달라지고 둘 다 남는지**(=C 배치 결함의 직접 재현 방지) + 정지 쪽 이름·attr 이 역사적 형태 그대로인지.

---

## §4 항목 1 관련 회귀 테스트 4건 (`wavesplat/tests/test_shipping_defaults.py`)

- `test_init_method_has_exactly_one_definition` — `DEFAULT_CONFIG['init_method'] is INIT_METHOD_DEFAULT`
- `test_yaml_loader_fallback_tracks_default_config` — `optimizer:` 만 있고 `init_method` 가 없는 임시 yaml 을 실제로 로드해 폴백이 배포값을 따르는지 검사(**과거 `'v2'` 였다면 실패했을 테스트**)
- `test_class_fallback_is_the_legacy_route_not_the_shipping_default` — `SequentialUpdater({}) == 'v2_konly'` 를 **고정**. 미래에 누가 "중복이니 병합"하려 하면 여기서 걸린다.
- `test_s8_probe_pin_is_the_pre_s9_value_not_the_current_default` — `LAMBDA_L1_PRE_S9 == 1e-3 != LAMBDA_L1_DEFAULT`. **핀을 "낡은 값"으로 오해해 "고치는" 것을 막는 테스트.**

---

## §5 NOT-DONE — 손대지 않은 것과 이유

| 대상 | 이유 |
|---|---|
| **`wavesim/config.py:76 ScenarioCfg.ship_speed_kts`** (진짜 dead config, 어디서도 안 읽힘) | 삭제하면 `WaveSimConfig` 를 직렬화/재구성하는 기록·러너가 깨질 수 있고, `wavesim` 을 실제로 선속의 주인으로 만드는 것은 **생성 동작 변경**이다. 지시서의 "저위험 항목만" 범위 밖. **권고**: 항목 3 의 `ship_speed_mps` 를 진짜 노브로 승격시키면서 이 필드를 은퇴시키는 것이 옳은 종착점이나, 별도 사전등록이 필요하다. |
| **`init_method` 두 상수 병합** | §1-b — 병합이 곧 동작 변경. **안 고치는 것이 정답인 경우**로 판단. |
| **λ 값 변경** | S8 재현성. 뜻만 명시. |
| `CLAUDE.md` / `MEMORY.md` 의 "둘 다 dead default" 서술 | 문서 수정은 내 임무 범위 밖 — **§6 에 정정으로 보고**하고 반영은 사용자·D1 에 맡긴다. |

---

## §6 ★내 정정·반증 (별도 절)

1. ⛔**"`sequential_update.py` 의 `init_method` 폴백은 dead default" 는 거짓이다.**
   지시서와 `CLAUDE.md`(*"둘 다 실행경로 도달 불가 … V3 2026-08-13"*)가 함께 틀렸다. 배포 CLI(`python3 -m wavesplat.run`) 경로에서만 참이고, 그 경로에서는 `DEFAULT_CONFIG` 가 항상 키를 주입하므로 자명하게 참이다. **범위를 붙이지 않은 문장이 측정보다 넓었다** — 이 저장소가 반복해서 물린 바로 그 부류.
   ⟹ 정확한 문장 = **"`init_method` 폴백은 배포 CLI 경로에서만 도달 불가하고, 직접 생성자를 쓰는 연구 러너 9곳에서는 살아 있다."**
2. ★**"단일출처화" 처방 자체가 여기서는 틀렸다.** 지시서가 준 수리 방향을 그대로 따랐다면 **9개 기록 배치를 재라우팅하고, 배포된 적 없는 제3의 거동을 만들었을 것**이다. CLAUDE.md 의 *"처방도 검증하라"* 가 그대로 적용됐다.
3. ⚠**λ 하드코딩은 "씨앗"이 아니라 이미 발화한 결함이었다.** 값이 아니라 **주석이 거짓**이었고, 그 주석이 가리키는 `configs/estimation.yaml:12` 의 값(0.001)도 지금은 존재하지 않는다. 지시서는 "배포 무해하나 씨앗"이라 했는데, 배포 무해는 맞지만 **문서로서는 이미 틀린 상태**였다.
4. ⚠**항목 3 의 수리는 완전하지 않다.** 파일명 충돌은 닫았으나 `ship_speed_mps` 는 검증되지 않는 선언값이다. "선속을 파일명에 넣었다"는 참이고, **"생성기가 선속을 안다"는 거짓**이다. 후자로 읽히지 않도록 독스트링에 ⛔를 박았다.

---

## §7 재현

```bash
cd /home/vexit/wavesplat/wavesplat
python3 -m pytest wavesim/tests/ wavesplat/tests/ -q     # 217 passed
git -C /home/vexit/wavesplat diff --stat                  # 6 files, +235 -15
```
변경은 **작업트리에만** 있다. 커밋·스테이징 0회.
