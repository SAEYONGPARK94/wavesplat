# V1 — 코드 변경 감사: 커밋 `d673892` (경로 통일, `auto_switch_thresh` 8 → 0)

2026-08-22 · 감사자 V1 · **추정 런 0회**(에피소드 미실행. 원자 캐시를 주입해 확산 제안을
건너뛴 Phase-1 초기화 프로브 + 전체 테스트 실행만 했다).
프로브 = `v1_probe.py` / `v1_probe_real.py`, 원자료 = `v1_probe_{NEW,OLD}.json` ·
`v1_probe_real_{NEW,OLD}.json`.
`OLD` = `git archive d673892^` 로 뽑은 배포 **직전** 트리를 별도 `PYTHONPATH` 로 임포트.

⚠**감사 시점 HEAD 는 `d673892` 가 아니라 `ee6f6a8`** (W15 사전등록 커밋이 그 위에 있다).
패키지 파일은 두 커밋 사이에 변하지 않았다(diff 는 `tmp/` 만).

---

## 판정 표

| # | 항목 | 판정 | 한 줄 |
|---|---|---|---|
| 1 | 단일출처가 진짜인가 | **WOUND** | **실행 경로는 진짜 단일출처**(실측). 그러나 **독스트링·주석 3곳이 아직 "default 8"** 이고, 저장소가 λ 를 위해 만들어 둔 **바로 그 드리프트 방지 시험이 확장되지 않았다** |
| 2 | 실측 게이트가 살아 있는가 | **PASS** | 배포 생성자 `make_updater(DEFAULT_CONFIG)` 로 GT-free 프레임 투입 → `_auto_real_gate=True` · 라우터 **미실행**(`_auto_n_selected=None`) · `ensemble_k=1` · 배포 직전 트리와 **비트 동일**(모델 4텐서 + 예측장) |
| 2b | 우회 경로 | **WOUND** | 신규 1건: **`n_selected==0` 이 이제 `learned` 로 간다**(구: `v2_konly`) ⟹ 3개 난수 성분 축퇴 폴백. 기존 3건(명시 `init_method` · `real_domain_gate=False` · 빈 프레임)은 사전 존재 |
| 3 | 되돌리기가 되는가 | **PASS** | `auto_switch_thresh=8` 이 `d673892^` 기본값과 **비트 동일**(경로·`final_n`·`ensemble_k`·kx/ky/w/phi·9×9 예측장 전부 일치, n_atoms 7·20 양쪽). `AUTO_SWITCH_THRESH_PRE_UNIFY == 8` ✓ |
| 3b | 되돌리기 **헬퍼** | **FAIL** | **`pre_varpro_config()` 가 갱신되지 않았다.** 자기 독스트링이 *"스위치가 하나가 아니라 **둘**이다"* 라고 경고하는데, 08-22 로 스위치는 **셋**이 됐다. `cfg.update(pre_varpro_config())` 는 임계 0 을 그대로 남긴다 ⟹ **배포 전 거동을 복원하지 못한다** |
| 4 | 부수 효과 (median-of-3 사망) | **WOUND** | **사망 확정(실측)**: 배포 기본값에서 `ensemble_k` = **1/1/1/1/1/1/1** (n_atoms 0·1·2·7·8·20·30), 배포 직전 트리는 n<8 에서 **3**. 그리고 **배포 기록 어디에도 적혀 있지 않다** |
| 5 | 테스트가 계약을 고정하는가 | **WOUND** | 공허하지는 않다(값을 못 박는 시험이 따로 있다). 그러나 **주 시험이 같은 상수에서 기대값을 유도**해 *어떤 값에서도 통과*하고, 실측 게이트 시험이 **배포 생성자를 안 쓴다**, 앙상블 사망을 고정하는 시험이 **0개** |
| 6 | 226 passed | **PASS** | `226 passed, 2 warnings in 1496.17s`, exit 0 (직접 실행) |

---

## 1. 단일출처 — `WOUND`

### 실행 경로는 진짜다 (MEASURED)

`d673892` 의 **비주석 변경 줄 전수**(`git diff -U0` 에서 주석·빈 줄 제거):

```
+  auto_switch_thresh: 0                                    # configs/estimation.yaml
+AUTO_SWITCH_THRESH_DEFAULT = 0
+AUTO_SWITCH_THRESH_PRE_UNIFY = 8
-        thresh = int(cfg.get('auto_switch_thresh', 8))
+        thresh = int(cfg.get('auto_switch_thresh', AUTO_SWITCH_THRESH_DEFAULT))
+                                                  AUTO_SWITCH_THRESH_DEFAULT)   # import
+    'auto_switch_thresh': AUTO_SWITCH_THRESH_DEFAULT,       # DEFAULT_CONFIG
+        cfg['auto_switch_thresh'] = raw['optimizer'].get(...)   # yaml 로더
+    parser.add_argument('--auto_switch_thresh', ...)            # CLI
+        config['auto_switch_thresh'] = args.auto_switch_thresh
```

⟹ **읽는 곳은 `sequential_update.py:806` 단 하나**이고 나머지는 전부 그 상수를 가리킨다.
런타임 확인:

| 검사 | 값 |
|---|---|
| `AUTO_SWITCH_THRESH_DEFAULT` / `_PRE_UNIFY` | `0` / `8` |
| `DEFAULT_CONFIG['auto_switch_thresh'] is AUTO_SWITCH_THRESH_DEFAULT` | `True` (같은 객체) |
| `load_yaml_config('configs/estimation.yaml')` | `0` |
| yaml 리터럴 | `0` |
| **키가 없는 레거시 yaml** | `0` (= `DEFAULT_CONFIG` 로 폴백, 표류 불가) |
| CLI `--auto_switch_thresh` | 존재, `config` 병합 **뒤**에 적용 ⟹ yaml 을 이긴다 |

`configs/default.yaml`·`server.yaml`, 두 `scripts/` 디렉터리 전부에 이 키는 **0회**.
패키지·`scripts/` 안에 라우팅용 리터럴 `8`/`0` 은 **실행 코드로는 남아 있지 않다**.

### 그런데 문서가 어긋났다 (MEASURED)

리터럴 `8` 이 **세 곳**에 살아 있고, 전부 "현재 기본값"이라고 읽힌다:

| 위치 | 문구 |
|---|---|
| `wavesplat/optimizer/sequential_update.py:791` | `n_selected >= auto_switch_thresh (default 8 -- the Stage-1 crowding signal, frozen …)` |
| `wavesplat/optimizer/sequential_update.py:797` | `Config knobs (all optional): auto_switch_thresh (int, default 8),` |
| `wavesplat/run.py:47` | `GT-free regime router (learned proposal n_selected, threshold 8)` |
| (+) `configs/estimation.yaml:45-46` | `init_method: auto  # 'auto' (… n_selected, threshold 8; …)` |

런타임 확인: `'default 8' in SequentialUpdater._auto_select_init.__doc__` → **True**,
`AUTO_SWITCH_THRESH_DEFAULT` → **0**.

★**이것이 왜 중대한가**: 저장소는 λ 에서 정확히 이 사고를 당한 뒤
`test_no_stale_lambda_literal_in_default_config_comment` 라는 **전용 회귀 시험**을 만들어 두었다
(*"Docstrings are what people read."*). 이번 커밋은 **같은 결함을 같은 파일에 새로 만들고
그 시험을 확장하지 않았다.** `sequential_update.py:791` 은 게다가 *"frozen from the design regimes,
not tuned on nRMSE"* 라는 **정당화 문장을 그대로 달고 있다** — 새 값 0 은 그 문장이 서술하는 물건이 아니다
(W11/W13/W14 = nRMSE 로 고른 값).

### 등록부가 확장되지 않았다 (MEASURED)

`wavesplat/tests/test_shipping_defaults.py:43-52` 의 `SHIPPING_DEFAULTS` 는 이 저장소가 선언한
**배포값 등록부**이고, `test_default_config_pins_shipping_value`(parametrize)와
`test_estimation_yaml_mirrors_default_config` 가 그것을 자동으로 훑는다.
`auto_switch_thresh` 는 **등록부에 들어가지 않았다**. 대신 신규 시험이 등록부 **밖에서** 같은 패턴을
손으로 다시 썼다 ⟹ 배포값을 적는 자리가 **하나 더 늘었다**.
⚠같은 날 배포된 `varpro_steps`·`learned_init_iters` 도 **등록부에 없다**(선행 결함, 이번 커밋 소관 아님).

---

## 2. 실측 게이트 — `PASS` / 우회 경로 `WOUND`

### 직접 검증 (MEASURED)

★시험 파일이 쓰는 `SequentialUpdater(...)` 가 아니라 **배포가 실제로 쓰는 생성자**
`make_updater(dict(DEFAULT_CONFIG, …))`(= `run.py:214`)로 GT-free 프레임(64점×3프레임, `gt_field=None`)을 넣었다.

| `real_estimator` | 생성자 | `_auto_real_gate` | `init_method` | `_auto_n_selected` | `final_n` | `ensemble_k` |
|---|---|---|---|---|---|---|
| `bank_mmse` (배포 기본) | `MedianEnsembleUpdater` | **True** | `bank_mmse` | **None** = 라우터 미실행 | 0 (bank 는 가우시안 없음) | 1 |
| `v2_konly` (탈출구) | `MedianEnsembleUpdater` | **True** | `v2_konly` | **None** | 14 | 1 |

전역 RNG 를 명시 고정(`torch.manual_seed(2026)`)하고 배포 직전 트리와 대조:
**두 팔 모두 `params` 비트 동일 · 예측장 비트 동일.**
⟹ **실측 경로는 이 커밋에 손대지 않았다.** 게이트는 산다.

⚠**부수 관찰(내 첫 프로브의 자기결함)**: 전역 RNG 를 고정하지 않은 첫 회차에서 `v2_konly` 실측 팔이
`final_n` 14 vs 15 로 갈렸다. 이것은 코드 변경이 아니라 **해석적 팔이 전역 RNG(프로세스 시작 시
비결정적 torch 시드)에 민감**하다는 기존 성질이다(그래서 median-of-k 가 존재했다). 재시드 후 소멸.

### 우회 경로 전수 (MEASURED, 코드)

게이트는 `_initialize_inner` 안 `if self.init_method == 'auto': if frames and
self.config.get('real_domain_gate', True) and not _frames_have_gt(frames):` 이다. 뚫리는 조건:

| # | 경로 | 상태 |
|---|---|---|
| a | 명시 `init_method`(`'learned'`·`'v2_konly'`·…) ⟹ `'auto'` 블록 자체를 안 탄다 | **의도된 계약**. 코드 주석 `:525-539` 이 함정으로 명시, `test_unifying_via_init_method_default_would_lose_the_real_gate` 가 고정 |
| b | `real_domain_gate=False` | 문서화된 탈출구, 시험 있음 |
| c | `_frames_have_gt` 가 **값 검사**다 — 실측 프런트엔드가 비퇴화 `gt_field` 를 붙이면 통과 | **선행 위험**, 이번 커밋 무관 (`real_data.py:310` 은 `gt_field=None` 확인) |
| d | **`frames` 가 비면** `if frames and …` 가 거짓 ⟹ 게이트를 건너뛰고 라우터로 간다 | **선행**. 단 ⚠결과가 바뀌었다 — 아래 e |
| e | ★**신규**: 임계 0 이면 `n_selected == 0` 도 `0 >= 0` = **`learned`** | **이번 커밋이 만든 변화** |

**e 의 실측**(합성, `learned_atoms` 를 빈 배열로):

| n_atoms | NEW 경로 | NEW `final_n` | OLD 경로 | OLD `final_n` |
|---|---|---|---|---|
| **0** | **learned** | 9 | v2_konly | 15 |
| 1 / 2 / 7 | learned | 6 / 7 / 9 | v2_konly | 15 / 15 / 15 |
| 8 / 20 / 30 | learned | 10 / 18 / 24 | learned | 10 / 18 / 24 |

`n=0` 일 때 `learned_init.learned_init_params:371-379` 의 **축퇴 폴백**(난수 3성분, 진폭 0.01)이
발화한다. 그 코드 주석은 스스로 *"never expected … the proposal always returns atoms"* 라고 적어 두었다
— 임계 8 시절에는 라우터가 그 경우를 **해석적으로 돌려보냈기 때문에** 참이었다.
`pool_and_select:174` 은 `en.size == 0 or en.max() <= 0` 에서 실제로 빈 집합을 반환할 수 있다.
⟹ **`MEASURED`(경로 도달 가능·거동 변화) + `ARGUED`(실제 발생 빈도는 미측정)**.
✅**체크포인트 부재 경로는 안전**하다 — `_auto_select_init:820-831` 예외 처리가 `'v2_konly'` 를
**명시 반환**하므로 임계와 무관하다.

---

## 3. 되돌리기 — `PASS` / 헬퍼 `FAIL`

### 비트 수준 확인 (MEASURED)

동일 프레임·동일 원자 캐시·`torch.manual_seed(123)`/`np.random.seed(123)` 고정.
**NEW**(HEAD, `auto_switch_thresh=8` 명시) vs **OLD**(`d673892^`, 키 없음 = 폴백 8):

| n_atoms | 경로 NEW/OLD | `final_n` | `ensemble_k` | kx·ky·w·phi | 9×9 예측장 |
|---|---|---|---|---|---|
| 7 | `v2_konly` / `v2_konly` | 14 / 14 | **3 / 3** | **비트 동일** | **비트 동일** |
| 20 | `learned` / `learned` | 18 / 18 | 1 / 1 | **비트 동일** | **비트 동일** |

`AUTO_SWITCH_THRESH_PRE_UNIFY == 8` ✓ (`test_pre_unify_threshold_still_reproducible` 가 7→`v2_konly`,
8→`learned`, 20→`learned` 로 거동까지 고정).
⟹ **되돌리기는 실제로 되고, 라우터 층에서 비트 동일이다.**

### 3b. 그러나 되돌리기 **헬퍼**가 거짓말을 한다 (MEASURED) — `FAIL`

```python
>>> pre_varpro_config()
{'varpro_steps': 0, 'learned_init_iters': 500}      # auto_switch_thresh 없음
```

`sequential_update.py:78-85` 독스트링 축자:
> *"2026-08-22 배포 **전** 거동을 그대로 복원하는 설정 조각. ★스위치가 하나가 아니라 **둘**이다.
> 한쪽만 끄면 배포된 적도 측정된 적도 없는 제3의 팔이 된다. **되돌릴 때는 반드시 이 함수를 쓰라.**"*

같은 날 세 번째 스위치가 배포됐는데 이 함수는 갱신되지 않았다. 지시대로
`cfg.update(pre_varpro_config())` 를 쓰면 **임계 0 이 남아** 배포 전이 아닌 **제4의 팔**
(VARPRO 없음 + 결합적합 500 + 항상 학습)이 된다 — 함수가 경고하는 바로 그 실패를
**함수 자신이 유발한다**. ⚠수리는 한 줄(`'auto_switch_thresh': AUTO_SWITCH_THRESH_PRE_UNIFY`)이지만,
그러면 이름(`pre_varpro_config`)이 내용과 어긋나므로 **개명이 함께 필요**하다.

---

## 4. 부수 효과 — median-of-3 앙상블 사망: `WOUND`

### 사망 확정 (MEASURED)

`MedianEnsembleUpdater.initialize` (`sequential_update.py:1636-1639`):

```python
route = self._primary.init_method
real_gate = bool(getattr(self._primary, '_auto_real_gate', False))
if self.k_requested <= 1 or route != 'v2_konly' or real_gate:
    self.ensemble_k = 1                      # ← 멤버 1개로 끝
```

임계 0 ⟹ 합성에서 `route` 는 **항상 `'learned'`** ⟹ 조건 성립 ⟹ 앙상블 미구축.
실측(배포 기본값, `make_updater`): `ensemble_k` = **1** (n_atoms 0·1·2·7·8·20·30 **7/7**).
배포 직전 트리는 n_atoms ∈ {0,1,2,7} 에서 **3**.
실측 경로는 사양상 원래 1 이었다.

⟹ ★**`analytic_ensemble_k: 3` 은 이제 배포 구성에서 아무 일도 하지 않는다.**
살아 있는 도달 경로는 둘뿐: (i) `auto_switch_thresh` 를 `max_n`(=30)보다 크게 **명시**,
(ii) 확산 체크포인트 부재 → `v2_konly` 폴백(퇴화 모드).

### 어디에도 적혀 있지 않다 (MEASURED, 열람 범위 명시)

내가 연 것: 커밋 메시지 · `run.py:57-64` · `configs/estimation.yaml:49-52` ·
`MedianEnsembleUpdater` 독스트링 · `sequential_update.py:59-75`(새 상수 주석) ·
`wavesplat/tests/**` 전수 grep · `SHIPPING_DEFAULTS`.

| 문서 | 지금 뭐라고 적혀 있나 |
|---|---|
| 커밋 메시지 `d673892` | **언급 0회** |
| `run.py:58-60` | *"3 = the pre-registered, confirmed value; 1 disables it"* — 0 이 이미 disable 했다는 말이 없다 |
| `configs/estimation.yaml:50-52` | *"median-of-k field ensemble on the ANALYTIC fallback of the 'auto' router only (1 = off)"* — 여전히 작동하는 것처럼 읽힌다 |
| `MedianEnsembleUpdater` 독스트링 | *"SCOPE — deliberately narrow, three exclusions"* 를 나열하지만 **네 번째(임계 0)** 가 없다 |
| `test_shipping_defaults.SHIPPING_DEFAULTS` | `"analytic_ensemble_k": 3` 을 *"The deployed operating point"* 로 계속 못 박는다 |
| 시험 | 앙상블 사망을 고정하는 시험 **0개** |

⟹ **배포 기록 기준으로는 미기록이 맞다.** 유일한 기록은 커밋 **이후** 같은 배치에서
`tmp/W15_DEPLOY_VERIFY_2026-08-22/AMENDMENT_1.md` · `V2_kill_unify.md` 가
**대조 팔 설계 문제로서** 다룬 것뿐이고, **배포 결과로서는 아니다**.

### ★그리고 이 사실이 커밋의 근거를 깎는다

`gates_w11.py:18` · `gates_w13.py:16-17` · `gates_w14.py:16-17` 실측:

```
'ANALYTIC': {'init_method': 'v2_konly'}      # 명시 ⟹ make_updater 가 plain SequentialUpdater
'LEARNED' : {'init_method': 'learned'}
```

`make_updater:1702-1705` 는 `init_method == 'auto'` 일 때만 앙상블을 붙인다 ⟹
**W11·W13·W14 의 `ANALYTIC` 팔은 median-of-3 없이 돌았다 = 배포 해석적 경로가 아니었다.**
(`LEARNED` 팔은 배포 학습 경로가 원래 앙상블을 안 받으므로 충실하다 — **비대칭 핸디캡**이다.)
이 기제는 내가 코드로 확인했다(`MEASURED`).
그 위에서 **V2 가 재집계한 수치**(배포 해석적이 강제 해석적보다 37/44 −14.5%, 그 44 에피소드에서
헤드라인 역전 16/44 +30.2%)는 **내가 재도출하지 않았다 = `UNVERIFIED`**.

⟹ `sequential_update.py:59-75` 의 주석 블록이 그 비교 수치를 **패키지 안에 축자로 박아 두었고**,
이제 그것이 저장소에서 가장 오래 남을 사본이 된다. **주석에 범위 단서가 없다.**

---

## 5. 테스트 — `WOUND`

### 갱신된 3개

**(a) `test_auto_router_follows_the_shipped_threshold`** — ⚠**값을 고정하지 못한다.**
```python
want = 'learned' if n_atoms >= AUTO_SWITCH_THRESH_DEFAULT else 'v2_konly'
```
기대값을 **코드가 읽는 바로 그 상수에서 유도**한다 ⟹ 상수가 0·8·999 무엇이든 통과한다.
고정하는 것은 *"라우터가 그 상수를 `>=` 로 쓴다"* 는 **기제**뿐이다.
게다가 DEFAULT=0 이므로 `else` 가지는 **도달 불가**, 시험은 한쪽 결과만 훑는다.
⟹ 공허(vacuous)는 **아니다**(아래 b·d 가 값을 못 박으므로). 그러나 **b 나 d 를 지우면
남은 시험이 어떤 임계도 조용히 승인한다.** 이름이 *"the shipped threshold"* 라 이 사실이 가려진다.

**(b) `test_unified_router_always_picks_learned_on_synthetic`** — 값 고정 ✓
(`assert AUTO_SWITCH_THRESH_DEFAULT == 0`). ⚠**이름이 측정보다 넓다**: `_auto_select_init([])` 를
**빈 프레임 리스트**로 호출한다 ⟹ `initialize()` 를 안 타고 **합성/실측 구분에 닿지 않는다.**
"on synthetic" 은 이 시험이 보인 것이 아니다. (갱신 3개 **전부** `initialize()` 를 우회한다.)

**(c) `test_pre_unify_threshold_still_reproducible`** — **가장 튼튼하다.** 기대값이 리터럴
(`7→v2_konly, 8→learned, 20→learned`)이라 상수에서 유도되지 않는다. 공허 아님 ✓

### 신규 2개

**(d) `test_router_threshold_is_single_sourced_and_unified`** — 값·되돌리기값·yaml 미러를 못 박는다 ✓
⚠미검사: **yaml 로더 경로**(`load_yaml_config`)와 **CLI 인자 존재**. 둘 다 이 커밋이 새로 깐 배선인데
시험이 없다(내가 런타임으로 확인했다: 정상).

**(e) `test_unification_did_not_disable_the_real_gate`** — 공허 아님(게이트를 없애면 `_auto_real_gate`
가 False 가 되어 실패한다) ✓ ⚠**그러나 `SequentialUpdater(dict(DEFAULT_CONFIG, …))` 로 만든다 —
배포가 쓰는 생성자는 `make_updater`(`run.py:214`)다.** 저장소 상시 교훈 *"출하 팔에서도 재라"* 미적용.
내가 §2 에서 `make_updater` 로 재서 **통과**를 확인했으므로 지금은 무해하지만, **시험이 그것을
고정하지는 않는다.**

### 시험 수 산술 (MEASURED)
구 `test_auto_router_threshold_is_8` = parametrize 3건 → 신규 함수 3개(순증 0) + `test_shipping_defaults` 신규 2개
= **224 + 2 = 226** ✓ 커밋 메시지와 일치.

### 빠진 계약
- 앙상블 사망 고정 시험 **0개** (§4).
- 독스트링 드리프트 가드 **미확장** (§1) — λ 용 가드가 바로 옆에 있는데도.

---

## 6. 테스트 실행 — `PASS`

```
$ python3 -m pytest wavesim/tests/ wavesplat/tests/ -q
226 passed, 2 warnings in 1496.17s (0:24:56)     # exit 0
```

---

## ★ 오케스트레이터가 놓친 것

> 아래는 커밋 메시지·코드 주석·시험 어디에도 없는 것들이다.

1. ★★★**`pre_varpro_config()` 를 갱신하지 않았다** (`FAIL`). 저장소가 스스로 *"되돌릴 때는 반드시 이
   함수를 쓰라"* 고 지시하는 헬퍼가 이제 **배포 전 거동을 복원하지 못한다**. 스위치가 둘에서
   **셋**으로 늘었는데 함수는 둘을 반환한다. 그 함수의 독스트링이 경고하는 *"제3의 팔"* 을
   이제 **그 함수 자신이 만든다.**
2. ★★★**독스트링 드리프트를 같은 커밋에서 새로 만들었다** — `sequential_update.py:791`·`:797`,
   `run.py:47`(+`estimation.yaml:45`)이 여전히 *"default 8" / "threshold 8"* 이다. 커밋 메시지는
   *"★임계를 리터럴이 아니라 상수에서 읽게 바꿨다 — 리터럴이 배포와 어긋나는 것이 이 저장소의
   반복 결함이다"* 라고 적었는데, **실행 리터럴만 고치고 문서 리터럴은 넷 다 남겼다.**
   `:791` 은 *"frozen from the design regimes, not tuned on nRMSE"* 라는 **정당화까지 달고 있고**,
   그 문장은 새 값 0 에 대해 **거짓**이다. λ 용 가드
   (`test_no_stale_lambda_literal_in_default_config_comment`)를 확장하면 잡힌다.
3. ★★**`SHIPPING_DEFAULTS` 등록부에 넣지 않았다** — 배포값을 적는 자리가 줄지 않고 **늘었다**.
   등록부에 넣었으면 `test_default_config_pins_shipping_value`·`test_estimation_yaml_mirrors_default_config`
   두 시험이 **자동으로** 값과 yaml 미러를 훑었을 것이고, 신규 시험 하나가 불필요했다.
   ⚠같은 날 배포한 `varpro_steps`·`learned_init_iters` 도 등록부 밖이다.
4. ★★**`n_selected == 0` 의 거동이 바뀌었다** — `0 >= 0` 이므로 이제 `learned` 로 가고
   `learned_init_params` 의 **축퇴 폴백(난수 3성분)** 을 밟는다. 임계 8 시절 이 경우는 해석적으로
   돌아갔고, 폴백 주석의 *"never expected"* 는 **그 라우팅 덕분에** 참이었다. 실측 확인:
   `n_atoms=0` → `route='learned'`, `final_n=9`(구: `v2_konly`, 15).
5. ★★**앙상블 사망을 배포 기록에 안 적었다** — 게다가 `analytic_ensemble_k: 3` 이 **네 개 문서에서
   여전히 작동 중인 것처럼 서술**된다(`run.py` 주석 · yaml 주석 · `MedianEnsembleUpdater` 독스트링의
   "three exclusions" · `SHIPPING_DEFAULTS`). **죽은 배포 기본값은 λ 3중 기재와 같은 부류의 씨앗**이다.
6. ★★★**근거가 된 비교가 해석적 팔에 불리했고, 그 수치가 패키지 주석에 축자로 박혔다** —
   W11·W13·W14 의 `ANALYTIC` 팔은 명시 `init_method='v2_konly'` 라 **median-of-3 를 못 받았다**
   (내가 코드로 확인). `LEARNED` 팔은 원래 앙상블을 안 받으므로 **핸디캡이 한쪽에만** 걸렸다.
   V2 가 커밋 **이후** 이것을 잡아 W15 를 재설계했지만, **커밋 메시지와
   `sequential_update.py:59-75` 주석은 수정되지 않았다.** 그 주석이 저장소에서 가장 오래 남을 사본이 될 텐데
   **범위 단서가 한 줄도 없다.** ⚠V2 의 재집계 수치 자체는 내가 재도출하지 않았다(`UNVERIFIED`).
7. ★**`gates_w15.py` 가 손복사 리터럴 `8` 을 다시 넣었다** —
   `'PRE': {'auto_switch_thresh': 8}`. `AUTO_SWITCH_THRESH_PRE_UNIFY` 를 임포트하면 될 것을
   **커밋이 제거했다고 선언한 바로 그 패턴**으로 되돌렸다(현재 값은 일치하므로 무해, 그러나 씨앗).
   ⚠부수: `ALL-ANALYTIC` 임계가 `gates_w15.py`·`PREREG.md` 에서 **999999**, `AMENDMENT_1.md`·
   `V2_kill_unify.md` 에서 **99999** 로 갈린다(둘 다 `max_n=30` 초과이므로 거동 동일, **문서만 불일치**).
8. ⚠**빈 프레임 경로**(`if frames and …`)는 게이트를 건너뛴다 — 선행 결함이나, 이제 그 낙하지점이
   `v2_konly` 가 아니라 `learned` 다(항목 4 와 같은 뿌리).
9. ⚠**`_frames_have_gt` 는 값 검사**다 — 실측 프런트엔드가 비퇴화 `gt_field` 를 붙이면 게이트가
   침묵한다. 선행 위험이고 이번 커밋과 무관하나, 게이트를 *"우회 불가"* 라고 부르는 코드 주석
   (`:551-555` *"it cannot be bypassed by a new real front-end"*)은 **정확히는 참이 아니다**.

## 감사 범위 (열람=주장)

**열었다**: `git show d673892` 전문 · `sequential_update.py`(상수 블록 / `_initialize_inner` /
`_auto_select_init` / `_initialize_learned` / `MedianEnsembleUpdater` / `make_updater` / `pre_varpro_config`) ·
`run.py`(`DEFAULT_CONFIG` / `load_yaml_config` / CLI / `make_updater` 호출) · `learned_init.py`
(`pool_and_select` / `learned_init_params`) · `real_data.py`(gt_field) ·
`configs/*.yaml` · `test_deploy_auto_v1_m3.py` · `test_shipping_defaults.py` ·
`gates_w11/w13/w14/w15.py` · `W15/PREREG.md` · `W15/AMENDMENT_1.md`(§1 일부) ·
`reference_citation_bans.md` 전문.
**열지 않았다**: `W11`·`W13`·`W14` 의 `results.jsonl`(⟹ 커밋이 인용한 수치를 **재도출하지 않았다**) ·
`V2/V3/V4` 감사 문서 본문 · `scripts/` 개별 러너.
**돌리지 않았다**: 에피소드 추정 런 0회. 프로브는 원자 캐시 주입 Phase-1 초기화뿐(확산 제안 미실행).
