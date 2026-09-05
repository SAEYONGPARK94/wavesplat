# `T18-FACTORIAL` 개정문 1 — **결과 0행 · 추정기 런 0회 시점의 사전 개정**

> **동결 시점: 결과 0행 · 스모크 0행 · 추정기 런 0회.** 이 문서는 `PREREG.md`(동결 `601ef1b`)를
> **구현하면서** 발견된 것만 담는다. ⛔**판정 문턱·라벨·확률·집계·방향은 한 글자도 바뀌지 않는다.**
> ⛔사전등록의 어떤 수도 이 개정으로 갱신되지 않았다(`gates_t18.verify_registered()` 가 전건 재생산).

---

## §1 ★★`G-MAXN` — **역할이 둘로 쪼개졌다** (사전등록보다 **넓게** 구현한다)

**사실**: 2026-09-05 커밋 `a6eb5ba` 가 `max_n` **한 키가 하던 두 역할**에 이름을 하나씩 줬다.
사전등록이 동결된 `601ef1b` 는 그 **바로 다음** 커밋이고, `PREREG §5.4` 는 분리가 아직
일어나지 않은 상태를 서술한다.

| | 종전 | 현재 |
|---|---|---|
| ① 제안 풀링 상한 | `max_n` | `learned_init.POOL_CAP_DEFAULT` / config `pool_cap` (`resolve_pool_cap`) |
| ② 성장 천장 | `max_n` | `sequential_update.GROWTH_CEILING_DEFAULT` / config `growth_ceiling` (`resolve_growth_ceiling`) |

⛔**배포 거동은 비트 불변**이다 — 둘 다 30 이고 레거시 `max_n` 은 지금도 **둘 다** 설정한다
(`resolve_cap` 우선순위). `SequentialUpdater.max_n` 은 `growth_ceiling` 의 **별칭 property** 로 남았다.

### 1.1 개정 내용
사전등록 §6.1 의 `G-MAXN` 은 *"다섯 팔의 `cfg['max_n']` 되읽기와 `SequentialUpdater.max_n`
객체 되읽기"* 였다. 그 판은 이제 **역할 ② 하나만** 본다 — ①이 다른 이름 뒤로 갔기 때문이다.
⟹ 게이트를 **셋을 되읽는 것**으로 넓힌다(⛔완화가 아니라 확장이다):

```
pool_cap_readback      = learned_init.resolve_pool_cap(upd.config)
growth_ceiling_readback= sequential_update.resolve_growth_ceiling(upd.config)
max_n_alias            = upd.max_n                       (별칭 property 되읽기)
통과조건 = 셋 다 팔 간 동일 ∧ 배포 핀 30 과 일치 ∧ alias == growth_ceiling
```
⚠**왜 확장이 필요한가**: 분리 이후에는 `pool_cap=10, growth_ceiling=30` 같은 **비대칭 구성이
가능하다**. 사전등록 판의 게이트는 그 구성을 **통과시킨다**(`upd.max_n` 은 30 이므로).
그런 팔이 섞이면 이 배치는 사전등록이 §5 에서 일부러 배제한 **성분 수 인자와 교락**된다.

### 1.2 ⛔`PREREG §5.4` 의 **전제 한 줄이 낡았다** — 그러나 **결정은 바뀌지 않는다**
§5.4 는 *"단일요인으로 만들려면 상수를 먼저 둘로 쪼개야 하고, 그것은 `wavesplat/**` 수정이다
⟹ 배포된 적도 측정된 적도 없는 제3의 팔"* 이라 적었다. **그 쪼개기는 이미 일어났다**(`a6eb5ba`,
회귀 시험 `wavesplat/tests/test_component_caps.py`).
⟹ ⛔**그래도 이 배치는 제3인자를 넣지 않는다.** 사유가 §5.4 하나가 아니기 때문이다:
- **§5.2-1**(모의 코퍼스가 다르다 — 캡 상향은 자국 84% 런에서 구조적 no-op)과
- **§5.2-2**(`M4` 가 이미 **반대 방향**으로 측정했다: `CAPACITY-HURTS`)와
- **§5.2-3**(검정력·비용: q80 0.7109 → 0.7500, 런 +80%)는 **전부 그대로 성립**한다.

⟹ ★**바뀐 것은 「불가능하다」가 「가능하지만 여전히 하지 않는다」로 내려간 것**이고,
그 사유를 여기 적어 둔다. ⛔결과를 보고 이 문단을 다시 쓰지 마라.

---

## §2 `G-GROW` — ⛔호출이 아니라 **분기**를 센다 (구현 방식을 못박는다)

사전등록 §5.4 · §6.2 가 코드 주석을 축자로 인용해 요구한 것이다:
> *"A per-call counter on this function does NOT detect that — the prune half still runs. **Count the BRANCH.**"*

**구현**(`run_t18.py`): `SequentialUpdater.growth_ceiling` 을 **읽기 관찰 property** 로 바꾼다
(값은 그대로 돌려주고 읽은 시점의 `(n_gaussians, ceiling)` 만 기록). 그 속성을 읽는 지점은
`sequential_update.py:1697`(분기 조건)과 `:1734`(`n_add` 계산) **둘뿐**이므로, 한 호출 안의
읽기 로그가 다음을 **정확히** 준다:

| 행의 열 | 뜻 |
|---|---|
| `grow_calls` | `_adaptive_prune_grow` 호출 수 (⛔이것만으로는 분기를 못 본다) |
| `grow_branch_entries` | ★**성장 분기 진입** = 이른 반환을 통과한 호출 수 |
| `grow_threshold_pass` | 잔차 문턱까지 넘어 `n_add` 를 계산한 호출 수 |
| `n_grow_atoms` | ⚠`add_gaussians` **총합** — 성장 분기 **밖**의 호출도 포함한다(정직하게 표기) |

⛔거동 불변: 세 훅 전부 진짜 함수의 반환값을 그대로 돌려준다. 그 증명은 추론이 아니라
**`G-ANCHOR-B22`/`G-ANCHOR-B16`**(동결 행과 비트 동일)이다 — 동결 배치는 이 훅 없이 돌았다.

---

## §3 행 스키마가 사전등록 §6.3 보다 **넓다** (게이트가 요구하는데 목록에 없던 열)

사전등록 §6.2 의 `G-DOSE` 는 `shift_max` 를 요구하는데 **§6.3 의 열 목록에 없고**,
`run.py` 의 요약 dict 이 `_snap_k_to_grid` record 의 `shift_max`·`n_atoms` 를 **버린다**.
⟹ 러너가 그 메서드를 **관측 훅으로 감싸** record 전체를 잡고 두 열을 더한다:
`k_snap_shift_max` · `k_snap_n_atoms`.
같은 사유로 다음 열도 더했다(전부 **되읽기**이고, 게이트가 실제로 쓴다):
`pool_cap_readback` · `growth_ceiling_readback` · `max_n_alias`(§1) ·
`grow_calls` · `grow_branch_entries` · `grow_threshold_pass` · `n_grow_atoms`(§2) ·
`deploy_readback`(§6.1 `G-DEPLOY` 가 요구하는 9종 `cfg` 되읽기) ·
`propose_calls` · `n_pool_expected` · `n_pad_dropped`(`B-22` 규약 계승, `G-ATOMSRC` 의 원자료) ·
`effective`(`gates_b19.effective_from_updater`) · `updater_class` · `code_md5_row` ·
`zero_n_frames`·`zero_t_first`·`zero_t_last`·`Hs_true`·`Tp_true`(`zero_nrmse` 가 함께 준다).
⛔**빼지 않았다** — 사전등록 §6.3 의 열은 전부 그대로 있다.

---

## §4 등록 예측 `Q_REG`·`Q_SYN` 의 **입력은 등록된 4자리 값**이다

사전등록 §3.5 의 스크립트는 `MED_Q1, SD_Q1, MED_PR, RHO = 0.4879, 0.3290, 0.6655, 0.7138` 을
**리터럴로 고정**하고 그 위에서 예측을 냈다. 동결 jsonl 에서 **전정밀도로 재계산**하면
`Q_REG` 가 `0.003752 → 0.0037468` 로 **5자리째에서** 갈린다(라벨 확률은 인쇄 정밀도로 불변).
⟹ **등록된 예측은 등록된 입력의 함수**이므로 `gates_t18.PRED_INPUTS` 가 그 4자리 값을 쓰고,
⛔재계산은 버리지 않는다 — `verify_registered()` 가 동결 행 재계산과 **4자리 일치를 강제**한다
(§12-1 *"재계산으로 핀한다"*). ⚠부기: §0.4 의 `sd` 는 **모표준편차(ddof=0)** 여야 값이 맞는다
(표본표준편차로는 0.3324 다). 재계산은 ddof=0 을 쓴다.

---

## §5 스모크에서 **범위가 좁아지는 게이트 셋** — ⛔조용한 완화가 아니다

`G-CORPUS`·`G-N`·`G-MESH` 는 표본의 **모양**을 보는 게이트라 2 에피소드 스모크에서는
성립할 수 없다. ⟹ 스모크 모드에서 그 셋의 **범위를 스모크 부분집합으로** 좁히고,
**이름에 `[스모크 범위]` 를 박는다**(verdict 에 그대로 인쇄된다). 나머지 22개 게이트는
스모크에서도 **본 판정과 같은 조건**으로 돈다 — 특히 사전등록 §7 이 요구한
`G-ANCHOR-B22`·`G-ANCHOR-B16`·`G-SNAPFIRE`·`G-PAIRSRC`·`G-BLOCK` 이 그렇다.
⛔스모크 verdict 은 **배치 디렉터리 밖**에 쓰고 라벨을 인쇄하지 않는다(`label = None`).

---

## §6 구현 순서·파일 구성 (사전등록 §12 와 다른 점을 적는다)

- 사전등록 §12-2 는 `analyze → gates → run` 이라 적었고, 실제 구현은 **`gates → analyze → run`**
  이다(작업 지시). ⛔러너가 마지막이라는 핵심 규율은 지켜졌고, **행 스키마는 여전히
  `analyze_t18.py` 독스트링이 정본**이다(러너가 그것을 따른다).
- 사전등록 §12-1 은 `consts_t18.py` 를 따로 두라 했으나 **`gates_t18.py` 안에 넣었다**
  (`design_constants()`·`REGISTERED`·`verify_registered()`). 사유: 이 저장소가 여섯 번 물린
  결함이 *"같은 기능이 두 파일에 구현돼 한쪽만 계측"* 이다 ⟹ **단일출처를 쪼개지 않는다.**
- 사전등록 §3.3 의 양성대조 문턱 `32` 는 **리터럴로 적지 않았다**(§12-4) —
  `THRESH_CTRL = k_two(48)` 가 **계산**하고 그 값이 32 다.
- `SEC-C` 층은 별도 결과 파일(`results_t18_secc*.jsonl`)로 나가고 분석기가 합쳐 읽는다.

---

## §7 이 개정이 **바꾸지 않은 것** (명시)

⛔PRIMARY 통계량·방향·집계·n · 문턱(32/31) · 등가 밴드 `ln(1.05)` · `CEIL` 핀(1.306) ·
라벨 8개와 발화 순서 · 라벨 확률표 · MDE(0.7109/0.6911/0.8529) · 표본 48 · 시드 42·43·44 ·
`SEC-C` 45·46·47 × 앞 16 · 팔 5개와 `ANA-N` 선택 · §9.2 금지 20항 · §9.3 공시 9항 ·
그리고 ★**§3.6**(이 설계가 자기 질문에 답하지 못할 공산이 크다는 결과 전 진술).
