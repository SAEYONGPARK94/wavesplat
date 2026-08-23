# RV-6 배포 검증 — `bank_refit_every_s: None → 5.0` (2026-08-23)

**추정 런 0회.** 근거 = 코드 열람(줄 단위) · `results_b3.jsonl`+`results_b3b.jsonl` 216행 ×
`true_zero_perlead.json` 재집계 · `results_b6c.jsonl`+`verdict_b6.json` 재계산 · `git` 이력 ·
**회귀 시험 변이 3건 실행**(⚠아래 §0 참조).

**열람 범위** = `wavesplat/optimizer/sequential_update.py`(`:55-160,995-1130,1747-1760`) ·
`wavesplat/run.py`(`:18-100,117-200,655-690`) · `configs/estimation.yaml` ·
`wavesplat/tests/test_config_exposure.py`(전문) · `test_bank_mmse_port.py:190-250` ·
`test_shipping_defaults.py:40-100` · `scripts/gen_deploy_constants.py` · `thesis/deploy_constants.tex` ·
`tmp/B3_REFIT_2026-08-23/`(PREREG·gates_b3/b3b/b3c·analyze_b3b·RESULT_B3/B3B/B3C·FINAL·RV3/RV4/RV5) ·
`tmp/B6_BUDGET_2026-08-23/` 전체 · `tmp/B7_PROP_TRACK_2026-08-23/PREREG·gates·run·FROZEN_md5` ·
`reference_citation_bans.md` 전문 · `scripts/d3_d17_ctrl_closedloop.py:107-121` ·
`scripts/d3_required_horizon.py:1-12`.
⛔**열지 않은 것**: `bank_mmse.py` 내부 · `zero_ref.py` · `B-5`/`B-2` 러너 · `scene_station.py`.

---

## ★0. 운영 사항 — 패키지 파일을 **건드리지 않았다**

검증 시작 시점에 **주 세션의 `run_b7.py` 가 돌고 있었다**(PID 220863, `timeout 4800`).
저장소는 *"배치 중 다른 세션이 임포트 경로 파일을 수정"* 으로 S13·S13b 에서 이미 대가를 치렀다.
⟹ 항목 6(시험 파괴)은 **패키지 사본**(`scratchpad/wsmut`, 1.8 MB, `results`/`data` 는 심볼릭)에서
수행했다. 검증 종료 시각 실측:

```
md5sum -c 기준선 5파일 → 전건 OK
git status --porcelain | grep -v '\.h5$'
  ?? wavesplat/results/gifs/                            (검증 시작 전부터 존재)
  ?? wavesplat/tmp/B7_PROP_TRACK_2026-08-23/results_b7.jsonl   (주 세션 B-7 산출물)
```
⟹ ✅**내가 만든 변경 0건.**

---

# ★ 판정 표

| # | 검증 항목 | 판정 | 한 줄 |
|---|---|---|---|
| 1 | **`τ=5` 선택 논거** | ⚠**WEAK** | 수치는 **전건 재현**(28/36, p=5.97e-04). 그러나 ★**그 결정 규칙은 어느 PREREG 에도 없다**, ★**Holm 으로 바꾸면 τ=10 도 통과**, ★**14 s 지평이 긴 τ 를 구조적으로 벌한다**(τ=10 은 채점 lead 의 **69%가 FROZEN 과 비트 동일**) |
| 2 | **점유율 계산** | ✅**SOUND** | 1.854/0.997/0.385/0.273 **네 값 전부 소수 3자리까지 재현**. 재적합 14/7/2/1 은 `refit_calls` 원자료와 일치하고 `refit_readback` 이 노브를 확증한다. ⚠단 `t_init` 을 넣은 것이 **일회비용을 정상비용으로 계상** = τ 간 격차를 압축한다(정직한 방향은 보수적) |
| 3 | **단일출처** | ⚠**WEAK** | 런타임 사슬(모듈→`DEFAULT_CONFIG`→클래스 폴백)은 **진짜**. 그러나 ⛔**yaml 키는 죽은 미러**(로더가 한 번도 읽은 적 없다) · ⛔**`SHIPPING_DEFAULTS` 미등록** · ⛔**`deploy_constants.tex` 미등록** · ⛔**낡은 주석 4곳(그중 하나는 `--help` 문면)** |
| 4 | **되돌림** | ✅**SOUND** | 배포 커밋이 **거동 값 한 개**만 바꿨다(세 소비처 + yaml 미러). `pre_2026_08_23_config()` 만으로 08-22↔08-23 사이 거동이 복원된다. `pre_2026_08_22_config()` 는 **그 이전**으로 갈 때만 필요하고 독스트링이 정확히 그렇게 적었다 |
| 5 | **`DEPLOY_IMPACT` census** | ✅**SOUND**(2분 낡음) | 15:54 정정판이 옳다. 내 독립 전수 스윕에서 **추가 누락 0건** — 유일한 신규는 census **2분 뒤** 생긴 `B-7`(의도적으로 배포값 사용). `COMPARE_REAL_SYNTH/real_estimate.py` 는 `init_method='learned'` 라 게이트 밖 = **무관** |
| 6 | **시험이 막는가** | ✅**SOUND** | 변이 **3/3 검출**. ★M2(폴백을 옛 `None` 으로 되돌림)를 **거동 시험**이 잡았다 = 조용한 회귀가 실제로 막힌다. ⚠단 `test_bank_refit_fallback_matches_shipping_default` 의 **첫 assert 는 공허**하고, yaml 검사 3건은 **거동을 못 지킨다**(§3) |
| 7 | **범위 진술** | ⚠**WEAK** | *"실제 센서 잡음 미측정"* 말고 **네 가지가 더 빠졌다**: ★**동기 재적합 정지**(update() 한 번이 1.35–2.53 s = 프레임 14–25개) · ★**정확도는 1 Hz 투입, 예산은 10–15 Hz** · ★**τ=5 는 신선 창에서 한 번도 안 쟀다** · ★**14 s 지평 의존** |

## ★ 권고 — **조건부 유지 (KEEP-WITH-CONDITIONS)**

> **되돌리지 마라. 단 `RESULT_B6C §2` 의 *"유일한 배포점"* 과 *"전부 사전등록"* 두 문장은 철회하고,
> §7 의 네 범위 제한을 배포 주석에 넣어라.** 사전등록된 것(`ANY-REFIT-HELPS`: 1·2·5·10 s **전부**
> FROZEN 을 이기고 FROZEN 은 참 제로필드 대비 **무기술** 16/36 p=0.797)만으로도 *"재적합을 켠다"* 는
> 결정은 지지된다. 흔들리는 것은 **5 냐 10 이냐** 뿐이고, 두 값의 차이는 **정확도 4창 vs 예산 1.4배**다.
> ⚠**되돌려야 할 유일한 시나리오** = 실시간 제어 루프에 실제로 물릴 때. 그때는 τ 를 고르는 문제가 아니라
> **재적합을 `update()` 밖으로 빼는 문제**다(§7-A).

---

# 1. `τ=5` 선택 논거 — ⚠ WEAK

## ✅1.1 수치는 전건 재현된다 (MEASURED)

`results_b3.jsonl`+`results_b3b.jsonl` 의 `per_lead` 를 사전등록 통계량
(`fc_mean` = lead 2–14 평균; 저장된 `fc_mean` 과 내 재계산이 **부동소수 비트 동일**)로 접고
`true_zero_perlead.json` 의 같은 구간 평균을 기준으로 짝 부호검정:

| 팔 | 이김 | 상대 중앙 | p(단측) | `RESULT_B6C` 기재 |
|---|---|---|---|---|
| `FROZEN` | 16/36 | +3.26% | 0.797 | (RV-3 ⑦과 일치) |
| `REFIT1` | **34/36** | −23.57% | 9.71e-09 | 34/36 ✅ |
| `REFIT2` | **32/36** | −20.68% | 9.71e-07 | 32/36 · 9.7e-07 ✅ |
| ★`REFIT5` | **28/36** | −12.20% | **5.97e-04** | 28/36 · **6.0e-04** ✅ |
| `REFIT10` | **24/36** | −6.49% | **0.0326** | 24/36 · 0.033 ✅ |

⟹ ★**주 세션의 재도출은 정확하다. 내가 깨지 못했다.**

## ⛔1.2 그러나 **그 결정 규칙은 어느 사전등록에도 없다** (MEASURED, 문서 전수)

- `gates_b3.py` PRIMARY = `REFIT2` **vs `FROZEN`** ≥26/36.
- `gates_b3b.py` 판정 = 세 팔 **vs `FROZEN`** ≥24/36 → `ANY-REFIT-HELPS`. **네 팔 전부 통과**(36·36·35·35).
- `gates_b3c.py` = 신선 창 복제, **vs `FROZEN`**.
- `gates_b6.py`·`gates_b6b.py`·`gates_b6c.py` = **예산 전용**. 정확도 기준 **0개**.

⟹ ***"참 제로필드 대비 + 가족단위 α" 로 τ 를 고른다*** 는 규칙은 **`RESULT_B6C.md`(15:36)에서 처음
등장한다** — 즉 **네 팔의 p 값을 전부 본 뒤**다. `grep -n "zero_ref\|제로필드\|ZERO"` 로 여섯 개
게이트 파일과 두 PREREG 를 훑으면 이 기준은 **한 줄도 없다**(유일한 등장은 `gates_b3.py:12` 의
*"짝 비교라서 `ZERO_REF` 결함에 면역이다"* = **정반대 취지**).

⟹ ⛔**배포 커밋 메시지와 `RESULT_B6C §2` 의 *"근거 사슬 … 전부 사전등록"* 은 과대 진술이다.**
저장소가 S13 에서 정확히 이 부류로 물렸다(*"요약이 사전등록보다 넓게 나갔다"*).
✅**정직한 문안** = *"사전등록은 `τ ∈ {1,2,5,10}` 을 구분하지 않는다. 5 를 고른 근거는 결과 이후에
세운 기준이다."*

## ⛔1.3 분모 10 은 **문제가 아니다** — 문제는 **Bonferroni 냐 Holm 이냐**다 (MEASURED)

RV-5 `C-20` 이 분모가 문서마다 4/7/10/16 이라고 지적했고, `B-7`(15:56)은 **11** 을 쓴다.
오늘 같은 12 레코드를 본 `results*.jsonl` 은 **15개**다. 그런데 **분모는 판정을 못 바꾼다**:

| m | α | τ=5 (p=5.97e-4) | τ=10 (p=3.26e-2) |
|---|---|---|---|
| 1 | 0.05 | 통과 | **통과** |
| 2·4·7·10·11·15·16 | 0.025 ~ 0.0031 | **통과** | 탈락 |
| ≥84 | ≤5.95e-4 | 탈락 | 탈락 |

⟹ τ=5 는 **m ≤ 83 이면 무조건 통과** · τ=10 은 **m ≥ 2 면 무조건 탈락**.
**분모 10/7/16 중 무엇을 쓰든 결론이 같다** ⟹ 이 축에서 주 세션은 **결백하다.**

★**그러나 절차를 바꾸면 뒤집힌다.** 같은 4팔 가족에 **Holm 단계하강**(저장소가 같은 날
`rv4/final.py` 의 Holm-75, `rv4/perlead.py` 의 Holm-15 에서 **실제로 쓴 절차**)을 적용하면:

```
REFIT1  34/36 p=9.71e-09  Holm-α=0.0125  PASS
REFIT2  32/36 p=9.71e-07  Holm-α=0.0167  PASS
REFIT5  28/36 p=5.97e-04  Holm-α=0.0250  PASS
REFIT10 24/36 p=3.26e-02  Holm-α=0.0500  ★PASS
```
⟹ ⛔***"τ=10 은 기술 주장이 다중성에서 죽는다" 는 절차 선택의 산물이다.***
CLAUDE.md 상시경고 *"같은 자료가 규약 하나로 판정이 뒤집힌 적이 여러 번 있다"* 의 재발이고,
**그 규약이 결과 뒤에 골라졌다**(§1.2).

## ★★1.4 더 센 반증 — **14 s 지평이 긴 τ 를 구조적으로 벌한다** (MEASURED, 신규)

재적합은 `frame.t − last_fit_t ≥ τ` 에서 발화하고 `last_fit_t` 는 초기화 창의 끝이다.
⟹ **lead < τ 인 구간은 정의상 `FROZEN` 과 같은 상태**다. 36창 전건에서 비트 동일 여부를 셌다:

| 팔 | `FROZEN` 과 **비트 동일**인 lead | 채점 구간(2–14, 13개) 중 비중 |
|---|---|---|
| `REFIT1` | {0,1} | **0/13** |
| `REFIT2` | {0,1,2} | 1/13 |
| `REFIT5` | {0..5} | 4/13 (31%) |
| ★`REFIT10` | {0..10} | ★**9/13 (69%)** |

⟹ ***`REFIT10` 의 `fc_mean` 은 69%가 무기술 팔의 값이다.*** 용량-반응 곡선의 상당 부분은
**적합 나이 곡선이 아니라 희석 곡선**이다.

**민감도** — 각 팔을 *자기 첫 재적합 이후* 구간에서만 채점하면(참 제로필드 대비):

| 팔 | 구간 | 이김 | 상대 중앙 | p |
|---|---|---|---|---|
| `REFIT5` | lead 5–14 | 31/36 | −13.91% | 6.5e-06 |
| ★`REFIT10` | lead 10–14 | ★**29/36** | ★**−14.95%** | ★**1.56e-04** |

(같은 구간의 `FROZEN` 은 각각 13/36·12/36 = 무기술.)
⟹ ★**희석을 걷어내면 τ=10 이 α=0.005 를 통과하고 효과 크기는 τ=5 보다 오히려 크다.**
⚠**단서(정직성)**: 두 팔의 채점 구간이 서로 다르므로 **팔 간 직접 비교가 아니다** — 이것은
*"τ=10 의 탈락이 지평 절단에 기인하는가"* 에 대한 민감도이지 τ=10 우세의 증거가 아니다. `ARGUED`.

## ⚠1.5 τ=2 의 배제는 **살아남는다** (공정성)

`τ=2` 를 뺀 근거(점유율 중앙 0.997)는 규약에 민감하다 — 일회 `t_init` 을 빼면 **0.886 = "빠듯"**
이지 ⛔가 아니다. 그러나 **최악 레코드**는 두 규약에서 **1.474 / 1.288 = 둘 다 1 초과**이고
`B-6 PREREG` 가 *"최악 레코드도 함께 보고한다 — 중앙만으로 배포 결정을 하지 않는다"* 를
**결과 전에** 적었다. ⟹ ✅**τ=1·τ=2 배제는 사전등록 문언으로 방어된다. 내가 깨지 못했다.**

## ⟹1.6 §1 결론

**살아있는 문장** = *"τ ∈ {5,10} 둘 다 예산과 기술을 만족한다. 5 는 정확도 최적(28 vs 24창),
10 은 예산 최적(점유율 0.273 vs 0.385, 정지 빈도 절반)."*
**죽는 문장** = ⛔*"τ=5 가 둘을 동시에 만족하는 **유일한** 점"* · ⛔*"근거 사슬 전부 사전등록"*.

---

# 2. 점유율 계산 — ✅ SOUND (죽이지 못했다)

## 2.1 네 값 전건 재현 (MEASURED)

`results_b6c.jsonl` 12행 + `verdict_b6.json` 의 `t_refit` 으로
`duty(τ) = (t_init + n_refit(τ)·t_refit + n_frames·(t_upd + t_ctrl)) / 14` 를 다시 계산:

| τ | n_refit | 내 중앙/최악 | `RESULT_B6C` 기재 |
|---|---|---|---|
| 1 | 14 | **1.854 / 2.738** | 1.854 / 2.738 ✅ |
| 2 | 7 | **0.997 / 1.474** | 0.997 / 1.474 ✅ |
| 5 | 2 | **0.385 / 0.571** | 0.385 / 0.571 ✅ |
| 10 | 1 | **0.273 / 0.390** | 0.273 / 0.390 ✅ |

## 2.2 재적합 횟수 14/7/2/1 도 원자료 확인 (MEASURED)

`results_b3b.jsonl`·`results_b3.jsonl` 의 `refit_calls` 는 팔별로 **단일값**이고
`refit_readback` 이 노브가 실제로 들어갔음을 확증한다:

```
FROZEN [0]  readback [None]   REFIT1 [14] readback [1.0]
REFIT2 [7]  readback [2.0]    REFIT5  [2] readback [5.0]   REFIT10 [1] readback [10.0]
```
lead 1..14 · 발화 조건 `t − t_fit ≥ τ` 와 산술적으로 정확히 일치한다.
✅ `RESULT_B6B` 의 *"τ ≥ 15 는 발화 0회 = 동결과 동일"* 도 같은 산술로 맞다.

## ⚠2.3 두 가지 유보 (판정을 바꾸지는 않는다)

1. **`t_init` 을 정상비용으로 계상했다.** τ=10 에서 게시된 점유율의 **41%가 일회 초기화**다.
   방향은 **보수적**(더 나빠 보이게)이지만, τ 간 격차를 압축해서 τ=5 를 **실제보다 좋아 보이게**
   한다(초기화 제외 시 0.274 vs 0.152 = 1.8배인데, 포함 시 0.385 vs 0.273 = 1.4배).
2. **`t_refit` 은 τ=2 에서만 쟀다**(`gates_b6.TAU = 2.0`)고 모든 τ 에 재사용했다.
   `_bank_fit_window` 가 `bank_max_obs=600` 으로 상한을 두므로 **가정은 타당**하다(`ARGUED`,
   측정으로 확인되지는 않았다).

---

# 3. 단일출처 — ⚠ WEAK

## ✅3.1 런타임 사슬은 진짜다 (MEASURED)

배포 커밋 `23b945c` 의 **거동 diff 는 정확히 네 줄**이다:
`BANK_REFIT_EVERY_S_DEFAULT = 5.0` 신설 · `run.py` import 추가 ·
`DEFAULT_CONFIG['bank_refit_every_s'] = BANK_REFIT_EVERY_S_DEFAULT` ·
클래스 폴백 `.get('bank_refit_every_s', None)` → `.get(..., BANK_REFIT_EVERY_S_DEFAULT)`.
저장소 전체에서 `.get('bank_refit_every_s'` 는 **패키지 안에 `sequential_update.py:1021` 하나뿐**이고
(나머지는 `tmp/rx2_init_2026-08-05/snap/` 동결 스냅샷) **옛값 `None` 하드코딩 잔존 0건**이다.
⟹ ★**"한 곳에서 바꾸면 전부 따라오는가" 의 답은 예다** — 변이 M1 이 그것을 실행으로 확인했다(§6).

## ⛔3.2 그런데 **yaml 키는 처음부터 죽어 있었다** (MEASURED — 신규)

`configs/estimation.yaml:78` 은 `bank_refit_every_s: 5.0` 을 적지만
**`run.py:load_yaml_config()` 는 이 키를 읽지 않는다.** `sed -n '117,172p' run.py | grep -c
bank_refit_every_s` = **0**. 같은 블록이 `real_estimator`·`real_bank_ensemble_k` 는 읽는다.
`git log -S` 로 확인: 노출 커밋 `b9d5206`(08-20)이 `DEFAULT_CONFIG`·CLI·yaml **파일**은 건드렸으나
**로더는 건드린 적이 없다.**

**결과 셋:**
- yaml 을 `null` 로 고쳐도 **되돌아가지 않는다**(변이 M3 가 시험은 깨지만 거동은 안 바뀐다).
- CLI 는 `type=float, default=None` + `if args.… is not None:` 이라 **`None` 을 줄 방법이 없다**
  (`--bank_refit_every_s 0` 은 동결이 아니라 **매 프레임 재적합**이다).
- ⟹ ★**배포된 기본값을 끄는 표면이 파이썬 API 하나뿐이다.**
- ⟹ `test_config_exposure` 의 yaml assert 3건은 **거동이 아니라 문서를 지킨다** — 그런데
  모듈 docstring은 그 파일을 *"implemented-but-unreachable 을 막는 사슬"* 이라 부른다.
  ★**이 파일이 막으려던 결함 부류가 이 파일 안에서 재현됐다(3회차).**

## ⛔3.3 배포값이 **저장소가 지정한 등록부 두 곳에 없다** (MEASURED — 신규)

| 등록부 | 자기 선언 | `bank_refit_every_s` |
|---|---|---|
| `tests/test_shipping_defaults.py::SHIPPING_DEFAULTS` | *"배포값 적는 자리는 **여기 하나**여야 한다"*(축자, `:53`) | ⛔**없다** (08-22 의 세 키는 등록돼 있다) |
| `scripts/gen_deploy_constants.py` → `thesis/deploy_constants.tex` | *"논문은 배포 상수를 문자로 다시 적지 않는다"* | ⛔**없다**(`\DeployNumTests` 만 236 으로 갱신됐다) |

⟹ 논문이 이 값을 쓰려면 **손으로 적어야 하고**, 그 순간 `lambda_l1` 3중 기재와 같은 부류가 된다.

## ⛔3.4 낡은 주석 **4곳** — 하나는 사용자에게 보인다 (MEASURED)

| 위치 | 축자 | 상태 |
|---|---|---|
| ★`run.py:665-666` **`--help` 문면** | *"omit to keep the shipping **frozen-fit** behaviour"* | ⛔거짓 — 생략하면 5 s 재적합 |
| `run.py:78-91` | *"None (the shipping default, **unchanged here**)"* · *"Hence exposure only — **the default is NOT changed**"* | ⛔거짓 |
| `sequential_update.py:1003-1005` | *"the default (None) keeps the fit frozen, **which is exactly the protocol the real-sea numbers were measured under**"* | ⛔거짓 |
| `sequential_update.py:1070-1071` | *"the default None keeps the init fit frozen, matching the validated pure-forecast protocol"* | ⛔거짓 |

★저장소는 이 부류에 이미 회귀 시험을 갖고 있다 —
`test_shipping_defaults.py::test_no_stale_lambda_literal_in_default_config_comment`
(*"2026-08-13 에 발견한 독스트링 표류를 막는다"*). **새 키에는 그 대응물이 없다.**
그리고 2026-08-22 검수 V1 이 지적한 *"독스트링 거짓 정당화"* 가 **하루 만에 네 번 재발**했다.

---

# 4. 되돌림 — ✅ SOUND (죽이지 못했다)

`git show 23b945c` 의 거동 diff 는 §3.1 의 네 줄이 **전부**다(나머지는 주석·시험·결과 파일).
2026-08-22 의 세 스위치(`varpro_steps`·`learned_init_iters`·`auto_switch_thresh`)는 **건드리지 않았다**.
⟹ **`pre_2026_08_23_config()` 하나로 08-22 배포 직후 상태가 정확히 복원된다.**
`pre_2026_08_22_config()` 는 *그 이전*으로 갈 때만 필요하고, 독스트링이 축자로 그렇게 적었다
(*"2026-08-22 의 세 스위치까지 함께 되돌리려면 … 같이 쓰라"*). **문언과 코드가 일치한다.**

★**부수 사실(강화)**: 실측 경로에서는 08-22 의 세 스위치가 **원리적으로 무관**하다 —
실측 게이트가 `bank_mmse` 로 보내므로 `varpro`/`learned_init`/라우터 임계는 도달하지 않는다.
⟹ **실측 재현에는 `pre_2026_08_23_config()` 만으로 충분하고, 그것이 완전하다.**

---

# 5. `DEPLOY_IMPACT` census — ✅ SOUND (2분 낡음)

## 5.1 15:54 정정판이 옳다 (MEASURED, 독립 재도출)

주 세션의 첫 census(15:52)는 `B-2 SHIP` 이 오염된다고 적었고 **15:54 커밋 `00e7c8e` 가 스스로
정정**했다(`cfg.update()` 는 dict 메서드). 내가 독립적으로 다시 셌다:
실측 장면 빌더를 만지는 **123개 `.py`** 중 **추정기 `update()` 를 부르는 것**은

```
run_b3.py  run_b3b.py  run_b3c.py  run_b4.py  run_b6.py  run_b6c.py   ← census 의 여섯
run_b7.py                                                            ← census 2분 뒤 생성
tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py:124               ← ✅무관 (아래)
```

- ⚠`run_b7.py` — census(15:54)보다 **2분 늦게**(15:56) 만들어졌다. `SHIP-TRACK` 팔이 키를
  명시하지 않지만 **PREREG 가 *"재적합 5 s = 2026-08-23 배포"* 라 의도적으로 적었다** ⟹ 오염 아님.
  다만 **census 는 이제 일곱 개**다.
- ✅`real_estimate.py` — `arms_config.ARM_DELTAS` 가 `init_method='learned'` 를 고정하므로
  실측 게이트(`init_method=='auto'` 안)에 **도달하지 않는다** ⟹ `_bank` 가 `None` 으로 남고
  이 키와 무관하다. (CLAUDE.md 가 이 파일을 *"실측 경로"* 로 부르지만, **bank 경로는 아니다.**)

⟹ ★**census 에 내가 추가할 누락은 0건이다.** 주 세션의 정정은 옳고 완전하다.

## 5.2 합성 불변도 코드로 확인 (MEASURED)

`update()` 의 **첫 줄**(`sequential_update.py:1123`)이 `if self._bank is not None:` 이고
`_bank` 는 `_bank_fit_window` 에서만 채워지며 그 유일한 초기화 호출부는 실측 게이트
(`:638-639`, `not _frames_have_gt(frames)`) 안이다. ✅ `DEPLOY_IMPACT §3` 이 옳다.

## ⚠5.3 census 가 안 적은 것 — **배포된 기본값에 출하 소비자가 없다**

`wavesplat/real_data.py`(패키지의 실측 프론트엔드)를 import 하는 것은 **`tests/test_model.py` 뿐**이고,
`python3 -m wavesplat.run` 은 HDF5 합성 에피소드만 받는다.
⟹ ★**현재 이 배포값을 실제로 쓰는 것은 `tmp/` 하네스뿐이다.** 배포 위험이 그만큼 낮다는 뜻이자,
*"배포했다"* 는 말이 **출하 경로 변경**이 아니라 **기본값 선언**이라는 뜻이다. 문서에 없다.

---

# 6. 시험이 실제로 막는가 — ✅ SOUND

⚠**패키지 사본에서 수행**(§0). 대상 = `test_config_exposure.py` 7건 +
`test_bank_mmse_port.py::test_bank_refit_default_refits_and_revert_freezes`.
**사본 기준선 = 8 passed (110 s).** (참고: 실제 트리에서 두 파일 전체는 **17 passed / 326 s**.)

| 변이 | 무엇을 흉내냈나 | 결과 |
|---|---|---|
| **M1** `BANK_REFIT_EVERY_S_DEFAULT` 5.0→2.0 | 값 표류 | ⛔**2 failed** — `…default_is_the_2026_08_23_deploy`, `…single_sourced_and_deployed` |
| ★**M2** 폴백을 `.get(…, None)` 로 되돌림 | ★**조용한 옛-거동 회귀**(`DEFAULT_CONFIG` 는 5.0 그대로) | ⛔**2 failed** — 소스텍스트 시험 + ★**거동 시험** `test_bank_refit_default_refits_and_revert_freezes` 가 *"배포 기본값에서 재적합이 발화하지 않았다"* 로 잡았다 |
| **M3** yaml `5.0`→`null` | 미러 표류 | ⛔**2 failed** — 두 yaml 미러 시험 |

⟹ ★**M2 가 핵심이다**: `lambda_l1` 을 무효화시킨 바로 그 결함(폴백이 배포와 갈라짐)이 여기서는
**거동 수준에서** 잡힌다. 이 배포의 회귀 방어는 **진짜다.**

## ⚠6.1 그래도 두 군데는 헐겁다

1. **공허한 assert.** `test_bank_refit_fallback_matches_shipping_default` 의 첫 줄
   `upd.config.get('bank_refit_every_s', BANK_REFIT_EVERY_S_DEFAULT) == BANK_REFIT_EVERY_S_DEFAULT`
   는 `.get` 의 기본인자로 **비교 대상 자신을 넘긴다** ⟹ 코드가 무엇이든 **항상 참**이다.
   그 시험을 실제로 지탱하는 것은 뒤의 **소스텍스트 검사 두 줄**이다(M2 가 그것에 걸렸다).
2. **yaml 검사는 거동을 못 지킨다**(§3.2). M3 는 시험을 깨뜨리지만 **런타임 거동은 비트 동일**이다
   ⟹ 그 세 assert 는 *"문서가 일치한다"* 의 증거이지 *"설정 표면이 살아 있다"* 의 증거가 아니다.

## ✅6.2 사후 확인

```
md5sum -c (패키지 5파일) → 전건 OK          git status --porcelain (h5 제외) → 내 변경 0건
```

---

# 7. 범위 진술 — ⚠ WEAK (네 가지가 빠졌다)

배포 주석과 `RESULT_B6C §2` 는 세 가지를 적었다(합성 잡음 0.02 m · 4 사이트 바닥 p 0.0625 ·
CPU 1코어). **다음 넷이 없다.**

## ★★A. 동기 재적합 정지 — 예산 판정이 **재적합을 끈 채** 측정됐다 (MEASURED)

`_bank_update` 는 `self._bank_fit_window(*self._bank_obs)` 를 **`update()` 안에서 직접, 동기로**
부른다. 스케줄러도 백그라운드도 없다. 그런데:

- `run_b6.py:66` 축자 주석: *"**재적합이 없던 프레임만** 갱신 단가로 센다"* ⟹ `t_upd` 는
  구조적으로 재적합 프레임을 제외한다.
- `run_b6c.py` 는 **배포 전(15:35)** 에 `DEFAULT_CONFIG`(당시 `None`)로 돌았다 ⟹
  ★**`CONTROL-FRAME-OK 12/12` 는 재적합이 한 번도 발화하지 않은 구성에서 나온 판정이다.**

배포값에서 실제로 일어나는 일(레코드별 `t_refit` ÷ `dt_frame`):

| | 값 |
|---|---|
| 정지 길이 | **1.35 – 2.53 s** |
| 프레임 간격의 배수 | **중앙 20.8배, 최악 25.3배** |
| 빈도 (τ=5, dt≈0.083 s) | **약 60프레임마다 1회** |
| 배포 전(`None`) | **0회** |

⟹ ★***이 배포는 `update()` 에 5 초마다 1.7 초짜리 정지를 새로 도입한다.*** 점유율 0.385 는
그것을 **평균으로 지워버린다**. `B-6 PREREG` 자신의 문언 — *"프레임 예산을 넘으면 재적합과
무관하게 추적 자체가 실시간이 아니다"* — 을 재적합 프레임에 적용하면 **12/12 가 초과**한다.
✅**정직한 문안** = *"제어 프레임 예산은 재적합이 없는 프레임에서만 만족된다. 재적합 프레임은
프레임 마감을 20배 초과하며, 그 해결은 τ 조정이 아니라 재적합을 `update()` 밖으로 빼는 것이다."*

## ★B. 정확도는 **1 Hz 투입**, 예산은 **10–15 Hz** (MEASURED)

- 정확도(B-3/B-3b/B-3c): lead 1..14 **1 s 간격** = 14회 `update()`, 프레임당 12점,
  창 크기 569–573점(RV-3) ⟹ `bank_max_obs=600` **상한에 안 닿는다**.
- 예산(B-6c): `n_frames = round(14 / dt_frame)` = **140–210 프레임**(레코드 고유 10–15 Hz).

⟹ 두 배치가 **다른 시스템을 서술한다.** 배포 프레임률에서는 40 s 기억창이 ~4,800–6,000점이 되어
매 재적합이 `rng(0)` **600점 부분추출** 위에서 일어난다 — ★**정확도가 측정된 적 없는 구성**이다.
(⚠재적합 *횟수*는 τ 가 정하므로 불변이고, `t_refit` 도 600 상한 때문에 거의 불변일 것이다 —
바뀌는 것은 **무엇을 적합하는가**다. `ARGUED`.)

## ★C. `τ=5` 는 **신선 창에서 한 번도 안 쟀다** (MEASURED)

`results_b3c.jsonl`(신선 창 {0,2,4})의 팔은 **`FROZEN` 36 · `REFIT2` 36** 뿐이다.
`REFIT5`/`REFIT10` 은 **창 {1,3,5} 에만 존재**한다.
⟹ 배포 커밋 메시지가 근거 사슬에 적은 `B-3c(신선 창 36/36 복제)` 는 **τ=2 의 복제**이지
배포값 τ=5 의 복제가 아니다. ⛔**증거 전이 과대.**
✅ 남는 것: *"어떤 주기든 듣는다"* 는 신선 창에서 복제됐다 — **값이 아니라 기제**.

## ★D. 14 s 지평 의존 (MEASURED, §1.4)

τ=10 은 채점 lead 의 69%가 `FROZEN` 과 비트 동일이고, **τ ≥ 15 는 발화 0회**다.
⟹ 이 배치가 잰 것은 *"무한 지평의 정상상태 적합나이"* 가 아니라
***"신선한 적합에서 출발한 14 초 창"***이다. 배포는 무한 지평이다.

## ⚠E. 그 밖에 고정된 채 상호작용이 안 쟀진 값

`real_bank_ensemble_k=3`(재적합 비용 3배의 원인) · `bank_memory_s=40`(재적합 창의 90%가
직전 창과 겹친다) · `bank_max_obs=600` · 창 {1,3,5}. **τ 와의 상호작용 0건.**

---

# ★★ 내가 깨지 못한 것 (공정성)

1. **`τ=5` 의 28/36 · p=6.0e-04 은 정확하다.** 네 팔 전부, 소수점까지 재현됐다.
   `RESULT_B6C` 의 정확도 열과 점유율 열은 **여덟 값 전부** 원자료에서 다시 나온다.
2. **재적합 횟수 14/7/2/1 은 원자료·산술 양쪽에서 옳다.** `refit_readback` 이 노브를 확증한다.
3. **단일출처는 런타임에서 진짜다.** 옛값 하드코딩 **0건**, 변이 M1 이 실행으로 확인.
4. **되돌림이 완전하다.** 배포 diff 가 값 하나뿐이고, 실측 경로에서는 08-22 스위치가 무관하다.
5. **census 정정(15:54)이 옳다.** 내 독립 전수 스윕에서 **추가 누락 0건**.
   `real_estimate.py` 는 게이트 밖이라 무관하고, `B-7` 은 의도적으로 배포값을 쓴다.
6. **회귀 시험 3/3 검출.** 특히 조용한 폴백 회귀(M2)를 **거동 시험**이 잡는다.
7. **τ=1·τ=2 의 배제는 사전등록 문언(최악 레코드)으로 방어된다.**
8. **`B-6c` 의 8점 목표치는 진짜로 코드에서 읽었다** —
   `scripts/d3_d17_ctrl_closedloop.py:108-111` 축자 *"each foil, 4 points … Returns Q (8·n_steps, 3)"*
   확인. *"통과할 때까지 목표를 바꿨다"* 라는 내 첫 의심은 **틀렸다**.
9. **게이트 수 표기가 이번엔 전부 맞다**(B-3 8/8 · B-3b 8/8 · B-3c 9/9 · B-6 7/7 · B-6b 6/6 ·
   B-6c 6/6, `verdict_*.json` 대조). RV-5 `C-2` 부류의 재발 없음.
10. **`B-7` 의 `FROZEN_md5.txt` 가 배포 후 해시(`536d36e9…`)를 정확히 기록**했다 — 동결 사슬 유지.

---

# ★ 처방 (우선순위)

| # | 처방 | 비용 |
|---|---|---|
| **P-1** | `RESULT_B6C §2` 와 배포 주석에서 ⛔*"유일한 배포점"* · ⛔*"전부 사전등록"* 을 철회하고 §1.6 문안으로 교체 | 편집 |
| **P-2** | 배포 주석에 §7-A(동기 정지 20배) · §7-B(1 Hz vs 10–15 Hz) · §7-C(τ=5 신선창 미측정) · §7-D(지평) 추가 | 편집 |
| **P-3** | ⛔**낡은 주석 4곳 수정** — 특히 `run.py:665` **`--help` 문면** | 4줄 |
| **P-4** | `SHIPPING_DEFAULTS` 와 `gen_deploy_constants.py` 에 키 등록(+`\DeployBankRefitEverySec`) | 3줄 |
| **P-5** | yaml 로더에 키 추가(`cfg['bank_refit_every_s'] = raw['optimizer'].get(…, DEFAULT_CONFIG[…])`) **또는** yaml 에서 키를 빼고 *"이 키는 파이썬 API 전용"* 이라 적기. 지금은 **둘 다 아니다** | 1–2줄 |
| **P-6** | CLI 에 `--bank_refit_every_s frozen`(또는 `--no_bank_refit`) 을 붙여 **끄는 표면**을 만들기 | 3줄 |
| **P-7** | 공허한 assert 교체 + *"낡은 frozen 문구 금지"* 회귀 시험 추가(`lambda` 대응물) | 10줄 |
| **P-8** | (다음 배치) `REFIT5`·`REFIT10` 을 **신선 창 {0,2,4}** 에서 72런 — τ=5 배포값의 유일한 미복제 고리 | 72런 |
| **P-9** | (설계) 재적합을 `update()` 밖으로 — 그것이 τ 논쟁 자체를 없앤다 | 구조 |
