# P1 WATCHDOG 검증 — `BRIEF.md` 전 수치·코드 주장 재도출

작성: 2026-08-11 · 대상: `wavesplat/tmp/P1_real_path_2026-08-11/BRIEF.md`
규율: 판단 없음, 검증만. 이 파일 외 수정 0건.
작업 트리 상태(검증 시점): `sequential_update.py` 커밋 상태(수정 없음) ·
`configs/estimation.yaml`, `wavesplat/run.py` 는 **미커밋 수정**(= 오늘의 `lambda_l1` 1e-3→1e-4 배포).

--------------------------------------------------------------------------------
## 0. 요약 표 (전 18항목)

| # | 절 | 주장 | 판정 |
|---|---|---|---|
| 1 | §0.1 | 두 λ 예측면 **비트 동일** (probe 재실행) | **OK** |
| 2 | §0.1 | S9 희소 50/64 · −33.0% | **OK** |
| 3 | §0.1 | S9 혼잡 27/32 · −9.9% | **OK** |
| 4 | §0.2 | `sequential_update.py:455-475` = real-domain gate 라우팅 | **OK**(범위 주의) |
| 5 | §0.2 | `run.py:434` = 실측 경로에 가우시안 성분 없음 | **OK** |
| 6 | §0.3 | 실측 nowcast nRMSE **0.49** | **OK** |
| 7 | §0.3 | 14 s 리드 평균 **0.76** | **OK**(키 명시 필요) |
| 8 | §0.3 | 합성 비교치 "희소 ~0.014 · 혼잡 ~0.18" | 🔴**MISMATCH(규약)** |
| 9 | §1 | `sigma0 = 0.05*Hs_obs` 가 `:840` 하드코딩 | **OK** |
| 10 | §1 | 그 값 **설정 불가**(config/CLI/env 경로 부재) | **OK** |
| 11 | §1 | `real_bank_ensemble_k` 기본 **3** | **OK** |
| 12 | §1 | `bank_memory_s` 기본 **40.0** | **OK** |
| 13 | §1 | `bank_refit_every_s` 기본 **None** | **OK** |
| 14 | §1 | 역사 2배: 0.498/0.666/0.718 vs 0.858/0.927/1.048 | 🔴**MISMATCH(출처)** |
| 15 | §2 | `d3_compare_gifs.py:161` ⟹ 레코드당 창 **6개** | **OK** |
| 16 | §2 | `d3_falcon_real_operator.RECORDS` = **5개** | **OK** |
| 17 | §2 | `measure_blind_vs_frozen.py:27` 동결 창 **3개** | **OK** |
| 18 | §2 | n=5 최소 양측 p=0.0625 · n=3 최소 양측 p=0.25 | **OK** |
| 19 | §2 | `.nc` 13개 vs RECORDS 5개 = 미등록 8개 | **OK**(제외 사유는 코드 주석 1줄만) |

**MISMATCH 2건 · STALE 0건.**

--------------------------------------------------------------------------------
## 1. 🔴MISMATCH #1 — §1 역사적 2배 수치의 **출처가 틀렸다**

BRIEF §1:
> nowcast 0.498/0.666/0.718 vs 0.858/0.927/1.048, `tmp/bankmmse_port_2026-07-29/m3_field.json`

**재도출 결과 — 한 파일에서 나오지 않는다. 두 팔의 출처가 다르다.**

| 팔 | 값 | 실제 출처 (키까지) |
|---|---|---|
| bank_mmse (median-of-3) | 0.49824 / 0.66569 / 0.71800 | `tmp/bankmmse_port_2026-07-29/m3_field.json` → `/{AA15a_w5,AA15b_w3,YS17_w0}/m3_field/nowcast_nrmse` ✅ |
| v2_konly (단일런) | 0.85827 / 0.92677 / **1.04834** | 🔴`tmp/confirm2_2026-07-27/real_gif_info.json` → `/scores_animation/nowcast_nrmse`, `/additional_windows/{AA15b_w3,YS17_w0}/scores_animation/nowcast_nrmse` (`deployed.route_after_gate == 'v2_konly'` 로 확인) |

`m3_field.json` 에는 **bank 팔만** 들어 있다(키 = `members/{42,1042,2042}` + `m3_field`).
v2_konly 수치는 이 파일에 **존재하지 않는다**.
`blind_vs_frozen.json` 에도 없다(그 파일의 두 팔은 `frozen`=GT-파라미터 bank, `blind`=관측전용 bank).

**수치 자체는 전부 재현됨. 파일 인용만 틀림.**

⚠**상속된 오류**: 같은 단일-파일 인용이 `sequential_update.py:461-462` 코드 주석에도 있다
(`... 0.858/0.927/1.048, tmp/bankmmse_port_2026-07-29/m3_field.json`). BRIEF 가 그것을 그대로 옮긴 것.

**"약 2배" 배율 재도출** (v2_konly ÷ bank):

| 창 | 비 |
|---|---|
| AA15a w5 | 0.85827 / 0.49824 = **1.723** |
| AA15b w3 | 0.92677 / 0.66569 = **1.392** |
| YS17 w0 | 1.04834 / 0.71800 = **1.460** |
| 중앙 | **1.460** |

⟹ 실측 3창 중 **2 배를 넘는 창은 0개**. "약 2배"는 상한(1.72)에 붙인 표현이며 중앙은 **1.46배**.
(⛔이 역시 코드 주석 `~2x worse` 에서 상속됨.)

--------------------------------------------------------------------------------
## 2. 🔴MISMATCH #2 — §0.3 "합성 희소 ~0.014 · 혼잡 ~0.18" 은 **한 규약으로 재현되지 않는다**

출처 미기재. `results.jsonl` 에서 두 규약 모두 계산:

| 팔 | 레짐 | 에피소드-중앙(=S9 규약) | 런-단위 중앙(2016런) |
|---|---|---|---|
| `AUTO` (λ=1e-3) | sparse | 0.02024 | 0.02145 |
| `AUTO` (λ=1e-3) | crowded | 0.17794 | **0.17841** |
| `AUTO-L1` (λ=1e-4, **오늘의 출하값**) | sparse | 0.01625 | **0.01431** |
| `AUTO-L1` (λ=1e-4, 출하값) | crowded | 0.15829 | 0.15532 |

- "희소 ~0.014" → **`AUTO-L1` 의 런-단위 중앙 0.01431** 에서만 나온다.
- "혼잡 ~0.18" → **`AUTO` 의 런-단위 중앙 0.17841** 에서만 나온다. 같은 팔(`AUTO-L1`)이면 **0.155 = ~0.16**.

⟹ 두 숫자가 **서로 다른 팔**에서 왔고, 둘 다 S9 헤드라인이 쓰는 **에피소드-중앙 규약이 아니다**.
(에피소드-중앙 규약이면 `AUTO-L1` = 희소 **0.016** · 혼잡 **0.158**.)

⚠단, 이 항목은 "실측 0.49 vs 합성 O(0.01–0.2) = 자릿수 격차"라는 §0.3 결론의 방향을 바꾸지 않는다.
필요한 것은 **팔·규약 명기**.

--------------------------------------------------------------------------------
## 3. 항목별 재도출 상세

### ① §0.1 비트 동일 — `OK` (직접 재실행)
`python3 tmp/S9_lr_x_l1_2026-08-11/probe_real_path.py` 를 오늘 재실행. 출력:

```
lambda_l1=1e-03  라우트=bank_mmse  실측게이트=True  cfg 반영=1e-03  init 1.3s  field std=0.52998
lambda_l1=1e-04  라우트=bank_mmse  실측게이트=True  cfg 반영=1e-04  init 0.9s  field std=0.52998
★두 예측면이 **비트 동일**한가: 예 — lambda_l1 은 실측에 도달하지 않는다
```
`np.allclose(..., rtol=0, atol=0)` = 비트 동일. **재현 OK.**
(부수 확인: 두 런 모두 `real_domain_gate` 발화 → `bank_mmse`, `cfg` 에는 λ 가 정상 반영됨
⟹ "λ 가 config 에 안 들어갔다"가 아니라 **그 경로가 λ 를 안 쓴다**.)

★**독립 2차 확인**: `tmp/bankmmse_port_2026-07-29/real_gif_info.json` 의
`/additional_windows/s9_2026-08-11/` 항목이 최상위 2026-07-29 측정과 **모든 점수 비트 동일**
(`nowcast_nrmse` 둘 다 `0.488482266664505`). 서로 다른 날·다른 λ 에서 같은 값 ⟹ 같은 결론.

### ② ③ §0.1 S9 합성 — `OK` (독립 코드 재집계)
`analyze_s9.py` 를 쓰지 않고 새로 작성. 규약: 에피소드당 3시드 **중앙** → 같은 에피소드 내 짝 Δ →
Δ 부호 세기. (검증: 모든 (cell,episode) 조합이 정확히 3시드 = 672조합 × 3 = 2016런, 결측 0.)

| 대조 | 부호 | Δ중앙 | 상대 | 양측 부호검정 p |
|---|---|---|---|---|
| `AUTO-L1 − AUTO` 희소 n=64 | **50/64** | −0.005247 | **−33.04%** | **7.069e-06** |
| `AUTO-L1 − AUTO` 혼잡 n=32 | **27/32** | −0.020060 | **−9.86%** | **1.1307e-04** |

RESULT.md(50/64, −0.00525, −33.04%, 0.000007 / 27/32, −0.02006, −9.86%, 0.000113)와 **전건 일치**.
BRIEF 의 "−33.0% / −9.9%" 도 일치.

★**상대% 규약 주의**(BRIEF 미기재): −33.04% 는 **에피소드별 상대변화의 중앙**이다.
`Δ중앙 / AUTO중앙` 로 계산하면 희소 **−25.92%** · 혼잡 **−11.27%** 로 달라진다.

### ④ §0.2 `sequential_update.py:455-475` — `OK` (범위 주의)
현재 코드 실측 행 번호:

| 행 | 내용 |
|---|---|
| 445 | `# Escape hatch for deliberate research use: real_domain_gate=False.` |
| **446-447** | `if frames and self.config.get('real_domain_gate', True) \` / `and not _frames_have_gt(frames):` ← **게이트 판정 자체** |
| **455** | `if self.real_estimator == 'bank_mmse':` |
| **475** | `return self._initialize_bank(frames, verbose)` |

⟹ 인용 범위 `455-475` 는 **bank_mmse 라우팅 분기**를 정확히 감싼다.
⚠단 **게이트 조건식(446-447)은 그 범위 밖**이다. "게이트가 잡아 라우팅한다"를 한 범위로 말하려면
`446-475` 가 정확. (기능적 주장은 참 ⟹ `OK`.)

### ⑤ §0.2 `run.py:434` — `OK`
```
434:        # None on the real-domain bank-MMSE route, which has no Gaussian components
435:        # (see 'bank_*' keys in init_result / 'bank_members' below).
436:        'n_gaussians': (updater.model.n_gaussians ...
```
행 번호·내용 일치.

### ⑥ ⑦ §0.3 실측 성능 — `OK`
출처 = `tmp/bankmmse_port_2026-07-29/real_gif_info.json` (최상위 = AA15a w5, `arm=deployed`,
`deployed.route_after_gate='bank_mmse'`, `real_domain_gate_fired=True`, `bank_members=3`).

| BRIEF | 재도출 | 키 |
|---|---|---|
| nowcast 0.49 | **0.488482** | `/scores_animation/nowcast_nrmse` (= `/scores_protocol_1s_leads/nowcast_nrmse`, 동일) |
| 14 s 평균 0.76 | **0.756212** | `/scores_animation/mean_nrmse` (85프레임 균등, `lead_max_s=14.0`) |

⚠키 구분 필요: `/scores_protocol_1s_leads/mean_nrmse` = **0.751856 → 0.75**.
"0.76"은 **animation(85프레임) 평균**에서만 나온다. 프로토콜 1 s 리드 15점 평균은 0.75.

### ⑨ ⑩ §1 `sigma0` 하드코딩·설정 불가 — `OK`
```
839:        self._bank_Hs_obs = 4.0 * float(np.std(z))
840:        sigma0 = 0.05 * self._bank_Hs_obs
841:        self._bank = [fit_bank_mmse(P, z, self.depth, sigma0, kp_hint=None, seed=s) ...
```
- 행 번호 **840 정확**. 리터럴 `0.05` · `4.0` 둘 다 하드코딩.
- 전 저장소 `sigma0` grep: `self.config.get(...)` 경로 **0건**. `os.environ`/`getenv` 경로 **0건**
  (`sequential_update.py` 의 env 사용은 `S8_L1_TRACK_ONLY`, `S8_L1_PHASE1_ONLY` 두 개뿐).
- `run.py` CLI 인자 목록에도 없음. `load_yaml_config` 가 읽는 키 목록에도 없음.
⟹ **"설정 불가" 참.**

★**완전성 보완(FYI, 판단 아님)**: `bank_mmse.py:188-196` 은 `smult ∈ {1.0, 3.0, 10.0}` 을
**검증분할로 선택**해 `s2 = (smult * sigma0)**2` 을 쓴다. 즉 유효 사전 잡음은 `sigma0` 가 아니라
`smult × sigma0` 이고, 배율 3종은 데이터가 고른다. `sigma0` 계수를 c 배 하면 격자 `{c,3c,10c}` 로
이동한다 — 격자가 겹치므로(예: c=3 이면 `{3,9,30}`) 계수 스윕이 **선택 격자와 교락**될 수 있다.

★부수 사실: `real_domain_gate`, `bank_memory_s`, `bank_refit_every_s`, `bank_max_obs` 는
`load_yaml_config` 와 CLI **어디에도 노출되지 않는다**. yaml 로 조절 가능한 bank 키는
`real_estimator`, `real_bank_ensemble_k` **2개뿐**. §1 의 "게이트를 여는 방법은 이미 있다"는
**config dict 를 직접 넘길 때만** 참(yaml/CLI 로는 불가).

### ⑪ ⑫ ⑬ §1 bank 손잡이 기본값 — `OK`

| 키 | BRIEF | 코드 실측 |
|---|---|---|
| `real_bank_ensemble_k` | 3 | `run.py:54` `'real_bank_ensemble_k': 3` · `configs/estimation.yaml:36` `3` · `sequential_update.py:793` `config.get(..., DEFAULT_REAL_BANK_ENSEMBLE_K)` (1..5 검증) ✅ |
| `bank_memory_s` | 40.0 | `sequential_update.py:801` `float(self.config.get('bank_memory_s', 40.0))` ✅ |
| `bank_refit_every_s` | None | `sequential_update.py:802` `self.config.get('bank_refit_every_s', None)` ✅ |

(추가: `bank_max_obs` 기본 600 — `:835`.)

### ⑮ §2 레코드당 창 6개 — `OK`
`scripts/d3_compare_gifs.py:161`: `i0 = int(np.linspace(lo, hi, 6).astype(int)[win])` — 행 번호·내용 정확.
동일 관용구가 6개 파일에 더 있음(`d3_required_horizon.py:206`, `d3_static_field.py:165`,
`make_gif_bank_deployed.py:75`, `measure_blind_vs_frozen.py:40`, `make_real_estimation_gif.py:84`,
`run_rolling_refit.py:46`) ⟹ **6창 규약은 실측 하네스 전체에서 일관.**

### ⑯ §2 RECORDS 5개 — `OK`
실제 import 로 확인, `len(RECORDS) == 5`, 키 순서 = `YS17, AA15a, AA14, AA15b, LJ18g`.
BRIEF 의 나열(YS17 · AA15a · AA14 · AA15b · LJ18g)과 **완전 일치**.
⟹ 최대 5 × 6 = **30창** 산술도 OK.

### ⑰ §2 동결 창 3개 — `OK`
`tmp/bankmmse_port_2026-07-29/measure_blind_vs_frozen.py:27`:
`WINDOWS = [('AA15a', 5), ('AA15b', 3), ('YS17', 0)]` — 행 번호·내용·개수 전부 일치.

### ⑱ §2 부호검정 최소 p — `OK` (`math.comb` 재계산)

| n | 최소 양측 p = 2·C(n,0)/2^n | 최소 단측 p |
|---|---|---|
| 3 | **0.25** | 0.125 |
| 5 | **0.0625** | 0.03125 |
| 6 | 0.03125 | 0.015625 |
| 30 | 1.863e-09 | 9.313e-10 |

BRIEF 의 0.0625(n=5) · 0.25(n=3) **일치**.
⚠참고(BRIEF 미기재): 레코드를 블록으로 보되 **레코드 안 6창을 창-단위 부호로 쓰면 n=6 에서
최소 양측 p = 0.03125** 이므로 단일 레코드 내부만으로는 α=0.05 가 형식상 가능해진다 —
단 그 6창의 독립성은 BRIEF 가 지적한 대로 미검증. (사실 기재만, 설계 제안 아님.)

### ⑲ §2 `.nc` 13개 vs RECORDS 5개 — `OK` / 제외 사유는 **코드 주석 1줄**
`find data/real_bergamasco -name "*.nc"` = **13개** (BRIEF·메모리 문서 "13 레코드"와 일치).

| 사이트 | 파일 | 등록 |
|---|---|---|
| AA_2014 | `Surfaces_20140327_091000.nc` | ✅ AA14 |
| AA_2015 | `Surfaces_20150305_103500.nc` | ✅ AA15a |
| AA_2015 | `Surfaces_20150515_090000.nc` | ✅ AA15b |
| YS_2017 | `Surfaces_20170513_050000.nc` | ✅ YS17 |
| LJ_2018 | `Surfaces_20180103_093938_Grid2_short.nc` | ✅ LJ18g |
| LJ_2018 | `Surfaces_20180103_093938_Grid1_short.nc` | ❌ |
| BS_2011 | `Surfaces_20111001_161800_short.nc` | ❌ |
| BS_2011 | `Surfaces_20111004_113800_short.nc` | ❌ |
| BS_2011 | `Surfaces_20111004_130700_short.nc` | ❌ |
| BS_2011 | `Surfaces_20111004_153000_short.nc` | ❌ |
| BS_2013 | `Surfaces_20130922_130001_short.nc` | ❌ |
| BS_2013 | `Surfaces_20130925_121501_short.nc` | ❌ |
| BS_2013 | `Surfaces_20130930_102001_short.nc` | ❌ |

**미등록 8개 = BS_2011(4) + BS_2013(3) + LJ_2018 Grid1(1).** 5 + 8 = 13 ✅

**제외 사유 단서 — 코드 주석 딱 한 줄** (`scripts/d3_falcon_real_operator.py:65`, RECORDS 직전):
> `# only records whose patch can host a (possibly scaled) forward fan`

⟹ 기준 = **패치가 (축소하더라도) 전방 fan 을 담을 수 있는가**. 등록 5개의 note 가 그 근거를 적고 있다
(`65x85 m — fan scaled ~0.4x`, `220x240 m — real-scale Falcon fan`, `70x86 m, valid 0.43` 등).
BS/LJ-Grid1 각각에 대한 **개별 제외 사유는 코드·문서 어디에도 없음**(=`DOC-ONLY`).
관련 정황(메모리 `reference_bergamasco_stereo_dataset.md`): BS01 Hs 0.30/Tp 3.0 (짧은 wind-sea),
LJ01 Hs 10.0/Tp 13.3 (λ~277 m, **개구비<1**). 8개 파일은 `tmp/` 의 옛 파일럿 스크립트
(`tmp/bs_confirm.py`, `tmp/operability_curve.py`, `tmp/lj_forecast_check.py` 등)에서는 실제로 쓰였다
⟹ **읽히지 않는 데이터가 아니라, 이 하네스에 등록되지 않은 데이터**다.

--------------------------------------------------------------------------------
## 4. 부수 확인 (요청 외, 사실 기재)

- 오늘의 배포는 **미커밋 작업 트리 변경**: `configs/estimation.yaml` `lambda_l1: 0.001 → 0.0001`,
  `run.py` `DEFAULT_CONFIG['lambda_l1'] 0.001 → 0.0001` + yaml 폴백을 `DEFAULT_CONFIG` 참조로 변경.
  ⟹ BRIEF §0.1 의 "오늘 배포한 `lambda_l1` 1e-3→1e-4" **사실 확인**.
- `results.jsonl` = 2016행, 7셀 × 96에피소드 × 3시드, 결측 0. 셀 =
  `AUTO, AUTO-L1, AUTO-QTR, AUTO-QTR-L1, AUTO-NT, AUTO-SEED-NULL, AUTO-DUP`.
  에피소드 96 = 희소 64 + 혼잡 32.
