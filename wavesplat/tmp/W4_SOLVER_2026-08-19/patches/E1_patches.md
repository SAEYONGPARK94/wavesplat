# E1 — 코드 결함 패치 준비 (2026-08-19)

**상태: 준비 완료 · 적용 0회.** `wavesplat/**` · `wavesim/**` · `configs/**` 는 **한 글자도 수정하지 않았다.**
검증은 전부 **스크래치패드 복사본**(`/tmp/.../scratchpad/patched/`)에서 수행했다.

- 저장소 HEAD: `348d0f2` (2026-08-19)
- 표적 파일 md5 **작업 전=작업 후** `[M]`
  - `wavesplat/optimizer/sequential_update.py` = `63e6362b7c2f0e66e1a4ea3dd38ed075`
  - `wavesplat/run.py` = `e546e834a414571dfd7d80248d043331`
- 작업 중 추적 파일 변경 **0건**(`git status --porcelain -- wavesplat wavesim configs` = 미추적 `test_shipping_defaults.py` 한 줄뿐, 내가 만든 것 아님) `[M]`
- `results.jsonl` **열람 0회**(사전노출 회피). 결과 파일 스캔은 **W4 배치 디렉터리를 제외**하고 수행.

---

## ⛔ 적용하면 안 되는 시점 — 지금이다

**2026-08-19 08:20 에 시작된 W4 배치 1,344런이 실행 중이다** `[M]`
(`pid 50916 tmp/W4_SOLVER_2026-08-19/run_w4.py run`, 워커 8개).
그 하네스는 **행마다 임포트 경로 파일의 md5 를 기록**하고 게이트 `G-8` 이 그것을 검사한다.

⟹ **런이 전부 끝나고 `results.jsonl` 이 동결된 뒤에, 사용자 승인 아래에만 적용할 것.**
런 도중 적용하면 (a) 코퍼스가 두 코드 버전에 걸쳐 오염되고 (b) `G-8` 이 실패하며
(c) S13/S13b 에서 이미 대가를 치른 **"더러운 작업트리"** 사고가 세 번째로 재발한다.

적용 절차(런 종료 후):
```bash
cd /home/vexit/wavesplat                       # ← 저장소 루트 (diff 경로 기준)
git apply --check wavesplat/tmp/W4_SOLVER_2026-08-19/patches/atan2_sign.diff
git apply --check wavesplat/tmp/W4_SOLVER_2026-08-19/patches/rmse_near.diff
git apply         wavesplat/tmp/W4_SOLVER_2026-08-19/patches/atan2_sign.diff
git apply         wavesplat/tmp/W4_SOLVER_2026-08-19/patches/rmse_near.diff
cp wavesplat/tmp/W4_SOLVER_2026-08-19/patches/test_phase_convention.py \
   wavesplat/wavesplat/tests/test_phase_convention.py      # 상시 스위트에 편입
cd wavesplat && python3 -m pytest wavesim/tests wavesplat/tests -q   # 204 → 207 기대
```
**두 diff 는 이미 `git apply --check` 를 통과했다**(드라이런, 쓰기 0) `[M]`.

---

## 패치 1 — `amplitude_refit` 의 `(a,b) → φ` 부호 결함

### 1.1 결함 `[M]`

| 항목 | 위치 |
|---|---|
| 결함 라인 | `wavesplat/optimizer/sequential_update.py:1335` `phi_refit = np.arctan2(b, a)` |
| 동반 오류 독스트링 | `wavesplat/optimizer/sequential_update.py:1268` `phi_i = atan2(b_i, a_i)` |
| 전방모델 | `wavesplat/model/spectral_gaussian.py:6,124-127` `z = Σ wᵢ·cos(kxᵢx + kyᵢy − ωᵢt + φᵢ)` |
| 옳은 규약의 근거 | `w·cos(θ+φ) = (w cosφ)·cosθ − (w sinφ)·sinθ` ⟹ `a = w cosφ`, **`b = −w sinφ`** ⟹ **`φ = atan2(−b, a)`** |

### 1.2 ★"다른 호출부를 깨지 않는가" — `atan2` 전수 `[M]`

패키지 내 `arctan2/atan2` **14개 site 를 전수 열람**했다. LS `(a,b)→φ` 변환은 **6곳**이고,
**5곳이 이미 `atan2(−b, a)`**, 결함 site 만 홀로 반대였다:

| site | 식 | 판정 |
|---|---|---|
| `model/tnse.py:216` | `np.arctan2(-b, a)` | ✅ 옳음 (독스트링 `:183` 동일) |
| `optimizer/initializer.py:448` | `np.arctan2(-b, a)` | ✅ 옳음 (독스트링 `:422`) |
| `optimizer/relax_estimator.py:68` | `np.arctan2(-b, a)` | ✅ 옳음 |
| `optimizer/cs_initializer.py:160` | `np.arctan2(-c_sin, c_cos)` | ✅ 옳음 (독스트링 `:21`) |
| `optimizer/deep_relax.py:232` | `torch.atan2(-b, a)` | ✅ 옳음 |
| **`optimizer/sequential_update.py:1335`** | **`np.arctan2(b, a)`** | ⛔ **유일한 이탈** |

나머지 site 는 **방위각** `atan2(ky, kx)` 로 `(a,b)` 규약과 무관하다
(`wavesim/hos_simulator.py:345`, `wavesim/wave_generator.py:39`, `wavesim/tests/test_wave_generator.py:293-294`).

⟹ **diff 는 함수 지역 변수 `phi_refit` 한 개에만 닿고**, 그 값의 유일한 소비처는
`:1356-1361` 의 `self.model.phi.copy_(...)` 다. `w_refit`(`:1334`)·`residual_before`(`:1353`)·
`residual_after`(`:1354`)는 **전부 무관**. **다른 호출부 0곳.** `[M]`

### 1.3 ★결함이 지금까지 조용했던 이유 `[M]`

`residual_after` 는 `A @ coeffs`(= cos/sin 계수 직접)로 계산된다(`:1354`).
계수는 **옳고**, 틀린 것은 그 계수를 `(w, φ)` 로 바꾸는 마지막 한 줄뿐이다.
⟹ **함수 자신의 보고서는 부호에 완전히 눈이 멀어 있다.** 실측:

| | `residual_after` (함수 보고) | `(w,φ)` 왕복 잔차 | 신호 rms |
|---|---|---|---|
| 현행(`atan2(b,a)`) | **5.554e-16** | **0.824359** | 0.698327 |
| 패치(`atan2(-b,a)`) | **5.554e-16** | **5.25e-16** | 0.698327 |

★**틀린 왕복 잔차 0.8244 > 신호 rms 0.6983 (1.18배)** = **아무것도 예측하지 않는 것보다 나쁘다.**
그리고 `residual_after` 는 **두 경우에 비트 동일**하다 — 왜 이 결함이 잡히지 않았는지의 전부다.

### 1.4 ★과거 오염 재검증 — **독립 재수행, 전건 재현, 오염 0건** `[M]`

세 팀의 전수 확인을 **내가 처음부터 다시 돌렸다**. 결과는 뒤집힌 항목 **0개**.

| # | 점검 | 명령/증거 | 결과 |
|---|---|---|---|
| V1 | 유일 실행 게이트 | `run.py:398 if cfg.get('amplitude_refit', False):` | ✅ **기본 False** |
| V2 | CLI 노출 | `grep add_argument run.py \| grep refit` | ✅ **0건** |
| V3 | YAML 노출 | `grep -rn refit configs/` (루트·패키지 양쪽) | ✅ **0건** |
| V4 | `DEFAULT_CONFIG` | `grep -n refit run.py` → 396–403, 482 뿐 | ✅ **키 없음** |
| V5 | 작업트리에서 True 설정 | 저장소 전수 `grep -rnI amplitude_refit` × `true\|=1` | ✅ **0곳** (걸린 4줄은 전부 오늘의 W4 감사 문서 자기언급) |
| V6 | `git log -S` 7패턴 (코드 경로 한정) | `-- wavesplat wavesim scripts configs` | ✅ **전부 0 커밋** |
| V6b | `git log -G"amplitude_refit['\"]?[]:= ]+True"` | 코드 경로 한정 | ✅ **0 커밋** |
| V6c | 문자열을 건드린 **모든** 커밋 | `874419a`(2026-04-29 도입) · `e4209ee`(07-27) · `5a9841a`(07-29, bank 조기반환 추가) | ✅ **셋 다 활성화하지 않는다** |
| V7 | `results/**/*.json` 624개 | 파이썬 재귀 스캔 | ✅ **비어있지 않은 `refit` 0건**(빈 dict 조차 0건) |
| V9 | `tmp/**/*.json*` 523개 (**W4 제외**) | 동일 | ✅ **`"refit"` 키를 가진 파일 0개** |
| V8 | 테스트 커버리지 | `tests/test_bank_mmse_port.py:146 assert upd.amplitude_refit([new]) == {}` | ✅ **bank 조기반환(`:1281`) 경로뿐 — 수학에 도달 0회** |
| V11 | 전체 스위트 | 현행 트리 **204 passed**, 패치 사본 **204 passed** | ✅ **양쪽 동일 ⟹ 기존 스위트의 이 결함 커버리지 = 0** |

⚠**부재증명의 한계** `[A]`: 커밋되지 않은 REPL/일회성 세션이 `config={'amplitude_refit': True}` 로
돌렸을 가능성은 원리적으로 배제 못 한다. 다만 그런 세션은 **결과 파일도 남기지 않았다**(V7·V9).
**저장소에 기록된 어떤 수치도 이 결함을 통과하지 않았다.**

⟹ ★**"하나라도 뒤집히면 그게 최우선 산출" — 뒤집힌 것은 없다.** 이 패치는 위생 수리이지 정정이 아니다.

### 1.5 diff — `patches/atan2_sign.diff`

**1글자**(`b` → `-b`) + 독스트링 4줄. 실질 변경 라인 **1개**.
적용 후 위치: `:1338`(코드), `:1268`(독스트링).

### 1.6 영향 범위 `[M]`

- **실행 경로**: `run.py:398` 옵트인 하나뿐. 기본 OFF ⟹ **배포 동작 변화 0**.
- **과거 결과**: 재계산·재해석 대상 **0건**(V7·V9).
- **API/시그니처**: 불변. 반환 dict 키 불변.
- **다른 추정기**: 무관(1.2절 전수).
- ⚠**켜면 무엇이 달라지나** `[A]`: 이 함수는 여태 **켜면 손해**였다(왕복 잔차 > 신호 rms).
  수리 후에는 k 고정 하 **벌점 없는 닫힌형 `(w,φ)` 해**가 된다.
  ⛔단 **W2 의 `PROBE-A` 가 이 축을 이미 널로 측정했다**(추정기 자신의 k 위에서 +0.3%, 6/16).
  **버그를 고치는 것과 축을 되살리는 것은 별개다** — 수리를 근거로 축을 재개하지 마라.

---

## 패치 2 — `RMSE_near` 가 계산되는데 저장되지 않는다

### 2.1 소실 지점 — **`run.py` 가 이미 버린다** `[M]`

배치가 보고한 대로 **하류 스크립트만의 문제가 아니다.** 경로를 전부 열었다:

| 단계 | 위치 | 상태 |
|---|---|---|
| 생산 | `evaluation/metrics.py:433` `metrics['RMSE_near'] = rmse_n` (`rmse_spatial_split`, split 80 m, `:97-122`) | ✅ **매 GT 프레임 채워짐** |
| 프레임 반입 | `run.py:282 compute_all_metrics(...)` → `run.py:380 frame_results.append(frame_metrics)` | ✅ **`frame_results` 안에 살아 있다** |
| ⛔**1차 소실** | `run.py:424-433` 집계 블록이 `elapsed_ms`·`RMSE_z`·`nRMSE`·`Hs_error` **넷만** 평균낸다 | ⛔ **`avg_RMSE_near` 없음** |
| ⛔**2차 소실** | `run.py:626-627` `save_result = {k: v ... if k not in ('frame_results', 'spectrum_params')}` | ⛔ **`frame_results` 통째로 삭제 후 `json.dump`** |
| 하류(별건) | `scripts/run_b1_full_comparison.py:51-54 _metrics_keys()` 에 `'RMSE_far'` 는 있고 `'RMSE_near'` 는 없음 | ⛔ 별도 1토큰 수리 필요 |

⟹ **저장된 결과 파일은 구조적으로 근거리에 눈이 멀어 있었다.**
CLAUDE.md 의 *"근거리가 설계상 검출 불가"* 는 하류 스크립트가 아니라 **`run.py` 에서 이미 결정된다.**
`metrics.py:433` 은 **46일간** 값을 채우고 있었고 **아무도 그것을 꺼내지 않았다.**

### 2.2 수리 — 한 줄 `[M]`

`frame_results` 에 살아 있으므로 **집계 한 줄이면 회수된다**. 위치는 summary dict 안,
`'avg_Hs_error'` 바로 뒤(적용 후 `run.py:478`). 기존 `avg_obs_loss`(`:475-477`) 와 **같은 인라인 패턴**:

```python
'avg_RMSE_near': (float(np.mean([r['RMSE_near'] for r in frame_results]))
                  if (frame_results and has_gt) else float('nan')),
```

- `has_gt` 가드 = 형제 `avg_rmse`·`avg_nrmse` 와 **정확히 같은 도달성**(`run.py:426`).
- summary **최상위 키**이므로 `:635` 의 `frame_results` 삭제를 **통과한다**. ★이게 핵심이다 —
  `frame_results` 를 살리는 방식으로 고치면 결과 파일 크기가 폭증하고 기존 소비자를 흔든다.
- ⛔`RMSE_far` 는 이번 diff에 **넣지 않았다**(지시 범위는 near). 필요하면 동일 형태 한 줄.

### 2.3 검증 `[M]`

- `compute_all_metrics` **직접 호출** → `'RMSE_near' in metrics` = **True** (값 0.014103).
- 추가된 식을 합성 `frame_results` 5프레임에 평가 → 프레임 평균과 **정확히 일치**.
- 가드 3분기(`frames=[]`, `has_gt=False`, 둘 다) → **전부 `nan`**, 예외 없음.
- 패치 사본의 `wavesplat.run` **임포트 성공** + `avg_RMSE_near` 가 `run_estimation` 소스에 존재.
- 스키마 소비자 조사: `summary.keys()`·`result.keys()`·`fieldnames` 를 쓰는 코드 **0건**
  ⟹ **키 추가는 순수 가산(additive), 기존 소비자 무해.** `[M]`
- 전체 스위트 **204 passed**(패치 사본).
- ⚠**미측정** `[A]`: 실제 에피소드 end-to-end 런은 **하지 않았다**(배치 CPU 보호).
  근거는 소스 경로 열람 + 위 단위 검증이다. 적용 후 첫 런에서 `avg_RMSE_near` 키 존재를 확인할 것.

### 2.4 선택적 동반 수리 (이번 diff 에 **미포함**) `[M]`

`scripts/run_b1_full_comparison.py:52` 의 리스트에 `'RMSE_near'` **한 토큰** 추가.
⛔단 그 파일로 근거리 주장을 하려면 **`WaveSplat` 120–220 프레임 vs `ETKF`/`GPR-1` 301 프레임의
프레임 불일치**(CLAUDE.md 인용금지 사유 1)를 먼저 해결해야 한다. **키만 고쳐도 그 파일은 여전히 인용 불가.**

---

## 회귀 테스트 — `patches/test_phase_convention.py`

### 3.1 설계

저장소 규율 **"테스트가 배포 기본값을 전혀 지키지 않는다"**(과거 BLOCKER)를 정면으로 겨냥했다.
이 테스트는 **실제 배포 모듈·실제 클래스·실제 public 메서드**를 호출한다. 재구현 0줄.

- 알려진 `(A, φ)` 4성분(네 사분면 전부, `0/±π/2/±π` 회피 — 그 점에서는 두 규약이 일치한다),
  5프레임 × 64관측, **무잡음**. 관측은 **모델 자신의 float32 k/ω 로**, `amplitude_refit:1310-1313` 과
  **같은 연산 순서**로 생성한다(⚠float64 로 승격하면 `ω*t` 의 float32 반올림만으로 잔차 5e-9 이 남아
  1e-12 문턱을 삼킨다 — 그러면 테스트는 부호가 아니라 numpy 캐스팅을 재게 된다).
- `np.arctan2` 를 **스파이로 감싸** 모듈이 실제로 계산한 float64 `φ` 를 `:1356-1361` 의
  float32 절단 **전에** 읽는다 ⟹ 진짜 **machine-zero(≤1e-12)** 단언이 가능해진다.
- 3개 테스트:
  1. `test_refit_is_an_exact_least_squares_fit` — **전제조건**. `residual_after ≤ 1e-12`.
     ⚠**두 규약 모두 통과한다**(부호맹). 회귀 테스트가 아니라, 케이스가 무잡음·양호조건임을
     증명해 아래 실패를 **부호에만** 귀속시키기 위한 것이다.
  2. `test_phase_convention_roundtrip_is_machine_zero` — ★**본 회귀 테스트**.
     (a) 모듈의 float64 `(w,φ)` 왕복 **≤1e-12**, (b) 모델에 실제로 기록된 float32 상태 왕복 **≤1e-6**
     (float32 저장 바닥), (c) 진짜 `(A,φ)` 복원 오차 **≤1e-5**(자기일관적 가짜해 배제).
  3. `test_wrong_convention_is_detected` — ★**음성 대조**. 같은 훅으로 규약을 되뒤집고
     왕복 잔차가 **신호 rms 를 넘는지** 단언. 이게 없으면 "그냥 통과하는 테스트"와 구별되지 않는다.

### 3.2 실행 결과 `[M]` — ⛔실행하지 않은 것을 통과라 쓰지 않았다

```
현행(미패치) 트리   /home/vexit/wavesplat/wavesplat
  → 1 passed, 2 failed
    ✅ test_refit_is_an_exact_least_squares_fit      (residual_after 5.554e-16 — 부호맹)
    ❌ test_phase_convention_roundtrip_is_machine_zero
         "(w, phi) round-trip residual 8.244e-01 > 1e-12 (signal rms 0.698)"
    ❌ test_wrong_convention_is_detected
         되뒤집으면 5.251e-16 = 옳은 규약 ⟹ 음성 대조가 정확히 반대 부호를 지목

패치 사본        /tmp/.../scratchpad/patched
  → 3 passed in 1.20s
전체 스위트      패치 사본 204 passed (83.6 s) · 현행 트리 204 passed (79.1 s)
```

⟹ ★**대칭이 완전하다**: 미패치에서 정확히 실패하고, 패치에서 정확히 통과하며,
음성 대조가 양쪽에서 부호를 정확히 반대로 지목한다. **204/204 가 양쪽에서 같다는 것은
기존 스위트에 이 결함의 커버리지가 0 이었다는 직접 증거다.**

### 3.3 배치 방침

지금은 `tmp/W4_SOLVER_2026-08-19/patches/` 에 있고 `pytest.ini` 기본 수집 범위 밖이다
(= **현재 204 카운트를 흔들지 않는다**). 패치 적용 시 `wavesplat/tests/test_phase_convention.py`
로 옮겨 상시 스위트에 편입할 것(204 → 207).

---

## MEASURED / ARGUED 요약

| 주장 | 라벨 |
|---|---|
| `:1335` 는 전방모델과 부호가 어긋난다 | `[M]` (왕복 잔차 0.824 vs 5.25e-16) |
| 틀린 잔차가 신호 rms 를 넘는다 | `[M]` (0.8244 > 0.6983) |
| `residual_after` 는 부호에 눈이 멀었다 | `[M]` (양쪽 5.554e-16, 비트 동일) |
| 나머지 5개 LS site 는 이미 옳다 | `[M]` (14 site 전수 열람) |
| 과거 결과 오염 0건 | `[M]` (V1–V9, 결과 파일 1,147개 스캔) — 한계는 `[A]` (§1.4) |
| diff 가 다른 것을 깨지 않는다 | `[M]` (204 passed × 2, `git apply --check` 통과) |
| `RMSE_near` 는 `run.py` 에서 이미 소실된다 | `[M]` (`:424-433` 미집계 + `:626-627` 삭제) |
| 한 줄로 회수된다 | `[M]` (단위 검증) / e2e 런 확인은 `[A]` — 적용 후 확인 필요 |
| 수리해도 `amplitude_refit` 축은 여전히 널 | `[A]` (W2 `PROBE-A` 인용) |
