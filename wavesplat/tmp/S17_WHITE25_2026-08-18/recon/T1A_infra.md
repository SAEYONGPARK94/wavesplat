# T1-A · 인프라 정찰 (런 0회)

**작성** 2026-08-18 · **에이전트** T1-A · **범위** = 아래 "열람 목록" 에 있는 파일만.
**런 0회 준수**: 추정기·필터 `update()` 호출 0회, 에피소드 로드 0회, `.h5` 접근 0회.
단 §2.3 의 배경 앙상블 검증은 **객체 생성만** 한다(에피소드·필터 사이클 없음) — 이는
`run_s16.py:313-315` 가 스스로 *"RUN-FREE, objects only"* 라 라벨한 것과 **동일한 부류**이며
그 사실을 표기한다.

## 열람 목록 (주장 범위 = 이것뿐)
| 파일 | 범위 |
|---|---|
| `/home/vexit/wavesplat/wavesplat/tmp/S16_BASELINE_2026-08-14/run_s16.py` | 전체 653줄 |
| `…/S16_BASELINE_2026-08-14/letkf_spec.py` | 전체 284줄 |
| `…/S16_BASELINE_2026-08-14/analyze_s16.py` | 전체 |
| `…/S16_BASELINE_2026-08-14/RESULT.md` · `NOTES.md`(발췌) · `AMENDMENT_1.md`(A-1~A-9) · `PREREG.md` §2·§3·§4·§7·§8·§9·§10·§11 | 해당 절 |
| `…/S16_BASELINE_2026-08-14/results.jsonl` | 1,920행 **집계만**(팔별 `wall_s`·시드 산포·스키마) |
| `…/S16_BASELINE_2026-08-14/{FROZEN_md5.txt, SCRIPTS_md5.txt, code_fingerprint.json, smoke2_report.json}` | 전체 |
| `/home/vexit/wavesplat/wavesplat/wavesplat/baselines/letkf_grid.py` | 전체 269줄 |

---

# 1. `WHITE25` 의 정확한 정의

## 1.1 답: `ARMS` dict 한 줄 + **러너 한 줄**(`white_amp` 전달)이면 된다 — 단 `kind='spec'` 이어야 한다

지시서가 제시한 후보 `dict(kind='spec', background='white', loc=25.0)` 는 **`kind`/`background`/`loc` 세
키에 관해서는 맞다**. `white_amp` 를 안 적으면 `LETKFGridSpec.__init__` 기본값 `0.1`
(`letkf_spec.py:130`)이 그대로 들어가므로 **`WHITER50` 의 loc 만 25 로 바꾼 팔**이 되는데,
`RESULT.md:200` 의 이월 2("백색 × loc 25")가 요구하는 것이 정확히 그것이다. ⟹ **`ARMS` 한 줄로 충분.**

```python
'WHITER25': dict(kind='spec', background='white', loc=25.0),   # white_amp 는 기본 0.1
```

⚠**이름 권고**: S16 의 명명 규약은 `WHITE*` = **재고** `LETKFGrid`(`ARMS['WHITE50']`
= `run_s16.py:70`), `WHITER*` = **서브클래스** `LETKFGridSpec` 의 백색 대조(`run_s16.py:69`)다.
`RESULT.md:200` 이 쓴 이름 `WHITE25` 를 그대로 쓰면 **재고 클래스로 오독**된다.
⟹ **`WHITER25` 로 명명하고 RESULT 의 별칭을 각주로 달 것**(MEASURED: `run_s16.py:65-72`).

## 1.2 `kind='spec'` vs `kind='stock'` 이 실제로 갈아끼우는 것 (코드 라인)

분기는 **`run_s16.py:206-214` 단 한 곳**이다.

| | `kind='stock'` (`run_s16.py:207-208`) | `kind='spec'` (`run_s16.py:210-214`) |
|---|---|---|
| 클래스 | `wavesplat.baselines.letkf_grid.LETKFGrid` | `letkf_spec.LETKFGridSpec` (배치 디렉터리, 패키지 무수정) |
| 초기 배경 | `letkf_grid.py:88` `np.random.randn(Ne, n_state) * 0.1` — **전역** 스트림 | `letkf_spec.py:156` `self.ensemble = self._draw_background(Ne, centre=True)` — **전용** `default_rng(filter_seed)` (`letkf_spec.py:140`) |
| 전역 RNG | 배경 draw + 가산팽창 + 관측 thinning 이 **한 스트림** | `letkf_spec.py:144-146` 이 base `__init__` 앞뒤로 전역 상태를 **save/restore** ⟹ 전역은 `letkf_grid.py:120 np.random.choice`(thinning) **만** 소비 |
| 가산팽창 | `letkf_grid.py:164-165` `X_pert += np.random.randn(...)*add_inflation` — 예보 **후**, 전역 스트림 | `letkf_spec.py:152-153` 이 base `add_inflation=0` 으로 죽이고, `letkf_spec.py:214-219 update()` 가 **예보 한 스텝 전에** 같은 배경모형에서 뽑아 더한다 |
| 중심화 | 없음 | `letkf_spec.py:209-210` 앙상블 평균 제거 |
| `background_diagnostics()` | **없음** ⟹ `run_s16.py:215` 이 `background_diag=None` 기록 (MEASURED: `WHITE50` 384행 전부 `None`) | 있음 ⟹ G-7 등방성 게이트 대상 |
| `effective_config` | `run_s16.py:282-292 _stock_readback` (하드코딩 `white_amp:0.1`) | `letkf_spec.py:276-283 readback` (**객체 되읽기**, 규율 4) |

⟹ `kind` 를 갈면 **세 가지가 동시에** 바뀐다: ①배경모형 ②RNG 스트림 소유권(= thinning 부분집합
동일성) ③가산팽창 적용 지점. **②+③ 의 총 크기는 S16 이 이미 쟀다**: `WHITER50 vs WHITE50`
= **37/64, med(rel) −0.1%** (`RESULT.md` E-3, 사전등록된 측정).
⟹ **`WHITER25` 는 반드시 `kind='spec'`** 이어야 `SPEC25`·`WHITER50` 와 **thinning 비트 동일**이
되어 `loc` 이 유일 요인이 된다(`PREREG.md:107-110`).

---

# 2. ★`WHITEHS50` — **`WHITER50` 과 중복이 아니다**

## 2.1 결론 (MEASURED)

> **중복 아님.** `WHITER50` 은 `white_amp = 0.1 m`, `WHITEHS50` 은 `white_amp = Hs/4`
> (**BF4 0.30 m · BF5 0.50 m**) ⟹ 배경 스프레드가 **3배 / 5배** 다르다.
> ⟹ **이 배치의 절반은 살아 있다.**

지시서가 인용한 *"`WHITER50` 대비 교락 0.1%"* 는 **다른 양**이다. 그 0.1% 는
`WHITER50 vs WHITE50` = **서브클래스 코드경로 대 재고 클래스**(§1.2 의 ②+③)의 크기이지
(`RESULT.md` E-3), **스프레드 축이 아니다**. `WHITEHS50` 은 그 축 위에 없다.

## 2.2 코드 경로 — `white_amp` 가 스프레드를 결정하는 지점

`letkf_spec.py:182-194`:
```
def _draw_background(self, n, centre=True, target_std=None):
    if self.background == 'white':
        amp = self.white_amp if target_std is None else target_std     # :193
        X = self._rng.standard_normal((n, self.n_state)) * amp          # :194
    else:
        tgt = (self.Hs_bg / 4.0) if target_std is None else target_std  # :196
```
- **초기 앙상블**(`letkf_spec.py:156`)은 `target_std=None` 으로 호출 ⟹ 백색은 `white_amp`,
  JONSWAP 은 `Hs_bg/4` 가 스프레드가 된다. **여기가 두 팔의 유일한 갈림길이다.**
- **가산 팽창**(`letkf_spec.py:217-218`)은 `target_std=self._add_inf` 를 **명시**하므로
  `white_amp` 와 **무관**하다 ⟹ `WHITER50` 과 `WHITEHS50` 의 팽창 항은 **동일**하다.
- `white_amp` 는 생성자 kwarg(`letkf_spec.py:130`)이고 `READBACK_KEYS`(`letkf_spec.py:270`)에
  **이미 들어 있다** ⟹ `effective_config` 에 자동 기록된다(감사 가능). MEASURED: S16 384행의
  `WHITER50.effective_config.white_amp == 0.1`.

## 2.3 객체 검증 (에피소드 0 · 필터 사이클 0 · `run_s16.py:313` 과 같은 부류)

`LETKFGridSpec(egx=40, egy=30, ne=100, depth=50, add_inflation=0.005)` 를 생성만 해서
`background_diagnostics()` 를 읽었다:

| 구성 | `std(η)` | `std(η̇)/std(η)` | 1-lag_x | 등방성 | `_add_inf` |
|---|---|---|---|---|---|
| `WHITER50` (`white_amp=0.1`) | **0.09957** | 1.0009 | +0.000 | PASS | 0.005 |
| `WHITEHS50-BF4` (`white_amp=0.3`) | **0.29870** | 1.0009 | +0.000 | PASS | 0.005 |
| `WHITEHS50-BF5` (`white_amp=0.5`) | **0.49784** | 1.0009 | +0.000 | PASS | 0.005 |
| `SPEC50-BF4` (`Hs_bg=1.2`) | 0.29847 | **1.0794** | **+0.834** | PASS | 0.005 |

- `max|3·E(WHITER50) − E(WHITEHS50-BF4)| = 4.44e-16` ⟹ **두 백색 팔의 앙상블은
  부동소수 반올림까지 정확히 비례**(같은 `_rng` draw × 다른 amp). ⟹ **`WHITEHS50 vs WHITER50`
  은 진폭 단일요인의 교과서적 대조**다.
- 팽창 draw std = **0.004942 (양 팔 동일)** ⟹ 팽창 교락 0.
- `readback(WHITEHS50)['white_amp'] == 0.3`, `['background'] == 'white'`, `['Hs_bg'] == None`.
- S16 전량 실측과 정합(`RESULT.md` §1.1 / `NOTES.md:90`): `WHITER50` 0.09984 · `SPEC50` 0.29966.
  (여기 값이 소수 4째에서 다른 것은 `n_ensemble=100` 대 600 의 표본오차다.)

## 2.4 ⚠ "스프레드 정합" 의 정확한 범위 (RESULT 에 반드시 적을 것)

`WHITEHS50` 이 `SPEC50` 과 맞추는 것은 **η 의 스프레드뿐**이다. **MEASURED**:
- 백색은 η 와 η̇ 이 **독립**이고 `std(η̇)/std(η) = 1.0009`;
- `SPEC50` 은 분산관계 정합 교차공분산을 갖고 그 비가 **BF4 1.0794 · BF5 0.8786**(`NOTES.md:280-281`).
⟹ `SPEC50 vs WHITEHS50` 이 재는 "구조" 는 **{공간 상관길이, η–η̇ 분산관계 결합, η̇ 스프레드 비}**
3개 묶음이다. **단일요인이라 부르지 마라.** (`WHITEHS50 vs WHITER50` 은 진짜 단일요인이다.)

## 2.5 2×2 중 3칸 (4칸째는 없다)

| | 스프레드 0.1 | 스프레드 Hs/4 |
|---|---|---|
| **백색 구조** | `WHITER50` ✅S16 보유 | **`WHITEHS50`** ← 신규 |
| **JONSWAP 구조** | ❌ 없음 | `SPEC50` ✅S16 보유 |

⟹ 주효과 2개는 추정되지만 **상호작용은 추정 불가**.
4칸째(`SPECLO50`)를 원하면 **`Hs_bg=0.4` 를 넘기면 된다** — `_build_spectrum`(`letkf_spec.py:159-180`)
은 `Tp_bg` 만 쓰고 `Hs_bg` 는 **오직 `letkf_spec.py:196` 의 재척도 목표**로만 쓰이기 때문이다.
⛔단 그러면 `readback.Hs_bg = 0.4` 가 되어 **§4.2 오라클 선언과 어긋난 기록**이 남는다.
⟹ 필요하면 `letkf_spec.py` 를 **수정하지 말고** S17 디렉터리에 얇은 서브클래스
(`letkf_spec17.py`)를 새로 두고 `bg_std` 를 명시 kwarg 로 노출할 것. **선택 사항**(비용 +4.1 h).

---

# 3. 최소 diff 명세 (`run_s17.py`)

⛔**`wavesplat/**` 은 수정 대상이 아니다.** MEASURED: `letkf_spec.py` 는 패키지 밖 배치 파일이고
`LETKFGrid` 는 상속만 한다(`letkf_spec.py:81,122`). S17 도 같은 규율을 지킨다.
⛔**`letkf_spec.py` 자체도 수정하지 마라** — 수정하면 §3.5 의 비트 재현 앵커가 무의미해진다.

## 3.1 경로 (`run_s16.py:31-46`)
| 상수 | S16 | S17 |
|---|---|---|
| `HERE` | 자동 | 자동 (`tmp/S17_WHITE25_2026-08-18`) |
| `OUT` | `HERE/results.jsonl` | 동일 (**신규 파일**) |
| `SMOKE_OUT/SMOKE_REPORT/FP_PATH` | 동일 패턴 | 동일 패턴 |
| — | — | ★**신규** `S16 = ROOT/tmp/S16_BASELINE_2026-08-14/results.jsonl` (열 수입용) |
| — | — | ★**신규** `S16DIR` 을 `sys.path` 에 넣어 `import letkf_spec` (복사 금지) |
| `S13`/`S13B` | `run_s16.py:45-46` | **그대로**(`SHIP` 열) |

## 3.2 팔 (`run_s16.py:65-72`)
```python
ARMS = {
    'WHITER25':  dict(kind='spec', background='white', loc=25.0),                 # 이월 2
    'WHITEHS50': dict(kind='spec', background='white', loc=50.0, white_amp='hs4'),# 이월 1
}
ARM_ORDER = ('WHITEHS50', 'WHITER25')   # 순서 = 답이 급한 질문 먼저(§3.7)
```
S16 의 5팔(`SPEC25/50/100`,`WHITER50`,`WHITE50`)은 **재실행 0회, 열로 수입**한다(§3.4).

## 3.3 `run_letkf` — **바뀌는 줄은 2개뿐** (`run_s16.py:209-214`, `:265-279`)
```python
# (a) 생성자에 white_amp 전달 — 209-214 블록
wa = spec.get('white_amp', 0.1)
wa = float(loader.Hs_true) / 4.0 if wa == 'hs4' else float(wa)
f = letkf_spec.LETKFGridSpec(egx, egy, background=spec['background'],
        Hs_bg=float(loader.Hs_true), Tp_bg=float(loader.Tp_true),
        white_amp=wa, filter_seed=filter_seed, depth=loader.depth,
        n_ensemble=N_ENSEMBLE, mult_inflation=1.0, add_inflation=0.005,
        loc_radius=spec['loc'])
# (b) 반환 dict 에 최상위 필드 추가 — 265-279 블록
                white_amp=float(f.white_amp), bg_std_target=float(wa),
```
⚠`Hs_bg`/`Tp_bg` 를 백색 팔에도 넘기는 것은 **무해**하다(`letkf_spec.py:167-169` 가 비-JONSWAP 이면
즉시 반환) 그리고 `background_diagnostics.eta_target` 이 채워져 **G-7 이 `std(η) ≈ Hs/4` 를 직접
검사**할 수 있게 된다. ⛔단 `readback.Hs_bg` 가 백색 행에도 찍히므로 **RESULT 에 "백색 팔의
`Hs_bg` 는 스펙트럼에 쓰이지 않는다" 를 명기**할 것. (넘기지 않는 선택도 가능 — 그 경우
`eta_target=None`, 대신 새 최상위 필드 `bg_std_target` 으로 검사.)

## 3.4 필드 게이트 (`run_s16.py:76-83`)
```python
REQUIRED_LETKF = (...S16 의 28개 그대로..., 'white_amp', 'bg_std_target')
REQUIRED_SHIP  = 그대로
```
MEASURED: S16 행의 최상위 키 33개 확인(`episode … zero_shell`), 셸 키는 전부 **`obs:` 접두어**
(`obs:0_5 … obs:150_inf` 9개). ⛔`ship:` 고리는 S17 에서도 재지 않는다(규율 9 / `PREREG.md:126-128`).

## 3.5 G-5 재현 앵커 — ★**여기가 S17 설계의 급소**
S16 의 G-5 는 ①`SHIP` 재실행 vs S13 ②`WHITE50` vs S13 `LETKF_PLUS` 였다(`PREREG.md:245`).
S17 은 **S16 의 5개 열을 수입**하므로 **그 열들에 대한 앵커가 새로 필요하다**:

| | 앵커 | 비용 | 근거 |
|---|---|---|---|
| **G-5a′** | `SHIP@42` 2에피소드 재실행 vs S13 저장값 `\|Δ\|=0` | ≈40 s | S16 실측 `\|Δ\|=0.0` (`smoke2_report.json.g5_ship`) |
| **G-5b′** ★ | **`WHITER50@42` 2에피소드 재실행 vs S16 `results.jsonl` 저장값 `\|Δ\|=0`** | ≈76 s | S16 5열 수입을 정당화하는 **유일한** 앵커. 같은 클래스·같은 배경·같은 코퍼스 |
| (선택) G-5c′ | 위를 128 에피소드로 확장 | 1.36 h | S16 이 G-5b 를 128/128 로 확장한 전례(`RESULT.md` §4) |
| **G-8′** ★ | 패키지 md5 동일성 | 0 s | **MEASURED(오늘)**: S16 `code_fingerprint.json` 의 **87 파일 전부 변경 0 · 결실 0 · 신규 repo `.py` 0** ⟹ 수입 열은 지금 코드와 동일 상태에서 나온 것 |

⚠**G-8 은 정의를 바꿔야 한다.** `run_s16.py:126-129` 가 지문에 `letkf_spec.py`+`run_s16.py` 를
포함하므로 S17 행의 `code_md5` 는 S16 행(`5bba1a48…`)과 **구조적으로 다를 수밖에 없다**.
⟹ 행에 **두 값**을 적을 것: `code_md5`(배치 내 유일성, S16 규칙 그대로) + ★`code_md5_repo`
(= `wavesplat/**`+`wavesim/**`+`configs/*` **85개만**의 digest). 후자를 S16 의
`code_fingerprint.json` 에서 같은 부분집합으로 재계산한 값과 **일치 검사**하면
"수입 열과 신규 열이 같은 패키지 상태에서 나왔다" 가 증명된다.
⚠추가로 `letkf_spec.py` 의 md5 를 **`7991d24a8a4e31580d888ddf766ec1ca`**(`SCRIPTS_md5.txt`
19:38 항목)로 **하드 assert** 할 것 — 복사본을 두지 말고 S16 디렉터리에서 import.

## 3.6 나머지 게이트
| | 처리 |
|---|---|
| G-1/G-2/G-4 (표본·`SHIP` 수입·λ) | `run_s16.py:349-401` **그대로 재사용** — 코퍼스 128 동일 |
| **G-6′** 시드 민감도 | §5.2 참조. ⚠S16 의 8.216%/13.549% 는 **`SPEC50`** 값이다. 백색 팔은 다르다 |
| G-7 | 백색 팔은 `_sigma_k=None` ⟹ `omega_rms_spectrum=nan`·`n_modes_nonzero=0` 이 **정상**. 검사할 것은 ①`std(η) ≈ white_amp`(±5%) ②앙상블 평균≈0 ③**1-lag ≈ 0**(백색이므로 **>0.5 가 아니라 <0.05 를 요구**) ④등방성 `\|corr\| ≤ 0.15` |
| **G-9′** ★공짜 강화 | S16 은 팔 간 `zero_nRMSE` 일치만 봤다. S17 은 **신규 행의 `zero_nRMSE`·`zero_shell` 이 S16 같은 에피소드 값과 `<1e-12`** 인지 **파일 간**으로 검사할 수 있다 — 코퍼스·프레임마스크 동일성의 무료 증명 |
| G-10 | 신규 `.h5` 0 · 시드 원장 **[1589–1818] 불변** — S16 과 동일 |

## 3.7 실행 순서
`main_run`(`run_s16.py:596`)의 seed-major 순서를 유지하되 `ARM_ORDER` 를 **`WHITEHS50` 먼저**로
둘 것: 그것이 S16 이 못 판정한 **§6.6(구조 vs 스프레드)** 을 4.1 h 시점에 열어준다.
⛔**단독 점유 필수**(`AMENDMENT_1` A-9: 동시 실행이 같은 팔을 11.7배 느리게 만들었다).

---

# 4. 분석기 — **수정 필요**(기존 코드로는 계산 불가)

## 4.1 `analyze_s16.py` 가 실제로 계산하는 것 (MEASURED, 전문 열람)
| | 대조 | 팔 | 셸 |
|---|---|---|---|
| comparator 선택 | `argmin_a #{SHIP < a}`, `POOL=('SPEC25','SPEC50','SPEC100')` (`:26`), 동점 규칙 = med(rel) 큰 쪽 → loc 작은 쪽 (`:206`) | — | 전역 |
| 보조 comparator | `POOL_ALL` 5팔 (`:27`) | — | 전역 |
| **R-1** | `r* < ZERO` + `losses_best ≤ losses_WHITE50` (`:222-232`) | `r*` | 전역 |
| **R-2** | **`SPEC50 < WHITER50` 고정**(`R2_ARM='SPEC50'`, `:38`, AMENDMENT_1 B2) | 2팔 | 전역 |
| PRIMARY | `SHIP < r*` + IUT `SHIP < ZERO` (`:245-252`) | | 전역 |
| SECONDARY | `S-1a r*<ZERO`, `S-1b SHIP<r*` @ **`obs:150_inf` 하나**(`:28`), **Holm-2** (`:104-112`, `:270`) | | 셸 |
| 헤드라인 규칙 | §8-5 > §8-4 ⟹ `obs:0_5…obs:75_100` **면적가중** (`:283-297`) | | 셸 |
| E-1 | 9셸 × `('SHIP', best, 'WHITER50', 'WHITE50', ZERO)` **하드코딩** (`:305-330`) | | 셸 |
| E-2/E-3/E-4/E-5 | loc 용량반응 · `best vs WHITER50` · `WHITER50 vs WHITE50` · 희소 거울 · 셀분해 | | 전역 |

**셸 접두어**: `R.SHELL_KEYS`(`run_s16.py:55-56`) = `obs:` 전용. `ship:` 은 **어디에도 없다**.
**median-of-3**: `base_med`(`analyze_s16.py:41-59`)가 **기준선 팔**을 필터시드 median 으로,
`R.ship_med3`(`run_s16.py:167-178`)가 **`SHIP`** 을 추정기시드 median 으로 접는다.
**스칼라와 9셸 전부** median 을 취한다. 시드 3개가 안 차면 `PROVISIONAL` 스탬프(`:190-195`).
**Holm**: `holm()`(`:104-112`) step-down, **SECONDARY 2개에만** 적용. R-1/R-2/PRIMARY 는 미보정.

## 4.2 S17 이 필요한 대조 — **하나도 기존 코드로 안 나온다**
`POOL`·`R2_ARM`·E-1 열이 전부 하드코딩이고, `load()`(`:62-77`)는 **`HERE/results.jsonl` 한 파일만**
읽는다. ⟹ **`analyze_s17.py` 신규 작성**이 필요하다. 단 **재사용 가능한 부품이 많다**:
`base_med` · `ship_med3` · `sign_one_sided` · `effects`(3규약) · `holm` · `report`(실현 n 의 정확 α 병기)
· `field_gate` · `keys_of/col/zcol` — 약 120줄. **로직을 다시 쓰지 말고 복사할 것**(규약 드리프트 방지).

바꿀 곳 정확히 3개:
1. `load()` : **두 jsonl 병합** — `raw[episode][arm][filter_seed]` 에 S16 5팔 + S17 2팔을 함께 적재.
2. 대조 목록 : 아래 표를 **런 전에 동결**.
3. G-8 : 파일별 `code_md5` 유일성 → **`code_md5_repo` 일치성**으로 교체(§3.5).

## 4.3 S17 이 물어야 할 대조 (제안 — T1-B/사전등록이 확정)
| | 대조 (혼잡 n=64, 짝 부호검정 단측, 문턱 41/64) | 요인 | S16 참조값 |
|---|---|---|---|
| **F-1 구조** | `SPEC50 < WHITEHS50` | JONSWAP 구조 (loc·η스프레드 정합) | — |
| **F-2 스프레드** | `WHITEHS50 < WHITER50` | 진폭 0.1→Hs/4 **단일요인**(§2.3 비례 증명) | — |
| **F-3 loc(백색)** | `WHITER25 < WHITER50` | `loc` **단일요인**(thinning 비트 동일) | — |
| **F-4 R-2@25** | `SPEC25 < WHITER25` | loc 25 에서의 loc-정합 R-2 | `SPEC50<WHITER50` = 28/64 +3.5% |
| **C-1 정합검사** | F-1⊕F-2 가 S16 R-2(**28/64, +3.5%**)를 재구성하는가 | — | ✅기록 존재 |
| **C-2 ★위험** | comparator 풀에 `WHITER25` 를 넣으면 `r*` 가 바뀌는가 | — | S16 `r*=SPEC25`, `SHIP<SPEC25` 59/64 |

⚠**C-2 를 반드시 사전등록하라.** `RESULT.md` §3.2 는 백색 팔에서 **loc 이 지배적**임을 이미
보였다(`SPEC25<WHITER50` 59/64 의 정체가 loc). `WHITER25` 가 `SPEC25` 보다 강한 기준선일
가능성이 실질적이고, 그러면 **PRIMARY 의 comparator 가 바뀌어 −24.4% 가 더 깎인다**.
결과를 본 뒤 풀을 정하면 `PREREG §11-1` 위반이다.

---

# 5. 비용 — **런당 38.2 s, 2팔 median-of-3 = 8.1 h**

## 5.1 `wall_s` 실측 (S16 `results.jsonl` 1,920행, 에러 0, **단독 점유**)
| 팔 | n | min | p25 | **median** | p75 | max | 합계 |
|---|---|---|---|---|---|---|---|
| `SPEC25` | 384 | 40.7 | 41.1 | 41.2 | 41.5 | 44.4 | 4.41 h |
| `SPEC50` | 384 | 40.9 | 41.2 | 41.4 | 41.6 | 44.4 | 4.42 h |
| `SPEC100` | 384 | 41.0 | 41.3 | 41.5 | 41.6 | 44.0 | 4.43 h |
| `WHITE50`(재고) | 384 | 39.4 | 39.7 | 39.8 | 39.9 | 46.8 | 4.26 h |
| ★`WHITER50`(백색·spec) | 384 | 37.7 | 38.0 | **38.2** | 38.3 | 82.2 | 4.10 h |
| **전체** | 1,920 | | | | | | **21.62 h** |

층(희소/혼잡)별 median 차이는 0.1 s 이하 ⟹ 층은 비용축이 아니다.
`loc` 은 비용에 무관하다(`letkf_grid.py:216` `gaspari_cohn(dists, loc_radius)` 는 반경과 무관하게
`(n_grid, M)` 전체를 계산) ⟹ **`WHITER25` 도 38.2 s** (INFERRED-from-code + S16 3개 loc 의 SPEC
median 이 41.2/41.4/41.5 로 사실상 동일한 것이 방증).

## 5.2 S17 예상
| 시나리오 | 런 | 벽시계 |
|---|---|---|
| **권고: 2팔 × 128ep × 3시드** | **768** | **8.15 h** (min 기준 8.05 h) |
| 2팔 × 128ep × **1시드** | 256 | **2.72 h** |
| 2팔 × **혼잡 64ep** × 3시드 | 384 | 4.07 h |
| 게이트+스모크 (G-5a′ 2 + G-5b′ 2 + G-6′ 2팔×2ep×3시드) | ~16 | **≈10 분** |

⚠`RESULT.md:199-200` 의 *"각 ≈1.5 h"* 는 **단일시드 128ep**(=1.36 h) 값이다. **G-6 이 발동한
median-of-3 체제에서는 팔당 4.07 h** 다. ⟹ **RESULT 의 이월 견적은 3배 과소.**

### ★median-of-3 가 필요한가 — 새 실측
S16 원자료에서 **에피소드별 필터시드 3개의 상대표준편차**를 직접 계산했다(n=128 에피소드):
| 팔 | median rel_sd | p90 | max |
|---|---|---|---|
| `SPEC25` | 0.844% | 2.16% | 3.93% |
| `SPEC50` | **4.495%** | 12.32% | 25.81% |
| `SPEC100` | **8.333%** | 27.25% | 42.44% |
| ★`WHITER50` | **0.247%** | 0.52% | **0.76%** |
| `WHITE50` | 0.267% | 0.49% | 1.01% |

⟹ **백색 배경은 시드에 거의 무감**하고 **G-6 문턱 2% 를 전 분위에서 밑돈다**.
`PREREG G-6` 문언대로면 **단일시드가 허용**된다(2.72 h, **5.4 h 절약**).
⛔**그럼에도 median-of-3 를 권고한다**: 비교 상대인 S16 의 `SPEC*`·`WHITER50` 열이 median-of-3 이고,
`RESULT.md` §6.2 가 **"한쪽만 median 인 비대칭은 우리에게 유리하다"** 를 이미 의무기재 항목으로
올려놨다. 5.4 h 를 아끼려고 그 비대칭을 되살리는 것은 나쁜 거래다.
⟹ **G-6′ 을 스모크에서 새 팔로 실측하고**(2 에피소드 × 3시드 × 2팔 ≈ 8분), 문턱을 밑돌더라도
**대칭성 사유로 median-of-3 를 사전등록**할 것.

## 5.3 생성·시드
**신규 `.h5` 0 · 시드 원장 소모 0** — S16 과 **동일한 128 에피소드**
(`data_s9_sparse_2026-08-11` 64 + `data_s8_crowded_2026-08-10`/`data_s9_crowded_2026-08-11` 64,
`run_s16.py:49-50`)를 쓴다. 원장 **[1589–1818] 불변**.

---

# 6. 요약 — MEASURED / INFERRED 분리

**MEASURED**
1. `WHITEHS50` ≠ `WHITER50`. 스프레드 0.1 vs Hs/4(0.30/0.50) = **3배/5배**, 앙상블은 정확히 비례
   (`max|3a−b| = 4.4e-16`). 지시서가 인용한 "교락 0.1%" 는 **코드경로 축**(`WHITER50 vs WHITE50`)이지
   스프레드 축이 아니다.
2. `white_amp` 는 이미 생성자 kwarg(`letkf_spec.py:130`)이자 `READBACK_KEYS`(`:270`) 항목이다
   ⟹ 팔 정의는 **dict 한 줄 + 러너 한 줄**.
3. `kind` 는 배경모형·RNG 스트림 소유권·팽창 적용지점 **3가지를 동시에** 갈아끼운다(`run_s16.py:206-214`).
   ⟹ 새 팔은 **반드시 `kind='spec'`**.
4. 패키지는 S16 이후 **무변경**(87 파일 md5 전건 일치, 신규 repo `.py` 0) ⟹ S16 5열 수입이 정당하다.
5. 런당 38.2 s. 2팔 median-of-3 = **8.15 h**. RESULT 의 "각 1.5 h" 는 단일시드 값.
6. 백색 팔 시드 rel_sd median **0.247%**(max 0.76%) ⟹ G-6 문턱 2% 미만.
7. `analyze_s16.py` 는 S17 대조를 **하나도** 계산하지 못한다(POOL·R2_ARM·E-1 하드코딩, 단일 파일 로드).

**INFERRED**
- `WHITER25` 의 런당 비용이 `WHITER50` 과 같다(loc 이 `gaspari_cohn` 비용에 무관, SPEC 3개 loc 의
  median 이 41.2/41.4/41.5 인 것이 방증) — 스모크 2런으로 확인 가능.
- `WHITER25` 가 `SPEC25` 보다 강한 기준선일 가능성이 실질적이다(`RESULT.md` §3.2 의 loc 지배).
  ⟹ comparator 풀 규칙을 **런 전에** 동결하지 않으면 §11-1 위반 위험.
