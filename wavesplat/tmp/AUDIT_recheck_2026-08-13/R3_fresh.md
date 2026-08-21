# R3 — 신선 3라운드 감사 (2026-08-13)

**규율**: READ-ONLY (이 파일만 작성). 하위 에이전트 0회. 무거운 실험 0회(GPU 미사용).
`tmp/S13b_seed_robust_2026-08-13/` 미접근.

앞 라운드(A1–A7, V1–V3, FIX1–FIX2)는 **머리말 40행씩만** 읽어 "무엇을 이미 봤는지" 확인용으로만 썼다.
아래 판정은 전부 내가 직접 연 파일·직접 돌린 재도출에서 나온다.

---

## 열람 범위 (= 주장 범위)

**코드 (전문 또는 명시 구간)**
- `wavesplat/evaluation/metrics.py` (전문 455행)
- `wavesplat/baselines/letkf_grid.py` (전문 269행) · `wavesplat/baselines/etkf_grid.py` (전문 207행)
- `wavesplat/run.py` (1–140, 190–300, 330–400행) · `wavesplat/data_loader.py` (124–131, 77–97행)
- `wavesplat/optimizer/sequential_update.py` (30–50행)
- `wavesplat/model/spectral_gaussian.py` (1–130행)
- `wavesim/wave_generator.py` (1–135행) · `wavesim/gt_generator.py` (전문 상단) ·
  `wavesim/pipeline.py` (56–110, 150–240행) · `wavesim/noise_model.py` (1–95행) ·
  `wavesim/mesh_adequacy.py` (61–95행) · `wavesim/config.py` (mesh 부)
- `configs/estimation.yaml` · `configs/default.yaml` · `configs/server.yaml` (3종 전문)
- `wavesplat/tests/` 11개 파일 (grep 전수 + `test_metrics.py` 전문) · `wavesim/tests/` (파일명·개수)
- `tmp/S13_oos_2026-08-12/run_s13.py` (전문 230행)

**데이터 (헤더/속성 + 소량 슬라이스만)**
- `data_s8_crowded_2026-08-10/` · `data_s9_crowded_2026-08-11/` · `data_s9_sparse_2026-08-11/`
  → 128개 전부의 `config/wave_components`·`gt_grid_*` 속성; 그 중 4개만 `gt_wave_field` **1 프레임**,
  1개만 `lidar_points` 31 프레임. **전량 로드 0회.**

**결과 파일**
- `results/*.json` **96개 전수 + `results/direction1|2|3/*.json`** 를 스크립트로 집계 스캔
- `results/b1_full_comparison.json` · `results/multiseed_extended.json` ·
  `results/100ep_ws_letkf{,_partial}.json` · `results/gpr_comparison.json` 는 **개별 재도출**
- `paper/wavesplat.tex` (grep + 45–70, 100–106행) · `docs/PRESENTATION_SUMMARY.md` (grep)
- `tmp/DEBATE_blankslate_2026-08-12/r2_letkf_{strength,round2,round3}.json`
- `tmp/S13_oos_2026-08-12/results.jsonl` (704행 전량) · 같은 곳 `RESULT.md` (grep 12행)
- `memory/reference_citation_bans.md` (49·76·80행)

**안 본 것 (주장하지 않음)**
- `tmp/` 의 나머지 120여 배치 디렉터리 · `thesis/` · `scripts/` 413개 · 학습 체크포인트
- 실측(`data/real_bergamasco/`) 경로 일체 · `wavesim/hos_simulator.py` 본문
- 앞 라운드 보고서 본문(머리말 40행 초과분)

---

# 1. BLOCKER

## B-1 (신규) — 합성 GT 가 추정기의 모델식과 **완전히 같다**. 불일치는 0.25% 가 아니라 **0.000%**

**MEASURED.** 4개 에피소드에서 저장된 `frames/gt_wave_field` 한 프레임을
`config/wave_components` 로부터 `Σ Aᵢ cos(kxᵢx + kyᵢy − ωᵢt + φᵢ)` 로 재구성해 비교:

| 에피소드 | max&#124;Z_저장 − Z_평면파합&#124; | max&#124;ω − √(g·k·tanh(kd))&#124; |
|---|---|---|
| `BF4_15comp_seed32` (혼잡) | **0.000e+00** | 4.4e-16 |
| `BF4_2comp_seed48` (희소) | **0.000e+00** | 0.0 |
| `BF5_30comp_seed55` (혼잡) | **0.000e+00** | 2.2e-16 |
| `BF2_1comp_seed0` (공개 코퍼스) | **0.000e+00** | 0.0 |

**기제 (코드):** `wavesim/gt_generator.py:38` 이
`_, _, Z, NX, NY, NZ = single_gerstner(...)` — 즉 **`X_disp`·`Y_disp` 를 버린다**.
Gerstner 변위는 `wavesim/wave_generator.py:99-100` 에서 계산되지만
GT 필드에도, **레이캐스트 메시에도** 들어가지 않는다(`wavesim/pipeline.py:157` GT,
`:165` 메시 — 둘 다 같은 `evaluate_field` 호출). `Q=0.5` 는 `config/wave_components[:,5]` 에
저장돼 있으나 **어디에도 쓰이지 않는다**.

추정기의 전방모델은 `wavesplat/model/spectral_gaussian.py:127`
`z = Σ w·cos(phase)` + 분산관계 하드제약 — **문자 그대로 같은 식**.

**따라서 반증되는 문장 2개**
1. `wavesplat/model/spectral_gaussian.py:16` — *"The intentional model mismatch with Gerstner GT
   is O(ε²) ≈ 0.25% at BF4"* → **실측 0%**. 코드 주석이 근거 없이 방어막을 자처하고 있다.
2. `MEMORY.md` RESUME — *"⚠희소 −93% 는 표현 아티팩트(합성=평면파 합, 모델=평면파 합)"* →
   **희소에 한정된 진술이 아니다.** 혼잡(15·30 comp)도 동일하게 진실이 모델류 안에 **정확히** 있다.
   `max_n=30` 이므로 30comp 까지 표현 상한도 안 걸린다. ⟹ 혼잡 −28.8% 헤드라인도
   *"진실이 가설공간 안에 정확히 있는 우주에서의 최적화 성능"* 이며, 물리 모델링 우수성의 증거가 아니다.

**부수 결과:** 상시지시 [`feedback_validation_nonlinearity.md`](검증에 비선형 필수)가
S7·S8·S9·S13 코퍼스에서 **한 번도 충족된 적이 없다**(비선형 성분 = 정확히 0). HOS 경로
(`wavesim/hos_simulator.py`, `cfg.hos.enabled`)는 존재하지만 이 코퍼스들은 쓰지 않았다.

> **ARGUED (내 해석):** 이건 "데이터가 조작됐다" 가 아니다. 선형 중첩 우주는 합법적인 1단계다.
> 문제는 **저장소가 스스로 "Gerstner 비선형과의 의도된 불일치가 있다" 고 세 군데에 적어놓은 것**
> (모델 docstring · `CLAUDE.md` 설계 결정 · 논문 서술)이고, 그 문장이 리뷰어 방어로 쓰이면
> 곧바로 무너진다. 정직한 문장은 *"평가 GT 는 분산관계-정합 평면파 중첩이며 추정기의 가설공간과
> 동일하다. 따라서 보고되는 오차는 전적으로 식별·최적화 오차다"*.

---

## B-2 (신규) — 배포 기본값 **3개를 동시에 되돌려도 테스트 111/111 통과**

**MEASURED.** 아래를 임포트 직후 몽키패치하고 `pytest wavesplat/tests` 실행:

| 되돌린 것 | 원래 | 되돌린 값 | 이 값이 배포된 근거 |
|---|---|---|---|
| `sequential_update.LAMBDA_L1_DEFAULT` | 1e-4 | **1e-3** | 2026-08-11 S9 배포 (커밋 `40e22f6`/`5f0846c`) |
| `run.DEFAULT_CONFIG['init_method']` | `auto` | **`v2_konly`** | 2026-07-26 라우터 flip |
| `run.DEFAULT_CONFIG['tracking_mode']` | `w_phi` | **`all`** | `estimation.yaml:22` 출하값 |

→ **`111 passed in 38.12s`**. **깨진 테스트 0개.**

**기제:** `wavesplat/tests/` 와 `wavesim/tests/` 전체에 **`DEFAULT_CONFIG` 문자열이 0회**,
**`configs/` 경로 문자열이 0회** 등장한다(grep 전수). 모든 테스트가 `lambda_l1`·`init_method` 를
**명시 인자로 넘긴다**(예: `test_integration.py:78,114` 는 `lambda_l1: 0.0`,
`test_deploy_auto_v1_m3.py:133` 은 `{'init_method':'auto'}`).

**왜 BLOCKER 인가:** `sequential_update.py:38-39` 주석이 스스로
*"이 결함 부류(상수가 두 곳에 적혀 낡는 것)가 이 프로젝트에 이미 **세 번** 비용을 물렸다"* 라고
적어 놓았고, 실제로 **S7·S8 의 재실행·확장이 이것 때문에 무효** 판정을 받았다
(`reference_citation_bans.md:76`). 그런데 **그 결함을 잡는 테스트는 여전히 0개**다.
196개가 통과한다는 사실은 배포 구성에 대해 **아무것도 보증하지 않는다**.

*처방(1파일 추가로 충분):* `DEFAULT_CONFIG['lambda_l1'] == LAMBDA_L1_DEFAULT == yaml['optimizer']['lambda_l1']`
· `DEFAULT_CONFIG['init_method'] == 'auto'` · `SequentialUpdater({}).lambda_l1 == 1e-4` ·
`load_yaml_config('configs/estimation.yaml') ⊇ DEFAULT_CONFIG` 4줄.

---

# 2. MAJOR

## M-1 (★ 앞 라운드 결론 **뒤집힘**) — "0.319 를 내는 파일이 저장소에 없다" 는 **거짓**

**재검증 대상:** `reference_citation_bans.md:80` 이 2026-08-13(=앞 라운드 FIX)에 기록한
> ★같은 감사에서 **살아남은 문장**은 *"논문 헤드라인 분모 0.319 를 내는 파일이 저장소에 없다"* 쪽이다.

및 `CLAUDE.md`
> ⛔"WS 0.223 / ETKF 0.319 / 13.4 ms / 132.6 ms" 인용 금지 — **어느 결과 파일에서도 재현되지 않고**(감사 2026-08-12)

**MEASURED — `results/b1_full_comparison.json` (git 추적, 커밋 `e57528c`, 163 KB, 100 에피소드)에서
네 수치가 전부 나온다:**

| 금지된 수치 | 재도출값 (n=100 평균) |
|---|---|
| WS 0.223 | **0.2229** |
| ETKF 0.319 | **0.3189** |
| 13.4 ms | **13.38** |
| 132.6 ms | **132.64** (중앙 110.09 — 논문 `wavesplat.tex:1387` 의 *"median 110 ms, mean 132.6 ms"* 와 일치) |
| "30% 낫다" | **30.1%** |

파일은 ETKF 구성을 안에 기록해 두었다: `etkf_config = {grid:[40,30], n_ensemble:100,
mult_inflation:1.0, add_inflation:0.005}`.

**⟹ 판정: 뒤집힘.** 08-12 감사도, 앞 라운드도 **이 파일을 열지 않았다**(A1 은 *"results/ 하위 어떤
파일도 안 열었다"* 라고 스스로 적어놓았고, 08-12 감사가 연 것은 `results/100ep_ws_letkf.json`).
금지의 **사유가 사실이 아니다.**

**그러면 무엇이 진짜 문제인가 (MEASURED):** 같은 WaveSplat 열(0.2229)이
`results/100ep_ws_letkf.json` 에서는 **LETKF 0.2278 과 51/100, Wilcoxon p=0.989 = 널** 이다.
즉 **동일한 WaveSplat 런**이 ETKF 대비 −30.1%, LETKF 대비 **무승부**다.
결함은 **출처 부재가 아니라 기준선 선택**이다.

> **처방 (ARGUED):** 금지를 **해제**하지 말고 **재기술**할 것.
> 허용문 예: *"2026-06-24 공개 구성(`results/b1_full_comparison.json`, ETKF grid 40×30/Ne=100)에서
> WS 0.2229 vs ETKF 0.3189 (−30.1%). **단 같은 WS 열은 LETKF 대비 0.2229 vs 0.2278 = 널
> (51/100, p=0.99)** 이므로 −30% 는 약한 기준선을 고른 결과다."*
> 지연 수치(13.4/132.6 ms)의 금지는 **별도 근거(08-05 게스트시계 오염·경합 측정)** 로 유지된다 —
> 나는 그 근거를 재검증하지 않았다.

## M-2 (신규) — 논문 헤드라인 **0.115** 는 `results/` 어디에도 없다

**MEASURED (범위: `results/*.json` 96개 + `results/direction{1,2,3}/*.json` 전수 자동 집계).**
`nRMSE`/`avg_nRMSE` 를 팔별로 모아 n≥20 인 집계 중 평균이 [0.10, 0.13] 인 것은
`results/hos_vs_linear_gap.json` 의 무관한 항목(n=21, 0.1014) **하나뿐**이다.
n≥90 짜리 100-에피소드 집계는 **정확히 세 파일**이고 값은 전부 0.219–0.322 대다:

| 파일 | WS | 상대팔 |
|---|---|---|
| `b1_full_comparison.json` | **0.2229** | ETKF **0.3189** · GPR-1 0.2610 |
| `multiseed_extended.json` | **0.2193** | ETKF **0.3213** · GPR-3 0.2530 |
| `100ep_ws_letkf.json` | **0.2229** | LETKF **0.2278** |

논문 `paper/wavesplat.tex:50,103` 의 **`0.115 vs 0.332` (65.5%)** 는 이 셋 어디에도 없다.
논문 자신이 `:2147` 에서 ETKF 를 `0.319 → 0.332` (메시 재생성) 이라 적으므로 **분모 0.332 는
재생성 코퍼스 산출**로 보이나, **그 파일이 `results/` 에 없다**.
분자 0.115 는 `MEMORY.md` 2026-07-25 항목의 *"위상수정 100ep 0.199→**0.1145**"* 와 맞물린다
⟹ **분자와 분모가 서로 다른 배치에서 왔을 가능성**이 높다. (나는 `tmp/` 를 전수 수색하지 않았다 —
`0.1145` 문자열 grep 은 무관한 히트만 반환.)

⛔`CLAUDE.md` 는 **재현되는 쌍(0.223/0.319)** 을 금지하고, **재현 안 되는 쌍(0.115/0.332)** 은
금지목록에 없다. **금지 대상이 뒤바뀌어 있다.**

## M-3 (신규) — `configs/` 3종이 **실제 코퍼스를 기술하지 않고**, 자기 검사기가 둘 다 탈락시킨다

**MEASURED.** 저장소 자신의 `wavesim/mesh_adequacy.check()` 를 셋업 그대로 돌렸다
(BF4 15comp, Hs 1.2 / Tp 7.0, `PTS_PER_LAMBDA_OK=6.0`):

| 설정 | mesh | `pts_per_lambda` | `var_under_resolved` | **verdict** |
|---|---|---|---|---|
| `configs/default.yaml` | 40×30 | 1.236 | **33.7%** | **INADEQUATE** |
| `configs/server.yaml` | 80×60 | 2.515 | 8.8% | **DEGRADED** |
| 실제 S8/S9 코퍼스 (h5 속성) | **200×167** | 7.070 | 0.0% | ADEQUATE |

- 200×167 은 `tmp/S8_deploy_confirm_2026-08-10/gen_s8.py:17` · `tmp/S9_.../gen_s9.py:18` 에만 있다.
  **`configs/` 에는 없다.**
- `configs/server.yaml` 의 `gt_grid: nx 120 / ny 90` 은 실제 데이터(`80×60`)와 **불일치**.
  `default.yaml` 의 `80×60` 만 맞는다.
- `CLAUDE.md`/`generate_dataset.py:7` 는 여전히 `--config configs/default.yaml` 을 안내한다.
  `mesh_adequacy.enforce(strict=False)` 는 **경고만** 하므로, 안내대로 재생성하면
  2026-07-23 에 🔴 로 진단됐던 **메시 아티팩트가 조용히 재도입된다**.

## M-4 (신규, ARGUED + MEASURED 뒷받침) — LETKF 배경 앙상블에 **파 스펙트럼 사전분포가 없다**. 튜닝 스윕이 유일하게 안 건드린 축

**MEASURED (코드):** `letkf_grid.py:88` 과 `:268` — 앙상블 초기화는
`np.random.randn(Ne, n_state) * 0.1`, 즉 **백색잡음**. 가산 인플레이션도 백색
(`letkf_grid.py:165`). 저장소 전체 grep 결과 **`letkf.ensemble` 을 JONSWAP/스펙트럼에서
채우는 코드 경로가 0개**(필터 내부 대입 3곳 외 없음).

**MEASURED (스윕 범위):** R2 강화 스윕이 실제로 건드린 축은
`max_obs`(200/800) · 격자(40×30/80×60) · `loc_radius`(10/25/50/75/100/200) ·
인플레이션(add 0/0.002) · `n_ensemble`(100/300/600/1200) · `obs_noise`(0.01/0.02/0.05/0.1/0.2).
**배경 공분산의 스펙트럼 형태는 없다.**

**ARGUED:** 선형 전방연산자는 백색 초기장을 *분산관계에는 맞지만 파수적으로 평평한* 바다로
키운다. 진실은 협대역 JONSWAP 이므로 배경 스펙트럼이 오설정돼 있고, 관측이 프레임당 **중앙 49점**
(240×200 m 영역, 점당 ≈980 m²)뿐이라 필터가 이를 교정할 정보를 못 받는다.
S13 에서 LETKF/LETKF_PLUS 가 **제로필드를 겨우 8–14% 밖에 못 이기는**
(중앙 Δ −0.0208 / −0.0343, 제로 0.2500) 이유의 유력 후보다.
문헌 표준(파 nowcasting ETKF)은 기후학적 파 스펙트럼에서 앙상블을 뽑는다.
⟹ **"튜닝된 LETKF 를 이겼다" 는 문장은 이 축이 열리기 전까지는 상한이 아니다.**

---

# 3. MINOR

| # | 발견 | 근거 |
|---|---|---|
| m-1 | **yaml 폴백 `init_method` 가 배포값과 불일치** — `run.py:99` 는 `raw['optimizer'].get('init_method', 'v2')`. `DEFAULT_CONFIG['init_method']='auto'`(`run.py:43`). `lambda_l1` 만 `run.py:94` 에서 *"폴백은 DEFAULT_CONFIG 를 따라야 한다(S9 교훈)"* 로 고쳐졌고 **형제 줄들은 안 고쳤다**. 키를 뺀 yaml 은 조용히 `v2`(출하도 아니고 문서화도 안 된 추정기)를 받는다. | `run.py:89-99` |
| m-2 | **죽은 yaml 키 7개** — `optimizer.max_steepness` · `init.tp_factor` · `init.grid_nx` · `init.grid_ny` · `eval.converge_threshold` · `eval.split_distance` · `eval.unobserved_min_dist` 는 `load_yaml_config` 가 **읽지 않는다**. yaml 로 튜닝했다고 믿으면 결과가 동일하게 나온다. | `run.py:81-123` vs `configs/estimation.yaml:38,58-71` |
| m-3 | **`metrics.rmse_shell`·`shell_fraction` 은 호출자 0개(죽은 코드)** — `SUPPORT-TRADEOFF`(7회 재현)를 만든 셸 계산은 `run.py:352-378` 의 **인라인 중복 구현**이다. 두 구현은 수식상 동치이나, 긴 docstring 으로 감사 대상처럼 보이는 쪽은 안 쓰인다. | grep 전수: `evaluation/metrics.py` 밖 호출 0 |
| m-4 | **측정층 테스트 공백** — `hs_error`·`tp_error`·`spectral_r2`·`rmse_spatial_split`·`rmse_shell`·`shell_fraction`·`directional_r2`·`dispersion_concentration`·`sigma_units_to_hs_units` = **테스트에서 참조 0회**. `test_metrics.py` 는 **2개뿐**이고 둘 다 `rmse_observed` 전용. 헤드라인 지표 `nrmse` 의 정의(=RMSE/Hs)를 고정하는 단위 테스트가 없다. | `wavesplat/tests/test_metrics.py` 전문 |
| m-5 | **`use_confidence`(출하 기본 true)가 GT 유래 값을 먹는다** — 신뢰도는 `compute_confidence(incidence, distance)` 이고 `incidence` 는 **참 표면 법선**에서 계산(`pipeline.py:193,229`), h5 `frames/lidar_confidence` 로 저장, `data_loader.py:87` 이 그대로 관측 가중치로 전달. 실측에는 대응물이 없다. **완화:** S13 에서 끄는 쪽이 낫다(혼잡 SHIP 이 10/64 승, 중앙 Δ −0.0263) ⟹ 헤드라인을 부풀리지 않는다. 다만 *"출하 팔은 GT-free"* 라는 서술엔 예외가 있다. | `noise_model.py:69-93`, `pipeline.py:229`, S13 `results.jsonl` |
| m-6 | **LETKF 평가격자(40×30)의 Nyquist 가 GT 성분 일부를 원리적으로 배제** — 격자 Nyquist λ = 12.3 m(x)/13.8 m(y) 인데 코퍼스 `lambda_min` ≈ 8.5 m. **단 영향은 국소적**: 128 에피소드 중 분산의 >50% 가 배제되는 건 **2개뿐**(`BF4_2comp_seed50` 99.8%, `seed56` 61.2%), 배제분율과 (LETKF−SHIP) 격차의 상관은 **r=0.227**. 꼬리는 설명하나(격차 중앙 0.223 vs 0.116) 헤드라인은 설명 못 한다. | 128개 `wave_components` 전수 계산 |
| m-7 | `100ep_ws_letkf.json` 과 `100ep_ws_letkf_partial.json` 은 **md5 동일**(`afe985f3…`). "partial" 이라는 이름이 오해를 부른다. | md5sum |
| m-8 | **Skill Score 는 nRMSE 의 독립 축이 아니다** — Hs=4σ 이므로 `SS ≈ 1 − (4·nRMSE)²`. `b1_full_comparison.json` 에서 실측 상관 **0.967(WS) / 0.924(ETKF)**. `metrics.py:9-14` 는 SS 를 *"nRMSE 만으로는 구별 못 하는 것을 드러낸다"* 고 광고하는데, 최소한 이 자료에서는 거의 동어반복이다. | `results/b1_full_comparison.json` 100행 재도출 |
| m-9 | 같은 파일에서 WaveSplat 의 **Skill Score 는 100 중 53 에서만 양수**(평균 +0.034). 논문 `:57` 의 *"95% CI [+0.42, +0.65]"* 는 **재생성 코퍼스** 주장이라 직접 모순은 아니지만, 그 코퍼스 파일을 `results/` 에서 찾지 못했다(M-2 와 동일 범위). | 동상 |

---

# 4. 깨끗하다고 확인한 영역 (범위 명시 — 이 범위 밖은 주장 안 함)

1. **nRMSE 정규화 분모의 팔 간 대칭** — S13 양 팔 모두 `loader.Hs_true`
   (`run_s13.py:151` LETKF, `run.py:285` WaveSplat). 팔마다 다른 분모를 쓰는 경로 **없음**.
   실측 σ-정규화와의 혼선은 `metrics.py:32-60` 이 명시적으로 경고·변환함수까지 제공한다. **양호.**
2. **프레임 정합** — LETKF `sel = t > 2.0*Tp`(`run_s13.py:157`) 와
   WaveSplat `get_tracking_frames(tp_factor=2.0)` → `timestamps[i] > 2.0*Tp`
   (`data_loader.py:126-130`) 는 **부등호·계수까지 동일**. A1 의 "frame-matched" 주장 **유지**.
3. **관측 입력의 동일성** — 두 팔 모두 같은 `EpisodeDataLoader` 프레임의
   `x_obs/y_obs/z_obs` 를 받는다(`run_s13.py:142-145` vs `run.py` 전 경로). 팔별 전처리 차이 없음.
4. **`obs_noise` 오설정은 실측상 무해** — 실제 수직 관측오차를 재니 **rms 0.0062 m**
   (1509점, 저장 z 대 해석적 GT), 필터는 0.05 m 를 가정 ⟹ R 이 분산 기준 **65배 과대**.
   그러나 R2 스윕이 이미 훑었고 효과가 평평하다: `noise0.01` 0.2443 / `noise0.02` 0.2423 /
   `0.05` 0.2418 / `0.2` 0.2460 (n=4). **불공정 축 아님 — 기각.**
5. **`max_obs=40` 관측 솎기도 무해** — 유효 관측 중앙 49점이라 솎임은 걸리지만,
   `obs200`/`obs800` 로 풀면 nRMSE 가 0.2578 → 0.2575 (−0.14%). **기각.**
6. **ETKF/LETKF 해석식 자체는 표준** — 평균 갱신 `x + Xᵀ C⁻¹ Y d/√R`,
   섭동 갱신 `((Ne−1)C⁻¹)^{1/2}` (Hunt et al. 2007 형태), Gaspari-Cohn(1999) 국소화 계수도
   교과서 다항식과 일치(`letkf_grid.py:18-35`). 행렬 방향·차원 전부 정합. **구현 결함 없음.**
7. **`rmse_observed`/`rmse_unobserved` 는 정확한 여집합** — 코드로 확인, 테스트로도 고정
   (`test_metrics.py`). NaN 처리는 관측 0개 / 마스크 공집합에서만 발생하며 **어느 쪽에도 유리하게
   기울지 않는다**(양 팔 동일 함수).
8. **`GPR 0.276` 재현** — `results/gpr_comparison.json`, n=10, 평균 **0.2756**.
   `CLAUDE.md` 의 *"GPR 0.276 만 재현됨"* 은 **유지**.
9. **"8.1× 빠르다"(LETKF 비교) 재현** — `100ep_ws_letkf.json` 중앙 116.5 ms / 14.2 ms = **8.18×**.
   (⛔ 별개인 "~10× / 11.0 vs 132.6 ms" 는 08-05 금지 유지.)
10. **테스트 수집 196개 확인** (`pytest --collect-only`, 1.61 s). `CLAUDE.md` 의 개수 주장 **유지** —
    단 B-2 가 보이듯 **개수는 배포 방어와 무관**하다.
11. **S13 자료 자체의 위생** — `results.jsonl` 704행, `error` 키 0, `(episode,arm)` 중복 0,
    `git_head` 단일. A1 의 F 항목 **유지**(내 독립 재계수).

---

# 5. 재검증한 앞 라운드 결론 — 판정

| 앞 라운드 결론 | 내 판정 | 근거 |
|---|---|---|
| **"논문 헤드라인 분모 0.319 를 내는 파일이 저장소에 없다"**(08-12 감사에서 *살아남은* 문장, 앞 라운드가 `reference_citation_bans.md:80` 에 명시적으로 재확인) | **뒤집힘** | `results/b1_full_comparison.json` 이 0.2229 / **0.3189** / 13.38 / **132.64** 를 전부 낸다. 금지 사유가 사실이 아님. 단 금지 자체는 **재기술 후 유지** 권고(§M-1) |
| A1 "S13 은 frame-matched" | **유지** | 부등호·계수까지 코드 대조 (§4-2) |
| A5 "196 tests" | **유지(수치)** / **무의미(함의)** | 수집 196 확인. 그러나 배포 기본값 3개를 되돌려도 111/111 통과 (§B-2) |
| `CLAUDE.md` "GPR 0.276 만 재현됨" | **유지** | `gpr_comparison.json` n=10 평균 0.2756 |
| `MEMORY.md` "희소 −93% 는 표현 아티팩트" | **유지하되 범위 확대** | 아티팩트는 희소 전용이 아니라 **코퍼스 전체**(§B-1) |

---

# 6. 내가 틀렸을 수 있는 곳 (선제 자백)

- **B-1**: 나는 4개 에피소드의 **각 1 프레임**만 대조했다(128개 전 프레임 아님). 그러나
  `gt_generator.py:38` 의 변수 폐기는 무조건 경로이므로, 반례가 있으려면 별도 GT 경로가 있어야 한다.
  `hos_simulator` 분기가 그 후보이고 **이 코퍼스들은 안 썼다**는 것만 h5 속성으로 확인했다.
- **M-2**: `tmp/` 전수 수색을 안 했다. 0.115/0.332 가 `tmp/` 어딘가에 있을 수 있다.
  내 주장은 **"`results/` 에 없다"** 로 한정된다.
- **M-4**: 스펙트럼 사전분포를 실제로 넣어 재봤다면 개선폭이 0 일 수도 있다.
  나는 **런을 0회** 했고, 이건 **ARGUED** 다. (`ne1200` 도 0.2322 까지밖에 못 갔다는
  기존 자료는 이 가설과 양립한다 — 멤버 수는 스펙트럼 오설정을 못 고친다.)
- **m-8**: SS 와 nRMSE 의 관계는 프레임 평균 때문에 **정확한 항등식이 아니다**(Jensen 갭).
  "거의 동어반복" 은 상관 0.92–0.97 에 근거한 진술이지 증명이 아니다.
