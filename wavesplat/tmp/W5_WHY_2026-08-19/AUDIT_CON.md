# AUDIT_CON — `W5 · WHY` 사전등록 부정측 감사 (런 0회)

**대상**: `tmp/W5_WHY_2026-08-19/PREREG.md` md5 `f5dd2e386cea1e537c14150462ed9268` · **판정 행 0개**
**시각**: 2026-08-19 KST · **감사자 실행 내역**(전부 비실험):
`run_w5` 모듈 import 시도 1회 · W4 `results.jsonl`(1,344행) 재집계 · `git log/status` · 60×60 선형대수 결정성 시험 1회.
**에피소드 셀 실행 0회.**

---

## 판정 요약

| # | 공격 | 판정 | 런 전 최소 변경 |
|---|---|---|---|
| **1** | W4 열 재사용의 정당성(§0.1) | **WOUND** (KILL 아님) | §0.1 의 근거를 프로브가 아니라 **정적 사문(死文) 증명**으로 바꾸고, **G-0 에 `L-VP@1595` 4셀을 추가**(≈1.5분) |
| **2** | 기대값 720 / 1620 | **산술 SURVIVES · 구현 KILL** | `arms_config_w5.py` 를 `run_w5.py` 가 읽는 API 로 맞추고 `W4AC.ARM_DELTAS`/`ARM_GATES` 를 **제자리 변경**(copy 금지) 후 md5 재동결 |
| **3** | `L-VP-ONLY` 단일요인성(§2) | **WOUND** | §2 요인칸을 "`learned_init_iters` (동반: clamp 투영 500회 소멸 · prune/grow 입력 변경)" 로 고치고 `clamp_*`·`adam_steps_grow`·`n_atoms_post_init` 를 D-항목으로 사전등록 |
| 짧게 | `G-8` "code_md5 1종" | **KILL (문언)** | §4 G-8 → "**출처별 1종**(NEW 1종 · REUSED = `a29340cee6c1de6b3bc08c089ceb8e61`)" |
| 짧게 | `D-4` `avg_RMSE_near` | **KILL** | `run_w5.run_one` 에서 `wavesplat.run.run_estimation` 을 감싸 값을 가로채고 `x80:` 접두어로 기록 |
| 짧게 | `P-1` 연역 가능성 | WOUND(정보량) | — (문언에 "강하게 예측됨" 명시 권고) |
| 짧게 | `S-2` 포화 vs 둘 다 부족 | WOUND | S-2 널을 `vp_obj_before/after`·`vp_k_shift` 와 **함께** 읽도록 §3 에 한 줄 추가 |

---

# 1. ★★W4 열 재사용은 정당한가 — **WOUND**

## 1.1 `MEASURED` — 패치 1(`amplitude_refit` 부호)은 이 배치 전 팔에서 **사문**이다
- `amplitude_refit` 의 **호출자는 저장소 전체에서 `run.py:402` 단 하나**이고 `run.py:398`
  `cfg.get('amplitude_refit', False)` 로 게이트된다(전수 grep: 패키지 밖 호출 0건, 나머지 히트는 테스트/문서).
- `arms_config.base_config()` 는 이 키를 넣지 않고, `configs/*.yaml`·`run.py` `DEFAULT_CONFIG` 에도 없다.
⟹ **팔 6개 전부에서 실행 불가.** 이것은 프로브보다 **강한** 증거다(프로브는 표본, 이건 경로 부재).

## 1.2 `MEASURED` — 패치 2(`avg_RMSE_near`)도 수치에 닿지 않는다
- `run.py:478` 은 **요약 dict 리터럴 안의 가산 항**이고, 모든 추정이 끝난 뒤 평가된다. RNG 소비 0.
- 예외 위험도 없다: `has_gt` 인 프레임은 전부 `compute_all_metrics`(`run.py:281`)를 거치고 그 함수는
  `metrics.py:433` 에서 `RMSE_near` 를 **항상** 채운다. `frame_results` 의 append 지점은 `run.py:380` 하나뿐.
- ⚠단 이 키는 `_pkg_files()` 글롭에 걸리는 **`tests/test_phase_convention.py` 추가와 함께** 패키지 지문을
  바꾼다 ⟹ 아래 G-8 문제로 이어진다.

## 1.3 `MEASURED` — 저장소 드리프트는 없다
- W4 이후 커밋은 `b1467bd`(위 두 곳) + `aeff4d8`(문서 전용) **둘뿐**.
- `git status --porcelain wavesplat/wavesplat/` 의 **추적 파일 수정 0건**(현재 트리 clean).
  W4 행의 `git_dirty_n=63` 은 `git status --porcelain` 이 `??` 를 세기 때문이며 tmp 미추적 디렉터리다.
- ★더 강한 앵커: **`code_md5` 가 1,344행 전부에서 단일값**(`a29340…`, 91파일 내용 해시) —
  `git_head` 보다 강하다(CLAUDE.md "더러운 작업트리를 못 잡는다" 의 정확한 처방).

## 1.4 `MEASURED-NEGATIVE` — 내가 세운 가설 하나는 **실패했다** (정직 보고)
가설: "VARPRO 는 `Phi_wᵀPhi_w`(8080×60 float64 GEMM, `varpro.py:185`)와 `torch.linalg.solve` 를 쓰므로
BLAS 스레드 수에 따라 비트가 갈릴 수 있다. 그런데 `run_one` 의 `os.environ.setdefault('OMP_NUM_THREADS','1')`
는 워커에서 torch import **뒤에** 실행되어 무력하다(`run_w4.py:480` · `run_w5.py:225`, 현재 셸 `OMP=None`,
`torch.get_num_threads()=8`). ⟹ 재사용된 `L-VP` 열과 새로 도는 `L-VP-ONLY` 가 비교 불가일 수 있다."
**실측**: 배포 규모 그대로 스레드 1/2/4/8 에서 `max|ΔA| = max|Δc| = 0.000e+00` (torch 2.12.1+cu126).
⟹ **이 공격은 성립하지 않는다.** 기각을 기록한다.

## 1.5 그래서 무엇이 남는가 (`ARGUED`)
§0.1 이 인용하는 근거는 **`L-SHIP` 1개 팔 · 시드 1개 · 4셀**이다. 그 프로브가 **실행하는 VARPRO 스텝은 0** 이고,
재사용 6개 팔 중 **5개를 한 번도 건드리지 않는다**. 그런데 **`P-2` 의 대조군이 바로 재사용된 `L-VP` 열**이다
— 즉 P-2 는 "오늘 도는 VARPRO" 대 "어제 돈 VARPRO" 의 차이를 측정한다.
1.1–1.4 덕분에 재사용은 **실제로는 정당하다**. 그러나 **PREREG 가 적어 둔 사유는 그 정당성을 주지 못한다.**
(`ZERO`·`ORA-K-CVX` 만은 예외적으로 이미 교차배치 앵커가 있다 — W4 `AMENDMENT_2 A2-5` 가 W3 값과 |Δ|=0 확인.)

### ⇒ 런 전 최소 변경
1. §0.1 의 첫 근거를 **"패치 1 은 호출 경로 부재로 사문 · 패치 2 는 RNG·제어흐름과 무관한 가산 키"** 로 교체하고,
   프로브는 **보조 확인**으로 격하.
2. **G-0 에 `L-VP@1595` 4셀 추가**(W4 실측 21 s/셀 ⟹ ≈1.5분). 이것이 재사용 열 중 **P-2 의 대조군을 직접** 친다.

---

# 2. ★★기대값 720 / 1620 은 맞는가 — **산술 SURVIVES · 구현 KILL**

## 2.1 `MEASURED` — 분해식은 맞다 (W4 1,344행 재집계)
| 팔 | `adam_steps_phase1` | `adam_steps_grow` | `adam_steps_other` | `vp_steps` |
|---|---|---|---|---|
| `L-ITER0` | **320** (192/192) | 50 (141행) / 0 (51행) | 0 | 0 |
| `L-SHIP` | **820** (192/192) | 50 (47행) / 0 | 0 | 0 |
| `L-VP` · `L-VP-NW` | **1220** (각 192/192) | 50 (47행) / 0 | 0 | **400** (192/192) |
| `L-STEP400` | 1220 (192/192) | 50 (37행) / 0 | 0 | 0 |

⟹ `POLISH=320` · `SHIP_ITERS=500` · VARPRO 400 의 **가법성이 실측**됐다.
`expected_phase1('L-VP-ONLY') = 320+0+400 = 720` · `('L-VP800') = 320+500+800 = 1620` 은 **옳다.**

## 2.2 `MEASURED` — 질문 ① 패치 순서 의존은 **충족돼 있다**
`run_w4.py:355` `vp.install(...)` → `:362` `ac.install_counters()` (코드 주석 그대로
*"2nd: encloses varpro's wrapper"*). 카운터가 varpro 래퍼를 **감싸므로** `_Phase('p1')` 안에서
`apply_varpro` 가 실행되고 그 Adam 스텝이 `adam_steps_phase1` 에 귀속된다. 순서가 뒤집히면 `adam_steps_other`
로 샜을 것이고, 실측 `other=0` 이 그것이 일어나지 않았음을 증명한다.
또 `varpro_refine`(`varpro.py:218-223`)의 루프에는 **early break 가 없다** ⟹ `steps_executed ≡ steps`,
즉 `==` 정확 게이트가 조기종료로 죽을 위험은 **0**.

## 2.3 `MEASURED` — 질문 ② `grow` 분리도 충족돼 있다
`L-ITER0` 는 `learned_init_iters=0` 인데 `adam_steps_phase1` 이 **192/192 에서 정확히 320**,
같은 행들의 `adam_steps_grow` 는 **141행에서 50**. ⟹ `_adaptive_prune_grow` 의 50스텝은 `p1` 로 새지 않는다.
(W4 `AMENDMENT_2 A2-2` 가 `L-ITER0 == 0` 이라는 **원리적 통과 불가** 게이트를 런 전에 고친 그 사건의 뒷면.)

## 2.4 ★★★`KILL` — 그런데 **그 공식을 읽을 코드가 없다**
동결된 `arms_config_w5.py` 로는 `run_w5.py` 가 **import 조차 되지 않는다**. 실측:

```
File "tmp/W5_WHY_2026-08-19/run_w5.py", line 116, in <module>   ARMS = _arms()
File "tmp/W5_WHY_2026-08-19/run_w5.py", line 102, in _arms      A5.register()
AttributeError: module 'arms_config_w5' has no attribute 'register'
```

`run_w5.py` 가 요구하는 이름 13개 — `register` · `NEW_SPECS` · `POLISH_STEPS` · `ALL_SPECS` ·
`expectation_table` · `_verify_single_source` · `PREREG_QUOTED` · `W4_MEASURED` · `SingleSourceError` ·
`AC` · `expected_adam_p1` · `expected_vp_steps` · `row_gate_verdict` — 중 **제공되는 것은 0개**다
(파일이 내보내는 것은 `expected_phase1` · `expected_vp` · `arm_config` · `ARM_DELTAS` · `VP_SPEC` · `NEW_ARMS` · `REUSE_ARMS`).

★그리고 이름 문제만이 아니다. `run_w5.py:534` 는
`all(ARMS[a]['cfg_from'] in A5.AC.ARM_DELTAS for a in NEW_ARMS)` 로 **W4 의 `ARM_DELTAS` 가 제자리에서
확장돼 있기를** 요구한다(`:535` 주석 *"W4 ARM_DELTAS extended in memory"*). 그러나 파일은
`ARM_DELTAS = dict(W4AC.ARM_DELTAS)` — **복사본**이다. 이 상태로 `W4.arm_cfg('L-VP-ONLY')` 가 불리면
`arms_config.py:112` 의 `raise KeyError('unknown W4 arm ...')` 로 죽는다.

⟹ **PREREG §4 의 핵심 약속("기대값은 한 곳에서만 정의하고 러너·워치독·분석기가 그것을 읽는다")이
현재 구현되지 않았다.** W4 가 네 곳 중복으로 하루에 네 번 물린 그 결함이, W5 에서는
"**한 곳에 정의됐지만 아무도 읽지 않는다**" 로 형태만 바뀌어 남아 있다.
⚠부수적으로 `watchdog_w5.py` 는 **파일 자체가 없다**(디렉터리 목록 확인) — G-9 감시 주체 미구현.

### ⇒ 런 전 최소 변경
`arms_config_w5.py` 를 `run_w5.py` 의 API 로 맞추고(위 13개 이름), 확장은 **제자리**로 —
`W4AC.ARM_DELTAS.update(...)` · `W4AC.ARM_GATES.update({a: {'adam_steps_phase1': ('==', expected_adam_p1(a))} ...})` —
그 뒤 `FROZEN_md5.txt` 재작성 + 런 전 git 커밋.
(⚠**게이트 문턱은 건드리지 마라.** 720/1620 은 위 실측으로 정당하다. 고칠 것은 배선뿐이다.)

---

# 3. ★`L-VP-ONLY` 는 단일요인인가 — **WOUND**

`learned_init_iters: 500 → 0` 이 바꾸는 것은 **넷**이고, 설정 dict 차이에 보이는 것은 **첫 번째뿐**이다.

| # | 무엇이 바뀌나 | 근거 | 실측 |
|---|---|---|---|
| (i) | 결합 적합 Adam 500스텝 소멸 | `sequential_update.py:645-649` | 의도된 처치 |
| (ii) | `clamp_steepness` 물리 투영 **500회 소멸** | `:650` (루프 넷째 문장) | ★**무해로 실측** — `L-SHIP` `clamp_hits_p1` **0/192**, `L-ITER0` `steepness_max_post_init` 최대 **0.0661** vs 천장 0.15 |
| (iii) | `_adaptive_prune_grow` 가 **결합적합 안 된 모델**을 받는다 | `:658-659` (`_initialize_learned` **안**) | ★**크게 갈린다** — 성장 발화 `L-ITER0` **141/192** vs `L-SHIP` **47/192**, `n_atoms_post_init` 중앙 **19 vs 16** |
| (iv) | VARPRO 의 **입력 k·N** 이 달라진다 | `varpro.py:386` (`_ORIG_INIT_LEARNED` **뒤**) | (iii)의 귀결 |

- (ii)는 **측정으로 방어된다**(이 배치에 유리한 결과다). ⚠단 완전 무해는 아니다: `L-VP`·`L-VP-NW` 는
  `clamp_calls_p1 = 501`(500 + VARPRO 자신의 1회)이고 `clamp_hits_p1` 가 각각 **6/192 · 2/192**,
  `steepness_max_post_init` 가 **정확히 0.15**(=천장)에 닿는다. **VARPRO 는 제약면을 민다.**
  `L-VP-ONLY` 는 더 나쁜 출발점에서 밀 것이므로 **발화율이 커질 수 있다** — 미측정.
- (iii)·(iv)는 **교락이 아니라 매개(mediator)** 다(처치의 하류). 대조 자체는 잘 정의돼 있다.
- ⛔**그러나 러너의 "single-factor audit"(`run_w5.py:428-451`)은 설정 dict 차집합**이다.
  (ii)–(iv)를 **원리적으로 볼 수 없고** `L-VP-ONLY` vs `L-VP` 에 대해 `delta=['learned_init_iters']` **PASS** 를
  인쇄한다. 이것이 저장소가 반복해 대가를 치른 **"의도표를 인쇄한다"**(E1, S13b) 의 재발 자리다.

## 3.1 ★배치에 유리한 선명화 (부정측이지만 적어 둔다)
`varpro.py:187` 이 매 스텝 `(a,b)` 를 **닫힌형 릿지로 다시 푼다** ⟹ `ab_to_wphi` 로 쓰이는 `w,φ` 는
초기값을 **전혀 물려받지 않는다**. 즉 `L-VP-ONLY` 가 결합 적합으로부터 물려받는 것은 **k 와 N 뿐**이다.
⟹ **`P-2` 의 정확한 뜻**: *"500스텝 결합 적합은 이어질 400스텝 k-전용 하강에게 더 좋은 분지(basin)를 주는가."*
§1.1 의 해설("500스텝을 400스텝으로 갈아끼우면 된다")은 이 문장으로 좁혀 적는 것이 정직하다.

### ⇒ 런 전 최소 변경
§2 요인칸 = "`learned_init_iters` (동반: clamp 투영 500회 소멸[W4 실측 무해] · prune/grow 입력 변경)",
그리고 `clamp_calls_p1`·`clamp_hits_p1`·`adam_steps_grow`·`n_atoms_post_init` 를 **D-5 로 사전등록**
(전부 이미 `counters` 에 기록되고 있으므로 비용 0).

---

# 짧게 본 것

## S-1. `G-8` "code_md5 1종" — **KILL (문언)**
§4 는 `G-8` 을 **중단 게이트**로 두고 "`code_md5` **1종**" 이라 적었다. 그런데 재사용 행은 W4 지문
`a29340cee6c1de6b3bc08c089ceb8e61` 를 지고 오고, 신규 행은 **그 값을 가질 수 없다** —
`b1467bd` 가 `wavesplat/tests/test_phase_convention.py` 를 추가했고 `_pkg_files()` 의 `wavesplat/**/*.py`
글롭이 그것을 잡기 때문이다. **문언대로 읽으면 배치가 스스로를 죽인다.**
`run_w5.py:653-658` 은 이미 출처별로 나눠 검사한다(*"split by provenance ON PURPOSE"*) ⟹
**러너가 동결된 PREREG 와 어긋나 있다**(러너는 권위가 아니다).
⇒ **최소 변경**: §4 G-8 → "**출처별 1종** — NEW 1종 AND REUSED == `a29340cee6c1de6b3bc08c089ceb8e61`".

## S-2. `D-4` `avg_RMSE_near` — **KILL**
`run_w5.run_one:240` 은 `W4.run_est` 를 *"the W4 driver, unmodified"* 로 부른다. 그 함수
(`run_w4.py:397-436`)는 **명시 키 목록으로 행을 만든다** — 그 목록에 `avg_RMSE_near` 도 `RMSE_near` 도 **없다.**
⟹ D-4 는 **아무것도 기록하지 못한다.** 이것은 PREREG 가 근거로 드는 바로 그 결함
(`metrics.py:433` ↔ `run_b1_full_comparison.py:51-54`)의 **새 자리에서의 재발**이다.
⚠추가로 `RMSE_near` 는 `metrics.py:97-119` 에서 **`grid_x < 80.0`** — x 반평면 분할이다.
`obs:` 셸도 `ship:` 고리도 아니다(CLAUDE.md "접두어까지 적을 것").
⇒ **최소 변경**(처방 검증 완료 — `run_w4.py:366` 시점에 `r` 안에 값이 실재한다):
`run_w5.run_one` 에서 `wavesplat.run.run_estimation` 을 감싸 반환 dict 의 `avg_RMSE_near` 를 모듈 전역에
넣고(`finally` 에서 undo), `row['x80:avg_RMSE_near']` 로 기록. **`wavesplat/**` · W4 파일 모두 무수정.**

## S-3. `P-1` 은 W4 자료로 연역 가능한가 — **연역 불가, 그러나 강하게 예측됨**
필요 조건은 `L-VP-ONLY < 0.157812`. 출발점 `L-ITER0 = 0.182566` ⟹ VARPRO 가 **−13.6%** 만 벌면 된다.
W4 실측 VARPRO 이득은 `L-SHIP` 의 k 위에서 **−25.4%**(0.157812→0.117716)다.
게다가 `L-STEP400`(결합적합 900스텝) 은 `L-SHIP` 대비 **38/64, −1.28% = 널** ⟹ **결합 적합의 한계가치는
500 에서 이미 고갈**돼 있다. ⟹ P-1 실패는 "k 품질이 결정적일 때"만 일어나는데, 그것이 곧 P-2 다.
⟹ **정보는 P-1 의 통과 여부가 아니라 `L-VP-ONLY` 가 0.117716 에 대해 어디 서느냐(P-2)에 있다.**
⚠P-1·P-2 는 **같은 새 열 하나에 문턱 두 개**다. Holm-2 는 임의 종속 하에서도 타당하므로 오류율 문제는
없지만, "질문 2개"로 홍보하지 말 것.

## S-4. `S-2`(800 vs 400) 널은 "포화" 와 "둘 다 부족" 을 구분하는가 — **구분 못 한다**
`ORA-K-CVX = 0.001298` vs `L-VP = 0.117716` = **×90.7** ⟹ 지형은 전혀 소진되지 않았다.
2× 용량에서의 널은 "스텝이 충분" 이 아니라 "국소 분지에 갇혔다" 와 **완전히 양립**한다.
⇒ **최소 변경**(비용 0, 필드는 이미 기록 중): §3 에 한 줄 —
*"`S-2` 널은 `vp_obj_before/after` 와 `vp_k_shift` 와 **함께** 읽는다. 목적함수는 계속 내려가는데 nRMSE 가
안 움직이면 판정은 `OBJECTIVE-SATURATION ≠ FIELD-SATURATION` 이고 '400 이면 충분' 이 아니다."*
(W4 `OPTIMIZER-EARNS` · S16 "솔버 이득이 필드에 닿는가" 와 같은 축.)

---

# 부록 A — 재집계 실측 (W4 `results.jsonl`, 1,344행, 무손실 재도출)

혼잡 64 에피소드 · 시드 median-of-3 · 에피소드 수준 med3 · 상대차 `(처치−대조)/대조` 중앙값:

| 팔 | med3 중앙 | PREREG §0 표기 | 일치 |
|---|---|---|---|
| `L-ITER0` | 0.182566 | 0.182566 | ✅ |
| `L-SHIP` | 0.157812 | 0.157812 | ✅ |
| `L-VP` | 0.117716 | 0.117716 | ✅ |
| `L-VP-NW` | 0.104503 | 0.104503 | ✅ |
| `ORA-K-CVX` | 0.001298 | 0.001298 | ✅ |
| `ZERO` | 0.250328 | (표 없음) | — |

| 짝 대조 | 승 | 중앙 상대차 |
|---|---|---|
| `L-VP < L-SHIP` | **52/64** | **−17.53%** |
| `L-ITER0 < L-SHIP` | 7/64 | **+23.34%** |
| `L-VP-NW < L-VP` | **43/64** | **−5.62%** |
| `L-VP < L-ITER0` | 59/64 | −36.23% |
| `L-STEP400 < L-SHIP` | 38/64 | −1.28% (널) |
| `L-VP < L-STEP400` | 48/64 | −17.30% |

⟹ **§0 의 모든 수치가 재현된다.** 사전등록의 "결과 전에 아는 것" 은 정확하다.
(`S-1` = `L-VP-ONLY < L-ITER0` 는 위 −36.23% 에 비추어 **사실상 기정사실** ⟹ 정보량이 낮다.)

# 부록 B — `MEASURED` / `ARGUED` 분리

**MEASURED** (코드 열람 또는 원자료 재도출):
1.1 호출 경로 부재 · 1.2 가산 키·`RMSE_near` 상존 · 1.3 커밋 2건/트리 clean/`code_md5` 단일 ·
1.4 스레드 비트불변(1·2·4·8, |Δ|=0) · 2.1 스텝 분해표 · 2.2 설치 순서 + early-break 부재 ·
2.3 `grow` 분리 · 2.4 **import 실패(스택트레이스)** · 3.(ii)(iii) clamp·성장 카운터 ·
S-2 `run_est` 키 목록 · `RMSE_near` 의 `grid_x<80` 정의 · 부록 A 전부.

**ARGUED** (측정 아님):
1.5 "프로브가 P-2 대조군을 못 친다" 의 위험 크기 · 3.(ii) "`L-VP-ONLY` 에서 clamp 발화율이 커질 수 있다" ·
S-3 의 P-1 예측 · S-4 의 "국소 분지" 해석.

# 부록 C — 처방 검증 (⚠"감사자가 옳은 결함에 틀린 수정을 처방한" A1 전례 대비)

| 처방 | 검증 |
|---|---|
| G-0 에 `L-VP@1595` 추가 | `probe_bitidentity.py` 는 `arm='L-SHIP'` 하드코딩 2곳뿐 ⟹ 루프화 가능. 비용 = W4 실측 21 s × 4셀 |
| `W4AC.ARM_DELTAS` 제자리 확장 | `arms_config.arm_config` 는 `ARM_DELTAS` 를 **런타임에** 조회(`:110-114`) ⟹ 제자리 update 로 충분, 파일 무수정 |
| `run_estimation` 래핑으로 `avg_RMSE_near` 회수 | 값은 `run_w4.py:366` 의 `r` 에 실재. 래퍼는 반환 dict 읽기 전용 ⟹ RNG·수치 무영향. `finally` undo 로 워커 누수 방지 |
| G-8 문언 개정 | 러너가 **이미** 그렇게 구현돼 있음(`:653-658`) ⟹ 코드 변경 0, 문서만 |
| §2 요인칸 · D-5 승격 | 카운터가 **이미 기록 중** ⟹ 런 비용 0 |

**⛔이 감사가 하지 않은 것**: MDE·검정력 재계산, `S-3`·`S-4` 셸 설계, 워치독(`watchdog_w5.py` 부재) 설계 검토,
`L-VP800` 의 벽시계 예산 검증. 세션 한도 때문에 상위 3건 + 짧은 4건에 집중했다.
