# WaveSplat Project — Claude Context

## Project Overview
Grid-free Spectral Gaussian wave field estimation from sparse ship-mounted LiDAR.
Target: OCEANS 2026 conference paper submission.

## Repository Layout
```
wavesplat/                   # Root package directory
├── wavesim/                 # Wave simulation + LiDAR
├── wavesplat/               # Estimation algorithm      (**250 tests**, 실측 2026-08-27)
│   ├── model/               # spectral_gaussian, dispersion, confidence
│   ├── optimizer/            # initializer, coarse_to_fine, sequential_update
│   ├── baselines/etkf_grid.py
│   ├── evaluation/metrics.py
│   └── run.py               # Main entry point
├── scripts/                 # 413 experiment/paper scripts
├── paper/                   # Ocean Engineering LaTeX (wavesplat.tex) + figures/ (5 png)
├── thesis/                  # thesis_main.tex — 학위논문
├── results/                 # 96 JSON result files (+ direction1/2/3 하위)
├── configs/                 # YAML configs (estimation.yaml, server.yaml, default.yaml)
├── data/                    # HDF5 episodes (110) + real_bergamasco/ (실측 **12** 레코드 — 원논문 축자, 고정 구조물 4곳. ⛔"13" 은 낡음)
├── tmp/                     # 배치별 사전등록·결과 (S1–S13, B-1~B-22, BLANKSLATE 등) — 연구 기록의 본체
│                            # ⛔**"하위 디렉터리 34개가 git 미추적" 은 낡았다(2026-08-27 실측 정정)** —
│                            # 08-21 커밋 `033e742` 가 34배치 502파일을 추적에 넣었다. 현재 미추적은
│                            # **4개뿐**(c2_cache·critic_v2·frf_lidar_2026-08-01/ws·m2m7, 전부 비-판정물).
│                            # 확인 = `git ls-files --others --exclude-standard --directory wavesplat/tmp/`.
├── docs/PRESENTATION_SUMMARY.md  # 발표 요약 (⛔줄 수를 인용하지 마라 — 편집마다 바뀐다.
│                            #  08-13 에만 562(HEAD)→640→683 으로 세 값이 돌아다녔다.)
└── requirements.txt
```
⚠**저장소 루트는 `/home/vexit/wavesplat`, 위 트리의 뿌리는 그 안의 `wavesplat/` 다.**
아래 명령은 전부 **`cd /home/vexit/wavesplat/wavesplat` 를 전제**한다.
(루트에는 별도의 `scripts/`·`tmp/`·`external/`·`ref_paper/`·`graphify-out/` 이 따로 있다.)

## Key Commands
```bash
cd /home/vexit/wavesplat/wavesplat     # ← 아래 전부 이 디렉터리 기준

# Run estimation  (⛔위치인자 아님. --adaptive_n 은 값을 받는다)
python3 -m wavesplat.run --episode data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n true

# Run tests  (실측 2026-08-27: **250 passed**, 0 failed, 0 skipped, 55.2 s)
#   ⚠논문의 \DeployNumTests 는 **손으로 고치지 말고** scripts/gen_deploy_constants.py 를 돌려라
python3 -m pytest wavesim/tests/ wavesplat/tests/ -v

# Generate paper figures/tables
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py

# Run comparisons
python3 scripts/run_p1_comparison.py       # Main 12-episode comparison
python3 scripts/run_gpr_comparison.py      # GPR 3-way comparison
python3 scripts/run_raydrop_comparison.py  # Raydrop model comparison
```

## Current State
⛔**이 절의 낡은 수치가 2026-08-12 전체 감사에서 BLOCKER 로 잡혔다.**
**연구 현황의 권위는 이 파일이 아니라 auto-memory (`MEMORY.md`) 와 `tmp/<배치>/RESULT.md` 다.**
- **250 tests** (`python3 -m pytest wavesim/tests/ wavesplat/tests/` 로 실측 2026-08-27).
  ⚠이 수는 자주 바뀐다 — **인용 전에 돌려서 확인할 것**(08-23 하루에 231→242→245).
  ⛔**"245" 는 낡았다**(08-23 값이 08-27 까지 이 파일에 남아 있었다).
- ⛔**"WS 0.223 / ETKF 0.319 / 13.4 ms / 132.6 ms" 인용 금지 — 단 사유가 2026-08-13 에 바뀌었다.**
  ⛔**"어느 결과 파일에서도 재현되지 않는다" 는 거짓이었다**(08-12 감사의 오류, 08-13 정정).
  `results/b1_full_comparison.json`(git 추적, 커밋 `e57528c`)을 집계하면 **네 수치가 전부 나온다**:
  `WaveSplat.avg_nRMSE` 평균 **0.2229** · `ETKF.avg_nRMSE` **0.3189** · `13.3760 ms` · `132.6387 ms`.
  **진짜 금지 사유는 셋이다**:
  1. **프레임 불일치** — 같은 파일에서 `WaveSplat.n_frames` 는 **120–220**, `ETKF`·`GPR-1` 은 **301**,
     그리고 **100/100 에피소드 전부에서 WS 가 더 적다**. 두 팔이 **같은 프레임에서 채점된 적이 없다**
     ⟹ 그 파일의 `84/100` 도 인용 금지. (S13 은 이 점검을 통과했다 — 0/128 불일치.)
  2. **기준선 선택** — 그 ETKF 는 `grid 40×30 · ne=100 · mult_inflation 1.0` 구성이다. 같은 `WaveSplat`
     열을 **튜닝된 LETKF** 상대로 재도출하면 `results/100ep_ws_letkf.json` 에서 **0.2229 vs 0.2278,
     51/100, p=0.9204 = 널**이다.
  3. **지연은 프로세스 간 비교** — 동일프로세스 1스레드 정직값은 **1.5–2.6×**, CPU 는 **AMD Ryzen 7 7800X3D**.
  (GPR 0.276 은 재현됨.)
- ⛔**"All phases complete" 금지** — 연구는 진행 중이고, 학위논문 제1장 기여 중 다수가 이후
  측정에서 무효화됐다. ✅**논문 편집 정지는 2026-08-28 에 전면 해제됐다**(사용자 축자 *"논문작업 전부 승인이야"*,
  08-22~28 의 46커밋 **사후 추인 포함**). ⚠단 `paper/` 의 **파기·개명 처분은 별개 결정**이다.
  상세 = `memory/feedback_no_paper_edits_until_research_done.md` · `tmp/STATUS_2026-08-28/STANDING_ORDER_AUDIT.md`.
  ★**교훈: 해제는 받은 그 자리에서 축자로 기록한다** — 08-22 커밋이 해제를 주장했으나 근거가 저장소에 없었다.
- 배포 상태·살아있는 판정·인용금지는 **`MEMORY.md` 와 `reference_citation_bans.md`** 를 볼 것.
- **2026-08-13 멀티에이전트 재감사** (7축 + 적대적 반증 3건):
  `wavesplat/tmp/AUDIT_recheck_2026-08-13/` — S13 판정 자체는 두 독립 재집계에서 전건 재현되나,
  요약 서술 다수가 측정보다 넓었다. ⛔**08-12 헤드라인 `85/100 p=4.8e-13` 인용 금지**
  (100ep 코퍼스가 레거시 생성기 = 위상/방향 draw 가 시드 0–4 뿐, **클러스터 바닥 p=0.031**;
  게다가 λ 와 2요인 교락). 결론 자체는 S13 이 수정 생성기 위에서 독립 방어한다.

## Important Design Decisions (출하 기본값 — 코드에서 확인)
⚠★**2026-08-22 에 출하 기본값이 셋 바뀌었다. 새 배치의 기준선 팔은 이 셋을 명시하라.**
| 키 | 이전 | 현재 | 근거 |
|---|---|---|---|
| `varpro_steps` | (키 없음) | **400** | W4·W5·`B` (VARPRO 배포) |
| `learned_init_iters` | 500 | **0** | W5 (결합 적합은 잉여) |
| `auto_switch_thresh` | 8 | **0** | W11·W13·W14 (합성 경로 통일 = 항상 학습) |
★**되돌리기는 `sequential_update.pre_2026_08_22_config()` — 스위치가 셋이다.**
⛔한둘만 끄면 배포된 적도 측정된 적도 없는 제3·제4의 팔이 된다.
⚠**`dict(DEFAULT_CONFIG)` 를 런타임 스냅샷하는 러너는 이 셋을 자동 흡수한다** — 사전등록에
*"explicitly, no fallback"* 이라 적어도 그 부류는 막지 못한다(S13·S13b·HOSGATE 가 그렇다).
⚠부수효과: 통일로 **`analytic_ensemble_k` 앙상블이 합성에서 구조적으로 죽었다**(라우터가 해석적을 안 고른다).
⚠**실측은 통일에서 제외**된다 — 실측 게이트가 `bank_mmse` 로 보낸다(W12: 학습 강제 시 예보 지평 0/5 패배).
⟹ ★**그래서 통일을 `init_method` 기본값이 아니라 임계로 했다** — 기본값을 바꾸면 게이트가 조용히 사라진다.

- `tracking_mode: w_phi` — kx/ky fixed during tracking, only w+φ updated (`estimation.yaml:22`)
- `init_method: auto` — GT-free 라우터 (`run.py:43`, `estimation.yaml:24`).
  ⛔**"v2_konly" 는 낡음** — 2026-07-26 flip 이후 기본값이 아니다(명시 지정 시에만 사용).
  ⛔**"둘 다 dead default" 는 거짓이다 (2026-08-22 실측 정정).**
  · `run.py` yaml 폴백 `'v2'` → ✅**dead 확인**(`optimizer:` 블록을 가진 yaml 이 `configs/estimation.yaml` 하나뿐이고 그게 명시한다).
  · `sequential_update.py` 폴백 `'v2_konly'` → ⛔**LIVE**. `init_method` 없이 생성자를 부르고 **실제로 추정을 돌리는 곳**이 있다:
    **배포 테스트** `wavesplat/tests/test_phase_convention.py:125` · **실측 경로** `tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py:107`
    (→ `:112 initialize` · `:124 update`) 외 다수.
  ⟹ ★**두 값을 하나로 병합하면 동작이 바뀐다** — 기록 배치가 조용히 재라우팅되고, `make_updater({})` 경로에 `'auto'` 가 들어가면
  **라우팅은 되는데 median-of-k 앙상블은 없는, 배포된 적도 측정된 적도 없는 제3의 팔**이 생긴다.
  ⟹ 병합하지 말고 **두 상수를 각각 명명·문서화**할 것(`INIT_METHOD_DEFAULT` / `INIT_METHOD_CLASS_FALLBACK`).
- `lambda_l1: 1e-4` — **유일 정의 = `optimizer/sequential_update.py:40 LAMBDA_L1_DEFAULT`**;
  `wavesplat/run.py:30` 이 import 하고 `configs/estimation.yaml:12` 이 미러한다.
  ⛔**"세 곳에 중복 기재" 는 낡은 지시** — `5f0846c` 에서 이미 단일출처화됐다.
  **배포 시점이 경로별로 다르다**(2026-08-13 git 재확인):
  · `40e22f6`(08-11 17:19) = `estimation.yaml` + `run.py` DEFAULT_CONFIG + `run.py:94` yaml 폴백
  · `5f0846c`(08-12 00:30) = `sequential_update.py` 폴백(그전까지 **여전히 1e-3**)
  ⟹ ⛔**"08-11 에 폴백이 1e-3→1e-4" 는 거짓.** 주입 지점은 `run.py:30` DEFAULT_CONFIG 다
  (`run.py:139-141` 이 먼저 병합).
  ⚠**기본값을 바꿀 때는 기준선 팔이 λ 를 명시하는지 반드시 확인.** 실측된 범위(V3, 2026-08-13):
  · **이미 무효 = 0건** — S5·S6·S7·S8·S9 는 배포 **전** 실행 + 하네스 비트결정성으로 이중 확인
    (복제 셀 Δ≡0 인데 처치−기준선 Δ≠0 이 S5 96/96·S6 96/96·S7 192/192·S8 192/192·S9 288/288).
  · **재실행 시 무효** — λ 미명시 러너(S1–S4 는 0회 명시, λ-처치 배치 S5·S6·S7·S8).
  · ⛔**배포 후 실행돼 실제로 오염된 것** — `tmp/BLANKSLATE_2026-08-12/bs1_shipping_default.py:37`
    (`v2_konly_100ep` 팔이 "100ep_ws_letkf.json 이 돌린 것"이라 주석하나 원본은 λ=1e-3),
    `tmp/DEBATE_blankslate_2026-08-12/r2_phase_attrib.py:31`(공개 컬럼 대비 비트일치 **0/30**),
    `r2_auto_rest70.py:47` ⟹ **"출하 auto 85/100" 은 `init_method` 단독이 아니라 λ 와의 2요인 묶음.**
  ⚠`sequential_update.py:426`·`:967` 에 **λ=0.001 하드코딩** 잔존 — 배포 후 의미가
  "출하 기본값"에서 "구 기본값"으로 조용히 바뀌었다.
- ⛔**`amplitude_refit` 위상 부호 결함 — 2026-08-19 수리됨(`b1467bd`).** `sequential_update.py:1335` 가
  `arctan2(b, a)` 였는데 전방모델이 `w·cos(θ+φ)` 라 `b = −w·sinφ` ⟹ `atan2(-b, a)` 가 옳다.
  저장소의 다른 LS `(a,b)→φ` 사이트 5곳은 이미 옳았고 이 줄만 이탈했다. **과거 오염 0건**(전수 확인:
  기본 False · 노출 0 · `git log -S/-G` 0커밋 · 결과 JSON 1,147개에 `refit` 키 0개).
  ★**왜 안 잡혔나**: 같은 함수의 자기진단 `residual_after` 가 **올바른 계수**로 계산되어 두 규약에서
  비트 동일하다 — **진단은 통과하고 결과만 틀린다.**
  회귀 테스트 `wavesplat/tests/test_phase_convention.py` 추가(204→**207**).
  ⟹ 일반 규율은 「측정 규율」의 **자기진단** 항목(같은 부류가 `B-2` 수심에서 재발).
- ✅**`RMSE_near` 는 이제 저장된다**(2026-08-19, `b1467bd`) — `run.py` 가 `avg_RMSE_near` 를 요약에 집계한다.
  그전까지 `metrics.py:433` 이 채운 값을 아무도 집계하지 않았고 `main()` 이 `frame_results` 를 삭제한 뒤
  덤프했다 ⟹ **저장된 모든 결과 파일이 근거리에 구조적으로 눈이 멀어 있었다**(약한 게 아니라 **검출 불가**).
  ⚠**그 이전 파일들에는 여전히 없다.**
- Adaptive N: energy-based pruning + residual-triggered growth. ⚠**효과는 NULL 로 측정됨**(08-03).
- Dispersion hard constraint: `ω = √(g·|k|·tanh(|k|·d))`

## 측정 규율 (이 저장소에서 반복적으로 대가를 치른 것)
- **열람 범위 = 주장 범위.** 실제로 연 파일만 근거로 삼고, 감사 범위를 명시할 것.
- **집계·방향·MDE 를 결과 전에 동결.** 같은 자료가 규약 하나로 판정이 뒤집힌 적이 여러 번 있다.
- ★★★**새 축을 제안하기 전에 그 코드가 배포 혼잡 경로에서 실제로 실행되는지 확인하라**
  (2026-08-15 novelty 탐색이 후보 4개 중 2개를 이 사유로 죽였다 — 누적 21축 사망).
  **`sequential_update._initialize_learned` 가 코스투파인 staging 을 우회한다**(독스트링이 스스로
  *"the coarse-to-fine N-staging is bypassed"* 라 적었다). 라우터가 **혼잡 90.6%** 를 그쪽으로 보낸다.
  ⛔**단 "실행되지 않는다" 는 거짓이다(2026-08-19 정정).** 우회되는 것은 **`coarse_to_fine.py` 하나뿐**이고
  거기서는 진짜 0회다(학습 경로 `coarse_to_fine.py:173` **0/130**, `v2_konly` 14/14). 그러나
  **같은 기능의 두 번째 구현이 `sequential_update.py:1065-1200` 에 살아 있고 조건절이 `'learned'` 를 포함한다**
  (`:1111-1116` `MIN_DK`, `:1193` 난수 위상, 호출 지점은 `_initialize_learned` 자신 `:660`).
  실측: `SHIP` **혼잡 `add_calls>0` = 17/96**(학습 경로만 **8/87 = 9.2%**), 사이트별 `:1188` 14 · `:173` 9.
  ★**정확한 문장 = "성장은 배포 혼잡 런의 82%에서 발화하지 않고, 발화해도 넣는 원자는 28/1,660 = 1.69% 다."**
  ⟹ **죽은 축들의 판정은 유지되고, 사유만 바뀐다**: *"도달 불가"* → **"지렛대 부족"**.
  ⚠**코드 독스트링(`:620-626`)은 원래 옳았다**(*"the same adaptive prune+grow"*) — **문서가 그 문장을 떨어뜨렸다.**
  학습 경로의 실제 분리 문턱은 `learned_dedup_dk=0.03`(`optimizer/learned_init.py:341`)이고
  `w,φ` 는 `:386 ls_fit_wphi(ridge 0.5)` 로 **이미 LS 적합**된다(`learned_konly` 기본 False).
  ★**축이 죽는 진짜 사유는 "안 돌아가는 코드" 가 아니라 "같은 기능이 두 파일에 중복 구현돼 한쪽만 계측한 것" 이다**
  (λ 3중 기재와 같은 결함 부류). ⚠**2026-08-19 W4 배치에서 이 부류가 하루에 네 번 재현됐다**
  (개입 기대값 320/820 이 러너 G-2·G-3 · `arms_config.ARM_GATES` · `watchdog.py` 네 곳에 중복)
  ⟹ ★**처방: 개입 기대값은 단일출처여야 한다.**
  ⟹ **처방: 원자 카운터로 런 0회 확인**(W3 설계가 그렇게 `SHIP-FITPHI` 0/3 개입을 런 전에 잡았다).
- ⚠**배포 팔의 분리 문턱 0.03 은 측정된 분해능 한계 양쪽보다 작다**(반경 0.042–0.074 · 방위 0.107–0.179)
  ⟹ 위험은 "바닥이 넓다" 가 아니라 **한 주엽 안 중복 원자**다. `merge_atoms` 는 **이미 구현·GT-free·
  끄면 비트 동일**이고 `learned_merge_frac` 기본값이 `None`(no-op), `configs/`·`run.py` 에 **미노출**이다
  (`run_estimation(config=…)` 로는 도달 가능, CLI/YAML 은 화이트리스트에 막힌다).
  ⛔**단 "공짜 티켓" 은 틀린 서술이다(2026-08-19 정정).** 저장소가 **2026-07-26 에 이미 쟀고 혼잡 20셀 0승**,
  `frac=0.375` 는 `no_harm` **FAIL**(p=0.03125)이었다. W4 가 `frac=0.5` 로 재측정: **9/47 lower, 동점 17,
  rel median +14.99% = 같은 방향(무익)**. ⚠*"문턱 0.03 보다 크게 올리면 된다"* 는 논증도 끊긴다 —
  측정 반경 0.0191·0.0287 은 **아직 0.042 미만인데 이미 해쳤고**, 0.042 에 닿으려면 `frac≈0.55` 인데
  **0.5 에서 이미 셀 45%가 비트동일 no-op, 그리고 상당수가 원자를 잃는다.**
- **"출하 팔에서도 재라."** 저장소에 "WaveSplat" 이라 불리는 서로 다른 추정기가 **최소 4개** 있다
  (출하 `SequentialUpdater` · `bank_mmse` · `mmse_bank_estimate` · `relax_step`+`tnse`).
  ⚠**`results/direction1/*` 를 뭉뚱그리지 마라** (2026-08-13 정정): `d1_predictable_zone.py` ·
  `d74_closed_loop_trajectory.py` · `d72_platform_aperture.py` · `d73_tank_scaled_aperture.py` 는
  `wavesim` 참조 **0건**(= LiDAR 0회 호출, 출하 시스템의 증거 아님)이지만, **`d1_crb.py` 는 다르다** —
  `:39` 가 `from wavesim.data_writer import load_episode`, `:149` 가 호출, `:154` 가
  `lidar_points`·`lidar_valid_mask` 를 관측으로 쓴다. ⛔**"direction1 전 계열이 LiDAR 0회" 는 거짓.**
- **전역 승리는 근/원 셸로 분해할 것** (`SUPPORT-TRADEOFF`).
  ⛔**재현 횟수를 서수로 세지 마라** — "6번째/7번째"가 문서마다 되감겼고, 원본 판정은
  "단일 요인(`tracking_lr==0`) × 단일 자 × 6회, **독립 처치 0건**" 이다.
  ⛔**`SUPPORT-TRADEOFF` 를 "우리 방법의 이득은 먼바다에 있다" 로 읽지 마라**
  (2026-08-15 전수 감사, `tmp/DEBATE_novelty_2026-08-14/AUDIT_tradeoff_axis.md`):
  **후보 10건 전부가 WaveSplat 내부 처치 대조이고 외부 기준선 행이 0개다**
  (`Rx6·Rx7·E-B·E-D·S7·S11·BLANKSLATE` = 추적기 on/off · `K6`·`B-1` = 오라클 · `E-C` = 팔 안 gap).
  허용 = **"우리 노브(주로 추적기 off)를 돌리면 근거리를 팔아 먼바다를 산다"**.
  ⚠**기준선 대비 자료는 3건뿐이고 전부 미확정**: HOSGATE `LETKF_PLUS` 재실행(9셸 전부 승, n=16 탐색적) ·
  S15 스모크(4짝 중 **1짝은 `obs:0_5` +7.7% 로 SHIP 이 진다**, `direction_deg=180` 에서만) ·
  `results/b1_full_comparison.json`(`far`·`unobserved` 만 있고 **`RMSE_near` 는
  `scripts/run_b1_full_comparison.py:51-54` 가 버려서** 근거리가 **설계상 검출 불가**).
  ⚠**셸(`obs:`)과 배-거리 고리(`ship:`)는 다른 양이다** — 같은 표본에서 1.8–4.1배 갈린다.
  어느 쪽을 쟀는지 **접두어까지 적을 것**(S13 이 여기서 사전등록을 이탈했다).
  ★**팔별 필드 존재를 PREREG 게이트로 검사할 것** — S13·HOSGATE 가 기준선 셸을 0행 기록해
  `S-2a` 를 두 번 못 쟀다. 본보기 = `tmp/S15_DIRECTION_2026-08-14` 의 **G-8/G-8b**.
- ★**`RMSE_near` 가 계산되는데 저장되지 않는다** — `evaluation/metrics.py:433` 이 채우고
  `scripts/run_b1_full_comparison.py:51-54` 의 키 목록이 버린다. **수리는 키 한 줄**이고,
  그전까지 그 파일로 근거리 주장을 하지 마라(부재증명 아님).
- ⛔**합성 GT 가 추정기 모델식과 100% 동일하다** (2026-08-13 실측, BLOCKER).
  저장된 `gt_wave_field` 를 `Σ A cos(kx·x+ky·y−ωt+φ)` 로 재구성하면 희소·혼잡·공개 코퍼스 4개
  에피소드 전부 **max|Δ| = 0.000e+00**. 원인은 `wavesim/gt_generator.py:38` 이 `single_gerstner` 의
  **변위 성분(X_disp/Y_disp)을 버리는 것** — Gerstner 변위(Q=0.5)는 생성되지만 GT 필드에도
  레이캐스트 메시에도(`wavesim/pipeline.py:157,165`) 들어가지 않는다.
  ⟹ `model/spectral_gaussian.py:16` 의 *"의도된 모델 불일치 O(ε²)≈0.25%"* 는 **실측 0%** 이고,
  "표현 아티팩트" 단서는 **희소뿐 아니라 코퍼스 전체**에 붙어야 한다.
  ⟹ 상시지시 **"검증에 비선형 필수" 는 S7~S13 에서 한 번도 충족된 적이 없었다.**
  ✅**2026-08-13 `HOSGATE` 192런이 이 라인에서 처음 충족했다** (`tmp/HOSGATE_2026-08-13/`, `ROBUST`).
  같은 성분으로 (선형 GT, HOS M=3 GT) **짝**을 만들어 재검정: **혼잡은 살아남고**(HOS 14/16 −25.4%
  vs LIN 14/16 −28.9%, 문턱 13), ★**희소는 표현이 하중을 받는다**(`SHIP` 희소 절대 nRMSE
  **0.0195 → 0.0700 = 3.6배**, 반면 `LETKF+`·`ZERO` 는 ±0.5% 무감).
  ⟹ ⛔**"희소 −93%" 를 알고리즘 우위로 인용 금지**(이제 추론이 아니라 **개입 측정**이다) ·
  ⛔**"비선형 재학습이 최고가치 축" 인용 금지**(혼잡이 버텼으므로 그 근거가 없다).
  ⚠범위: `mesh_verdict` 가 시드별로 갈려 `DEGRADED` 셀은 **사실상 2차 비선형성**이다
  (`gen_manifest.json`). HOS 도 **여전히 합성**이라 실측 갭은 안 닫힌다.
  ⚠**`wavesim/**` 은 무수정** — HOS 는 `config.hos.enabled` 로 켠다(기본 False).
- ⛔**테스트가 배포 기본값을 전혀 지키지 않는다** (2026-08-13 실측).
  `lambda_l1` 1e-4→1e-3 · `init_method` auto→v2_konly · `tracking_mode` w_phi→all 을 **동시에 되돌려도
  `111 passed`**. 테스트 전체에 `DEFAULT_CONFIG`·`configs/` 문자열이 **0회** 등장한다.
  ⟹ S7·S8 을 무효화시킨 바로 그 결함 부류에 **회귀 테스트가 0개**다.
- **요약은 사전등록보다 넓게 나가지 마라.** 2026-08-13 감사에서 S13 의 PREREG 는 정확했는데
  RESULT/메모리 요약이 부풀어 있었다("신선 시드 재확인"·"민감도 전부 통과"·"사전노출 24").
- **처방도 검증하라.** 감사 에이전트가 옳은 결함을 찾고 **틀린 수정**을 처방한 사례가 있다
  (A1 이 `obs:0_5` → `ship:10_20` 교체를 지시 = 두 번째 이탈이 될 뻔했다).
- **재실행 전에 추정기 시드 수를 확인하라.** S13 은 시드 1개(42)로 돌았고 PREREG 에 언급이 없었다.
  `S13b`(256런, 08-13)가 시드 43·44 를 붙여 **PRIMARY 는 지켰지만**(혼잡 median-of-3 61/64 −28.2%,
  세 시드 전부 문턱 위) **부차 주장 하나를 죽였다** — "성분수 단조 약화"는 **단일 시드의 모양**이었다.
  시드 간 CV 는 혼잡 중앙 **14.4%**·희소 **46.5%** 이고 **혼잡 18/64 에서 시드가 승패 부호를 가른다**.
- ⚠**행에 `git_head` 를 적어도 재현이 고정되지 않는다** — **더러운 작업트리를 못 잡는다**(S13·S13b 둘 다
  배치 중 다른 세션이 임포트 경로 파일을 수정했다). **임포트 경로 파일의 md5 를 동반 기록할 것.**
- ⚠**`effective_config` 를 입력 dict 복사로 만들지 마라(= 의도표)** — S13b 가 **객체 되읽기**로
  `analytic_ensemble_k` 앙상블이 실제로는 **런의 14%(36/256)에서만 작동**함을 드러냈다.
  의도표 방식으로는 보이지 않던 값이다.
- ★★★**자기진단이 통과하는 것은 판정이 불변이라는 뜻이 아니다.** 같은 부류가 두 번 나왔다:
  · `amplitude_refit` 위상 부호(위 「Important Design Decisions」 항목) — 같은 함수의 자기진단
    `residual_after` 가 **올바른 계수**로 계산돼 두 규약에서 **비트 동일**했다
    ⟹ *"진단은 통과하고 결과만 틀린다."*
  · ★**`B-2` 수심 `17.0`→`30.0`**(2026-08-26 발견, ⛔`B-2`…`B-2f` 는 **전부 17 로 돌았다**).
    당시 주석은 효과를 **분산관계에서** 재고 *"Δω = 0.0000–0.0012% ⟹ 수치적으로 무해"* 라 결론했다.
    **분산관계에는 옳았다.** 그런데 **하류 판정**에서 여럿이 널로 뒤집혔다 —
    층 분해 Fisher **p 0.0265→0.1591**(참 기준 0.0455→0.2475) ·
    `B-2b` σ 개입 **10/12 p=0.0193 → 7/12 p=0.3872**. (판정 라벨과 짝 비교의 **부호**는 견뎠다.)
  ⟹ ★★★**처방: 개입·오류의 효과를 그것이 들어가는 양에서 재지 말고 판정에서 재라.**
  값이 바뀌면 **그 값을 쓰는 판정을 재실행해 verdict 을 diff** 할 것.
  본보기 = `tmp/B2_REAL_2026-08-23/depth_recheck.py` — 두 구성을 **나란히 재집계**하고
  문서와 어긋나면 `AssertionError`(런 0회, 동결 결과 파일만 읽는다).
- ★**동결 목록은 배치의 모든 판정코드와 그 임포트를 덮어야 한다.** 검사기는 *"동결 뒤에 움직였다"* 만
  막았고 ***"동결한 적이 없다"*** 는 안 막았다 — ★**후자가 더 나쁘다**: 기준선이 없으면 `md5` 열이
  영원히 `OK` 를 찍어 **구조적으로 탐지 불가**다(전자는 최소한 탐지된다).
  2026-08-26 부터 `tmp/_audit/check_batch_artifacts.py` 의 `_judge_files()` 가 이를 강제한다 —
  배치 안의 `gates*`·`run*`·`analyze*` 와 **그것들이 임포트하는 배치-내 모듈**을 요구하고
  (`scene_station.py`·`zero_ref.py` 처럼 이름이 규약 밖이어도 잡힌다), 빠지면 `⚠미동결N` 이다.

## Language
User prefers Korean for discussion, English for code/technical terms.
