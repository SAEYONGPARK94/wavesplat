# W3_SOLVER_2026-08-15 — 진행 로그 (append-only)

세션 한도로 죽은 전례가 있어 **진행 중 계속 append** 한다.

## 2026-08-15 설계 착수

- 배경 = `memory/project_method_debate_2026-08-15.md` §5 (P0 `ORA-K-NOREG`, P1 `W3` 위상 수리) + §1 (닫힌형 lstsq 상한).
- 임무: 사전등록 + 러너 + 스모크(2 에피소드)까지. **전량 실행 금지.**

### 코드 열람 범위 (주장 범위 = 열람 범위)
- `wavesplat/run.py` (DEFAULT_CONFIG, `run_estimation`)
- `wavesplat/optimizer/initializer.py` (`_rx2_gate` 게이트 목록, `ORACLE_COMPONENTS`, oracle init)
- `wavesplat/optimizer/coarse_to_fine.py:150-200` (`RX2_GROW_FITPHI` 분기 = **배포 경로의 난수 위상**)
- `wavesplat/model/spectral_gaussian.py:100-140` (`forward` = Σ w·cos(kx·x+ky·y−ω·t+φ))
- `wavesplat/data_loader.py:1-80` (프레임/관측/GT 격자)
- `tmp/rx2_init_2026-08-05/run_rx2b.py` (frozen 러너 — CELLS/apply_cell/cell_config/run_one)
- `tmp/S4_phase_oracle_2026-08-08/run_s4.py` (오라클 사다리 전례)

### 확정 사실 (읽어서 확인)
1. `ORA-K` = frozen `run_rx2b.CELLS['ORACLE-K']` = `init_method='oracle'`, `n_gaussians=[n_gt]`,
   `loss_thresholds=[None]`, `max_iters_per_stage=[900]`, k **학습 가능**. (`__oracle__` 은 `init_k_lr_scale=0.0`.)
2. frozen `BASE_CFG` 에 **`lambda_l1` 이 없다** ⟹ `run_estimation` 의 `DEFAULT_CONFIG` 폴백을 받는다.
   그 폴백이 08-11 에 1e-3→1e-4 로 **배포되면서 바뀌었다** ⟹ ★**과거 4연속 널이 "λ=1e-3" 이라는 전제는
   그 배치들이 배포 전(08-11 이전)에 돌았기 때문에 성립**하고, **지금 그대로 재실행하면 λ=1e-4 가 된다**.
   ⟹ W3 는 **세 팔 모두 `lambda_l1` 을 명시**한다(1e-3 / 1e-4 / 0.0). (MEMORY 의 S7·S8 무효화 사건과 같은 함정.)
3. `RX2_GROW_FITPHI` 는 `initializer._rx2_gate` 로 **호출 시점에** `os.environ` 을 읽는다
   ⟹ 워커 프로세스 안에서 세팅해야 하고, frozen `apply_cell` 이 매 셀마다 4개 키를 **전부 '0' 으로 덮어쓴 뒤**
   셀 지정값을 넣는다(누출 방지). 이걸 그대로 쓴다.

### 2026-08-15 20:04 KST — PREREG 동결
- `PREREG.md` md5 `9f4e9c7e7bf534253b4049b7c91cca71`, git HEAD `5f0846c`.
- 동결 시점에 `results.jsonl` · `smoke_*.jsonl` **둘 다 부재** (확인함) ⟹ 어떤 런보다 먼저.
- 성격 = **self-attested freeze** (외부 타임스탬프 권위 없음).

### 설계 중 발견한 전제 결함 3건 (PREREG §1.1 에 기재)
- **F-1** frozen `run_rx2b.BASE_CFG` 에 `lambda_l1` 부재 ⟹ `DEFAULT_CONFIG` 폴백(08-11 에 1e-3→1e-4 배포).
  과거 4연속 널이 "λ=1e-3" 이었던 건 **배포 이전이라서**이고, 오늘 그대로 돌리면 1e-4 다.
  ⟹ 전 팔 λ 명시.
- **F-2** `_initialize_learned` 가 코스투파인 staging 을 **우회**한다(코드 주석이 그렇게 적음)
  ⟹ `auto`→`learned` 경로에서 `coarse_to_fine._beamform_residuals`(FITPHI 사이트 ①) **0회 호출**.
  살아 있는 사이트는 `sequential_update._adaptive_prune_grow`(사이트 ②)뿐이고 성장 트리거 조건부.
  ⟹ 원자 단위 카운터 필수 + `v2_konly` 짝(`KONLY`/`KONLY-FITPHI`) 추가.
- **F-3** 지정된 PRIMARY 는 "λ 가 오라클 팔을 해친다"이고, "4연속 널 = λ 인공물"은 **오라클 vs 배포**
  대비라야 한다 ⟹ `S-2` (`ORA-K-NOREG < SHIP`) 로 사전등록. PRIMARY 통과만으로 그 문장 금지.

### 문턱·검정력 (정확 이항, 지금 계산)
- 혼잡 n=32 ⟹ **k\*=23/32**, 정확 단측 α=**0.010030** (22/32 는 0.025048 로 **미통과**).
  power p=0.80 → **0.910**, p=0.85 → 0.984. MDE(power .80) = p≥**0.766**.
- 희소 n=16 ⟹ k\*=13/16, α=0.010635, MDE(power .80) = p≥0.853. (서술 층)

### 시드
- `1592, 1593, 1594`. 충돌 검사: `tmp/**/*.jsonl` 의 `seed|est_seed|filter_seed` 전수 스캔 →
  1500–1900 실사용 최대 **1591**(S15) ⟹ **충돌 0**. 잔여 [1595–1818].

### 2026-08-15 20:13 KST — AMENDMENT_1 동결 (판정 런 이전, `results.jsonl` 부재)
- md5 `db6d32039dc275ce45a5fff2b87ea647`. **PREREG.md 는 미수정**(md5 재검증).
- A-1 G-10 을 "유효 행 집합"(job_id 별 마지막 성공 행)으로 명시 + `NONDETERMINISM` 검사 추가.
  동기 = 스모크에서 CUDA OOM 1건(8워커가 GPU 하나를 물음).
- A-2 GPU/비GPU 위상 분리(수치 불변). A-3 `DEGENERATE` 대조일 때의 문장 규율.
  ⚠A-3 는 2에피소드 스모크 방향을 본 뒤 작성 — **주장을 약화시키는 방향으로만** 작동한다.

### 2026-08-15 — 스모크 완료 (49잡, 3.5분, 8워커)
- 게이트 G-2/3/5/6/7/8/9/11 **전부 PASS**. G-1·G-10 은 스모크라서 설계상 FAIL.
- ★★★**G-4 = 이 배치 설계의 소득**: `SHIP-FITPHI` 가 혼잡 **0/3 개입**(원자 0개), 세 시드 값이
  `SHIP` 과 **완전 동일** ⟹ 배포 팔에는 위상 수리가 닿는 지점이 없다(`_initialize_learned` 우회).
  `KONLY-FITPHI` 는 3/3 개입(원자 12–16, 게이트 `coarse_to_fine.py:173`).
  ⟹ **S-3 는 `UNDECIDED-NO-INTERVENTION` 예상, 실제 시험은 S-4.**
- ★★★**`ORA-K-CVX` 상한**(저장소 채점기): 혼잡 30comp **0.00201**, 희소 2comp **0.00144**
  (`ZERO` 0.240/0.250). rank full, cond 32.5/1.0, 창 잔차 0.00548 < 진실 0.00583.
  셸 평평(0.0023–0.0025) ⟹ 여기엔 `SUPPORT-TRADEOFF` 없음.
  ⚠**한계**: 합성 GT 와 모델이 **같은 식**이므로 이 상한은 **솔버 격차**만 잰다(모델계열 격차 아님).
- ⚠`ORA-K@L3` 이 혼잡 스모크 에피소드에서 `ZERO` 보다 나쁘다(0.2534 vs 0.2401)
  ⟹ 전량에서도 그러면 A-3 의 `(DEGENERATE-CONTROL)` 꼬리표 대상.
- 비용: 순차 상한 9.82 h, 8워커 효율 ×7.62 ⟹ **단일 위상 ≈1.3 h / 2위상 ≈1.8 h**.
- 분석기 리허설 `dryrun_smoke.py` → `dryrun_smoke.log`: 모든 게이트·부호검정·셸 경로가 실행됨.

### 2026-08-15 20:47 — 런 전 커밋 (`196050e`)
- `PREREG` · `AMENDMENT_1` · 감사 3인 · 스모크 원자료 · 코드 지문이 **전량 실행 이전에** 커밋됐다
  ⟹ 동결 성격이 `self-attested` → **`self-attested + git-anchored`**. (외부 권위는 여전히 없음.)

### 2026-08-15 20:48–20:51 — AMENDMENT_2 동결 (감사 13건 반영, 판정 행 0행)
- `AMENDMENT_2.md` md5 `6a46627308702d13a1ea592c78fb0cb7` (작성 20:48:28).
- 함께 동결: `analyze_w3.py` `4ff6fdc4…` · `dryrun_smoke.py` `2e9a41da…` · `run_w3.py` `b221ec29…`
  ⟹ **판정 경로가 처음으로 봉인 안에 들어왔다**(중립 U-2/W-6).
- 새 코드 지문 **`ff818418579590b976314c2f5ac3d7b4`**(87파일). 전량 행은 전부 이 digest 여야 한다.
- ⚠**pilot-informed**: 스모크 49행을 이미 본 상태에서 썼다(blind 아님). `results.jsonl` 부재 확인.
- 반영: CON B-1…B-7 · 중립 F-1/D-1/U-1/D-2/U-3 · W-3 스모크 · 추가로 W-2(경계 보수화) ·
  W-1/W-4/W-11 문장 정정. 미개정 6건은 이유를 `AMENDMENT_2 §미개정` 에 적었다.
- 계획 잡 **1,200 → 1,212**(`SHIP-DUP` 12). 러너 무런 게이트 G-1/G-7/G-9 **PASS** 재확인.
- 분석기 검증 2단: ① `dryrun_smoke.py`(스모크 원자료) ② **합성 1,212행**으로 `main()` 전 경로
  (Holm step-down · eff_thresh · 동점 인쇄 · DEGENERATE-ARM 꼬리표 · 코퍼스 분해 · §8 판정명 ·
  셸 · 상한표) 실행 확인 — 합성 파일은 scratchpad 에만 썼고 배치 디렉터리에 남기지 않았다.

### 2026-08-15 20:50 — W-3 스모크 (15comp 혼잡, 6잡, 0.5분) — **결과 2건**
1. ★**성장 분기는 발화하지 않는다.** `n_gt=15 < max_n=30` 인데도 `lp_calls=0`·`add_calls=0`·
   `N_final=15`(6/6 행) ⟹ 오라클 팔은 이 셀에서도 beamform 성장에 **들어가지 않는다**.
   ⟹ (i) G-4 의 "OFF 팔 전 행 `fitphi_atoms==0`" 이 **문제의 셀에서 확인**됐고,
   (ii) `ORA-K` 는 PRIMARY 층 전체에서 **3요인 묶음 그대로**(4요인이 되지 않는다) ⟹ CON W-3 해소.
2. ★★**B-6(a) 의 k 표류가 0 이 아니다.** `k_drift_rms` **0.053–0.097 rad/m**, max 0.29,
   **|k| 대비 중앙 7.5–28%**(그 에피소드 |k| 중앙 0.247) ⟹ ★**`ORA-K` 는 "오라클 k" 를
   초기화에서만 갖고 끝에는 잃는다.** ⟹ `ORA-K-CVX` 대비는 **진짜로 4요인 묶음**이고
   ⛔"72배 = 순수 솔버 격차" 는 인용 불가(AMENDMENT_2 B-6b 의 라벨을 쓴다).

### 남은 것
- ~~3에이전트 감사 대기~~ 완료 → 개정·재동결 완료 → **전량 실행**.
- 감사가 볼 만한 약점 후보(내가 아는 것):
  1. `ORA-K` 는 3요인 묶음(오라클 k + `[n_gt]` 단일단계 + 900스텝) — S-2 는 묶음 대비.
  2. 혼잡 32 가 **두 코퍼스 풀링**(indep 16 + s9 16). 코퍼스는 행에 기록하고 16/16 분해 병기.
  3. `ORA-K-CVX` 는 무가중·무추적 — 출하 손실은 confidence 가중.
  4. 추적기 ON 이라 S3/S4 의 `ORA-K-NT`(추적기 OFF) 바이트 재현이 **아니다**.

### 2026-08-15 20:20 — 마무리 점검
- `code_fingerprint.json` 갱신: digest **`696fa102bb05dd23f04ea18c8421ee12`**(87파일) = 스모크 전 행의
  `code_md5` 와 **일치**. 지문에 `run_w3.py` 와 `PREREG.md` 가 포함되므로, 배치 도중 러너를 고치면
  G-7 이 즉시 잡는다(의도된 설계).
- **git 커밋은 하지 않았다** — 사용자 지시 없이 커밋/푸시하지 않는 규율. M2 가 "다음 배치부터
  커밋+push" 를 권고했으므로 **사용자 승인 시 `tmp/W3_SOLVER_2026-08-15/` 전체를 `git add -f`**
  (이 경로는 gitignore 대상이라 `-f` 필요).
- 전량 실행은 **하지 않았다**(임무 지시). 스모크 2에피소드에서 멈춤.
