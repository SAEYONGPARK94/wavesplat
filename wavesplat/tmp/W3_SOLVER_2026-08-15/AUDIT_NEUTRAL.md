# W3_SOLVER — AUDIT_NEUTRAL (판정자, 2026-08-15)

**역할** 사전등록 조항 × 구현 기계적 대조. 어느 쪽도 편들지 않는다.
**열람 범위 = 주장 범위.** 연 것: `PREREG.md`(md5 재검증 `9f4e9c7e…` ✅) · `AMENDMENT_1.md`(`db6d3203…` ✅) ·
`FROZEN_md5.txt` · `run_w3.py` · `analyze_w3.py` · `dryrun_smoke.py` · `SMOKE.md` · `NOTES.md` ·
`smoke_results.jsonl`(49행 전수) · `smoke.log` · `dryrun_smoke.log` · `code_fingerprint.json` ·
그리고 저장소 원본 `wavesplat/run.py` · `optimizer/{initializer,coarse_to_fine,sequential_update}.py` ·
`model/spectral_gaussian.py` · `data_loader.py` · `evaluation/metrics.py` · `optimizer/learned_init.py` ·
frozen `tmp/rx2_init_2026-08-05/run_rx2b.py`.
**MEASURED** = 내가 파일을 열거나 계산을 재현한 것. **ARGUED** = 추론.
**READ-ONLY**: 이 파일 1개 외에 아무것도 쓰지 않았다. 전량 실행 안 함. 하위 에이전트 안 씀.

---

## 0. 결론 요약

**`GO-WITH-FIX`** — 사전등록 조항 대부분이 채점기에 **숫자까지 그대로** 있고, 문턱·표본·통계량·방향은
런 이전에 동결됐음이 파일시스템으로 뒷받침된다. 다만 **판정 경로의 마지막 조각(`analyze_w3.py`)이
동결 밖에 있고 스모크 방향을 본 뒤에 작성**됐다. 수정 6건은 전부 ≤10줄이고 **팔·문턱·표본·통계량·방향을
하나도 건드리지 않는다.**

---

## 1. 조항 × 구현 대조표

### 1.1 PRIMARY (문턱·α·통계량·방향·동점)

| PREREG 조항 | 요구 | 구현 위치 | 일치? |
|---|---|---|---|
| §4.1 n | 혼잡 n=32 | `analyze_w3.py:27` `THRESH={'crowded':(32,23)}`; `sign_test` `pass_` 에 `n == n_exp` (`:237`) | ✅ MEASURED |
| §4.1 문턱 | ≥23/32 | `analyze_w3.py:27` (k\*=23), `:237` `wins >= kstar` | ✅ |
| §4.1 α | 정확 이항 단측 0.010030 | `tail()` `:36-37` 정확 이항. **재계산: P(X≥23\|32,.5)=0.010031** (PREREG 은 절사 표기), **P(X≥22)=0.025051 > 0.025** ⟹ "22/32 는 통과 아님" 결론 유지 | ✅ MEASURED (6자리 절사 차) |
| §4.1 α 상수 | 0.025 단측 | `analyze_w3.py:26` `ALPHA = 0.025` | ✅ |
| §4.1 통계량 | 부호검정, 에피소드별 **중앙값 차의 부호** | `per_episode` `:203-208` 시드 중앙값 → `sign_test` `:220-242` | ✅ |
| §4.1 방향 | 단측, 음(처치<대조), 사전 지정 | `:227` `wins = (d<0).sum()`, `d = treat − ctrl` | ✅ |
| §4.1 검정력 | 0.737/**0.910**/0.984/0.999, MDE p≥0.766 | 재계산 **0.737 / 0.910 / 0.984 / 0.999**, MDE(.80)=**0.7653**, MDE(.90)=**0.7960** | ✅ MEASURED 전건 일치 |
| §4.2 희소 | n=16, 13/16, α=0.010635, MDE .853 | `THRESH['sparse']=(16,13)`; 재계산 α=**0.010635**, MDE=**0.8529** | ✅ |
| §4.1 **동점** | — | ⛔**PREREG 에 조항 없음.** 코드는 `wins=(d<0)` 엄격, `ties` 는 `:228` 에서 **계산만 하고 인쇄 안 함**, n 에서 제외 안 함 (= 동점은 처치의 패) | ⚠ **미등록** (§3 U1) |

### 1.2 사망조건·판정표

| 조항 | 요구 | 구현 | 일치? |
|---|---|---|---|
| §4.1 사망조건 | 23/32 미달 ⟹ `NOT-A-LAMBDA-ARTIFACT` | `:294` `'PASS' if pass_ else 'FAIL = NOT-A-LAMBDA-ARTIFACT'` — **평가되고 인쇄됨** | ✅ |
| §8 판정표 | P-1 × S-2 ⟹ `LAMBDA-ARTIFACT` / `LAMBDA-HURTS-ONLY` / `NOT-A-LAMBDA-ARTIFACT` | ⛔**두 칸이 코드에 없다.** P-1 통과 시 S-2 와 교차해 이름을 붙이는 코드 부재 ⟹ **사람이 사후에 고른다** | ❌ **누락** (§3 D1) |
| A-3 꼬리표 | 대조가 `DEGENERATE` 면 `(DEGENERATE-CONTROL)` **필수** | `:290` P-1 헤더에 부착 ✅ (대조 `ORA-K@L3` 에 한정) | ✅ (PRIMARY 만) |
| §4.3 퇴화 | ZERO 를 ≤50% 이기면 `DEGENERATE`, 그 팔의 우월 문장 금지 | `zero_condition` `:245-257` (`w > 0.5n`), 게이트 **이전에** 인쇄 `:267-275`. 단 SECONDARY 행에는 꼬리표 없음 | ⚠ 부분 (§3 D5) |
| §7 게이트 | 판정 전 전부 통과 | `gates()` 가 가설 인쇄 **앞**에서 돌고 `:277-279` 실패 시 `sys.exit(4)` | ✅ |

### 1.3 ★DCC (팔별 필수 필드) — S13·HOSGATE 가 두 번, S15 가 절반 놓친 지점

| 조항 | 요구 | 구현 | 일치? |
|---|---|---|---|
| §6 공통 24필드 | 전 팔 | `run_w3.py:99-103` `REQUIRED_COMMON` — PREREG 목록과 **1:1 대조 완료, 누락 0** | ✅ |
| §6 추정기 12필드 | 팔 #1–8 | `run_w3.py:104-107` `REQUIRED_EST` | ✅ |
| §6 CVX 7필드 | 팔 #9 | `run_w3.py:108-109` `REQUIRED_CVX` | ✅ |
| §6 규칙 | 한 팔이라도 한 행 빠지면 **판정 인쇄 없이** `SCHEMA-FAIL` | `analyze_w3.py:82-90` — 위반 시 `sys.exit(3)`, **G-3 이후 코드에 도달 못 함** | ✅ |
| ★셸 9키 | **모든 팔**(`ZERO`·`CVX` 포함) 3개 dict × 9키 | `missing_fields` `:590-595` 가 `shell_rmse`/`zero_shell`/`shell_frac` × `SHELL_KEYS` 를 **키 단위로** 검사 | ✅ |
| ★**전량 적용** | `results.jsonl` 전 행 | `viol = [... for r in rows]` `:82` — **유효 행 전체**, 팔 예외 없음 | ✅ |
| ★**실측 확인** | — | 스모크 48 성공행을 **내가 독립 재구현으로 재검사**: 위반 **0** (ZERO·CVX 포함), 비유한 셸 셀 **0**, `n_frames_scored` 10팔 전부 **160** | ✅ MEASURED |
| ★**경로 CLI 노출** | — | `analyze_w3.py:23` `RES` **하드코딩**, argv 는 `--dryrun` 만 파싱 ⟹ **다른 파일을 먹일 CLI 없음**. `run_w3.py:655` 도 `out` 이 모드로만 갈림(스모크 분리 ✅) | ✅ |
| ⚠예외 | — | `dryrun_smoke.py:13-15` 가 `A.RES` **와** `A.THRESH={'crowded':(1,1),'sparse':(1,1)}` 를 덮어쓴다. 동결 밖·지문 밖 | ⚠ (§3 U3) |

### 1.4 전 팔 `lambda_l1` 명시 (F-1 함정)

| 팔 | PREREG λ | 러너 코드 | 스모크 실측 `lambda_l1_effective`(객체) |
|---|---|---|---|
| `ORA-K@L3` | 1e-3 | `run_w3.py:68` | **0.001** ✅ |
| `ORA-K-NOREG` | **0.0** | `:69` | **0.0** ✅ |
| `ORA-K@L4` | 1e-4 | `:70` | **0.0001** ✅ |
| `ORA-K@L3-DUP` | 1e-3 | `:71` | **0.001** ✅ |
| `SHIP` | 1e-4 | `:73-75` | **0.0001** ✅ (앙상블 멤버 `[1e-4,1e-4,1e-4]`) |
| `SHIP-FITPHI` | 1e-4 | `:76-78` | **0.0001** ✅ |
| `KONLY` | 1e-4 | `:79-80` | **0.0001** ✅ |
| `KONLY-FITPHI` | 1e-4 | `:81-82` | **0.0001** ✅ |
| `ORA-K-CVX` | 0.0(무벌점) | `:428-432` `effective_config.lambda_l1=0.0`, `grad_steps=0` | ✅ |
| `ZERO` | n/a | `lambda_l1_intent=None` `:560-561` | ✅ |

- 폴백 금지: `run_w3.py:463` `assert 'init_method' in cfg and 'lambda_l1' in cfg` ✅.
- G-3 은 **의도 dict 가 아니라 살아 있는 객체**를 읽는다(`_readback` `:275-298`), 앙상블 멤버 불일치도 검사(`analyze_w3.py:96-99`), 위반 시 **중단** ✅.
- ⟹ **S7·S8 을 무효화한 그 함정(기준선이 처치값을 받음)은 이 배치에서 구조적으로 닫혀 있다.** MEASURED.

### 1.5 median-of-3 · 가짜복제 (S4 의 `48/48` 사건)

| 조항 | 요구 | 구현 | 일치? |
|---|---|---|---|
| §3.3 집계 순서 | 시드 중앙값 **먼저**, 그다음 에피소드 부호검정 | `per_episode` `:208` `np.median` → `sign_test` | ✅ |
| §3.3 독립 단위 | **에피소드** (episode×seed 금지) | `sign_test` 의 n = `len(eps)` = 에피소드 수. 96행이 32로 접힌다 | ✅ |
| §3.2 적용 팔 | est 8팔만 시드 3개; `CVX`·`ZERO` 는 `est_seed=null`, 복제 1 | `build_jobs:537` `seeds = SEEDS if kind=='est' else (None,)` | ✅ |
| §3.3 G-6 | 팔별 시드 퍼짐 인쇄, 퍼짐 0 이면 `deterministic=True` 로 표시, **중단 아님** | `analyze_w3.py:141-155`, est 팔만 순회, `ok &=` 없음 ✅. 퍼짐 0 이면 `<- DETERMINISTIC: median-of-3 is PSEUDO-REPLICATION` 인쇄 | ✅ |
| 실측 | — | 스모크에서 **8팔 전부 퍼짐 > 0** (오라클 팔 `8.8e-03`–`2.4e-02`) ⟹ **가짜복제 아님**. 오라클 팔도 `(w,φ)` 가 `torch.rand` 라는 §3.3 의 예측대로 | ✅ MEASURED |
| ⚠ | 행 필드 `deterministic` | `run_w3.py:562` `deterministic=(kind!='est')` = **의도 플래그**이지 §3.3 이 요구한 **측정값이 아니다**(측정은 G-6 인쇄에만 존재) | ⚠ (§3 D6) |

### 1.6 `effective_config` 객체 되읽기

- `install_updater_capture` `:249-265` 가 `wavesplat.run.make_updater` 를 감싼다.
- **검증 MEASURED**: `run.py:18` 이 `make_updater` 를 **run.py 네임스페이스로 import** 하고 `run.py:168` 에서
  전역 조회로 **정확히 1회** 호출한다(다른 호출 지점 없음, 테스트 제외) ⟹ 몽키패치가 **반드시** 잡고,
  `_UPDATER_BOX['u']` 가 덮어씌워질 여지 없음.
- `_readback` 이 `members[0]` 을 core 로 삼고 21개 속성 + `auto_selected` + **전 멤버 λ** 를 기록 ✅.
- 스모크 실측: `SHIP`→`learned` 일 때 `MedianEnsembleUpdater/n_members=1`, `v2_konly` 일 때 `n_members=3`
  ⟹ **라우터 분기까지 행에 남는다** ✅.

### 1.7 임포트 파일 md5 (G-7)

| 조항 | 구현 | 검증 |
|---|---|---|
| 각 행에 `code_md5`·`code_files_n` | `run_one:573` | 49행 전부 `696fa102bb05dd23f04ea18c8421ee12` / 87 ✅ |
| 단일 digest 아니면 중단 | `analyze_w3.py:157-161` `ok &= g7` | ✅ |
| 지문 범위 | `wavesplat/**/*.py` + `wavesim/**/*.py` + configs 3 + **`run_w3.py`·`PREREG.md`** | ✅ |
| **독립 재현 MEASURED** | `md5(json.dumps(files,sort_keys=True))` 를 내가 재계산 → **`696fa102…` 일치**; 87개 파일 **전부** 현재 디스크 md5 가 기록값과 **일치**(변경 0) | ✅ |
| ⚠지문 **밖** | `analyze_w3.py` · `dryrun_smoke.py` · `AMENDMENT_1.md` · 데이터 `.h5` · frozen `run_rx2b.py` | ⚠ (§3 U3/F1) |

### 1.8 재개·중복 방지

| 조항 | 구현 | 일치? |
|---|---|---|
| append + 완료 `job_id` 스킵 | `main:659-668` — `error` 없는 행만 `done` | ✅ |
| 스모크 분리 | `SMOKE_OUT` vs `OUT`, 분석기는 후자만 | ✅ |
| A-1 유효 행 = job_id 별 **마지막 성공 행** | `load:52-60` `ok[j]=r` 덮어쓰기 | ✅ |
| A-1 누락 0 / 미복구 0 | `g10 = not missing and not dup`, `missing = planned − have`(성공행 기준) | ✅ |
| A-1 실패 행 수·사유 **반드시 인쇄** | `:180-184` `Counter` 로 사유별 인쇄 | ✅ |
| A-1 `NONDETERMINISM` | 같은 `job_id` 성공행 2개 이상 & `avg_nRMSE` 상이 ⟹ `dup` ⟹ G-10 FAIL | ✅ |

### 1.9 ★동결 순서 (이 배치 신뢰의 핵심)

**MEASURED — 파일시스템 birth/ctime/mtime 전수:**

| 파일 | birth | mtime | 판정 |
|---|---|---|---|
| `PREREG.md` | **20:03:55.65** | 20:03:55.65 | birth==mtime ⟹ **생성 이후 한 번도 수정 안 됨**, md5 도 동결값과 일치 |
| `FROZEN_md5.txt` | **20:04:01.42** | 20:13:45 | birth 가 자기 주장 `frozen_at 20:04:01` 과 **초 단위 일치**; mtime 은 개정 md5 추가 시각(자기 기재와 일치) |
| `run_w3.py` | 20:08:23.74 | 20:08:23.74 | 동결 **이후** 작성, 이후 무수정 (= 사양이 구현보다 먼저) |
| `smoke_results.jsonl` | **20:08:27.25** | 20:11:58 | ★**어떤 런의 첫 행도 동결(20:04:01)보다 4분 26초 늦다** |
| `AMENDMENT_1.md` | 20:13:38.13 | 20:13:38.13 | 스모크 종료(20:11:58) **이후**, 무수정 |
| `analyze_w3.py` | 20:14:56.56 | 20:14:56.57 | 개정 이후 작성 |
| `results.jsonl` | **부재** | — | 배치 디렉터리에 `mode=full` 로그도 **0건**(`smoke.log` 헤더는 `mode=smoke`, `done 0`) |

- **AMENDMENT_1 이 "판정 행 0행" 상태였는가 → 예.** `results.jsonl` 은 지금도 존재하지 않고, 전량 런 로그도
  없으며, 유일한 원자료는 `smoke_results.jsonl`(49행, `mode=smoke`)이다. 개정문 스스로 A-1 은 스모크 OOM 에서,
  A-3 은 **스모크 방향을 본 뒤** 썼다고 **자진 신고**했다. A-3 는 주장을 **약화**시키는 방향으로만 작동한다(확인함:
  꼬리표 부착 + 문장 제약, 문턱·방향 불변). ARGUED: 이 자진 신고는 감사 관점에서 **감점이 아니라 가점**이다.
- **한계 (반드시 병기)**: 외부 타임스탬프 권위 없음, 배치가 **git 에 커밋돼 있지 않음**(NOTES §마무리), birth/mtime 은
  동일 사용자가 원리적으로 조작 가능. ⟹ PREREG §0 의 자기 규정 **"self-attested freeze"** 가 정확하다.
  ⛔**"preregistered" 라는 단어를 쓰면 안 된다**(PREREG §0 자신의 규율).
- ★**남은 구멍**: `analyze_w3.py` 는 **동결 목록에도, 코드 지문에도 없다**. 스모크가 P-1 방향을 보여준 뒤
  (20:11:58) 작성됐고(20:14:56), `results.jsonl` 이 생긴 뒤에 수정돼도 **어떤 게이트도 잡지 못한다**.
  상수는 오늘 내가 전수 대조해 PREREG 와 일치함을 확인했지만, **그 일치를 미래에 강제하는 장치가 없다**.

### 1.10 크기 3규약 · 셸 · 시드

| 조항 | 구현 | 일치? |
|---|---|---|
| §5 A 절대 중앙값 차 | `sizeA_median_abs` `:238` | ✅ |
| §5 B 에피소드별 상대차 중앙값 | `sizeB_median_rel_pct` `:239` | ✅ |
| §5 C 풀링 상대차 | `sizeC_pooled_rel_pct` `:240` | ✅ |
| 셋 다 병기 | P-1 `:295`, SECONDARY `:322` — **A·B·C 전부 인쇄** ✅. 희소 서술 행 `:333` 은 **A·B 만 인쇄**(C 는 json 에만) | ⚠ 경미 (§3 D7) |
| §5 셸 분해 의무 | `:367-384` P-1 + S-1~S-4, 9키 전부, 중앙값+승수 | ✅ |
| 셸 정합 | `run_w3.EDGES/BARE_KEYS`(`:57-60`) 가 `run.py:358` 의 `_edges`·키 포맷과 **동일** | ✅ MEASURED |
| §3.2 시드 | `SEEDS=(1592,1593,1594)` `:54`; G-9 이 분석기에서 **소모 == 기대** 검사(중단) | ✅ |
| 충돌 0 | **독립 재스캔 MEASURED**: `tmp/**/*.jsonl`(W3 제외)에서 1500–1900 실사용 **최대 1591**, 1592–1594 **충돌 0** | ✅ |
| 잔여 `[1595–1818]` | 원장과 정합 | ✅ |
| §1 표본 파일 | **48개 전부 존재 확인**(혼잡 32 = indep 16 + s9 16, 희소 16), 계획 잡 **1200** = 러너 인쇄와 일치 | ✅ MEASURED |
| G-9 "신규 `.h5` 0" | ⛔**어떤 코드도 검사하지 않는다**(러너가 h5 를 쓰지 않으므로 구성상 참) | ⚠ 미구현 (§3 D4) |

### 1.11 Holm-4

| 조항 | 구현 | 일치? |
|---|---|---|
| SECONDARY 4개에 적용 | `SECONDARY` `:30-33` 4개, `order` `:312` p 오름차순 | ✅ |
| α/4, α/3, α/2, α | `:315` `adj = ALPHA/(m-i)` (m=4 ⟹ .00625/.00833/.0125/.025) | ✅ |
| 정확 p 사용 | `tail()` 정확 이항 | ✅ |
| 추가 요건 | `holm = p ≤ adj **and** wins ≥ threshold` (보수적) | ✅ |
| ⚠단계하강 | **미구현** — 앞 가설이 떨어져도 뒤 가설이 독립적으로 통과할 수 있다(Holm 위반, 반보수적) | ⚠ (§3 U5) |
| ⚠m 고정 | `m = len(sec)` **가변** — 팔 축소로 SECONDARY 가 `NOT-RUN` 이면 m=3 ⟹ 사전등록한 "Holm-**4**" 보다 **느슨해진다** | ⚠ (§3 U5) |

### 1.12 조작검사 FITPHI (G-4) — 원자 단위

**MEASURED, 코드로 확인:**
- 사이트 ① `coarse_to_fine.py:172-173`(호출시 import) — 위상 원천은 `coarse_to_fine.py:241` 의
  `_linear_projection`, 이는 `coarse_to_fine.py:32` 의 **모듈 수준 import** ⟹ 프로브가 `C._linear_projection`
  을 패치(`run_w3.py:191,203`) ✅.
- 사이트 ② `sequential_update.py:1187-1188` — 위상 원천은 `:1142`, 함수 **안**에서 `:1110` 으로 import
  ⟹ `I._linear_projection` 패치로 커버 ✅.
- `_linear_projection` 은 **성분 1개당 스칼라** `(amplitude, phase)` 반환(`initializer.py:417-426`)
  ⟹ 프로브의 `float(np.float32(ph))` 가 성립 ✅ (배열이면 조용히 빈 집합이 될 뻔한 설계인데, 실제로는 스칼라).
- `add_gaussians` 는 **호출 지점이 2곳뿐**이고 둘 다 키워드 호출이며 서명이 래퍼와 정확히 일치
  (`spectral_gaussian.py:178-179`) ⟹ **원자 추가를 빠짐없이 관측** ✅.
- 실측: `KONLY-FITPHI` 3/3 개입(원자 12–16, `coarse_to_fine.py:173|T`), `SHIP-FITPHI` 혼잡 **0/3**,
  OFF 팔 8개 전 행 0 ✅. `SHIP` v2_konly 라우팅 행은 `coarse_to_fine.py:173|**F**' :6` 로 **게이트가 꺼진 채
  호출됐음까지** 기록 — 조작검사가 대칭으로 작동한다.
- ⚠경미: 프로브의 위상 집합에 `_linear_projection` 의 퇴화 반환 `0.0`(`initializer.py:442`)이 섞일 수 있고,
  `GROW_RANDOM_PHI=False` 경로였다면 `torch.zeros` 와 충돌한다. 실측 48행에서 OFF 팔 오염 0 ⟹ 위험 무시 가능.

### 1.13 `ORA-K-CVX` 동일성 (같은 관측·같은 채점기·같은 프레임 정합)

| 축 | 추정기 팔 | `ORA-K-CVX` | 동일? |
|---|---|---|---|
| 적합 관측 | `run.py:161` `get_init_window(tp_factor=2.0)` | `run_w3.py:365` **동일 호출** | ✅ |
| 채점 프레임 | `run.py:198` `get_tracking_frames(tp_factor=2.0)` | `run_w3.py:329`(`_zero_and_shape`) **동일 호출** | ✅ |
| 채점 함수 | `run.py:282` `compute_all_metrics(pred,gt,gx,gy,Hs_true,obs_x,obs_y)` | `run_w3.py:404` **같은 함수·같은 인자 순서** | ✅ |
| nRMSE 정규화 | `metrics.py:422` `nrmse(pred,gt,Hs_true)` | 동일 경로 | ✅ |
| 격자·정합 | `loader.gt_grid_x/y`, `np.meshgrid(gx,gy)` | **동일** (`run_w3.py:328,392`) | ✅ |
| 셸 | `run.py:358` edges/키 | `run_w3.py:57-60` **동일 edges**, `obs:` 접두어만 추가 | ✅ |
| 물리 | 열 `[A,omega,kx,ky,phi,Q]`(`data_loader.py:58`) | `kx=gtc[:,2], ky=gtc[:,3]`, ω=분산관계, 위상 `kx·x+ky·y−ω·t` = `initializer.py:421` 규약 | ✅ |
| 실측 정합 증거 | — | 10팔 전부 `n_frames_scored=160`; rank **60/60·4/4**, `cvx_gtomega_nrmse` 가 주값과 소수 5자리 일치 | ✅ MEASURED |
| **비대칭(우리에게 불리)** | 추적 이득 O, confidence 가중 O, 추적 프레임 관측도 소비 | 추적 **없음**, **무가중**, **초기화 창만** 사용 | PREREG §2.1·§9.7 이 **명시함** ✅ |

⟹ **판정: 동일성 조건 충족.** 유일한 비대칭은 전부 CVX 에게 불리한 방향이고 사전등록에 적혀 있다.
⚠단, SMOKE.md §3 의 한계 서술이 결정적이다 — 합성 GT 와 모델이 **같은 식**이므로 CVX 는 **솔버 격차**를
재는 것이지 **모델계열 격차가 아니다.** (08-13 "희소 −93% 는 표현 아티팩트" 와 동종 함정.)

---

## 2. `SHIP-FITPHI` 가 개입 0/3 이면 S-3 는 무엇으로 보고되는가

**답: 사전등록돼 있다. 사후 결정 불필요.**

- 조항: PREREG **§2.2 마지막 문단** — *"어떤 팔의 개입 부재 비율이 ≥50% 이면 그 팔이 참여하는 SECONDARY 는
  판정 없이 `UNDECIDED-NO-INTERVENTION` 으로 인쇄된다(게이트 G-4)"* + **§7 G-4 행** *"실패 시: 해당 SECONDARY
  → `UNDECIDED-NO-INTERVENTION`"*.
- 구현: `analyze_w3.py:120-121`(`dead` 집합) → `:308-310`(`verdict` 덮어쓰기) → `:318`(`holm_pass=False`)
  → `:321` 인쇄. **MEASURED: 분기 존재.**
- 전량 실행 시 S-3 는 **`UNDECIDED-NO-INTERVENTION` 이 될 공산이 크다**(ARGUED). 기제는 코드로 확인됨:
  `sequential_update.py:622-624` 가 *"the coarse-to-fine N-staging is **bypassed**"* 라고 자기 기술하고,
  스모크 혼잡 3/3 이 `learned` 로 라우팅됐다. 라우팅이 `v2_konly` 로 떨어진 희소 1행에서는 **개입 36원자**가
  실제로 발생했다 ⟹ 개입 유무는 **라우터가 정한다**.
- ★**단, 사전등록이 정의하지 않은 두 번째 층이 남는다**: `SHIP` 과 `SHIP-FITPHI` 는 `learned` 라우팅
  에피소드에서 **비트 동일**(스모크 3/3, `0.237897/0.225657/0.254893` 완전 일치) ⟹ 그 에피소드는 **정확한 동점**이다.
  PREREG 에 **동점 규칙이 없고**, 코드는 동점을 패로 센다. ⟹ 개입률이 50%를 **넘더라도** 남은 동점 때문에
  23/32 는 사실상 도달 불가. **S-3 는 통계적으로 이미 죽어 있다** — 이걸 `UNDECIDED` 가 가려 준다.
- ⟹ **정직한 보고 문장**(ARGUED, 권고): *"배포 팔에서 위상 수리는 **효과가 없다**"* 가 아니라
  *"배포 팔에는 위상 수리가 **닿는 지점이 없다**(라우터가 `learned` 를 고르면 사이트 ①이 0회 호출된다)"*.
  실제 시험은 **S-4(`KONLY` 짝)** 다.

---

## 3. 불일치 목록

### A. 약속했는데 없는 것 (MISSING)

| # | 조항 | 실태 | 무게 |
|---|---|---|---|
| **D1** | §8 판정표 | `LAMBDA-ARTIFACT` / `LAMBDA-HURTS-ONLY` 를 내는 코드 **부재**. P-1 실패 이름만 인쇄. 판정명이 **사람 손**으로 결정된다 | ★★ |
| **D2** | §3.1 "코퍼스 **16/16 분해** 서술 병기" | `corpus` 는 전 행에 기록되나 분석기에 **코퍼스별 분해 코드 0줄** | ★★ |
| **D3** | §10 비용상한 시 SECONDARY 축소 → `NOT-RUN` | **도달 불가**: G-1 이 **전량 계획**과 대조하므로 팔을 하나라도 빼면 `GATES FAILED` → `sys.exit(4)`, PRIMARY 판정도 못 본다 | ★ (축소 안 하면 무해) |
| **D4** | G-9 "신규 `.h5` 0" | 검사 코드 없음(구성상 참) | ☆ |
| **D5** | §4.3 "그 팔이 들어간 어떤 우월성 문장도 금지" | 퇴화 표시는 인쇄되나 SECONDARY 행에 꼬리표 없음. **스모크에선 `KONLY`·`KONLY-FITPHI` 둘 다 혼잡에서 ZERO 에 졌다** ⟹ S-4 는 **양쪽 다 퇴화**일 수 있다 | ★★ |
| **D6** | §3.3 `deterministic` | 행 필드는 **의도**(`kind!='est'`), §3.3 이 요구한 **측정 퍼짐**은 G-6 인쇄에만 존재 | ☆ |
| **D7** | §5 크기 3규약 | 희소 서술 행이 **A·B 만** 인쇄(C 는 json 에만) ⟹ 로그만 보고 희소 크기를 인용하면 규약 위반 | ☆ |

### B. 있는데 사전등록 안 된 것 (UNREGISTERED)

| # | 내용 | 무게 |
|---|---|---|
| **U1** | ★**동점 규칙**. PREREG 에 조항이 없다. 코드는 동점을 n 에 남기고 패로 센다(보수적). `ties` 는 계산되나 **인쇄 안 됨**. **S-3 의 운명을 이 미등록 규칙이 결정한다**(§2 참조) | ★★★ |
| **U2** | ★**`analyze_w3.py` 가 동결 밖·지문 밖**. 스모크가 P-1 방향을 보여준 뒤 작성(20:14:56 > 20:11:58). 상수는 오늘 전수 일치 확인했으나 **미래의 수정을 막는 장치가 전무** | ★★★ |
| **U3** | `dryrun_smoke.py` 가 `A.RES` **와 문턱**을 `(1,1)` 로 덮어쓴다. 동결 밖·지문 밖. `results.jsonl` 이 생긴 뒤 이 파일을 한 줄만 고쳐 돌리면 **문턱 1/1 짜리 판정**이 나온다 | ★★ |
| **U4** | Holm 단계하강 미구현 + `m` 가변(§1.11) — 사전등록보다 **느슨해질 수 있는** 방향 | ★★ |
| **U5** | G-4 경계 불일치: §2.2("부재 ≥50% ⟹ UNDECIDED") vs §7 표("개입 ≥50% ⟹ 통과"). 정확히 16/32 에서 두 조항이 충돌하고, 코드는 §7 판을 구현 | ☆ |
| **U6** | `BASE_CFG` 가 `lr/lambda_smooth/tracking_*/device` 를 **명시 고정**. frozen `run_rx2b.BASE_CFG` 는 이들을 폴백에 맡겼다. **값은 오늘의 `DEFAULT_CONFIG` 와 전건 일치 확인**(lr 0.01 · λ_s 1e-4 · steps 10 · `w_phi` · 0.005 · cpu) ⟹ 수치 동일, 그러나 §2 의 "frozen BASE_CFG 그대로"에 대한 **추가**이고 방향은 옳다(F-1 재발 방지) | ☆ |
| **U7** | `lambda_l1_effective` 가 `None` 인 행(캡처 실패)이 나오면 G-3 이 `float(None)` 으로 **예외 종료**(FAIL 인쇄가 아니라 트레이스백). 조용한 통과는 아니므로 안전 방향 | ☆ |

---

## 4. 실행 권고

**MEASURED (지금 상자 상태)**: GPU `808 / 16376 MiB`, util **0%**, compute app **0개** = **비어 있음**.
CPU 16코어, 여유 RAM 21 GiB, load 0.02.

**GPU OOM 의 정체 (코드로 확인)**: `learned_init.py:131` 이
`device = 'cuda' if torch.cuda.is_available() else 'cpu'` 로 **하드코딩**돼 있어 `cfg['device']='cpu'` 를
**무시한다**. 그래서 `auto`→`learned` 잡만 GPU 를 물고, 8워커가 4080S 하나를 동시에 물어 49잡 중 1잡이 죽었다.
⟹ **`AMENDMENT_1 A-2` 의 위상 분리는 타당하다**(수치 불변: 코드·시드·cfg 동일, 바뀌는 건 동시 실행 수뿐).
⛔**`CUDA_VISIBLE_DEVICES=''` 로 CPU 강제하지 말 것** — 학습 제안기의 수치가 바뀌고 스모크와 이질적이 된다.

**권고 순서** (총 **≈1.7–1.8 h**, ARGUED — 스모크 팔별 최소 `wall_s` × 잡 수 ÷ 실측 병렬효율 ×7.62):

1. **수정 6건 먼저**(전부 분석기·문서 측, `run_w3.py` 는 **절대 손대지 말 것** — 지문에 들어 있어 G-7 이 즉시 막는다):
   **F1** `analyze_w3.py`·`dryrun_smoke.py` md5 를 `FROZEN_md5.txt` 에 **추가**(가장 값싸고 가장 중요) ·
   **F2** §8 판정명 자동 산출 + `(DEGENERATE-CONTROL)` 꼬리표 · **F3** `ties` 인쇄 + 동점 규칙을
   `AMENDMENT_2` 로 **명문화**(현재 구현 그대로 = 동점은 패, 보수적) · **F4** Holm `m=4` 고정 + 단계하강 ·
   **F5** 코퍼스 16/16 분해 인쇄 · **F6** `dryrun_smoke.py` 에 `assert RES.endswith('smoke_results.jsonl')`.
   ⟹ `AMENDMENT_2` 는 **문턱·표본·통계량·방향·팔을 하나도 바꾸지 않는다**(F4 는 오히려 엄격해진다).
2. `run_w3.py gates` (런 0회) — 단, **`code_fingerprint.json` 을 덮어쓴다**. 첫 행 이전에만 돌리고
   digest 가 `696fa102…` 인지 확인할 것(달라졌으면 트리가 움직인 것).
3. **Phase A** `WORKERS=8`, 8팔(GPU 미사용) = **912잡 ≈ 1.0–1.1 h**.
4. **Phase B** `WORKERS=3`, `SHIP,SHIP-FITPHI` = **288잡 ≈ 0.6–0.7 h**.
5. **Phase C** 같은 명령 인자 없이 1회 = 실패 잡 재시도(분 단위).
6. ⚠**위상을 동시에 돌리지 말 것.** 둘 다 같은 `results.jsonl` 에 append 하는데 행이 ~2.7 KB 라
   교차 기록 위험이 있고, 위상 분리의 목적(GPU 경합 회피) 자체가 무너진다.
7. 50행쯤 쌓이면 `analyze_w3.py --dryrun` 을 **한 번** 돌려 G-2/G-3/G-4 를 실제 full 행에서 확인
   (1.8 h 를 태우기 전에).
8. ⛔`elapsed_ms` 는 인용 금지(PREREG §5·§9.4). 위 시간은 **배치 비용**(`wall_s`)이지 지연이 아니다.

**운영 위험 2건 (사전 고지)**: (a) **G-5 는 96쌍 전부 `|Δ|=0` 을 요구하는 하드 중단 게이트**다.
스모크 3/3 은 정확히 0 이었지만, 한 쌍이라도 흔들리면 사전등록상 **배치 전체가 중단**된다.
(b) **G-1 은 전량 계획과 대조**하므로 중간에 팔을 줄이면 판정이 **전혀** 인쇄되지 않는다(D3).
⟹ 줄이지 말고 전량 돌리는 것이 유일하게 안전한 경로다(비용이 1.8 h 뿐이므로 문제 없음).

---

## 5. 최종 판정

# `GO-WITH-FIX`

**근거(요약)** — 문턱 23/32·α 0.010030(재계산 0.010031)·검정력 0.910·MDE 0.766·방향 음·단측이
**채점기에 숫자까지 그대로** 있고(§1.1), DCC 가 **전 팔·전 행·셸 9키까지 실제로 걸리며 경로가 CLI 로
노출되지 않고**(§1.3), 전 팔 λ 가 **객체 되읽기로** 검증되며(§1.4), median-of-3 이 에피소드 단위를
지키고(§1.5), 코드 지문·시드 원장·재개 중복 방지가 작동하고(§1.7–1.8), **동결이 모든 런보다
4분 26초 앞선다는 것이 birth time 으로 뒷받침**된다(§1.9). `ORA-K-CVX` 는 같은 관측·같은 채점기·같은
프레임 정합을 쓰며 비대칭은 전부 자기에게 불리한 방향이다(§1.13).

**FIX 가 붙는 이유** — 판정 경로의 마지막 조각(`analyze_w3.py`)이 **동결 밖·지문 밖이고 스모크 방향을
본 뒤에 작성**됐으며(U2), **동점 규칙이 미등록인 채로 S-3 의 운명을 결정하고**(U1), §8 판정명이
**사람 손에 남아 있다**(D1). 셋 다 결과가 나온 뒤에는 고칠 수 없다 — **지금, 첫 행 이전에** 고쳐야 한다.
