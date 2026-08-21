# FIX3 — 3라운드 회귀검증 잔여 결함 실제 수정 (2026-08-13)

**지시서**: `R3_regression.md`(주) · `R3_fresh.md` · `V1_refute_nodisp.md`
**편집 파일 10개 / 신규 파일 3개.** 하위 에이전트 0회. 새 실험 런 0회(§6 은 스크래치패드 산출물 복원).

**대체 숫자를 발명하지 않았다** — 근거 없는 값은 지우고 `⛔RETRACTED` 로 표시했고, 이력은
HTML 주석 또는 취소선으로 보존했다.

---

## 1) [BLOCKER] `memory/project_s13_oos_2026-08-12.md` — 정정 5건 도달

| # | 위치 | 변경 | 근거 |
|---|---|---|---|
| 1.1 | frontmatter `description` | "기여 #2b 복권"·"λ 지분 37%" 제거 → 정정 5건 요지 + "λ 지분 규약별 28.9–45.1%(기본 규약 37.2%)" 로 교체 | R3 §B-1 · §M-1 |
| 1.2 | 본문 맨 위(frontmatter 닫는 `---` 뒤) | `S13_oos/RESULT.md` 머리 배너와 **동일 5건 정정 배너** 삽입 + PRIMARY 무사·시드 1개·G-3 미검사 단서 | `RESULT.md:3-29` · A1/A6/A7/V1/V2 |
| 1.3 | `:23` 사전노출 | ~~"사전노출 24 제외해도 유지"~~ + ⛔정정 3 (실제 **128/128**, S-2e 는 노출의 19%만 제거) | 배너 정정 3 |
| 1.4 | `:29-38` S-2d 절 | 제목 "⛔절반 철회", #2b 행 취소선 → **`UNDECIDED`**, "대칭 감사가 값을 했다" 서사 철회. **#3 `DEAD` 는 ✅유효 명시** | V1 §7·§5 |
| 1.5 | `:40-42` λ 절 | 제목을 허용형으로 교체, ~~"신선 시드 재확인"~~ 철회, ⛔"상한 37%" 금지 병기 | 배너 정정 2 · R3 §M-1 |
| 1.6 | `:50` S-2a 단서 | ⛔정정 4·5 추가 — `obs:0_5` 는 사전 지정 아님(PREREG 는 `ship:10_20`), S-2a 미검정 ⟹ "전 SECONDARY 통과" 인용 금지 | 배너 정정 4·5 |
| 1.7 | `:57-58` "바꾼 것" | 3·4번 항목 취소선 + 정정본 | 위와 동일 |

⚠**원 수치는 전부 보존**(취소선 안). ✅`MEMORY.md` 는 **손대지 않았다** — 링크 120개 전부 실존 재확인.

## 2) [BLOCKER] `results/direction1/**` 배너의 거짓 사실 — 스크립트별로 정확히 재작성

**직접 확인한 사실**:

| 스크립트 | `wavesim` | LiDAR 호출 | 관측 |
|---|---|---|---|
| `scripts/d1_crb.py` | `:39` **직접 임포트** | `:149` `load_episode` 호출 | `:154` `lidar_points`·`lidar_valid_mask` |
| `scripts/d1_predictable_zone.py` | `d1_crb` 경유 | ✅**호출함**(`:29` 가 `load_ep`/`collect_obs` 임포트, `:60-61` 사용) | 동상 |
| `d1_crb.validate_aperture_law` (`:270-`) | — | ❌ | 상자 균일난수 M=300, L=50 m, 단일 성분 |
| `d74_closed_loop_trajectory.py` | `d1_crb` 경유(전이) | ❌ **0회** | `:69-80` 플랫폼 주변 상자 균일난수 `L_FOOT=20 m` × 30점 |

| # | 파일 | 변경 |
|---|---|---|
| 2.1 | `results/direction1/REPORT.md` | "`wavesim` 임포트 0회 / LiDAR 0회 / 상자 균일난수" **삭제** → **절×코드×관측출처 표**로 교체. 살아 있는 사유만 남김: ① CRB 계산이라 **출하 추정기를 0회 실행**(§4 "estimator" 열은 `spacetime_gp_30ep.json` 의 ST-DF-GP 인용, `d1_crb.py:322-326`) ② 참 성분·정합 차수 가정 ③ `d=50 m`·`σ=0.02 m` 가정. `19.97×19.96 m` 는 **d74 값**임을 명시하고 실제 개구 `31.6×59.7 … 71.9×53.2 m`(`crb_results.json` 재도출) 기재 |
| 2.2 | `results/direction1/REVIEW_LOG.md` | 동상. Step 1(`d1_crb.py`)·Step 2(`d1_predictable_zone.py`) 둘 다 실 LiDAR 사용임을 명시, 예외는 `fig_aperture_law.png` 하나 |
| 2.3 | `results/direction1/closed_loop_trajectory_VERDICT.md` | 배너를 **`d74` 기준**으로 한정. "LiDAR 0회 호출"·"상자 균일난수"·`L_FOOT=20 m`·`tnse`(`:34`)는 **유지(참)**, "`wavesim` 임포트 0회"는 **전이 임포트 존재**로 정정, "`results/direction1/**` 계열" 일괄 적용은 **철회**(같은 디렉터리의 두 파일에는 적용 불가) |
| — | `results/direction3/platform_aperture_VERDICT.md` | **지시대로 그대로 두었다**(d72 기준이라 "LiDAR 0회 호출" 참) |

## 3) [MAJOR] `S9/WARNING_2026-08-13.md` §2 — S9 에 대한 거짓 경고 교체

- §2 제목·본문을 **"λ 오염은 S9 에 해당하지 않는다"** 로 교체.
  근거 인용: `run_s9.py:96` `S5.S4.EXTRA['AUTO'] = {'lambda_l1': L1_SHIP}` (`:68` `L1_SHIP = 1e-3`),
  `:90-95` E1 수리 주석. **해당 배치 = S5–S8** 로 명시(대조 근거 `run_s7.py:77` · `run_s8.py:79`
  *"EXTRA 없음"* — 둘 다 직접 열어 줄번호까지 확인).
- §3 첫 항목도 "S9 는 λ 수정 불필요, S5–S8 재실행 시에만 1줄 수정" 으로 조정.
- §1(폐기된 `SUPPORT-TRADEOFF` 게이트)·§3 나머지는 **유지**.
- ✅이 파일은 `FROZEN_md5.txt` 목록에 **없다**(봉인 7파일 = PREREG/s9_constants/mde_prespec/MDE json/analyze_s9/gate/G0) ⟹ 봉인 무손상.

## 4) [MAJOR] `S13_oos/RESULT.md` 본문 ↔ 머리 배너 충돌 해소 (원 수치 전량 보존)

| 위치 | 변경 |
|---|---|
| `### S-2c` 제목 | "37%, 문턱 미저촉" → **"기본 규약 37.2%, 규약 범위 28.9–45.1%, 6/6 미저촉"** + ⛔"상한 37%" 금지 블록 |
| `:73` | ~~"λ 자체가 신선 시드에서 재확인(54/64)"~~ 취소선 + ⛔ + 배너 정정 2 참조. **수치 자체는 정확**함을 병기 |
| `### S-2d` 제목 | "기여 하나는 죽고 하나는 살아났다" 취소선 + **⛔절반 철회** |
| `:82-83` 표 셀 | #2b 행의 `62/64`·`2.3e-16`·`+70.2%`·`+777.8%` 취소선, 판정 `ALIVE`→**`UNDECIDED`**. #3 행에 ✅유효 표시 |
| 표 직후 | ⛔인용금지 블록 신규(ω 미시딩 기제 + FAIR 32런 널 + 원자료 경로) |
| `:88-91` | "#2b 는 낡은 파일이 틀렸다 / 대칭 감사가 값을 했다" 문단 전체 취소선 + 철회 사유(2026-04 Full 팔이 제로필드 미달) |
| `:93` | `### S-2a … (사전 지정 1개)` 취소선 + **라벨 철회 블록**(PREREG 는 `ship:10_20`, `0_5` 는 0회, 1.8–4.1배 차) |
| `:116-117` 결론절 | 3·4번 취소선 + 정정본 |

## 5) [MAJOR] 살아 있던 금지 수치 3곳

| # | 파일 | 변경 | 근거 |
|---|---|---|---|
| 5.1 | `docs/PRESENTATION_SUMMARY.md §4.3` | 12-ep 표 **셀 전량 ⛔RETRACTED**(§4.2 와 동일 방식) + **원본 표를 HTML 주석으로 이력 보존** + 상단 철회 사유 블록. `0.211`/`0.436`/`0.358`/`ms/f` 열이 LIVE 였다 | R3 §M-4 (`README.md:191`·`docs/CHANGELOG.md:7` 이 같은 수치를 이미 금지) |
| 5.2 | `README.md` "Full 100-episode Comparison" 표 | **GPR 행 제거**, 표에 `n` · `Source` 열 추가(WS/LETKF 둘 다 `100ep_ws_letkf.json`, n=100), 검정 이름 명기(양측 이항 부호검정). GPR 은 각주로 강등 — **`0.276`, n=10, `gpr_comparison.json`**, 동반 팔 `WS-P1 0.2325` · `ETKF 0.4499` 병기 | 직접 재도출: 10 조건, GPR 평균 **0.2756** |
| 5.3 | `ref_paper/trending_2025/review_factbased_v3.md` | 파일 머리에 **철회표 배너**(5행) + `:24`(v2_konly default) · `:29-31`(0.223/0.319/0.276 · 13.4/132.6 ms · 57 tests) · `:231`(13ms, 0.223) **인라인 취소선 + ⛔** | R3 §4(ii) · `reference_citation_bans.md` |

## 6) [MAJOR] #2b 를 뒤집은 32런 개입 실험 — **원자료 복원 완료** (찾았다)

스크래치패드에서 `fair.py` · `fair.jsonl`(32행) · `fair.log` 를 **찾아** 저장소로 복사했다.
신규 디렉터리 **`tmp/AUDIT_recheck_2026-08-13/fair_ablation/`**:

| 파일 | md5 | 내용 |
|---|---|---|
| `fair.py` | `af8277fbf03cd8cc5e02db4813cef1ea` | 개입 러너(monkeypatch, 원본 그대로 — 경로 하드코딩 포함) |
| `fair.jsonl` | `e60717a891c1ce96672fc64963a5666f` | **32행 원자료** (`ep · strat · {fair,ship,nod}_{init,fin}`) |
| `fair.log` | `dcb4aa3904819d93fbc6a890113d4b46` | 러너 stdout(에피소드별 벽시계) |
| `analyze_fair.py` | (신규) | 집계기 — V1 §7 규약을 코드로 옮긴 것 |
| `README.md` | (신규) | 무엇을 돌렸나 · 재현 명령 · 결과 표 · 한계 |

**독립 재도출로 `V1 §7` 표 전건 일치 확인**(런 0회, `analyze_fair.py` 실행):

| 층 | 대조 | med(rel) | 패/n | p |
|---|---|---|---|---|
| 희소 | `NODISP` vs `SHIP` | +721.4% | 15/16 | 5.19e-04 |
| 희소 | `FAIR` vs `SHIP` | +7.3% | 9/16 | **0.804 (널)** |
| 희소 | `NODISP` vs `FAIR` | +358.5% | 15/16 | 5.19e-04 |
| 혼잡 | `NODISP` vs `SHIP` | +62.2% | 16/16 | 3.05e-05 |
| 혼잡 | `FAIR` vs `SHIP` | −0.7% | 8/16 | **1.000 (널)** |
| 혼잡 | `NODISP` vs `FAIR` | +54.8% | 16/16 | 3.05e-05 |

`init_nrmse` 중앙: 희소 SHIP 0.0121 / FAIR 0.0299 / NODISP 0.2416 ·
혼잡 SHIP 0.1546 / FAIR 0.1486 / NODISP 0.2755. **전부 V1 §7 과 일치.**
표본 커버리지 실측: 8 구성(`BF4/BF5 × {2,5,15,30}comp`) × 시드 `{32,36,48,52,56,60}` = 32.
`git_head = 5f0846c`. 생성 0회, 시드 소모 0개.

⟹ **R3 §M-6 해소** — `p=0.804 / p=1.000` 이 이제 파일에서 재도출된다.
⚠`fair.py:24` 의 `OUT` 경로가 스크래치패드로 하드코딩돼 있다(원본 보존을 위해 고치지 않고 README 에 경고).

## 7) [MINOR] `configs/estimation.yaml:18-19` "Keep in sync" 주석 교체

현재 구조를 실측해 반영:
- **유일 정의** = `wavesplat/optimizer/sequential_update.py:40` `LAMBDA_L1_DEFAULT = 1e-4`
- `run.py:30` 이 그 이름을 **import** 해 `DEFAULT_CONFIG['lambda_l1']` 에 넣음
- `run.py:94` yaml 폴백이 **`DEFAULT_CONFIG` 를 읽음**(세 번째 손사본 아님)
- 이 yaml 키는 **미러**이며 `--config` 로 넘길 때만 발효 — 낡을 수 있는 유일한 사본
- 낡은 "Keep in sync" 는 `5f0846c` **이전** 3사본 배치의 서술이었음을 명기

---

# 자기검증

| # | 항목 | 결과 |
|---|---|---|
| 1 | 문법 | `yaml.safe_load('configs/estimation.yaml')` ✅ (`lambda_l1 = 0.0001`) · `py_compile` `fair.py`·`analyze_fair.py` ✅ |
| 2 | 봉인 | `md5sum -c FROZEN_md5.txt` **4개 디렉터리 전건 OK** (S7 7/7 · S8 8/8 · S9 7/7 · S13b 1/1). `WARNING: 1 line is improperly formatted` 는 매니페스트 말미의 **타임스탬프 각주**(사전 존재, 손상 아님) |
| 3 | `git status --short` | 금지구역 이탈 **0**. `paper/`·`thesis/`·`hydrofoil/` 에 내 변경 **0**(`paper/PAPER_IMPROVEMENTS.md` 는 앞 라운드 것, 원고 아님). `CLAUDE.md` 는 **사용자가 직접 수정**(내 문자열 `3라운드 정정` grep = 0). `S13b_seed_robust_2026-08-13/` 는 **읽기 전용 md5 검증만** |
| 4 | auto-memory 링크 | `MEMORY.md` 의 `.md` 링크 **120개 전부 실존, 결손 0**. `project_s13_oos_2026-08-12.md` 는 파일명 불변이라 `:4`·`:28` 링크 유효 |
| 부 | 마크다운 표 | 편집 10파일 열수 검사 — 내가 만든 파손 **0**(검출된 2건은 `|k|`·셀 내 `|` 를 가진 **사전 존재** 표) |
| 부 | 생성기 | `generate_ppt.py`·`generate_labmeeting_pptx.py`·`generate_graphical_abstract.py` 는 `PRESENTATION_SUMMARY.md` 를 **읽지 않는다**(grep 0) ⟹ §4.3 편집이 게이트에 영향 없음 |

# 손대지 않고 남긴 것

| 항목 | 이유 |
|---|---|
| `wavesplat/paper/**`(`wavesplat.tex`·`cover_letter.tex`·`submission/**`)·`thesis/thesis_main.tex`·`hydrofoil/paper/**` | ⛔지시상 읽기만. 금지 수치 잔존은 **알려진 상태**(R3 §4(i)) |
| `tmp/S13b_seed_robust_2026-08-13/**` | ⛔배치 실행 중(`results_s13b.jsonl` mtime 계속 갱신). **읽기 전용 `md5sum -c` 만** 수행 |
| `FROZEN_md5.txt` 등재 파일 전부(S7·S8·S9 의 `PREREG.md`·`analyze_*.py`·`gate.py` 등) | ⛔봉인. 폐기된 `SUPPORT-TRADEOFF` 게이트는 `WARNING_2026-08-13.md` §1 로 계속 대체 |
| `/home/vexit/wavesplat/CLAUDE.md` · `memory/MEMORY.md` | ⛔사용자 직접 관리. ⚠단 **CLAUDE.md 는 이미 사용자가 `:114-118` 에서 direction1 오류를 정정해 두었다** — 내 §2 수정과 일치 |
| `results/direction3/platform_aperture_VERDICT.md` | 지시대로 유지. ⚠단 **"`wavesim` 임포트 0회" 는 d72 에서도 전이 임포트가 있어 엄밀히는 부정확**(`d72:26` → `d1_crb:39`). **"LiDAR 0회 호출"·"상자 균일난수" 는 참**이므로 판정은 안 바뀐다 — 기록만 남긴다 |
| `README.md:203` raydrop `10/10` · `:206-211` 절제 표(`w/o Confidence … essential`) | R3 §m-1. FIX1 이 "판단 대상" 으로 공시한 항목이고 이번 지시 목록 밖. ⚠**기여 #3 `DEAD` 와 모순은 실재** |
| `docs/strategic_repositioning.md` "WaveSim … 196 unit tests"(R3 §m-2) · `CLAUDE.md` `562줄`(§m-3) · "배치 24개"(§m-4) · 출처 오귀속 5건(§m-5) | 이번 지시 목록 밖(MINOR). §m-3·m-4 는 `CLAUDE.md` 라 금지구역이기도 하다 |
| `R3_fresh.md` 가 새로 제기한 B-1(합성 GT = 모델식 동일) · B-2(기본값 3개 되돌려도 111/111) · M-1(`b1_full_comparison.json` 이 0.319 를 낸다) 등 | **이번 지시는 `R3_regression.md` 의 7항목**이었다. `R3_fresh` 의 신규 판정은 아직 적대적 검증을 안 거쳤으므로 문서에 반영하지 않았다 — 특히 M-1 은 **금지 사유를 뒤집는** 내용이라 사용자 결정이 필요하다 |
