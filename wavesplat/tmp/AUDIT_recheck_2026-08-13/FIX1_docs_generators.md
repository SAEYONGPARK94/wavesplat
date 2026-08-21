# FIX-1 — 인용금지 위반 실제 수정 (문서 · 발표생성기 · 판정파일 · 구 메모리)

**작업일**: 2026-08-13 · **지시서**: `A3_citation_bans.md` · **금지목록**: `memory/reference_citation_bans.md`
**성격**: 제안이 아니라 **편집**. 대체 숫자는 **발명하지 않았고**, 금지목록이 처방한 값만 썼다.

---

## 0. 사용한 대체값 (전부 금지목록/감사가 처방한 것)

| 항목 | 금지된 표현 | 쓴 대체값 | 출처 |
|---|---|---|---|
| 정확도 헤드라인 | `0.223 vs 0.319 = 30.1%` | **WS 0.2229 vs LETKF 0.2278, 51/49, p = 0.9204 → 널** | `results/100ep_ws_letkf.json` 재도출 (감사 08-12) |
| GPR | `0.261` | **0.276** (재도출로 확인된 유일한 GPR 값) | `CLAUDE.md` / 감사 |
| 지연 | `13.4 ms` · `132.6 ms` · `~10×` | **동일 프로세스·1스레드 1.5–2.6×**, 중앙값+IQR 병기 | 금지목록 #1–#3 |
| CPU | `Intel i7-12700H` | **AMD Ryzen 7 7800X3D** | 금지목록 #5 |
| 승패 | `12/12` · `20/20` | **S13 704런 OOS: 혼잡 56/64 · 희소 63/64** | 금지목록 #9 |
| 단위시험 | `57 unit tests` (`39+18`) | **196 passed, 0 failed, 0 skipped** (85 wavesim + 111 wavesplat) | 본 세션 실측 `pytest -q` → `196 passed in 392.88s` |
| 초기화 기본값 | `init_method: v2_konly` | **`auto`** (`run.py:43` · `estimation.yaml:24`) | 금지목록 #6 (07-26 flip) |
| CRB 이동 이득 | `×4.4 CONFIRMED` | **"부호 생존, 크기 사망"** (복제 2.8배/3.22배) | 금지목록 #11 |
| 이동 vs 정지 | `platform motion unlocks…` | **"이동이 정보를 줄인다"** (정지 0.00986 vs cross 0.01615, 0/12) | 금지목록 #12 |
| LJ Grid2 개구 | `250x300 m` | **162×183 m** (클램프 교정 후 유효박스) | 금지목록 #14 |
| 실측 학습팔 | `insurance / 보험` | **INTERLOCK** 또는 "기여의 증거 없음" | 금지목록 #15 |

**대체값이 없는 것은 숫자를 발명하지 않고 `⛔RETRACTED (2026-08-13 감사)` 표시로 처리했다**
(Skill Score · Variance Ratio · Pearson ρ · Dir R² · Wilcoxon p · Cohen's d · 8.5× best-case ·
73.7 Hz · latency budget 14.6 ms / margin 85.4% · `0.211 vs 0.436 = 51.5%` · `31.7 ms` · `2.4×`).

---

## 1. 그룹 1 — 문서 (금지 수치 제거/철회)

| 파일 | 변경 위치 | 변경 내용 | 근거 |
|---|---|---|---|
| `wavesplat/README.md` | `:13-25` Key Results | 표 전체를 ⛔배너 + **검증된 재도출 표**(0.2229/0.2278, p=0.9204 널)로 교체. SS·GPR 0.261·Wilcoxon·d·20/20 은 ⛔RETRACTED. O(·) 복잡도만 유지 | bans #1–#4,#8 |
| | `:36` | `57 unit tests (39+18)` → **196 (85+111)**, 실측 표기 | bans #7 |
| | `:60,70-71` 트리 | `39 tests`/`18 tests` → **85**/**111** | 동상 |
| | `:77` 트리 | `initializer.py # Beamforming init (v1/v2/v2_konly)` → `Init router auto (shipping default) + v1/v2/v2_konly` | bans #6 |
| | `:165-178` | 100ep 3-way 표 → 재도출 표 + ⛔RETRACTED, 12-ep 절 전체 ⛔RETRACTED | bans #1–#4 |
| | `:229-234` Configuration | `init_method: v2_konly` → **`auto`** + ⛔주석 | bans #6 |
| `docs/PRESENTATION_SUMMARY.md` | 문서 머리 | ⛔전면 배너(헤드라인 전량 철회 + 검증 재도출 + 허용 지연 서술) | bans #1–#4 |
| | `:11-20` TL;DR | ⛔RETRACTED + 허용 서술로 교체 | 동상 |
| | `:85` | `v2_konly` 를 출하 기본값으로 읽히던 표기에 ⛔ + `auto` 명시 | bans #6 |
| | `:129` | ETKF `0.319 (100-ep)` → ⛔RETRACTED | bans #4 |
| | `:164-171` §4.1 | 6-지표 표 → ⛔배너 + 재도출 표. 관찰 4건에 ⛔ 라벨 | bans #4 |
| | `:182-188` §4.2 | BF별 표 **전 셀 ⛔RETRACTED**, 원본은 HTML 주석으로 이력 보존 | bans #4 |
| | `:211-214` | `12/12 승`·`8.5×`·`132.6/13.4 ms`·`~10×` → ⛔RETRACTED + 허용 서술(1.5–2.6×, S13 56/63) | bans #1–#3,#9 |
| | `:216-227` §4.4 | Wilcoxon/Cohen's d 표 전 셀 ⛔RETRACTED + 재도출 널 명시 | bans #8 |
| | `:314` §4.9 | `20/20 승` → ⛔RETRACTED (부호 관찰만 유지) | bans #9 |
| | `:340-347` §5 | 지연 열 ⛔RETRACTED + 허용 서술. `O(·)` 열은 해석적이라 유지 | bans #1–#3 |
| | `:403-418` 요약 박스 | 6-지표/통계/지연/20-20 전량 ⛔ + 검증 재도출 블록으로 교체 | bans #1–#4,#8,#9 |
| | `:459-462` 슬라이드 표 | 14·15·17번 항목 ⛔RETRACTED | 동상 |
| | `:512` Q6 | `Cohen's d=1.16 = large effect` ⛔RETRACTED | bans #8 |
| | `:531-539` 체크리스트 | "외울 수치" 6항목 ⛔RETRACTED + ✅검증값 병기 | bans #1–#4,#8,#9 |
| | 문서 말미 | "Table I~IV 와 1:1 매칭" 안심 문구에 ⛔정정 (표 자체가 무효 분모) | 감사 |
| `docs/STUDY_GUIDE.md` | `:14-35` 최신수치표 | 표 전체 ⛔배너 + **검증 재도출 표**(0.2229/0.2278/널, 1.5–2.6×, S13, 196 tests). 무효 15항목 명시 열거 | bans #1–#4,#8,#9 |
| | `:212` | `v2_konly` 에 ⛔ + `auto` 가 출하 기본값임 명시 | bans #6 |
| | `:242-249` §8 | 결과표 → ⛔배너 + 재도출 표, "30.1% 개선"·"73.7 Hz" ⛔ | bans #1–#4 |
| | `:358` | 슬라이드 13 지연·통계 ⛔RETRACTED | bans #1–#3,#8 |
| | `:373` | 보조 슬라이드 A6 `20/20 wins` ⛔ | bans #9 |
| | `:425-427` | "외워두기" 목록 ⛔RETRACTED + ✅검증값 | bans #1–#4,#8 |
| | `:486-488` Q-B | `31.7 ms`·`0.436→0.319` ⛔RETRACTED (설계 서술은 유지, S13 기준선 강화 사실로 보강) | bans #1,#4,#10 |
| | `:542` E-2 | "핵심 수치 암기" ⛔ + ✅검증값 | bans #1–#4 |
| | `:548-565` E-3 | 30초/2분 피치 ⛔RETRACTED 라벨 (원문은 이력 보존) | bans #1–#4,#8 |
| | `:574-585` E-4 | 필수 암기 목록 ⛔RETRACTED + ✅유효 항목(S13, 196 tests, A1 R̄, ETKF sweep) 분리 | bans #1–#4,#8,#9 |
| | `:658` E-7.3 | latency budget `14.6 ms` / margin `85.4%` ⛔RETRACTED (13.4 파생) | bans #2 |
| `docs/CHANGELOG.md` | 문서 머리 | ⛔이력문서 배너 + 인용금지 목록 + 검증 현행값 + `v2_konly→auto` 주의 | bans #2,#6,#9,#10 |
| | `:16` | `13.4 vs 14.2ms 해명` 에 ⛔ (해명 자체가 무효) | bans #2 |
| | `:203-205` | P1 Results 3줄에 ⛔ 마킹 + 상단 RETRACTED 배너 (원문 보존) | bans #2,#9,#10 |
| `docs/HANDOFF_OPUS_2026-07-26.md` | `:14` TL;DR | `failure-mode insurance, not accuracy` ⛔RETRACTED → **INTERLOCK / "기여의 증거 없음"**. 원문 취소선 보존. 같은 문서 `:28` 자기정정과 정합 | bans #15 |
| `docs/strategic_repositioning.md` | `:103-110` | "honest 대안"이 `30% nRMSE↓`·`~10× speed`·`39 unit tests` 를 인용 → **검증 재도출 문단으로 교체** + ⛔RETRACTED. `:104` over-claim 인용문에 "따라 쓰지 말 것" 라벨 | bans #3,#4,#7 |
| | `:83`, `:2132` | `39 unit tests` → **196** (실측) | bans #7 |

## 2. 그룹 2 — 발표/그림 생성기 (실행 시 큰 소리로 실패)

세 파일 모두 **`_blocked_banned_headline()` 게이트 함수**를 추가하고, 금지 수치를 쓰는 **블록 진입점**에서
호출한다. **파일 임포트/헬퍼 정의는 깨지지 않는다.** 원 리터럴은 **주석으로 이력 보존**했다.

| 파일 | 게이트 정의 | 차단점 | 리터럴 처리 | 실행 검증 |
|---|---|---|---|---|
| `wavesplat/scripts/generate_graphical_abstract.py` | `:12` (docstring 직후) | `:70` Panel 3 진입 | `nrmse=[0.223,0.319,0.295]`·`speed=[13.4,132.6,205.0]`·suptitle `30.1% / ~10x` 주석 처리, suptitle 은 수치 없는 문구로 교체 | ✅ `RuntimeError` at `:70`, 출력 파일 **미생성** |
| `wavesplat/scripts/generate_labmeeting_pptx.py` | `:46` (헬퍼 앞) | `:419` (첫 금지 사용 = Adaptive N 지연 문구) | `13.4 ms`·`0.319`·6-지표 표·`30.1%/~10×`·`12/12`·`132.6`·`p=1.1e-15`/`d=1.16`·`20/20`·요약 배너·`57 tests` 전부 주석 이력 + ⛔ 셀. 100ep 표는 **재도출값 0.2229/0.2278 + "51/49, p=0.9204 = 널"** 행으로 교체(표 행수 3→4 동기화). `57 tests`→**196** | ✅ `RuntimeError` at `:419`, pptx **미생성** |
| `scripts/generate_ppt.py` (저장소 루트) | `:228` (헬퍼 뒤, 슬라이드 생성 앞) | `:576` (첫 금지 사용 = Slide 19) | `132.6ms`·`0.319`·`0.261`·`13.4ms` 제목/예산·3-패러다임 표 속도/nRMSE 행·6-지표 슬라이드·"핵심 결과" 슬라이드(30.1%/10배/p/d)·마지막 배너 전부 주석 이력 + ⛔/재도출값 | ✅ `RuntimeError` at `:576`, pptx **미생성** |

## 3. 그룹 3 — 반증됐는데 CONFIRMED 로 남은 판정 파일 (철회 스탬프)

| 파일 | 변경 |
|---|---|
| `results/direction3/platform_aperture_VERDICT.md` | **맨 위 철회 배너**(단일 시드 · 복제 2.8/3.22배 · 실측 field-CRB 는 정지 승 0/12 · **LiDAR 0회 호출**). 제목줄에 `⛔ RETRACTED 2026-08-13`, `## Result — CONFIRMED` → `⛔CONFIRMED 철회됨`, `:18` `×4.4 sharper` 불릿 ⛔RETRACTED(원문 보존), `:32` 인용문 안 `×4.4` 에 ⛔ |
| `results/direction3/LEARNED_TRAJECTORY_REVIEW_2026-06-29.md` | **맨 위 철회 배너**(전제 "레버가 PROVEN real" 무효) + `:11` 문장 ⛔RETRACTED, 원문 취소선 보존 |
| `results/direction1/closed_loop_trajectory_VERDICT.md` | **맨 위 철회 배너**(실측 field-CRB 0/12 · 이중 오라클 · 먼바다 전용 `SUPPORT-TRADEOFF`) + **`direction1/**` 는 LiDAR 0회 호출 = 출하 증거 아님** 한 줄. `:33-35` (A) 불릿 ⛔RETRACTED, 원문 취소선 보존 |
| `results/direction1/REPORT.md` | (추가 지시) 맨 위에 **"LiDAR 0회 호출 = 출하 시스템 증거 아님"** 배너 — 원래 이 사실이 적혀 있지 않았음 |
| `results/direction1/REVIEW_LOG.md` | 동상 |

## 4. 그룹 4 — 지시문이 금지 수치를 요구하는 파일

| 파일 | 변경 |
|---|---|
| `paper/PAPER_IMPROVEMENTS.md` | **맨 위 경고 배너만** 삽입. §OPEN-1 Q1 옵션 (a) 가 "`0.319` 로 재계산 → `30% improvement`" 를 **지시**한다는 점을 명시하고, 검증 재도출(널)을 병기. **논문 편집 정지 중이라 지시를 실행하지 않았음**을 기록. 본문은 그대로 보존 |
| `scripts/d3_falcon_real_operator.py:84` | LJ18g note `'250x300 m storm swell…'` → `'162x183 m (valid box after clamp correction; the old "250x300 m" was a constant-clamp artifact)…'` + 근거 주석 |

## 5. 그룹 5 — 구 auto-memory 사본

`wavesplat/docs/memory/*.md` **12개 전부**에 ARCHIVED 배너 삽입 (삭제 없음).
YAML frontmatter 가 있는 11개는 **frontmatter 닫는 `---` 바로 다음**(=문서 본문 맨 위)에,
frontmatter 가 없는 `MEMORY.md` 는 **1행**에 넣었다 — frontmatter 를 깨뜨리지 않기 위함.

```
> ⛔**ARCHIVED / 낡은 사본 (2026-08-13 감사).** 현행 연구 기록의 권위는
> auto-memory(`~/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md`)와
> `tmp/<배치>/RESULT.md` 다. 이 디렉터리의 수치는 다수가 이후 측정에서 무효화됐다 — 인용 금지.
```

대상: `MEMORY.md`(`:9` `WS 51.5% better`) · `project_paper_plan.md`(`:22-24` `0.211 vs 0.436`/`13.4 ms`/
`12/12`,`20/20`) · `project_algorithm_design.md`(`:25,30`) · `project_open_questions.md`(`:11` `v2_konly`) ·
`project_baseline_design.md` · `project_code_doc_issues.md` · `project_eval_metrics.md` ·
`project_lidar_physics.md` · `project_research_decisions.md` · `feedback_dev_principles.md` ·
`feedback_discussion_format.md` · `user_phd_student.md`.

---

## 6. 자기검증 결과

### 6.1 금지 패턴 재-grep (그룹 1·2, 9개 파일)

분류기: `⛔ / RETRACTED / 인용금지 / 재현되지 않는 / 철회 / 하드코딩하고 있었다 / 이력 보존` 이 있는 줄과
그 블록 연속줄 = **철회 기록**, `#` 로 시작하는 파이썬 줄과 `<!-- -->` 안 = **주석 이력**, 나머지 = **잔존**.

| 파일 | **잔존(LIVE)** | 철회 기록 | 주석 이력 |
|---|---|---|---|
| `wavesplat/README.md` | **0** | 8 | 0 |
| `docs/PRESENTATION_SUMMARY.md` | **0** | 21 | 2 |
| `docs/STUDY_GUIDE.md` | **0** | 21 | 0 |
| `docs/CHANGELOG.md` | **0** | 5 | 0 |
| `docs/HANDOFF_OPUS_2026-07-26.md` | **0** | 1 | 0 |
| `docs/strategic_repositioning.md` | **0** | 3 | 0 |
| `scripts/generate_graphical_abstract.py` | **0** | 1 | 4 |
| `scripts/generate_labmeeting_pptx.py` | **0** | 2 | 19 |
| `scripts/generate_ppt.py` (루트) | **0** | 4 | 23 |
| **합계** | **0** | **66** | **48** |

⚠ `0.2229` / `0.2278` 은 금지목록이 **처방한 검증 대체값**이므로 잔존이 아니다.

### 6.2 `python3 -m py_compile`

```
OK  scripts/generate_ppt.py
OK  wavesplat/scripts/generate_labmeeting_pptx.py
OK  wavesplat/scripts/generate_graphical_abstract.py
OK  wavesplat/scripts/d3_falcon_real_operator.py
```

추가로 **세 생성기를 실제로 실행**해 게이트가 발화하고 산출물이 생성되지 않음을 확인했다
(`RuntimeError` × 3, `docs/*.pptx` · `paper/submission/graphical_abstract.*` 타임스탬프 불변).

### 6.3 `git status --short` — 금지구역 이탈 여부

**이탈 0건.** 다음은 **손대지 않았다**:
`paper/wavesplat.tex` · `paper/cover_letter.tex` · `paper/submission/**` · `thesis/thesis_main.tex` ·
`hydrofoil/paper/**` · `wavesplat/wavesplat/**`(패키지 코드) · `configs/**` ·
`tmp/S13_oos_2026-08-12/**` · `tmp/rx2_init_2026-08-05/**` · `tmp/S5*/S6*/S7*/S8*/S9*` ·
`/home/vexit/wavesplat/CLAUDE.md` · auto-memory 디렉터리.

세션 시작 시점 이미 `M` 이었던 파일(내 변경 아님): `CLAUDE.md` ·
`wavesplat/results/gifs/unified_auto_sparse_BF2_1comp.gif` · `wavesplat/scripts/realdata_analysis.py`.

신규 untracked 파일: **이 보고서 1개뿐** (`tmp/AUDIT_recheck_2026-08-13/FIX1_docs_generators.md`).

---

## 7. 판단이 필요해 손대지 않고 남긴 것

1. **논문/학위논문 원고 21 위치 / 6 파일** — `paper/wavesplat.tex`(9곳) · `paper/cover_letter.tex` ·
   `paper/submission/cover_letter.tex` · `paper/submission/graphical_abstract.md` ·
   `thesis/thesis_main.tex`(10곳) · `hydrofoil/paper/hydrofoil_feedforward.tex:1161`.
   **사용자 지시로 편집 정지 중.** ⚠ 이 파일들이 여전히 `0.319`·`13.4/132.6 ms`·`~10×`·`i7-12700H` 를
   담고 있고, **산출물 PDF/PPTX(`paper/wavesplat.pdf`, `thesis/thesis_main.pdf`,
   `docs/WaveSplat_Algorithm_Guide.pptx`)도 같은 수치를 담고 있을 것이 거의 확실하나 직접 열지 않았다.**
2. **`ref_paper/trending_2025/review_factbased_v3.md:24,30-31`** — 감사가 MINOR 위반으로 잡았으나
   이번 작업 대상 목록 밖이라 손대지 않았다(`v2_konly (default)` · `0.223/0.319/0.276` ·
   `13.4ms/132.6ms` · `57 tests`).
3. **`tmp/S7*/S8*/S9*/analyze_*.py` 의 `SUPPORT-TRADEOFF` 게이트** — 금지목록이 게이트 폐기를
   권고했으나 이 스크립트들은 **동시 열람 금지구역**이라 손대지 않았다.
   ⚠**재실행하면 폐기된 게이트 판정이 다시 인쇄된다.**
4. **금지목록 자체의 무결성 문제 4건** (A3 §4) — **해제 필요 1**(분산 하드제약 유의, S13 이 복권) ·
   **신규 금지 필요 1**(기여 #3 신뢰 가중치 DEAD) · **자기위반 1**("누적 40여 건" → 실측 75) ·
   **내부 모순 2**(`bans:38/:40` 의 "두 파일" → 세 파일, `run.py:26` → `run.py:30`).
   **auto-memory 는 금지구역**이므로 수정하지 않았다 — **사용자/메모리 담당이 처리해야 한다.**
5. **`README.md` 의 잔여 낡은 사실** — `results/ JSON (7 files)`(실제 96) · `scripts/`(실제 413) ·
   raydrop `10/10` · ablation `0.188/0.158/0.215`. 인용금지 목록에 없어 판단 대상으로 남겼다.
6. **`PRESENTATION_SUMMARY.md` §6.1 "Novelty 10항목"** — 감사가 "novelty 문헌 방어 0개" 라고 적었고
   S13 이 **기여 #3(신뢰 가중치)을 DEAD** 로 확정했으나, 금지목록에 **항목 자체가 없어서**
   (A3 §4 가 "신규 금지 필요" 로 분류) 처방된 대체 문구가 없다 ⟹ **손대지 않았다.**
