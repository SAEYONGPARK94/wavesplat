# R3 — 회귀 검증 (2026-08-13 수정분에 대한 신선한 눈)

**역할**: 회귀 검증자. 수정 작업에 참여하지 않았다.
**범위**: `git diff`(31파일) 전량 + auto-memory 5파일 + `tmp/S13_oos_2026-08-12/RESULT.md`
+ 신규 `WARNING_2026-08-13.md` 3개. `S13b_seed_robust_2026-08-13/` 는 **열지 않았다**.
**READ-ONLY** — 이 파일 외에 아무것도 수정하지 않았다.

## 판정: **결함 13건** (BLOCKER 2 · MAJOR 6 · MINOR 5)

수정의 **대부분은 사실과 맞고 재도출된다**(아래 §3, 27항목 재현). 숫자를 발명한 곳은
**1건**(§1 M-1)뿐이다. 그러나 **정정이 도달하지 않은 파일 2개**와 **새로 써넣은 거짓 사실 1건**이 있다.

---

# 1. 결함 목록

## [BLOCKER] B-1 — 정정이 S13 **메모리 문서에 도달하지 않았다**

`/home/vexit/.claude/projects/-home-vexit-wavesplat/memory/project_s13_oos_2026-08-12.md`
는 **`2026-08-13` 문자열이 0회**다(mtime 08-13 01:21 = 감사 이전). FIX2 가 편집한 4파일 목록에
없고, V1 반증 이후에도 손대지 않았다. 그런데 `MEMORY.md:28` 이 이 문서를 **S13 의 권위**로 링크한다.

이 문서가 지금 담고 있는 것(전부 08-13 에 철회된 문장):

| 위치 | 살아 있는 철회 문장 | 정정본 |
|---|---|---|
| frontmatter `description` | *"기여 #3 제거·**#2b 복권**(낡은 절제 파일 부호 반전). λ 지분 37%"* | #2b = `UNDECIDED` |
| `:33` 표 | `#2b 분산관계 하드 제약 … +70.2% / +777.8% … **ALIVE, 복권**` | ⛔인용 금지 |
| `:36` | *"★**대칭 감사가 값을 했다**"* | bans 가 "서사 폐기" |
| `:23` | *"**사전노출 24** 제외해도 유지"* | 실제 128/128 |
| `:41-42`, `:58` | *"★λ=1e-4 배포가 **신선 시드에서 재확인**됐다"* | ⛔거짓 판정 |

⚠**frontmatter `description` 은 인덱스·검색에 노출되는 필드**다 — 철회된 주장이 가장 먼저 읽힌다.
**수정안**: 이 파일 맨 위(frontmatter 닫는 `---` 다음)에 `S13_oos/RESULT.md` 와 **같은 정정 배너**를
넣고, `description` 에서 "#2b 복권"·"λ 지분 37%" 를 빼라.

## [BLOCKER] B-2 — `direction1` 배너에 **새로 써넣은 거짓 사실**

`results/direction1/REPORT.md:2-7` · `REVIEW_LOG.md:2-7` · `closed_loop_trajectory_VERDICT.md:8-12`
(그리고 `direction3/platform_aperture_VERDICT.md:8-9`)에 이번 라운드가 삽입한 문장:

> *"이 팔은 **LiDAR 광선·차폐·raydrop·FOV·range 를 0회 호출**하고(**`wavesim` 임포트 0회**),
> 관측을 상자 균일난수 + 해석식 `Σ A·cos` + N(0, 0.02) 로 만든다."*

**반증(코드 직접 열람)** — `REPORT.md` 자신이 헤더에 `Code: scripts/d1_crb.py` 라고 적어 둔 그 파일:

```
scripts/d1_crb.py:39   from wavesim.data_writer import load_episode      ← 임포트 0회? 거짓
scripts/d1_crb.py:149  d = load_episode(path)
scripts/d1_crb.py:154  lp = np.array(d['lidar_points']); vm = np.array(d['lidar_valid_mask'])
scripts/d1_crb.py:163  def collect_obs(ep, t_obs):   # ← 이 LiDAR 점군을 그대로 관측으로 쓴다
```

`REPORT.md:73-77` 과 `:88-92` 의 표(`BF4 5/15/30-comp`)는 **바로 이 경로**의 산출물이다
(`REPORT.md:117` *"depth not stored in episodes"* 가 에피소드 사용을 자백한다).
상자 균일난수 경로(`d1_crb.py:202-204`, `:282-284`)는 **같은 파일의 다른 실험**이지 전부가 아니다.

⚠`d72_platform_aperture.py` 는 `wavesim` 참조 **0건**이 맞다 ⟹ `direction3/platform_aperture_VERDICT.md`
의 같은 문장은 **참**이다. **거짓인 것은 `direction1/**` 3파일**이다.

**수정안**: `direction1/**` 배너에서 "`wavesim` 임포트 0회"·"LiDAR 0회 호출"·"상자 균일난수로 만든다"
를 빼고, **살아 있는 근거만** 남겨라 — "추정기가 출하 `SequentialUpdater` 가 아니라 `tnse`+`relax_step`"
+ "개구 19.97×19.96 m vs 출하 149–203 m"(이 둘은 확인 안 함, A3 근거 그대로). ⚠`CLAUDE.md:106`
의 *"`results/direction1/*` 계열은 **LiDAR 를 0회 호출**"* 도 같은 오류다(이번 라운드 이전부터 존재).

## [MAJOR] M-1 — **"λ 지분 상한 37%" 는 상한이 아니다** (유일한 발명 수치)

`results.jsonl` 704행에서 6규약 전부 재도출(내 독립 스크립트):

| 규약 | 마진차 | 직접비 |
|---|---|---|
| `med(rel)` | **37.2%** | 36.3% |
| `rel(med)` | 28.9% | 37.0% |
| `rel(mean)` | 36.9% | **45.1%** |

**최댓값은 45.1%** 다. "**상한** 37%" 는 어느 규약으로도 나오지 않는다.
게다가 `reference_citation_bans.md` **자기 문장 안에서** *"허용 = **λ 지분 상한 37%**(6 규약 전부
**28.9–45.1%**)"* 라고 상한과 범위를 나란히 적어 **자기모순**이다.
전파처 4곳: `bans (08-13 재감사)` · `MEMORY.md:5` · `MEMORY.md:28` ·
`project_audit_recheck_2026-08-13.md:74` · `S13_oos/RESULT.md:14`.
⚠방향이 **우리에게 유리한 쪽**(λ 기여를 작게 보이게 함)이라 더 위험하다.
**수정안**: `"λ 지분 = 37.2%(med(rel) 마진차), 규약 범위 28.9–45.1%, 6/6 이 사망문턱 50% 미만"`.

## [MAJOR] M-2 — `S9` 에 **틀린 재실행 경고**를 복사했다

신규 파일 `tmp/S9_lr_x_l1_2026-08-11/WARNING_2026-08-13.md` 는 S7/S8 의 것과 **바이트 동일**
(`diff` 무출력 ×2)이고, §2 가 이렇게 말한다:

> *"기준선 팔(`AUTO` 계열)이 `lambda_l1` 을 명시하지 않고 폴백에 맡긴다 … 지금 재실행하면
> 기준선 = 처치 가 되어 효과가 0 으로 붕괴한다."*

**S9 에서는 거짓이다**:
```
tmp/S9_lr_x_l1_2026-08-11/run_s9.py:90-96
# ★재현성 수리 2026-08-11 (E1 감사가 잡음). 이 배치가 돌 때 `AUTO` 는 `lambda_l1` 키를 **갖지…
S5.S4.EXTRA['AUTO'] = {'lambda_l1': L1_SHIP}      ← 기준선이 λ 를 명시한다
```
대조로 S7(`run_s7.py:77` *"EXTRA 없음 = 아무것도 덮지 않는다"*)·S8(`run_s8.py:79` 동일)은
경고가 **맞다**. 그리고 같은 라운드의 다른 산출물이 이미 정확히 그렇게 적어 두었다 —
`memory/project_s9_prereg_2026-08-11.md:38` *"**E1 이 고친 건 S9 뿐**"* ·
`project_audit_recheck_2026-08-13.md:118` *"재실행 시 무효 = … λ-처치 배치 **S5–S8**"*(S9 제외).
**수정안**: S9 의 WARNING 에서 §2 를 삭제하거나 *"S9 는 해당 없음 — `run_s9.py:96` 이 기준선 λ 를
명시한다(E1 수리)"* 로 바꿔라.

## [MAJOR] M-3 — `S13_oos/RESULT.md` 는 **머리만 고치고 본문은 그대로**다

머리에 정정 5건 배너가 들어갔지만 본문의 원문이 **⛔ 표시 없이** 살아 있다:

| 줄 | 살아 있는 문장 | 배너 항목 |
|---|---|---|
| `:73` | *"★동시에 **λ 자체가 신선 시드에서 재확인**됐다(54/64, p=1.0e-08)"* | 2 |
| `:82` | 표 `#2b … +70.2% … **ALIVE**` | 1 |
| `:88` | *"★★★#2b 는 낡은 파일이 틀렸다"* | 1 |
| `:93` | `### S-2a 근거리 고리 0_5 (**사전 지정** 1개)` | 4 |
| `:116-117` | **결론 절** *"3. 기여 #3 제거 · **기여 #2b 복권**" / "4. λ=1e-4 배포가 **신선 시드에서 재확인**(54/64)"* | 1, 2 |

결론 절이 배너와 정면으로 어긋난다. 이 저장소가 반복해서 대가를 치른 "헤드라인만 살아남는" 형태다.
**수정안**: 최소한 `:116-117` 결론 절과 `:82` 표 셀에 인라인 ⛔ 를 달아라.

## [MAJOR] M-4 — `PRESENTATION_SUMMARY.md §4.3` 표가 **철회되지 않고 남았다**

`docs/PRESENTATION_SUMMARY.md:226-241` — 12-에피소드 표가 셀 전량 LIVE:
`Avg | **0.211** | 14.2 | **0.358** | 166`, `BF2 1c … 0.436 | 154`.
같은 라운드가 **다른 두 파일에서 정확히 이 수치를 금지**했다:
`README.md:191` *"⛔RETRACTED — `0.211 vs 0.358 (41.1% better)`"* ·
`docs/CHANGELOG.md:7` *"인용 금지: `0.211 vs ETKF 0.436 (+51.5%)`"*.
§4.1/§4.2 는 셀을 ⛔ 로 갈아치웠는데 §4.3 만 빠졌다(아래 불릿만 철회됨).
⟹ **FIX1 §6.1 의 자기검증 "잔존(LIVE) **0**"** 은 이 표를 놓쳤다(분류기가 금지 리터럴 집합
`0.223/0.319/13.4/132.6` 만 봤고 `0.211/0.436/0.358` 은 안 봤다).
**수정안**: §4.2 와 같은 방식(셀 ⛔ + 원본 HTML 주석 이력 보존).

## [MAJOR] M-5 — `README.md:183` 이 **n=10 값을 100-에피소드 표에 넣었다**

```
### Full 100-episode Comparison
| WaveSplat | 0.2229 |            ← results/100ep_ws_letkf.json, n=100
| LETKF     | 0.2278 |            ← 같은 파일, n=100
| GPR       | 0.276  | (재도출로 확인된 유일한 GPR 값)   ← results/gpr_comparison.json, n=10
```
재도출: `gpr_comparison.json` 은 **10 조건**이고 `GPR mean = 0.2756`, 같은 파일의
`WS-P1 = 0.2325` · `ETKF = 0.4499` 다. 즉 **다른 코퍼스·다른 기준선**의 값을 라벨 없이 한 표에 섞었다.
(`docs/STUDY_GUIDE.md:9` 는 같은 3수치 `0.233·0.450·0.276` 을 **"구버전"** 이라 부른다 — 자기모순.)
**수정안**: 표에서 GPR 행을 빼고 각주로 *"GPR 0.276 = `gpr_comparison.json`, **10 조건**, 다른 표본"*.

## [MAJOR] M-6 — 판정을 뒤집은 **32런 개입 실험의 원자료가 없다**

`#2b 복권 철회`(B-1·M-3 의 원인이자 새 ban 항목)의 **유일한 실측 근거**는
`V1_refute_nodisp.md §7` 의 32런이다. 그런데 V1 자신이 *"monkeypatch, in-memory,
**저장소 파일은 건드리지 않았다**"* 라고 적었고, `ls tmp/AUDIT_recheck_2026-08-13/` 결과는
**`.md` 12개뿐** — 스크립트도 `results.jsonl` 도 없다.
⟹ `p=0.804 / p=1.000` 은 **재도출 불가**다. 이 저장소가 A1 에게 요구한 규율
(*"집계 스크립트를 파일로 남길 것"* — `A1_s13_rederive.md:333`)이 V1 자신에게 적용되지 않았다.
⚠판정 방향은 **보수적**(우리 기여를 죽이는 쪽)이므로 위험은 낮으나, `reference_citation_bans.md`
에 등재된 항목이 **재현 불가 근거** 위에 서 있다.
**수정안**: 몽키패치 스크립트 + 32행 결과를 `tmp/AUDIT_recheck_2026-08-13/` 에 저장하라(런 0회, 파일만).

## [MINOR] m-1 — `README.md` 내부 모순 2건 (같은 파일)
- `:32` 가 raydrop **`20/20` 을 ⛔RETRACTED** 로 선언하는데 `:203` 은 *"WaveSplat wins **10/10**
  under both uniform and slope raydrop"* 을 **LIVE** 로 둔다(같은 주장, 10ep×2).
- `:206-211` 절제 표가 *"w/o Confidence: 0.215 (+14.5%, **essential**)"* 로 남아 있다 —
  같은 라운드가 메모리에 **기여 #3 `DEAD`**(끄는 쪽이 낫다)를 확정했다.
  (FIX1 §7-5 가 "판단 대상으로 남겼다"고 공시 ⟹ 누락이 아니라 **의도적 보류**이나 모순은 실재.)

## [MINOR] m-2 — `strategic_repositioning.md` 가 **WaveSim 에 196 을 붙였다**
`:83` *"**WaveSim 시뮬레이터** … 100 episodes, **196 unit tests**"* · `:2132` 동일.
구값 `39` 는 **wavesim 단독**이었고(문장도 그렇게 자백한다), 오늘 실측은 **wavesim 85 / wavesplat 111**.
WaveSim 을 가리키는 문장의 정답은 **85** 다. **수정안**: 두 곳을 `85`(또는 "저장소 전체 196 중 wavesim 85")로.

## [MINOR] m-3 — `CLAUDE.md` 의 `562줄` 이 **같은 라운드에 의해 낡았다**
`CLAUDE.md:16` `docs/PRESENTATION_SUMMARY.md # 발표 요약 (562줄)`.
`git show HEAD:…` = 562 ✓ 이지만 **현재 파일은 617줄**(배너 삽입분). 종전 표기 `~600줄` 이 오히려 맞았다.

## [MINOR] m-4 — `CLAUDE.md` 의 "배치 **24개** 미추적" 이 자기 예시와 안 맞는다
실측: `wavesplat/tmp/` 아래 **미추적 디렉터리 34개**. `RESULT|PREREG` 를 가진 것만 세면 **24개**지만,
그 24 집합에는 **CLAUDE.md 가 예시로 든 `S11`·`AUDIT_full`·`B1` 이 들어 있지 않다**.
**수정안**: `"미추적 디렉터리 34개(그중 RESULT/PREREG 보유 24개)"`.

## [MINOR] m-5 — 출처 오귀속·용어 느슨함 4건
1. `scripts/generate_graphical_abstract.py:75` — `speed = None  # … shipping_default.json` 이라 적었으나
   `tmp/BLANKSLATE_2026-08-12/shipping_default.json` 의 팔은 **`shipping_default` vs `v2_konly_100ep`**
   (둘 다 WaveSplat)로 **ETKF/LETKF 타이밍이 없다** ⟹ WS-vs-ETKF 비를 그 파일에서 못 얻는다.
2. **"동일 프로세스·1스레드 1.5–2.6×"** — 1스레드 **핀 고정** 측정치는 `1.65× [1.63,1.68]`
   (`project_session_2026-08-05_night.md:42`)이고, `2.6×` 는 08-04 의 **"같은 런"** 값
   (`project_blank_slate_debate_2026-08-04.md:22`, WS 7.02 / ETKF 18.46 ms)이다. 두 조건을 하나의
   수식어로 묶었다. (방향은 보수적 — 종전 `~10×` 대비 축소.) 전파 ~10곳.
3. **`p = 0.9204`** 는 51/100 **양측 이항 부호검정**이다(정확 재현). 같은 자료의 **짝 Wilcoxon 은
   `p = 0.9890`**. 어느 문서도 검정 이름을 안 적었다 ⟹ "Wilcoxon p" 로 오독될 수 있다.
4. `CLAUDE.md:79` *"`sequential_update.py:426`·`:967` 에 λ=0.001 **하드코딩 잔존**"* — 둘 다
   `if os.environ.get('S8_L1_TRACK_ONLY'/'S8_L1_PHASE1_ONLY') == '1'` **안**이라 그 환경변수 없이는
   도달 불가다. 출하 경로가 1e-3 을 쓰는 것처럼 읽힌다.
5. `direction1/closed_loop_trajectory_VERDICT.md`·`direction3/platform_aperture_VERDICT.md` 가
   `# VERDICT …` + `# ⛔ RETRACTED …` 로 **H1 이 2개**가 됐다(렌더 깨짐은 없음).

---

# 2. 새로 **깨진 것** — 없음

| 점검 | 결과 |
|---|---|
| `pytest wavesplat/wavesim/tests/ wavesplat/wavesplat/tests/ -q` | ✅ **196 passed in 130.03s** (0 failed / 0 skipped) — 85 + 111 |
| `py_compile` 수정 `.py` 5개 | ✅ 전건 OK |
| 생성기 3종 **실제 실행** | ✅ 3/3 `RuntimeError` 발화, 산출물 타임스탬프 **불변** (`graphical_abstract.pdf` Jun 24 · `labmeeting_2026-07-12.pptx` Jul 12) |
| 게이트 앞 조기 저장 | ✅ 없음 — `prs.save`/`fig.savefig`/`img.save` 전부 게이트 **뒤**(1255 / 1790 / 107 / 1052 vs 게이트 419 / 576 / 70) |
| `docs/memory/*.md` 12개 배너 | ✅ 12/12, **frontmatter 닫는 `---` 뒤**에 삽입 — YAML 무손상 |
| `MEMORY.md` 링크 | ✅ **120개 전부 실존**, 고아 문서 **0** |
| 마크다운 표 구조 | ✅ 파손 없음(⛔ 셀 치환 시 열수 유지; `data_strong` 3행→4행도 `add_table(…, 4, 8, …)` 로 동기화) |
| `realdata_analysis.py` 회귀 | ✅ 호출자 16곳 전부 **키워드 인자** ⟹ 시그니처 확장 무해. `fps` 는 `tmax_frames` 경로에서 연속 프리픽스라 값 불변 |

⚠단 **생성기 게이트는 스크립트 중간**에 있어 금지 수치와 무관한 슬라이드까지 포함해 **덱 전체**가
죽는다(= fail-closed, 의도된 설계로 판단). 부작용으로 게이트 **뒤**에 넣은 ⛔ 셀 편집
(`generate_labmeeting_pptx.py:516-560,588-660,698,831-841,1129`, `generate_ppt.py:585-1780`)은
**도달 불가 코드**다 — 해가 되진 않으나 FIX1 이 "차단점" 과 "리터럴 처리" 를 둘 다 한 것은 중복이다.

---

# 3. 재도출 결과 — **27항목 중 23 일치, 4 불일치**

## ✅ 재현된 것 (23)

| # | 주장 | 재도출 | 방법 |
|---|---|---|---|
| 1 | `196` tests (85+111) | **196 passed** | 직접 실행 |
| 2 | WS `0.2229` / LETKF `0.2278` | 0.2229 / 0.2278 | `100ep_ws_letkf.json` 100행 평균 |
| 3 | `51/49` | 51 승 | 동상 |
| 4 | `p = 0.9204` | 0.9204107… | `binomtest(51,100,.5)` 양측 |
| 5 | 레짐 분해(폐기 근거) | ≤5comp 51/60 p=6.0e-07 · ≥15comp 0/40 p=1.8e-12 | 동상 |
| 6 | S13 혼잡 `56/64`, `p_adj 2.78e-10`, `−28.8%` | 56/64, 2.781e-10, −28.82% | `S13/results.jsonl` 704행 독립 재집계 |
| 7 | S13 희소 `63/64` | 63/64 | 동상 |
| 8 | 공개 LETKF 라면 `63/64 · −32.3%` | 63/64, −32.32% | 동상 |
| 9 | 704 = 128×5+64, 팔별 행수 | `SHIP/LETKF_PLUS/LETKF/NOCONF/NODISP` 128 + `L1E3` 64 | 동상 |
| 10 | `AMD Ryzen 7 7800X3D` | 일치 | `lscpu` |
| 11 | `162×183 m` | 일치 | `B3/RESULT.md:12,38,68,107` |
| 12 | `24.0 s (IQR 18.5–68.1)` | 일치 | `BLANKSLATE/RESULT.md:90` |
| 13 | `7.2 ms (IQR 6.3–17.4)` | 일치 | `BLANKSLATE/RESULT.md:91` |
| 14 | `34/34` · `26/34` | 일치 | `BLANKSLATE/RESULT.md:231-245` |
| 15 | 클러스터 바닥 `p=0.031` | 2⁻⁵ = 0.03125 ✓, 코퍼스 키가 `seed0–4` 뿐 ✓ | 산술 + JSON 키 |
| 16 | 시드 잔여 `[1581–1818]` | 최대 사용 **1580**, 1515–1580 **빈틈 0**, 1581+ 소모 **0** | `tmp/*/*.jsonl` **119파일** 전수 스캔(S13b 제외) |
| 17 | `40e22f6` = 08-11 17:19 | 일치 | `git log` |
| 18 | `5f0846c` = 08-12 00:30 | 일치 | `git log` |
| 19 | `40e22f6` 은 `sequential_update` 를 **안 바꿨다** | `git show 40e22f6:…/sequential_update.py:160` = `config.get('lambda_l1', 0.001)` ✓ | `git show` |
| 20 | 커밋 본문 "적용 (3곳)" | `estimation.yaml` + `run.py` DEFAULT_CONFIG + yaml 폴백 ✓ | 커밋 메시지 + `--stat` |
| 21 | `ablation_bf45.json` Full = `0.2757` | 0.2757(150 시드값 중앙) · 제로필드 ≈0.25 는 Hs=4σ 로 해석적 ✓ | 재계산 |
| 22 | `omega_free` 설치 경로 부재 | `sequential_update.py:550-553,635-638` 이 **kx,ky,w,phi 4개만** copy_ · `learned_init.py:324` 4+n 반환 ✓ | 코드 |
| 23 | 코드 줄 참조 12개 | `estimation.yaml:12/22/24` · `run.py:30/43/94/99/139-141` · `sequential_update.py:40/181/330/426/967` **전건 정확** | 코드 |

**부수 확인**: `results/*.json` 전수 스캔에서 nRMSE 계열 평균이 `[0.315, 0.323]` 에 드는 파일 **0개**
⟹ *"분모 `0.319` 를 내는 결과 파일이 없다"* 를 독립 경로로 지지.
**구조 카운트**: `scripts/` 413 ✓ · `results/*.json` 96(하위 포함 624) ✓ · `data/**.h5` 110 ✓ ·
`paper/figures/*.png` 5 ✓ · `real_bergamasco` 6사이트 **13 레코드** ✓ ·
`run.py:537` `--episode` required ✓ / `:565` `--adaptive_n` 은 `str` ✓ (CLAUDE.md 의 명령 정정이 옳다).

## ❌ 재도출 실패 (4)

| 주장 | 실측 | 등급 |
|---|---|---|
| `λ 지분 **상한** 37%` | 6규약 28.9–**45.1**%; 37.2% 는 한 규약의 점추정 | M-1 |
| `wavesim 임포트 0회`(direction1) | `d1_crb.py:39` 가 임포트하고 `:149` 가 호출 | B-2 |
| `PRESENTATION_SUMMARY (562줄)` | **617줄**(같은 라운드가 늘림) | m-3 |
| `배치 24개 git 미추적` | 미추적 디렉터리 **34개**(RESULT/PREREG 보유분만 24) | m-4 |

## ⚖ 과잉 철회 점검 — **과잉 없음**

- **#2b `UNDECIDED`**: 코드 근거(설치 지점이 ω 를 안 채운다)가 **독립적으로 검증됨**(재도출 #22).
  개입 실험 없이도 *"`NODISP` 는 분산 제약의 절제가 아니다"* 는 성립 ⟹ 철회는 정당(단 M-6).
- **#3 `DEAD` 유지**: `use_confidence` 가 손실 가중치 한 곳(`sequential_update.py:330`)만 건드리는 것
  코드로 확인 ⟹ `NODISP` 와 비대칭이라는 논거 성립.
- **`×4.4` / "platform motion unlocks"**: `MEMORY.md`·B-1 토론 기록의 `정지 0.00986 vs cross 0.01615, 0/12`
  와 일치 ⟹ 철회 정당.
- **`0.223/0.319/13.4/132.6`**: 재도출 #2·#5 + 0.319 부재 스캔이 지지 ⟹ 철회 정당.

---

# 4. 미수정 잔여

## (i) 의도적 보류 — 공시돼 있음

| 항목 | 근거 |
|---|---|
| `paper/wavesplat.tex` · `paper/cover_letter.tex` · `paper/submission/{cover_letter.tex, graphical_abstract.md}` · `thesis/thesis_main.tex` — 금지 수치 잔존 **확인함** | ⛔논문 편집 정지(`feedback_no_paper_edits…`) · FIX1 §7-1 |
| 산출물 PDF/PPTX(`paper/wavesplat.pdf`, `thesis_main.pdf`, `docs/WaveSplat_Algorithm_Guide.pptx`) | 위와 동상, 열지 않음 |
| `tmp/S7*/S8*/S9*/analyze_*.py` 의 폐기된 `SUPPORT-TRADEOFF` 게이트 | `FROZEN_md5` 봉인 ⟹ `WARNING_2026-08-13.md` 로 대체(단 M-2) |
| `PRESENTATION_SUMMARY.md:418` §6.1 "Novelty 10항목" | 금지목록에 항목 자체가 없어 대체 문구 부재(FIX1 §7-6) |
| `README.md:203,206-211` raydrop `10/10` · 절제 `0.188/0.158/0.215` | FIX1 §7-5 가 "판단 대상" 으로 공시(단 m-1 모순 실재) |
| `(08-03)` "분산 하드제약 유의" ban 항목 원문 유지 | V1 이 "판정 확정 → 유지" 로 종결 |

## (ii) 그냥 누락

| 항목 | 등급 |
|---|---|
| `memory/project_s13_oos_2026-08-12.md` 전체 (철회 5문장 + frontmatter) | **B-1** |
| `S13_oos/RESULT.md` 본문 `:73,:82,:88,:93,:116-117` | **M-3** |
| `PRESENTATION_SUMMARY.md §4.3` 12-ep 표 | **M-4** |
| `direction1/**` 배너의 거짓 사실 | **B-2** |
| `S9` WARNING §2 | **M-2** |
| `ref_paper/trending_2025/review_factbased_v3.md:24,29-31,231` — `v2_konly (default)` · `0.223/0.319/0.276` · `13.4ms/132.6ms` · `57 tests` **전부 잔존 확인** | MINOR (FIX1 이 "대상 목록 밖" 이라 공시했으나 **정정 없이 남았다**) |
| `configs/estimation.yaml:18-19` 주석 *"this constant is ALSO written in run.py … **Keep in sync**"* | MINOR — `5f0846c` 단일출처화 이후 낡음(`CLAUDE.md:71` 이 "세 곳 중복은 낡은 지시" 라고 정정한 것과 어긋남) |

---

# 5. 열람 범위 (= 주장 범위)

**연 것**: `git diff` 31파일 전량 · `AUDIT_recheck_2026-08-13/{A1,A3,A7,V1,FIX1,FIX2}`(전문) ·
`A2,A5,A6,V2,V3`(부분 grep) · auto-memory `{MEMORY.md, reference_citation_bans.md,
project_audit_recheck_2026-08-13.md, project_s13_oos_2026-08-12.md, project_s9_prereg_2026-08-11.md,
project_blank_slate_debate_2026-08-04.md, project_session_2026-08-05_night.md}` ·
`S13_oos/{RESULT.md, results.jsonl(704행), PREREG 미열람}` · `BLANKSLATE/RESULT.md` ·
`B3/RESULT.md` · `results/{100ep_ws_letkf.json, gpr_comparison.json, ablation_bf45.json}` ·
`wavesplat/{run.py, optimizer/sequential_update.py, optimizer/initializer.py, optimizer/learned_init.py}` ·
`scripts/{d1_crb.py, d72_platform_aperture.py}` · `run_s7.py/run_s8.py/run_s9.py`(λ 부분).

**안 본 것 (주장하지 않음)**: `tmp/S13b_seed_robust_2026-08-13/**`(실행 중, **금지구역**) ·
`paper/**`·`thesis/**` 원고 본문 · PDF/PPTX 산출물 · `A4` 전문 · `V1 §1-6` ·
`docs/STUDY_GUIDE.md` 전문(요지 4구간만) · 손상 slope 918파일 검사 · "저장소 전체 237 tests".
