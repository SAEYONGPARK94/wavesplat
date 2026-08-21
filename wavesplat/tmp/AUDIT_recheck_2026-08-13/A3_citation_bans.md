# A3 — 인용금지 목록 × 저장소 실사 (2026-08-13)

**READ-ONLY 감사.** 이 파일 외 어떤 파일도 수정/생성하지 않았다.
논문·학위논문 파일의 위반은 **"⛔편집정지"** 라벨을 붙이고 제안만 기재했다.

---

## 0. 스캔 범위 (열람 범위 = 주장 범위)

| 항목 | 값 |
|---|---|
| 루트 | `/home/vexit/wavesplat` |
| 포함 확장자 | `.md` `.tex` `.txt` `.py` `.html` `.yaml/.yml` (+ 히트 존재 확인용 `.json` 1회) |
| 스캔 파일 수 | **7,505** (md 719 · tex 6 · py 6,376 · txt 379 · html 15 · yaml 7; `external/` 3개 py 포함) |
| 제외 | `.git/**` · 바이너리(`.pdf .png .gif .pptx .h5 .nc .webm`) · `results/**` 의 **원시 측정 json**(지시대로 대상 아님) |
| **판정 제외(=기록으로 인정)** | `wavesplat/tmp/**` · `/tmp/**` 배치 기록 **6,229 파일** — 사전등록/감사/RESULT 문서는 금지사실을 *기록*하는 문서이므로 (ii) 로 분류 |
| grep 한 정규식 패턴 수 | **31** (아래 §1 표의 패턴 열) |
| 실제로 열어 문맥 확인한 파일 | 18 (`CLAUDE.md`, `README.md`, `PRESENTATION_SUMMARY.md`, `STUDY_GUIDE.md`, `wavesplat.tex`, `thesis_main.tex`, `strategic_repositioning.md`, `HANDOFF_OPUS_2026-07-26.md`, `platform_aperture_VERDICT.md`, `closed_loop_trajectory_VERDICT.md`, `direction1/REPORT.md`, `docs/memory/*.md` 5개, `generate_graphical_abstract.py`, `d3_falcon_real_operator.py`, `review_factbased_v3.md`) |

**주장 범위**: 위 확장자·위 루트에 대해서만 부재/존재를 말한다. PDF·PPTX·이미지 안의 텍스트는 **미검사**
(⚠ `paper/wavesplat.pdf` · `thesis/thesis_main.pdf` · `docs/WaveSplat_Algorithm_Guide.pptx` 는 tex/py 소스가
위반하므로 **산출물도 위반일 것이 거의 확실하나 직접 열지 않았다**).

---

## 1. 금지 항목 정규화 표 (금지 문구 / 사유 / 대체값 / grep 패턴 / 결과)

`reference_citation_bans.md` 전문(52줄) + `MEMORY.md` ⛔ 항목을 합쳐 정규화. 저장소에서 검색 가능한
**수치·고유문구가 있는 항목만** 패턴화했다(순수 서술형 금지 — 예 "다중시드 없는 에피소드-단위 델타" —
는 grep 대상이 아니라 설계 규율이므로 제외).

| # | 금지 문구 | 사유 | 허용 대체값 | grep 패턴 | 히트(비-tmp) | 진짜 위반 |
|---|---|---|---|---|---|---|
| 1 | `132.6 ms` 가 ETKF 비용 | 16스레드 unpinned 아티팩트 | 같은 런·1스레드 **10.2–10.3 ms** | `132\.6` | 34 | **19** |
| 2 | `13.4 ms` WS 지연 | 커밋 `77b8123` 이 11.0 으로 바꾸고 짝을 안 고침 | 중앙+IQR 병기 (`shipping_default.json`) | `13\.4` | 30 | **17** |
| 3 | **"~10× 빠르다"** | 스레드 미고정 아티팩트 | **"같은 런·1스레드에서 1.5–2.6×"** | `(~\|약 )?10 ?(×\|x\|배)( 이상)? ?(faster\|빠르)` | 13 | **11** |
| 4 | `WS 0.223 / ETKF 0.319` (=30.1%) | 분모 0.319 를 내는 결과 파일이 저장소에 **없다** (감사 08-12) | `100ep_ws_letkf.json`: WS 0.2229 vs LETKF **0.2278**, 51/49 **p=0.9204 널** | `0\.223\|0\.319\|30\.1` | 파일 60+ (json 포함) | **17** (요약·보고 파일 한정) |
| 5 | **"Intel i7-12700H"** | 실제 박스는 **AMD Ryzen 7 7800X3D** | AMD Ryzen 7 7800X3D | `12700H` | 4 | **4** |
| 6 | `init_method: v2_konly` 를 출하 기본값으로 | 2026-07-26 flip 이후 기본값은 `auto` | `auto` (`run.py:43`·`estimation.yaml:24`) | `v2_konly` | 34 | **3** |
| 7 | "57 tests" | 실측 **196** | 196 | `57 (unit )?tests` | 3 | **3** |
| 8 | `p=1.1e-15` · `Cohen's d=1.16` | #4 의 파생 통계 — 재현 안 되는 분모에서 나옴 | 재도출 전 인용 불가 | `1\.1e-15\|d ?= ?1\.16` | 11 | **11** |
| 9 | "12/12 wins" / "20/20" | 12ep 구버전(#12) + 라우팅 교락 | `S13` 704런 = 혼잡 56/64·희소 63/64 | `12/12\|20/20` | 10 | **4** |
| 10 | `0.211 vs 0.436 (51.5%)` · `31.7 ms` · `2.4×` | pre-tuning ETKF, 구버전 | 폐기 | `51\.5` | 6 | **3** |
| 11 | **"CRB ×4.4 CONFIRMED"** | **단일 시드**(`d72:50` seed=1); 복제 2.8배(8시드)·3.22배(30시드) | **"부호 생존, 크기 사망"** | `4\.4 ?(배\|x\|×)` | 4 | **3** |
| 12 | **"이동이 정보를 만든다"** / "platform motion unlocks" | 실측 관측기하 field-CRB: 정지 0.00986 vs cross 0.01615 (**0/12**) | **"이동이 정보를 줄인다"** | `platform motion\|cross-track` | 3 | **2** |
| 13 | `direction1`/`d72`/`d74` 수치를 출하 시스템 근거로 | LiDAR 광선·차폐·raydrop **전무**, `wavesim` 임포트 0회 | 출하 팔에서 재측정 | 파일경로 | 3 md | **3** |
| 14 | LJ Grid2 **"250×300 m 개구"** | 31.5% 상수 클램프 인공물 | 유효박스 **162×183 m** | `250.{0,4}300` | 2 | **1** |
| 15 | "insurance / 보험" (실측 학습기여) | 실측 학습팔 기여 **0** (07-26 GPU 재측정) | **INTERLOCK** 또는 "기여의 증거 없음" | `insurance\|보험` | 10 | **1** |
| 16 | "All phases complete" | 연구 진행 중, 기여 다수 무효 | 진행형 서술 | `All phases complete` | 1 | 0 (금지 기록) |
| 17 | "누적 40여 건" | 실측 **75개** | 75 | `40여 건` | 0 (repo) | 0 |
| 18 | "11,000 런" | 실측 **16,327** | 16,327 | `11,000` | 2 | 0 (오탐: 학습 장면 수) |
| 19 | `run.py:26` 위치 표기 | 커밋 `5f0846c` 로 무효 | `run.py:30` | `run\.py:26` | 0 | 0 |
| 20 | bank-MMSE 초기화 **"~2배"** | 재도출 1.723/1.392/1.460 | **중앙 1.46배** | `2배\|2× faster` | 13 | 0 (전부 오탐) |
| 21 | `0_5` 셸을 "배 근처"로 | 배-거리 중앙 **28.5–29.0 m** | "관측 지지집합 최내곽 셸" | `0_5.{0,40}배 근처` | 18 | 0 (전부 tmp 기록) |
| 22 | `SUPPORT-TRADEOFF` 게이트 운용 | λ=0 도 통과 = 위음성 입증 | 게이트 폐기 | `SUPPORT.TRADEOFF` (`.py`) | 13 | 0 (전부 tmp) |
| 23 | S6 `−0.02583`·`−21.8%` | 신선 n=32 서 21/32 p=0.110 널 | `SIGN-ONLY` | `21\.8\|0\.02583` | 2 | 0 (오탐) |
| 24 | S10 "희소 −38.4%" 널보정 없이 | 널에서도 53/64 | 널 보정 후 41/14 | `38\.4\|48\.1\|35\.8` | 1 | 0 (오탐: hydrofoil) |
| 25 | G1P1 `−0.0701`·`73.6%` | 신선값 −0.0275 (39%) | 배포 불가 | `0\.0701\|73\.6` | 4 | 0 (오탐) |
| 26 | `lam_p`/`aperture_m` 을 파장·개구로 | FFT 상자 인공물 / 불량셀 포함 bbox | part E 최대 깨끗 직사각형 146.5×167.0 m | `lam_p\|248\.8\|213\.3` | 11 | 0 (오탐) |
| 27 | `6.90×`·`14.24 ms`·`98.31 ms` | 다른 프로세스 | `shipping_default.json` | `6\.90\|14\.24\|98\.31` | 3 | 0 (오탐: 메시 dy 6.90 m) |
| 28 | E-D `−0.1417` · "0.707 이므로 과적합 아님" | 셸=관측거리 / 항등식 | — | `0\.1417\|0\.707` | 2 | 0 (오탐) |
| 29 | "교차 = 20–30 m" · "교차반경 75 m" | 실제 20–50 m 안 / 마스크 면적 균형 | "20–50 m 안" | `20.{0,3}30 ?m\|75 ?m` | 11 | 0 (코드 주석은 금지를 **올바로 기록**) |
| 30 | `ρ(final_n,nRMSE)=+0.316` | 원자료에 없음 | −0.729 (p=0.007) | `final_n.{0,30}0\.316` | 1 | 0 (금지 기록) |
| 31 | `R*=22 m` 범위벽 · `detection_rate` 배치지표 · "3/3" · "무결성 PASS" · "과적합 배제됨" | — | — | 각각 | 0 | 0 |

**오탐(iii) 걸러낸 건수: 총 히트 ≈ 268건 중 179건**(주로 `13.4`/`0.223` 같은 짧은 수치의 우연 일치,
`6.90 m` 메시 dy, hydrofoil 표의 `+11.8%`/`−38.4%`, `2배`(제어 대역폭·주파수), `12/12`(E-12 밀도 축 셀),
`4.4배`(진폭 폭주), `lam_p`(실측 특성화 스크립트의 정상 계산)).
**금지사실을 기록한 문서(ii): 42건**(전부 `tmp/**` 배치 기록 + `docs/DEBATE_2026-08-05_*` + `CLAUDE.md`).

---

## 2. 살아있는 위반 — 파일별

### 2.1 ⛔편집정지 (논문/학위논문 — 제안만)

| 등급 | 파일:줄 | 인용된 문장 | 위반 항목 | 제안(적용 금지) |
|---|---|---|---|---|
| BLOCKER | `paper/wavesplat.tex:53` | `(11.0~vs.~132.6~ms/frame on a single CPU thread, Intel i7-12700H)` | #1,#3,#5 | 문장 삭제 또는 "동일 프로세스·1스레드 1.5–2.6×", CPU = AMD Ryzen 7 7800X3D |
| BLOCKER | `paper/wavesplat.tex:104` | `$\sim$10$\times$ faster (11.0 vs.\ 132.6~ms/frame…)` | #1,#3 | 동상 |
| BLOCKER | `paper/wavesplat.tex:753-754` | `yielding 13.4~ms/frame on CPU (single-threaded, Intel i7-12700H)` | #2,#5 | 재측정값 + AMD |
| BLOCKER | `paper/wavesplat.tex:761,782,783` | `132.6~ms/frame` / 복잡도 표 `13.4~ms` `132.6~ms` | #1,#2 | 표 재생성 필요 |
| BLOCKER | `paper/wavesplat.tex:1371,1387` | `vs.\ 132.6~ms/frame for the tuned ETKF (median 110~ms, mean 132.6~ms)` | #1 | 캡션 재작성 |
| BLOCKER | `paper/wavesplat.tex:2113` | `WaveSplat's computational footprint (13.4~ms/frame, single CPU thread)` | #2 | — |
| BLOCKER | `paper/wavesplat.tex:2684` | `$10\times$ faster ($11.0$ vs.\ $132.6$~ms/frame), and is the only…` | #1,#3 | — |
| BLOCKER | `paper/wavesplat.tex:1890` | `\textbf{Avg} & 0.223 & …` | #4 | 분모 파일 부재 — 표 전체 재실행 |
| BLOCKER | `paper/cover_letter.tex:66` · `paper/submission/cover_letter.tex:47` | `(13.4 vs.\ 132.6 ms/frame, single CPU thread)` | #1,#2,#3 | — |
| BLOCKER | `paper/submission/graphical_abstract.md:34,41` | `"…10× faster…"` / `Table III latencies (13.4 / 132.6 / ~200 ms)` | #1,#2,#3 | — |
| BLOCKER | `thesis/thesis_main.tex:225-226` | `nRMSE **30.1%** 낮추고 (**0.223** vs. **0.319**), 처리 속도는 약 **10배** 빠른 **13.4 ms/프레임**` | #1–#4 | 제1장 기여문 재작성 |
| BLOCKER | `thesis/thesis_main.tex:289-290` | 동 영문판 | #1–#4 | — |
| BLOCKER | `thesis/thesis_main.tex:1449-1450,1460-1461,1466,1480-1481` | `13.4 ms/프레임 … ETKF (132.6 ms) … 약 10배 이상 … Intel i7-12700H` | #1,#2,#3,#5 | — |
| BLOCKER | `thesis/thesis_main.tex:1778,1838-1839,1873,1877,2035` | `0.319` / `0.223` 6-지표 표·BF별 표 | #4 | — |
| BLOCKER | `thesis/thesis_main.tex:2000` | `평균 13.4 ms/프레임으로 tuned ETKF 대비 약 10배 빠르다` | #2,#3 | — |
| BLOCKER | `thesis/thesis_main.tex:3035` | `End-to-end 지연 시간 예산 (Intel i7-12700H, CPU 단일 스레드)` | #5 | AMD Ryzen 7 7800X3D |
| BLOCKER | `thesis/thesis_main.tex:3043` | `\WS{} 추적 갱신 & 13.4 & 13.4\%` | #2 | — |
| BLOCKER | `thesis/thesis_main.tex:3507` | `(시뮬레이션 평균 0.223의 약 1.8배 이내)` | #4 | — |
| BLOCKER | `thesis/thesis_main.tex:3536-3537` | 결론: `30.1%` `0.223 vs 0.319` `10배` `13.4 vs 132.6` | #1–#4 | — |
| MAJOR | `thesis/thesis_main.tex:2126,2150,2156` (A2 신뢰 가중치 절제) | `A2: 신뢰 가중치 제거 (균일 가중치)` 를 **살아있는 기여**로 서술 | **신규 필요**(§4) | S13: 기여 #3 **DEAD**(혼잡 10/64·희소 16/64) ⟹ 기여 목록에서 제거해야 함 |
| MINOR | `hydrofoil/paper/hydrofoil_feedforward.tex:1161` | `(Intel~i7-12700H, single CPU thread)` | #5 | AMD Ryzen 7 7800X3D |

### 2.2 논문 아님 = 실제 수정 가능 (본 감사는 READ-ONLY, 수정 안 함)

| 등급 | 파일:줄 | 인용된 문장 | 위반 항목 | 대체값 |
|---|---|---|---|---|
| **BLOCKER** | `wavesplat/README.md:19` | `\| ETKF (tuned) \| 0.319 \| −0.702 \| 132.6 \| …` (헤드라인 표) | #1,#4 | `100ep_ws_letkf.json` = 51/49 p=0.9204 **널** |
| **BLOCKER** | `wavesplat/README.md:17` | `\| **WaveSplat (ours)** \| **0.223** \| **+0.034** \| **13.4** \|` | #2,#4 | — |
| **BLOCKER** | `wavesplat/README.md:21` | `- **30.1% lower nRMSE** than tuned ETKF, **14.6% better** than GPR` | #4 | — |
| **BLOCKER** | `wavesplat/README.md:22` | `- **~10× faster** than ETKF, **15× faster** than GPR` | #3 | "1.5–2.6×" |
| **BLOCKER** | `wavesplat/README.md:23` | `- Wilcoxon signed-rank test: p = 1.1×10⁻¹⁵, Cohen's d = 1.16 (large effect)` | #8 | 재도출 전 인용 불가 |
| **BLOCKER** | `wavesplat/README.md:36` | `- 57 unit tests, all passing (39 wavesim + 18 wavesplat)` | #7 | **196** |
| **BLOCKER** | `wavesplat/README.md:171,176` | `ETKF (tuned) \| 0.319 \| … 132.6 ms` / `13.4 ms/frame (ETKF: 132.6ms, ~10x faster)` | #1,#2,#3,#4 | — |
| **BLOCKER** | `wavesplat/README.md:233` | `` - `init_method: v2_konly` — beamforming for wavenumber… `` | #6 | `auto` (`run.py:43`) |
| MAJOR | `wavesplat/README.md:77` | `# Beamforming init (v1/v2/v2_konly)` | #6(약) | 라우터 `auto` 병기 |
| **BLOCKER** | `docs/PRESENTATION_SUMMARY.md:15` | `**30.1% 낮추고 약 10× 빠르게 (13.4 vs 132.6 ms/frame)**` | #1–#4 | — |
| **BLOCKER** | `docs/PRESENTATION_SUMMARY.md:129,166-167,185,188` | 6-지표·BF별 표의 `0.223`/`0.319` | #4 | — |
| **BLOCKER** | `docs/PRESENTATION_SUMMARY.md:213-214` | `평균은 **132.6 ms (median 110 ms)**, WS 13.4 ms.` / `**~10× speed advantage**` | #1,#2,#3 | — |
| **BLOCKER** | `docs/PRESENTATION_SUMMARY.md:225,411,512` | `Cohen's d=1.16 → large effect` / `Wilcoxon p=1.1e−15` | #8 | — |
| **BLOCKER** | `docs/PRESENTATION_SUMMARY.md:342-343,403,408,459,531-532` | 복잡도 표·요약 박스·발표 체크리스트 전부 | #1–#4 | — |
| MAJOR | `docs/PRESENTATION_SUMMARY.md:85,211` | `Dispersion-aware beamforming (v2_konly)` / `WaveSplat 12/12 승, best-case 8.5×` | #6,#9 | `auto` / S13 56·63/64 |
| **BLOCKER** | `docs/STUDY_GUIDE.md:19,24,245,358,426-427,542,544,557,562` | `Speed WS 13.4 / ETKF 132.6 → ~10× 빠름`, `p=1.1×10⁻¹⁵ Cohen's d=1.16 (large)`, `0.319`, `nRMSE 30.1% 감소(0.223 vs 0.319)` | #1–#4,#8 | — |
| MAJOR | `docs/memory/MEMORY.md:9` | `ETKF-grid baseline, 비교 완료 (WS 51.5% better)` | #10 | 이 디렉터리는 **구(舊) auto-memory 사본** — 현행 `~/.claude/.../MEMORY.md` 와 충돌 |
| MAJOR | `docs/memory/project_paper_plan.md:22-24` | `WaveSplat avg nRMSE: **0.211** vs ETKF **0.436** (51.5% better)` / `Speed: **13.4 ms/frame** (2.4× faster)` / `12/12 wins, 20/20 raydrop robustness` | #2,#9,#10 | — |
| MAJOR | `docs/memory/project_algorithm_design.md:25,30` | `Beamforming v2_konly → …` / `13.4 ms/frame 달성` | #2,#6 | — |
| MINOR | `docs/memory/project_open_questions.md:11` | `Beamforming v2_konly로 대체` | #6 | — |
| MAJOR | `docs/CHANGELOG.md:203-205` | `P1 avg nRMSE: **0.211** (… ETKF: 0.436, +51.5%)` / `12/12 episodes WaveSplat wins` / `Speed: 13.4 ms/frame (ETKF 31.7ms, 2.4× faster)` | #2,#9,#10 | 이력 문서지만 **철회 표시 없음** |
| MAJOR | `paper/PAPER_IMPROVEMENTS.md:19,21,24` | `WS=0.211 vs ETKF=0.436 → 51.5%` / `WS=0.223 vs ETKF=0.319 → 30.1% improvement` / `"30% improvement"로 수정` | #4,#10 | 이 문서가 **금지 수치로 고치라고 지시**하고 있다 |
| **BLOCKER** | `scripts/generate_graphical_abstract.py:57-58` | `nrmse = [0.223, 0.319, 0.295]` / `speed = [13.4, 132.6, 205.0]` | #1,#2,#4 | **그림 생성기 — 돌리면 금지 수치가 재생산된다** |
| **BLOCKER** | `wavesplat/scripts/generate_labmeeting_pptx.py:541,571,577,602,1088,1129` | `nRMSE 30.1% lower & ~10× faster (13.4 vs 132.6 ms)` · `ETKF (tuned): 132.6 ms (median 110)` · `12/12 승 … 13.4 ms vs 132.6 ms (~10× faster)` · `Cohen's d = 1.16 → large effect` · `57 tests pass` | #1–#4,#7,#8,#9 | 동상 |
| **BLOCKER** | `/home/vexit/wavesplat/scripts/generate_ppt.py:573,615,742,1519,1540-1541,1644,1648,1687-1693,1697,1735` | `Ne = 100 → 매 프레임 132.6ms` · `["속도","132.6ms",">200ms","13.4ms"]` · `nRMSE: WS 0.223 / ETKF 0.319 / GPR 0.261` · `개선율: WS가 ETKF 대비 30.1% 낮은 nRMSE` · `Cohen's d = 1.16` · `0.223 nRMSE \| 13.4ms/frame \| 30.1% better than ETKF` | #1–#4,#8 | 동상 (저장소 **루트** `scripts/`) |
| MAJOR | `results/direction3/platform_aperture_VERDICT.md:18,32` | `Cross-track platform motion collapses the directional CRB: 9.6° → 2.2° (**×4.4 sharper**)` / `(CRB ×4.4 sharper), while along-track surfing is worst` — 제목이 `CONFIRMED` | #11,#12,#13 | 복제 2.8배(8시드)·3.22배(30시드); 실측기하 field-CRB 는 **정지가 이긴다(0/12)** |
| MAJOR | `results/direction3/LEARNED_TRAJECTORY_REVIEW_2026-06-29.md:11` | `motion sharpens the directional CRB ×4.4, along-track (surfing) HURTS` | #11,#12 | 동상 |
| MAJOR | `results/direction1/closed_loop_trajectory_VERDICT.md:33-35` | `Platform motion DOES unlock reconstruction the stationary limit forbids — but only when the trajectory is optimized` | #12,#13 | ⛔이 팔은 LiDAR 광선 0회 + **이중 오라클**; 실측기하선 정지 0.00986 < cross 0.01615 |
| MINOR | `scripts/d3_falcon_real_operator.py:84` | `note='250x300 m storm swell lam~300 m — …'` (LJ18g 레코드 주석) | #14 | 유효박스 **162×183 m** (클램프 교정 후) |
| MINOR | `docs/HANDOFF_OPUS_2026-07-26.md:14` | `the learned arm's role is **failure-mode insurance, not accuracy**` | #15 | 같은 문서 :28 에 `🛑🛑 FINAL STATE` 자기정정 있음 — TL;DR 헤드라인만 미갱신 |
| MINOR | `docs/strategic_repositioning.md:107` | "honest" 대안 제안문이 여전히 `30% nRMSE↓ and ~10× speed` 를 인용 | #3,#4 | — |
| MINOR | `ref_paper/trending_2025/review_factbased_v3.md:24,30-31` | `v2_konly (default)` / `WS avg nRMSE: 0.223 vs ETKF 0.319 vs GPR 0.276` / `Speed: WS 13.4ms/frame … ETKF 132.6ms` / `57 tests all passing` | #1,#2,#4,#6,#7 | — |

**정상(ii) 로 판정한 대표 사례** — 위반 아님:
`CLAUDE.md:50-53,59`(금지를 명시 기록) · `docs/DEBATE_2026-08-05_LATENCY_AND_REPRESENTATION.md`(금지의 출처 문서) ·
`docs/PERSONA_DEBATE_novelty_phd_value_2026-07-26.md`("insurance" 폐기를 기록) ·
`wavesplat/run.py:349` · `wavesplat/evaluation/metrics.py:171`(코드 주석이 "75 m crossover" 를 **오류로 기록**) ·
`tmp/**` 전 배치 기록 · `docs/PAPER_STAGE2_RERUN_2026-08-03.md`·`docs/PERSONA_DEBATE_stage2_cdip_2026-08-03.md`(분산제약 p=0.69 기록).

---

## 3. 패키지 코드·설정은 깨끗

`wavesplat/wavesplat/**` · `wavesplat/wavesim/**` · `wavesplat/configs/**` 에 대해 패턴 #1–#4·#10 을
grep 한 결과 **히트 0**. 금지 수치는 **문서·발표 생성기·구 메모리 사본**에만 있다.

---

## 4. 금지 목록 자체의 무결성 (요청 5항)

| 판정 | 항목 | 근거 |
|---|---|---|
| **해제 필요** | `(08-03)` **"분산 하드제약 **유의**"(p=0.69)** — `reference_citation_bans.md:19` | **S13(08-12/13) 이 복권**: 절제하면 **혼잡 62/64 · 희소 62/64 나빠진다**(+70.2% / +777.8%). 2026-04 절제 파일이 **부호 반전**돼 있었다(`project_s13_oos_2026-08-12.md:33,35`). ⟹ 금지 사유(검정력 부족)가 **소멸**했다. 목록은 여전히 "누적, 전부 유효"(`:11`) 로 이 항목을 살려두고 있다 ⟹ **살아 있는 기여를 계속 죽이는 상태**. |
| **신규 금지 필요(누락)** | **기여 #3 "기하 신뢰 가중치"** | S13: 혼잡 **10/64** · 희소 **16/64**, −15.6%/−20.4% ⟹ **DEAD 확정**. 그런데 금지 목록에 **항목 자체가 없고**, `thesis_main.tex:2126,2150,2156` 과 `paper/wavesplat.tex` ablation 이 **여전히 기여로 서술**한다. |
| **자기위반** | `MEMORY.md` ⛔인용금지 절: *"누적 **40여 건**, 전부 유효"* | 같은 파일이 `bans:48` 에서 **"누적 40여 건" 을 금지**하고 실측 **75개** 라고 적어 뒀다. (repo 밖 파일이라 §2 표에는 미포함.) |
| **모순** | `bans:38`(S7) 과 `bans:40`(S8) 이 **"출하 yaml 한 줄" 금지 → 대체 '두 파일 중복 기재'"** | `bans:48`·`CLAUDE.md:60-61` 은 **세 곳**(`estimation.yaml:12`·`run.py:30`·`sequential_update.py:40`)이라고 정정했다. **대체값 자체가 낡았다** ⟹ `:38`·`:40` 의 "두 파일" 을 "세 파일" 로 갱신 필요. |
| **모순** | `bans:38`·`:40` 의 `wavesplat/run.py:26` | `:48` 이 `run.py:30` 으로 무효화. **같은 파일 안에서 상충하는 줄번호가 병존**. |
| 유지 | `SUPPORT-TRADEOFF` 게이트 폐기(`:48`,`:50`) | 08-12 감사(λ=0 통과=위음성) + BLANKSLATE(셸≠배거리) 로 **이중 지지**. 단 `tmp/S7·S8·S9/analyze_*.py` 가 여전히 게이트를 인쇄하므로, **재실행 시 낡은 판정이 다시 나온다**(⚠운용 주의, 위반은 아님). |
| 유지 | 나머지 전 항목 | 반증·복권 근거 미발견. |

---

## 5. 요약 수치

- 금지 항목 정규화: **31 패턴**(순수 설계규율형 금지 제외)
- 스캔 파일: **7,505**
- 총 grep 히트: **≈268**
- (iii) 오탐 제외: **179**
- (ii) 금지를 기록한 문서: **42**
- **(i) 살아있는 위반: 47개 위치 / 25개 파일** — BLOCKER 32 · MAJOR 10 · MINOR 5
  - 그중 ⛔편집정지(논문/학위논문/커버레터/그래피컬앱스트랙트): **21 위치 / 6 파일**
  - 즉시 수정 가능(README·docs·발표 생성기·구 메모리·VERDICT): **26 위치 / 19 파일**
- 금지 목록 무결성: **해제 필요 1 · 신규 필요 1 · 자기위반 1 · 내부 모순 2**
