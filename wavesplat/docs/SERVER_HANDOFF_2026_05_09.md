# 서버 핸드오프 노트 — 다른 서버에서 작업 이어가기

**작성일**: 2026-05-08 (이전 서버에서 마지막 작업)
**다음 작업일**: 2026-05-09 ~ 2026-05-13 (Lab meeting 발표 전까지)
**목적**: 다른 서버에서 *어제 컨텍스트를 그대로 복원*하고 Day 3-5 작업을 이어가기

---

## 0. 왜 이 문서가 필요한가

Claude Code 의 memory (`~/.claude/projects/.../memory/`) 는 **각 서버에 local-only** 입니다. 다른 서버에서 시작하면 memory 가 없습니다. 그래서:
- 모든 *영구적 정보* 는 git push 된 markdown 문서에 저장
- 새 서버에서는 이 문서들을 먼저 읽고 시작
- 이 핸드오프 노트가 *어제 무엇을 했는지 + 오늘 무엇을 할지* 의 single source of truth

---

## 1. 어제까지 한 일 (2026-05-08 작업 요약)

### 1.1 Strategic Repositioning §29-§31 추가
`docs/strategic_repositioning.md` 가 28 → 31 섹션으로 확장됐습니다.
- **§29**: Ehlers et al. 2025 (Phys. Rev. Fluids 10, 094901) PFT-PINN 정독 결과
- **§30**: Yan-Che-Li 2025 (arXiv:2510.14356) VOF-EnKF 정독 결과
- **§31**: 12 paper 정독 종합 + 5-카테고리 prior art landscape + 응용 매트릭스 + **main framing 격상**

**Main framing 격상안** (§31 의 핵심):
> "**Dispersion-Coupled Spatio-Temporal Inversion**: Vessel motion (synthetic aperture, Δx_SA = V·T_init = 11.3 m) and dispersion physics (ω = √(g·k·tanh(kd))) jointly resolve ocean wavenumbers 3–7× beyond the per-frame spatial Nyquist of shipboard LiDAR."

이게 paper 1 의 main contribution claim 입니다. 기존 "grid-free spectral Gaussian" framing을 이걸로 전면 교체.

### 1.2 Lab Meeting 2026-05-13 PPT 작업 시작
- **목적**: 외부 청자 (비전문가 포함) 20분 발표 — 우리 exploration phase + 새 framing publish plan
- **Output**: 19 main slides + 7 backup = 26 slides
- **Generator script**: `scripts/generate_labmeeting_2026_05_13.py` (Python, python-pptx)
- **PPT file**: `docs/labmeeting_2026_05_13.pptx` (530 KB)
- **Source content (markdown)**: `docs/labmeeting_2026_05_13_slides.md` — 19-slide outline + Day 3-5 작업 plan

### 1.3 결정된 사항 (어제 토론 결과)

| 결정 항목 | 답 | 이유 |
|---|---|---|
| Target venue | **Ocean Engineering 이상** (JFM/PRF upgrade option) | Algorithmic novelty 부족하지 않음, 정량 증명만 추가하면 OE 이상 가능 |
| Paper 1 framing | **α (algorithmic) + γ (empirical benchmark)**, β (system integration as main) 제외 | β 는 paper 2/학위논문으로, paper 1 은 algorithmic 강조 |
| Hydrofoil 통합 | paper 1 에 main contribution 아님, optional demonstration | β 제외 결정과 일관 |
| Wang 2025 처리 | PPT 에서 빼고 paper 작성 단계에서 정독 | PPT main 메시지가 Wang 2025 detail에 dependent 하지 않음 |
| Time investment | 충분 투자, 빨리 publish | PFT-PINN 위협 (1-3년 내 2D + real-time 발표 가능) |
| Prior art slide format | Top 5 paper detail (1 slide each) + backup A1-A7 | 외부 청자에게 "12편 정독" 신뢰 + 20분 timing |

---

## 2. 다른 서버 시작 — 단계별 명령어

### 2.1 Step 1: Repo 동기화

```bash
# 다른 서버에 이미 wavesplat 있다면:
cd /path/to/your/reserch  # wavesplat의 부모 디렉토리
git pull origin main

# 처음 clone:
git clone https://github.com/SAEYONGPARK94/wavesplat.git
cd wavesplat
```

**확인 사항**: 다음 두 commit 가 git log 에 있어야 합니다:
- `8717d21` — docs: §29-§31 strategic repositioning + lab meeting 5/13 PPT outline
- `cbf06cf` — feat(ppt): Lab Meeting 2026-05-13 PPT generator + 26 slides

```bash
git log --oneline -3
# 예상: cbf06cf, 8717d21, ...
```

### 2.2 Step 2: 환경 확인

```bash
# Python 환경 (python-pptx 필요)
python3 -c "import pptx; print('pptx version:', pptx.__version__)"
# 예상: pptx version: 1.0.2 (또는 비슷)

# 없다면:
pip install python-pptx

# WaveSplat 의존성 (test 돌릴 거면)
pip install -r wavesplat/requirements.txt
```

### 2.3 Step 3: PPT 파일 확인

```bash
ls -la wavesplat/docs/labmeeting_2026_05_13.pptx
# 예상: 약 530 KB

# 재생성 (필요 시):
cd wavesplat
python3 scripts/generate_labmeeting_2026_05_13.py
# 예상 출력:
#   saved: /path/to/wavesplat/docs/labmeeting_2026_05_13.pptx
#   slides: 26 (19 main + 7 backup)
```

---

## 3. Claude Code 시작 시 첫 prompt

다른 서버의 Claude Code 에 다음 prompt 를 붙여넣으세요:

```
어제 (2026-05-08) 다른 서버에서 진행한 WaveSplat 연구를 이어가자.
다른 서버이므로 memory 가 없어. 다음 두 문서를 순서대로 읽고 컨텍스트 복원해줘:

1. docs/SERVER_HANDOFF_2026_05_09.md (이 파일) — 어제까지 한 일 + 오늘 작업 plan
2. docs/strategic_repositioning.md §31 — main framing "Dispersion-Coupled Spatio-Temporal Inversion"
3. docs/labmeeting_2026_05_13_slides.md — Lab meeting PPT source of truth (19-slide outline + Day 3-5 작업 plan)

읽은 후, 핸드오프 노트의 "오늘 작업 plan" 섹션에 따라 진행하자.
오늘 작업: [여기에 오늘 날짜와 우선 작업 명시 — 예: "Day 3 (Slide 11-14 WaveSim+exploration phase) 다듬기" 또는 "Wang 2025 정독 + §32 추가"]
```

위 prompt 의 마지막 줄 [...] 부분만 그날 작업에 맞게 수정하면 됩니다.

---

## 4. 오늘 작업 plan (2026-05-09 ~ 2026-05-13)

### Day 3 — 5/9 (토) — WaveSim + exploration phase 다듬기

**대상 slides**: S11 (WaveSim physics + raydrop), S12 (100 episodes), S13 (WaveSplat 알고리즘), S14 (P1 12-ep result)

**작업 옵션 두 가지** — 둘 중 하나 선택:

**Option A: PowerPoint 에서 직접 다듬기**
- `docs/labmeeting_2026_05_13.pptx` 를 PowerPoint/Keynote 로 열기
- S11-S14 의 visual 강화 (figure 위치 조정, color, font)
- 필요하면 figure 추가 (예: paper figure PDF 캡처)
- 다른 이름으로 저장 (e.g. `labmeeting_2026_05_13_v2.pptx`) 하거나 같은 파일 덮어쓰기

**Option B: Generator script 수정 후 재생성** (reproducibility 우선)
- `scripts/generate_labmeeting_2026_05_13.py` 의 `build_s11()`-`build_s14()` 함수 수정
- bullets 수정, figure path 추가
- 재실행: `python3 scripts/generate_labmeeting_2026_05_13.py`

**추천**: Option A (시각 디자인은 PowerPoint 가 빠름) + 핵심 content 변경은 Option B 로 둘 다 sync.

### Day 4 — 5/10 (일) — Main framing slides 다듬기

**대상 slides**: S15 (spatial Nyquist 한계 발견), S16 (main framing), S17 (mechanism + identifiability)

**가장 중요한 day** — paper 1 의 main contribution claim 이 들어가는 slides.

**중점 사항**:
- S15 의 "spatial Nyquist paradox" 가 외부 청자에게 명확히 전달되는지 (suspense 형성)
- S16 의 main claim 이 1줄로 강력하게 전달되는지
- S16 의 mechanism diagram (vessel motion + dispersion) — 시각적 강화 필요. 직접 그리거나 figure 추가
- S17 의 identifiability decomposition 이 supporting role 로 자연스러운지

**미리 준비**:
- Mechanism diagram 스케치 — 손으로 먼저 그려보고 PowerPoint 에서 그리기 (또는 draw.io, Mermaid)
- Phase 1 ablation 의 4 분기 시각화 (single-frame / multi-frame-spatial-only / multi-frame-disp-postproc / multi-frame-disp-first)

### Day 5 — 5/11 (월) — Closing + backup + presenter notes 재작성

**대상 slides**: S18 (limitations + threat), S19 (timeline + Q&A), backup A1-A7

**작업**:
- S18 의 PFT-PINN 위협 응답 framing 재검토 — 청자가 받아들일 만한가?
- S19 의 timeline 이 현실적으로 보이는지 (overshoot 아닌가?)
- Backup A1-A7 의 short bullets 가 Q&A 시 *즉시 답변 가능* 한 형태인가?
- 모든 slides 의 presenter notes 한국어 자연스러운지 점검

### 5/12 (화) — Rehearsal

- 실제로 20분 timing 에 맞는지 측정 — slide 별 시간 기록
- 시각 디자인 미세조정
- Backup 으로 답할 만한 예상 질문 5-10개 list 작성

### 5/13 (수) — 발표 day

- 발표 시간: [확인 필요]
- 발표 자료: `docs/labmeeting_2026_05_13.pptx` (또는 v2)
- 백업: USB 또는 cloud (Dropbox/Google Drive)

---

## 5. Wang 2025 받으면 처리 방법

사용자가 Wang 2025 (Coastal Engineering 197) PDF 를 보내주실 예정입니다.

### 5.1 받자마자 즉시

```bash
# PDF 를 ref_paper/ 에 복사 (gitignored, 푸시되지 않음)
cp /path/to/wang_2025.pdf wavesplat/ref_paper/wang_2025_coastal.pdf

# Claude Code 에서:
# 1. Read ref_paper/wang_2025_coastal.pdf — 9-항목 체크리스트로 정독
# 2. §32 section 추가 to docs/strategic_repositioning.md
# 3. Case A/B/C 결정 (자세한 내용은 strategic_repositioning.md §8.3)
```

### 5.2 §8.2 의 9-항목 체크리스트

받자마자 다음 9 가지 확인:
1. **State vector**: η + ∂η/∂t? 또는 η + spectral coefficients?
2. **Observation operator H**: obs→grid (interpolation BEFORE assimilation) or grid→obs?
3. **Propagation**: FFT exact dispersion? WASSfast model 의 정확한 정의?
4. **Inflation**: 어떤 종류 (multiplicative, additive)?
5. **Localization**: 사용? r_loc?
6. **Outlier handling**: fuzzy logic 정확한 알고리즘
7. **Test dataset**: 어떤 simulator/실측 데이터? 우리와 비교 가능?
8. **Hyperparameter tuning protocol**: sweep? sensitivity?
9. **Code release**: GitHub 공개?

### 5.3 Case A/B/C 결정

| Case | 가능성 | 의미 | Paper framing |
|---|---|---|---|
| **A** | 50-60% | 우리 etkf_grid.py 가 Wang family | "We compare against Wang 2025 family ETKF" — 강한 quantitative claim |
| **B** | 30-40% | 핵심은 같지만 detail 차이 (fuzzy outlier 등) | "Approximates Wang 2025; full re-implementation = future work" |
| **C** | ~10% | Fundamentally 다른 ETKF 변형 | "Wang 2025 = different family. New baseline needed." |

### 5.4 Case 결정 후 PPT 업데이트

- Slide 6 (Wang-Pan EnKF-HOS) 의 contrast 항목 update — Wang 2025 가 어느 family 인지 명시
- 또는 Slide 6 을 Wang-Pan + Wang 2025 두 paper 다루도록 확장
- Generator script 의 `build_s6()` 수정 후 재생성

---

## 6. Figure 추출 + 삽입 (선택 작업)

5 detailed prior art slide (S5-S9) 의 figure 자리는 placeholder 입니다. 추출 옵션:

### Option 1: PowerPoint 에서 PDF 캡처 (권장)
- Adobe Reader / Preview / SumatraPDF 에서 paper PDF 열기
- 원하는 figure 영역 screenshot (Snipping Tool, Cmd+Shift+4 등)
- PowerPoint 의 placeholder rectangle 위에 paste

### Option 2: pdftoppm 으로 page 단위 PNG 추출
```bash
cd wavesplat/docs
mkdir -p figures/prior_art
pdftoppm -png -r 200 ../ref_paper/desmars2023_jfm.pdf figures/prior_art/desmars2023 -f 8 -l 9
# -f 8 -l 9: 8-9 페이지만, -r 200: 200 dpi
```
그 후 PNG 를 generator script 의 fig_path 에 삽입.

### 권장 figure (Slide 별)
- S5 Desmars 2023: Fig. 2 (setup) + Fig. 8 또는 Fig. 10 (results)
- S6 Wang-Pan: paper Fig 1 (EnKF-HOS pipeline) + predictable zone
- S7 Ehlers 2023 FNO: FNO architecture + SSP results table
- S8 Ehlers 2025 PFT-PINN: Fig. 2 (architecture) + Fig. 8 (MARATHON results)
- S9 Yan 2025 VOF-EnKF: Fig. 2 (pipeline) + Fig. 18 (plunging breaker)

---

## 7. Trouble shooting

### 7.1 PPT 가 열리지 않음
```bash
# PPT 재생성:
python3 scripts/generate_labmeeting_2026_05_13.py
# 그래도 안 되면 git checkout:
git checkout cbf06cf -- wavesplat/docs/labmeeting_2026_05_13.pptx
```

### 7.2 Generator script 에러
```bash
# python-pptx 버전 확인:
python3 -c "import pptx; print(pptx.__version__)"
# 1.0.2 권장. 다르면:
pip install python-pptx==1.0.2
```

### 7.3 Strategic doc 인덱스 깨짐
- `docs/strategic_repositioning.md` 는 31 sections 이어야 함 (line 2698 정도까지)
- §31 의 main framing 이 "Dispersion-Coupled Spatio-Temporal Inversion" 인지 확인
- 깨졌으면: `git log -- wavesplat/docs/strategic_repositioning.md` 로 commit 8717d21 또는 cbf06cf 시점 확인

### 7.4 Memory 가 비어 있는 경우
정상입니다. 다른 서버는 새 memory 로 시작. 위 Step 3 의 prompt 로 컨텍스트 복원하면 됩니다.

---

## 8. 작업 끝나면 git push

매 day 작업 끝나고 push 하는 패턴:

```bash
cd /path/to/wavesplat
git status

# 다듬은 PPT 파일 (사용자가 PowerPoint 에서 직접 수정한 경우)
git add docs/labmeeting_2026_05_13.pptx

# Generator script 수정한 경우
git add scripts/generate_labmeeting_2026_05_13.py

# md 파일 update
git add docs/labmeeting_2026_05_13_slides.md

# Strategic doc update (§32 등)
git add docs/strategic_repositioning.md

# 핸드오프 노트 다음 날 update
git add docs/SERVER_HANDOFF_2026_05_09.md  # 또는 새 핸드오프 노트

git commit -m "docs(ppt): Day N — content refinement + figures + Wang 2025 review"
git push origin main
```

---

## 9. 5/13 발표 후

**Phase 1 작업 시작** (paper 작성):
1. `scripts/run_super_resolution_ablation.py` 작성 (4-way ablation)
2. 100 episode 에서 ablation 실험
3. Theoretical bound 1-page derivation
4. Paper 본문 재작성 — Dispersion-Coupled Spatio-Temporal Inversion framing

**7월 말 decision gate**: Super-res factor 3-7× 정량 증명 → JFM 도전 vs OE submit.

---

## 10. 핵심 파일 위치 한눈에

| 파일 | 위치 | 역할 |
|---|---|---|
| Master strategic doc | `wavesplat/docs/strategic_repositioning.md` | 31 sections, 모든 prior art 정독 + main framing |
| Lab meeting PPT (output) | `wavesplat/docs/labmeeting_2026_05_13.pptx` | 26-slide 발표 자료 |
| Lab meeting PPT generator | `wavesplat/scripts/generate_labmeeting_2026_05_13.py` | Python script (재생성 가능) |
| Lab meeting PPT source MD | `wavesplat/docs/labmeeting_2026_05_13_slides.md` | 19-slide outline + Day 3-5 plan |
| 핸드오프 노트 (이 파일) | `wavesplat/docs/SERVER_HANDOFF_2026_05_09.md` | 다른 서버 시작 안내 |
| Reference papers | `wavesplat/ref_paper/` (gitignored, local-only) | 12 PDF papers |
| Memory (각 서버 local) | `~/.claude/projects/.../memory/` | Claude Code memory, 다른 서버 동기화 안 됨 |

---

## 11. 마지막 reminder

- **Memory 가 없는 다른 서버**라는 사실을 잊지 마세요. 모든 컨텍스트는 git push 된 doc 으로 복원.
- **Wang 2025 받으면** 정독 우선 (30분-2시간), §32 추가, Slide 6 update.
- **Day 3-5 작업** 은 generator script 수정 + PowerPoint 직접 다듬기 hybrid 권장.
- **Paper figure** 는 5 detailed prior art slides 에만 추가하면 충분 (S5-S9).
- **PFT-PINN 위협** 인지 — 빨리 publish + 응용 영역 차별화로 대응.

좋은 발표 되시길!
