# Paper Review 작업 시작 가이드

**이 문서의 목적**: 다른 서버에서 이 작업을 처음 본 사람이 바로 작업을 시작할 수 있도록 한다.

**선행 문서**: [WaveSplat_paper_review_2026-04-22.md](WaveSplat_paper_review_2026-04-22.md) — 상세 내용은 여기 참조

---

## Step 0 — 환경 확인 (5분)

```bash
cd /path/to/wavesplat
git status                     # clean인지 확인
git log --oneline -5           # 최근 커밋: 4ffe75d CHANGELOG v0.6.0 ...

# Python 환경
python3 -c "import torch, numpy, h5py, yaml; print('OK')"

# LaTeX 환경
which latexmk && which pdflatex
```

---

## Step 1 — 이전 작업 맥락 파악 (15분)

### 필수 읽기 (순서대로)

1. **상세 검토 문서**
   ```bash
   less docs/WaveSplat_paper_review_2026-04-22.md
   ```
   - §0 (작업 상태), §7 (체크리스트) 먼저
   - §2 (4개 문제), §5 (수정 스니펫) 자세히

2. **업데이트된 HTML 가이드** (브라우저에서 열기)
   ```
   docs/WaveSplat_Algorithm_Guide.html
   ```
   - Part 6: Fine Refinement (2-stage beamforming)
   - Part 7: Gradient chain rule, Adam element-wise, Normal 가정 warning
   - Part 8: τ_R 유도, Gerstner vs Linear
   - Part 9: Phase별 Adam 구성

3. **검토 대상 논문** (위치만 확인)
   ```
   paper/wavesplat.tex        # 메인 논문
   thesis/thesis_main.tex     # 박사 학위논문
   hydrofoil/paper/hydrofoil_feedforward.tex  # 문제 없음, 수정 제외
   ```

---

## Step 2 — 작업 분기 선택

네 가지 트랙 중 **하나**를 선택하여 진행:

### Track A — Paper 수정만 (빠름, 1-2일)
Priority 1만 수행. 추가 실험 없이 문서 정합성 확보.

→ **Step 3-A**로

### Track B — Paper 수정 + τ_R Sensitivity (3-4일)
Priority 1 + §6.1 sensitivity 실험. Paper에 appendix 추가.

→ **Step 3-A → 3-B**로

### Track C — Paper 수정 + 모든 추가 실험 (1주)
Priority 1 + §6.1 + §6.2. 가장 강한 rebuttal 자료.

→ **Step 3-A → 3-B → 3-C**로

### Track D — 추가 실험만 (paper 수정 미룸)
실험부터 돌려놓고, 결과 보고 paper 수정 범위 결정.

→ **Step 3-B → 3-C**로

---

## Step 3-A — Paper/Thesis 수정 (Priority 1)

### 작업 순서

```bash
# 브랜치 생성 (선택)
git checkout -b paper/v0.7-review-fixes

# 1. Paper 수정 (상세 문서 §5.1~5.5 스니펫 사용)
vim paper/wavesplat.tex
#   §3: Fine refinement 추가 (§5.1 스니펫)
#   §3.5: Confidence normal 가정 (§5.2 스니펫)
#   §5: A1 ablation 재해석 (§5.3 스니펫)
#   Limitations: Doppler + normal + 명칭 (§5.4 스니펫)
#   τ_R 유도 (§5.5 스니펫)

# 2. References 확인
grep -n "stokes1847\|holthuijsen2007" paper/references.bib
# 없으면 추가 (BibTeX 아래 참고)

# 3. 컴파일 테스트
cd paper
latexmk -pdf wavesplat.tex
# → wavesplat.pdf 생성 확인, warning/error 없는지 체크
cd ..

# 4. Thesis 동기화 (상세 문서 §5.7)
vim thesis/thesis_main.tex
#   Algorithm 1 Step 1.5 추가
#   Confidence normal 가정 (L1098 근처)
#   A1 해석 보강
#   Doppler limitations

cd thesis
latexmk -pdf thesis_main.tex
cd ..

# 5. Changelog 업데이트
vim docs/CHANGELOG.md
# v0.7.0 entry 추가 (아래 템플릿 참고)

# 6. 커밋
git add paper/wavesplat.tex thesis/thesis_main.tex paper/references.bib docs/CHANGELOG.md
git commit -m "Paper/thesis 수정: fine refinement 추가, normal 가정 명시, A1 재해석"
```

### 참고 — BibTeX 추가 (references.bib에 없을 경우)

```bibtex
@article{stokes1847,
  author = {Stokes, G. G.},
  title = {On the theory of oscillatory waves},
  journal = {Transactions of the Cambridge Philosophical Society},
  volume = {8},
  year = {1847},
  pages = {441--455}
}

@book{holthuijsen2007,
  author = {Holthuijsen, Leo H.},
  title = {Waves in Oceanic and Coastal Waters},
  publisher = {Cambridge University Press},
  year = {2007},
  isbn = {9780521860284}
}
```

### 참고 — CHANGELOG v0.7.0 템플릿

```markdown
## [0.7.0] — YYYY-MM-DD

### Fixed — 논문·학위논문 정합성 개선

#### Paper 재현성
- Fine refinement stage를 §3 Algorithm에 명시 (code-paper 불일치 해결)
  - 각 피크마다 41×41 fine grid (±0.2 rad/m, Δk_fine = 0.01)로 재탐색
  - 이전: coarse 81×81만 기술 → 재현자가 17% k 오차 겪음
- Paper L493-500 뒤에 fine refinement 단락 추가
- Thesis Algorithm 1에 Step 1.5 추가

#### Real-data 적용 한계 명시
- Confidence §3.5에 normal 가정 sim vs real 구분 추가
  - Sim: Gerstner true normal 사용
  - Real: flat-surface 가정 (n ≈ ẑ), grazing에서 cos θ 편향 ~9%
- Limitations에 Doppler 효과 추가 (0.5 m/s current → BF4에서 ~5% ω shift)
- "Fresnel-aware" 명칭 간소화 근거 Limitations에 명시

#### A1 Ablation 재해석 (§5)
- 9.4% nRMSE 개선을 "physical correctness trade-off"가 아니라
  "over-fitting artifact"로 재프레이밍
- R̄ = 0.51은 bound harmonic 기대값 <1%를 크게 상회 → 물리 signal 아님
- Hard constraint를 physics-based regularizer로 위치 재확립

#### τ_R 유도 보강
- Paper L544에 노이즈 + 모델 오차 하한 유도 추가
- Sensitivity analysis (Appendix) → [선택적, Track B/C]

### Commits
- [...] (작업 후 채움)
```

---

## Step 3-B — τ_R Sensitivity 실험 (Track B/C 진행 시)

### 스크립트 작성

```bash
# scripts/run_tau_sensitivity.py 작성
# 기존 run_p1_comparison.py 구조 재사용
cp scripts/run_p1_comparison.py scripts/run_tau_sensitivity.py
vim scripts/run_tau_sensitivity.py
# tau_R ∈ {0.02, 0.05, 0.08, 0.10, 0.15} sweep
# 결과를 results/tau_sensitivity.json에 저장
```

### 실행

```bash
# CPU 병렬 (서버 4 workers 가정)
python3 scripts/run_tau_sensitivity.py --config configs/server.yaml

# 예상: 100 eps × 5 tau × 2 raydrop = 1000 runs
# 각 ~30초, 4 workers → 100×5×2 × 30/4 = 7500s ≈ 2시간
```

### 결과 반영

```bash
# Figure 생성
python3 scripts/plot_tau_sensitivity.py  # 작성 필요

# Paper Appendix에 단락 + figure 추가
vim paper/wavesplat.tex
# Appendix 섹션에 τ_R sensitivity subsection
```

---

## Step 3-C — GPR + Confidence Ablation (Track C 진행 시)

```bash
# scripts/run_gpr_confidence_ablation.py 작성
cp scripts/run_gpr_comparison.py scripts/run_gpr_confidence_ablation.py
vim scripts/run_gpr_confidence_ablation.py
# 기존 GPR에 c_j weighted MSE 추가 버전 구현

python3 scripts/run_gpr_confidence_ablation.py --config configs/server.yaml
# 100 eps × 2 raydrop × {with_conf, without_conf} = 400 runs, ~반나절

# 결과에 따라 분기
# - GPR도 개선 → C3를 "standard weighting"으로 재프레이밍
# - WS에만 도움 → spectral representation 상호작용으로 설명
```

---

## Step 4 — 최종 검증 (모든 Track 공통)

```bash
# 1. 기존 테스트 통과 확인
python3 -m pytest wavesim/tests/ wavesplat/tests/ -v
# 기대: 57 passed

# 2. 기존 실험 재현성 확인 (random seed 고정)
python3 scripts/run_p1_comparison.py --seeds 0 --episodes 3
# 기존 results/p1_comparison.json과 수치 일치 확인

# 3. PDF 최종 컴파일
cd paper && latexmk -pdf wavesplat.tex && cd ..
cd thesis && latexmk -pdf thesis_main.tex && cd ..

# 4. 커밋 + Push
git add -A
git commit -m "Paper review fixes (v0.7.0): 재현성/정합성/한계 명시"
git push origin paper/v0.7-review-fixes  # PR 생성 등
```

---

## 문제 발생 시

### LaTeX 컴파일 실패
```bash
# 에러 로그 확인
cd paper
cat wavesplat.log | grep -E "Error|Warning"
# 주로 bibtex 미참조, 수식 닫기 누락 등
```

### 기존 실험 재현 실패
- `configs/` 파일 변경 확인
- `wavesplat/optimizer/` 커밋 이력 확인
- `git diff HEAD~10 wavesplat/optimizer/sequential_update.py` 등

### 스니펫 적용 위치 불명확
- 상세 문서 §5 (LaTeX 스니펫) 재확인
- Paper 라인 번호는 현재 기준: L419, L544, L1111 등
- `grep -n "pattern" paper/wavesplat.tex`로 실제 위치 확인

---

## 예상 소요 시간

| Track | 작업 | 시간 |
|-------|------|------|
| A | Paper + Thesis 수정 | 1~2일 |
| B | A + τ_R sensitivity | 3~4일 |
| C | B + GPR ablation | 1주 |
| D | 실험만 | 2~3일 |

---

## 세션 종료 시 상태 업데이트

작업 종료 후 상세 문서 §0과 §7의 체크리스트를 업데이트하고 커밋.

```bash
vim docs/WaveSplat_paper_review_2026-04-22.md
# §0: 진행 완료 추가
# §7: 체크리스트 [x] 표시

git add docs/
git commit -m "Paper review 진행 상태 업데이트"
```

---

**작업 시작 준비 완료. Step 1부터 진행.**
