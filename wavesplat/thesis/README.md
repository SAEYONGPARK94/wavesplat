# 박사학위논문 작업 디렉토리

## 논문 정보
- **제목**: 선박 탑재 LiDAR 기반 실시간 파랑장 추정과 그 제어 응용
- **저자**: 박세용
- **소속**: 서울대학교 대학원 조선해양공학과
- **지도교수**: (확인 필요)

## 파일 구조
```
thesis/
├── thesis_main.tex     # 본문 (메인 파일)
├── thesis_refs.bib     # 참고문헌 데이터베이스
├── figures/            # 논문 그림 (paper/figures/ 심볼릭 링크 또는 복사)
└── README.md
```

## 컴파일 방법 ★2026-08-24 실제로 검증됨

⛔**종전 문안(`xelatex` 권장)은 추측이었다.** 실측: **`pdflatex` 로 조판된다**
(kotex 이 `nanumtype1` Type1 폰트를 쓴다). 결과 **159쪽 · 오류 0 · 미정의 참조 0 · 미정의 인용 0**.

```bash
export PATH=$HOME/texlive/2026/bin/x86_64-linux:$PATH   # 이 기계의 TeX Live 위치
cd wavesplat/thesis
pdflatex -interaction=nonstopmode thesis_main.tex
bibtex   thesis_main
pdflatex -interaction=nonstopmode thesis_main.tex
pdflatex -interaction=nonstopmode thesis_main.tex       # 상호참조 수렴까지
```
또는 `bash build.sh` (같은 절차 + 오류/미정의 참조 요약을 찍는다).

### 이 기계의 TeX 환경 (2026-08-24 설치)
- `sudo` 없이 **사용자 영역**에 TeX Live 2026 설치: `~/texlive/2026` (`scheme-medium`).
- 추가 설치: `collection-langkorean`(kotex) · `newunicodechar` · `elsarticle`(= `elsarticle-harv.bst`).
- 재설치가 필요하면: `install-tl -profile <profile>` 후 위 세 개를 `tlmgr install`.

### 2026-08-24 에 고친 조판 차단 3건 (그 전까지 **한 번도 조판된 적이 없었다**)
1. `algorithmic` 의 `\FOR{...}` 뒤에 곧바로 `\COMMENT` → *"missing \item"* 으로 문서가 죽었다.
   ⟹ 주석을 루프 인자 안으로 옮겼다(뜻 보존).
2. 부록 C 가 참조하는 `figures_episodes/episode_*_seed0_uniform.png` **10개가 존재하지 않는다**
   (생성된 적 없음 — git 에도 90개뿐). 파일 부재는 `pdflatex` 을 **치명적으로** 죽인다.
   ⟹ `\epincl` 이 `\IfFileExists` 로 감싸 **보이는 자리표시자**를 인쇄한다.
   ⛔자리표시자는 그림을 만들어 주지 않는다 — 필요하면
   `python3 ../scripts/generate_all_episode_figures.py` 를 돌려라(⚠아래 범위 경고).
3. 자리표시자가 파일명의 `_` 를 수학모드로 읽어 죽었다 ⟹ `\detokenize`.

⚠**부록 C 의 범위**: 현재 있는 90개 그림은 **2026년 4월** 산출물이라
**2026-08-22 배포 변경 이전 구성**이다 ⟹ 지금 배포되는 추정기의 출력이 아니다.
전부 다시 만들려면 100 에피소드 × (WaveSplat·ETKF·GPR) 재실행이 필요하다.

## 챕터별 작업 현황

| 챕터 | 내용 | 상태 | 비고 |
|------|------|------|------|
| 1장 서론 | 연구 배경, 기여 | 초안 작성 | 내용 보강 필요 |
| 2장 관련 연구 | 문헌 조사 | 스켈레톤 | Paper 1 §II 기반으로 확장 필요 |
| 3장 WaveSplat | 방법론 + 결과 | 초안 작성 | Paper 1 기반, 세부 수치 확인 필요 |
| 4장 Hydrofoil | Feedforward 제어 | 스켈레톤 | Paper 2 기반으로 작성 필요 |
| 5장 수조 실험 | 실험 검증 | TODO | 수조 실험 후 작성 |
| 6장 결론 | 결론 + 향후 연구 | 초안 작성 | 보강 필요 |

## 작업 우선순위
1. **2장 관련 연구**: Paper 1의 §II를 박사 논문 수준으로 확장
2. **4장 Hydrofoil**: Paper 2 내용을 챕터 형식으로 편입
3. **1장 서론**: 두 논문을 잇는 narrative 강화
4. **그림 연결**: `ln -s ../paper/figures figures` 로 심볼릭 링크

## SNU 서식 확인 필요 사항
- [ ] 공식 여백 설정 (서울대 대학원 학위논문 작성 지침 확인)
- [ ] 한글 폰트 지정 (바탕체 또는 신명조)
- [ ] 표지/인준지 정확한 양식
- [ ] 참고문헌 스타일 (지도교수 지정 스타일)
- [ ] 학과명 정확한 표기
- [ ] 제출 시 PDF/HWP 여부

## 관련 파일
- Paper 1: `../paper/wavesplat.tex` (Ocean Engineering 투고 원고)
- Paper 2: Hydrofoil 논문 초안 (위치 확인 필요)
- 결과 데이터: `../results/`
- 그림: `../paper/figures/`
