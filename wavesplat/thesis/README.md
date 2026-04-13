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

## 컴파일 방법
```bash
# kotex (한글) 지원 위해 xelatex 또는 lualatex 사용
xelatex thesis_main
bibtex thesis_main
xelatex thesis_main
xelatex thesis_main

# 또는 latexmk
latexmk -xelatex thesis_main
```

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
