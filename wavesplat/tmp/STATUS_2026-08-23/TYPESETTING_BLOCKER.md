# ⛔조판 불가 — **사용자 조치 필요** (2026-08-23)

## 상태
`wavesplat/thesis/thesis_main.tex` 는 오늘 **광범위하게 수정**됐고(초록·1·2·3·5·6장),
LaTeX 환경 균형은 스크립트로 **전건 확인**했다:
`itemize 19/19` · `enumerate 9/9` · `equation 33/33` · `table 35/35` · `tabular 36/36` ·
`quote 2/2` · **중괄호 수지 0**.

⛔**그러나 실제 조판은 하지 못했다** — 이 기계에 **LaTeX 툴체인이 없다**:
```
$ which pdflatex   →  (없음)
```
⟹ ★**`.tex` 가 컴파일되는지, 새로 추가한 참조(`\ref{tab:b1_crowded}` · `\ref{tab:b11_hos}` ·
`\ref{sec:results_b11}` · `\ref{sec:init-bank}` · `\ref{sec:contributions}`)가 해석되는지
확인되지 않았다.**

⚠**R3 심사 위원장이 이것을 "이것 하나" 로 지목했다** — *"나머지 장을 맞추고 **한 번 조판하라**"*.
그리고 **PDF 가 2026-06-24 이후 없다**(`thesis/` 에 갱신된 `.pdf` 부재).

## 사용자 조치 (한 줄)
```bash
cd /home/vexit/wavesplat/wavesplat/thesis && pdflatex thesis_main.tex && pdflatex thesis_main.tex
```
(두 번 돌려야 상호참조가 해결된다. `bibtex`/`biber` 가 필요하면 그 사이에 한 번 더.)

## 오늘 새로 추가된 라벨·참조 (조판 시 확인 대상)
| 종류 | 이름 | 위치 |
|---|---|---|
| `\label` | `tab:b1_crowded` | 3장 대체 결과표 |
| `\label` | `tab:b11_hos` | 3장 비선형·신선 표 |
| `\label` | `sec:results_b1` · `sec:results_b11` | 3장 신설 절 |
| `\ref` | 위 넷 + `sec:contrib-limits` · `sec:init-bank` · `sec:contributions` | 1·3·6장 |

⛔**조판 전까지 "논문이 준비됐다" 고 말하지 마라** — 균형 검사는 **컴파일이 아니다**.
