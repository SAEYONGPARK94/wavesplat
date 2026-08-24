#!/usr/bin/env bash
# 학위논문 조판 — 2026-08-24 검증된 절차 (pdflatex, xelatex 아님).
# 쓰기: cd wavesplat/thesis && bash build.sh
set -u
export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"
command -v pdflatex >/dev/null || { echo "pdflatex 없음 — README.md 의 'TeX 환경' 절 참조"; exit 1; }
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -file-line-error thesis_main.tex > /tmp/thesis_p1.log 2>&1
bibtex thesis_main > /tmp/thesis_bib.log 2>&1
pdflatex -interaction=nonstopmode -file-line-error thesis_main.tex > /tmp/thesis_p2.log 2>&1
pdflatex -interaction=nonstopmode -file-line-error thesis_main.tex > /tmp/thesis_p3.log 2>&1
python3 - <<'PY'
import re
t = open('/tmp/thesis_p3.log', errors='replace').read()
err = len(re.findall(r'^! ', t, re.M))
uc  = len(re.findall(r'Citation .* undefined', t))
ur  = len(re.findall(r'Reference .* undefined', t))
m   = re.search(r'Output written on (\S+) \((\d+) pages, (\d+) bytes\)', t)
print(f'오류 {err} · 미정의 인용 {uc} · 미정의 참조 {ur} · '
      f'재실행 필요 {"예" if "Rerun to get" in t else "아니오"}')
print(m.group(0) if m else '⛔PDF 가 나오지 않았다 — /tmp/thesis_p3.log 를 읽어라')
print(f'Overfull {t.count("Overfull")} · Underfull {t.count("Underfull")} (조판 미세문제, 차단 아님)')
PY
