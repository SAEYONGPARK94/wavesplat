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
# ★[2026-09-02] **미정의 참조.인용을 차단으로 올린다.**
#   ⛔이 스크립트는 그것을 세어서 인쇄만 하고 rc=0 을 냈다 --- 그래서 오늘
#   내가 넣은 오타 라벨 둘이 조용히 통과했다(`sec:real`, `sec:regen`).
#   ⚠overfull 은 여전히 비차단이다(77건). 그러나 **미정의 참조는 인쇄면에 `??` 로
#   나온다** --- 조판 미세문제가 아니라 결함이다.
import sys as _s
print(m.group(0) if m else '⛔PDF 가 나오지 않았다 — /tmp/thesis_p3.log 를 읽어라')
print(f'Overfull {t.count("Overfull")} · Underfull {t.count("Underfull")} (조판 미세문제, 차단 아님)')
if err or uc or ur:
    print('⛔차단: 오류/미정의가 있다. 위 수를 0 으로 만들어라.')
    _s.exit(1)
PY
# ⛔[2026-09-02] **exit 코드를 셸로 전파한다.**
#   초판 수리는 python 안에서만 `exit(1)` 했고 셸은 그 뒤 명령의 rc 를 냈다
#   ⟹ 위조 시험에서 미정의 참조를 넣었는데 **rc=0 이 나왔다**.
#   ★차단은 "출력에 ⛔가 있다" 가 아니라 "종료 코드가 0 이 아니다" 여야 한다.
exit $?
