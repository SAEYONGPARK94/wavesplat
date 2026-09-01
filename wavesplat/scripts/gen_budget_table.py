#!/usr/bin/env python3
"""`paper/budget_table.tex` --- **프레임 예산 표**. ⛔런 0회: `T6` 동결 verdict 재인쇄다.

⛔왜(2026-09-02, 6차 감사 권고 8):

> *"Turn 3.5 into the table the batch already produced --- eight cells x {median
>  frame ms, IQR, init s, budget fraction} --- and either drop Table 4 or merge it.
>  The paper prints 5 of ~32 numbers as prose."*

★그 지적이 옳다: 배치는 8칸 x 4열을 이미 냈고 논문은 그중 다섯 개를 산문으로 적었다.
⚠**병기 의무**(동결 `disclosure` 에서 옮긴다, 축약하지 않는다):
  ⛔기준선 팔이 없다 --- 이 배치는 기준선을 돌리지 않았다. 비교 지연 주장 금지.
  ⛔철회된 프로세스간 수치(13.4 / 132.6 ms)와 **다른 양**이다. 비교 금지.
  ⚠라벨은 추적만 서술한다. 초기화는 어느 라벨에도 들어가지 않는다.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V = os.path.join(ROOT, 'tmp', 'T6_BUDGET_2026-09-02', 'verdict_t6.json')
OUT = os.path.join(ROOT, 'paper', 'budget_table.tex')

# 인쇄 순서 = 사람이 읽는 순서(장치 -> 초기화 경로 -> 층). ⛔알파벳순 아님.
ORDER = [('CPU', 'LEARN', 'crowded'), ('CPU', 'LEARN', 'sparse'),
         ('CPU', 'ANA', 'crowded'), ('CPU', 'ANA', 'sparse'),
         ('GPU', 'LEARN', 'crowded'), ('GPU', 'LEARN', 'sparse'),
         ('GPU', 'ANA', 'crowded'), ('GPU', 'ANA', 'sparse')]
NICE = {'LEARN': 'learned', 'ANA': 'analytic'}


def main():
    v = json.load(open(V))
    cells = v['cells']
    assert set(cells) == {'|'.join(k) for k in ORDER}, '⛔칸 집합이 바뀌었다'
    assert v['gates_pass'] == '10/10', f"⛔게이트가 통과 상태가 아니다: {v['gates_pass']}"

    L = ['% ⛔자동 생성 — 손으로 고치지 마라.',
         '%    생성: python3 scripts/gen_budget_table.py   (런 0회, T6 동결 verdict 재인쇄)',
         '']
    L += [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
          r'\caption{Measured cost of the deployed estimator, one process, one thread, '
          r'on an AMD Ryzen~7 7800X3D. Every cell is \(8\) episodes; the per-frame '
          r'column is the median over episodes of the per-episode median frame, with '
          r'the interquartile range, and the budget column is that median against a '
          r'\(10\)~Hz frame interval. Initialisation is a one-off cost and is not part '
          r'of the per-frame figure. \emph{No baseline arm was run in this batch}, so '
          r'no comparative timing claim follows from it, and these numbers are not '
          r'commensurable with the cross-process figures withdrawn in \ref{app:changes}, '
          r'R3. Pre-registered as T-6; all ten gates pass, including a per-run '
          r'read-back that the thread count was one and that the shipping constants '
          r'were in force.}',
          r'\label{tab:budget}',
          r'\setlength{\tabcolsep}{5pt}',
          r'\begin{tabular}{lllrrr}', r'\toprule',
          r'Device & Init.\ path & Stratum & Frame [ms] & \(10\)~Hz budget & Init.\ [s] \\',
          r'\midrule']
    prev = None
    for dev, path, stratum in ORDER:
        c = cells[f'{dev}|{path}|{stratum}']
        lo, hi = c['frame_ms_iqr']
        if prev and prev != (dev, path):
            L.append(r'\addlinespace[2pt]')
        prev = (dev, path)
        L.append(f"{dev} & {NICE[path]} & {stratum} & "
                 f"{c['frame_ms_median']:.2f} {{\\scriptsize[{lo:.2f},{hi:.2f}]}} & "
                 f"{100 * c['budget_frac_at_10hz']:.1f}\\% & "
                 f"{c['init_s_median']:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(OUT, 'w').write('\n'.join(L) + '\n')

    # ★사이드카 — 인용금지 래칫의 값 단위 면제(선례 = gen_abs_table.py).
    import re as _re
    vals = sorted({m.group(0) for ln in L for m in _re.finditer(r'\d+\.\d+', ln)})
    json.dump({'source': 'scripts/gen_budget_table.py',
               'note': 'T6 동결 verdict 재인쇄. 인용금지 래칫이 값 단위 면제에 쓴다.',
               'values': vals},
              open(os.path.join(ROOT, 'paper', 'budget_table.values.json'), 'w'),
              indent=1, ensure_ascii=False)
    ms = [c['frame_ms_median'] for c in cells.values()]
    print(f"✅8 칸 → {OUT}\n   프레임 중앙값 범위 {min(ms):.2f}–{max(ms):.2f} ms "
          f"(보고 칸 CPU|LEARN|crowded = {cells['CPU|LEARN|crowded']['frame_ms_median']:.2f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
