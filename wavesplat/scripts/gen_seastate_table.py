#!/usr/bin/env python3
"""`thesis/seastate_tables.tex` --- 해상상태 표 두 개. ⛔**코드에서 읽는다.**

⛔왜(2026-09-02, 학위논문 감사 B-9):
학위논문의 표 두 개가 **시뮬레이터에 없는 해상상태**를 인쇄하고 있었다:

| | 인쇄된 것 | `wavesim/pipeline.py:35-40` | 실제 에피소드 |
|---|---|---|---|
| BF3 | Hs 0.88 · Tp 5.0 | **0.6 · 5.5** | 0.6 · 5.5 |
| BF4 | Hs 1.88 · Tp 7.5 | **1.2 · 7.0** | 1.2 · 7.0 |

★그리고 **같은 장의 50쪽 앞**(§3.3.3)이 네 앵커를 **옳게** 적고 있었다.
파생 열(λ_p·c_p·c_g·ε)이 전부 오차를 상속했다 --- BF4 λ_p 를 **87.7 m** 로 인쇄했는데
저장소의 다른 모든 곳(F2 공시 포함)은 **76.5 m** 를 쓴다.

⟹ 이 스크립트가 `SEA_STATE` 를 **임포트**해서 두 표를 만들고, 에피소드 파일과 대조한다.
⛔손으로 적을 자리가 없다.

규약: 심해 근사(λ = gT²/2π, c_p = gT/2π, c_g = c_p/2)를 쓴다 --- 저장소의 다른 인용
(`λ_p 76.5 / 126.5 m`)과 같은 규약이어야 새 불일치를 만들지 않는다.
⚠d=50 m 유한수심 값은 BF5 에서 124.8 m 로 1.3% 낮다. 표 각주에 적는다.
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from wavesim.pipeline import SEA_STATE                              # noqa: E402

G = 9.81
NCOMP = {'BF2': '1, 2', 'BF3': '1, 2, 5', 'BF4': '5, 15, 30', 'BF5': '15, 30'}


def verify_against_corpus():
    """★코드 상수가 **실제 에피소드**와 맞는가. 어긋나면 죽는다."""
    bad = []
    for bf, p in SEA_STATE.items():
        hits = sorted(glob.glob(os.path.join(ROOT, 'data', f'episode_{bf}_*.h5')))
        if not hits:
            continue
        import h5py
        with h5py.File(hits[0], 'r') as f:
            a = f['config'].attrs
            if abs(float(a['Hs']) - p['Hs']) > 1e-9 or abs(float(a['Tp']) - p['Tp']) > 1e-9:
                bad.append(f"{bf}: 코드 ({p['Hs']}, {p['Tp']}) vs 파일 "
                           f"({float(a['Hs'])}, {float(a['Tp'])}) [{os.path.basename(hits[0])}]")
    if bad:
        raise SystemExit('⛔코드 상수가 코퍼스와 다르다:\n  ' + '\n  '.join(bad))
    return len(SEA_STATE)


def derive(Tp, Hs):
    lam = G * Tp ** 2 / (2 * np.pi)
    cp = G * Tp / (2 * np.pi)
    k = 2 * np.pi / lam
    eps = k * Hs / 2
    return lam, cp, cp / 2, eps


HDR = ['% ⛔자동 생성 — 손으로 고치지 마라.',
       '%    생성: python3 scripts/gen_seastate_table.py',
       '%    출처: wavesim/pipeline.py SEA_STATE (+ data/ 에피소드와 대조)',
       '%    ⚠심해 근사(lam = gT^2/2pi). d=50 m 유한수심은 BF5 에서 1.3% 낮다(124.8 m).',
       '%       저장소의 다른 인용(lam_p 76.5/126.5)과 **같은 규약**이어야 새 불일치가 없다.', '']


def main():
    n = verify_against_corpus()
    L = list(HDR)
    # ── 표 1: 급경사도
    L += [r'\begin{tabular}{lcccc}', r'\toprule',
          r'해상 상태 & $H_s$ [m] & $T_p$ [s] & $\epsilon$ & $\epsilon^2$ [\%] \\',
          r'\midrule']
    for bf in sorted(SEA_STATE):
        Hs, Tp = SEA_STATE[bf]['Hs'], SEA_STATE[bf]['Tp']
        _, _, _, eps = derive(Tp, Hs)
        L.append(f'{bf} & {Hs:.2f} & {Tp:.1f} & {eps:.3f} & {100 * eps ** 2:.2f} \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(ROOT, 'thesis', 'seastate_tables_1.tex'),
         'w').write('\n'.join(L) + '\n')
    # ── 표 2: 파장.속도
    L = list(HDR)
    L += [r'\begin{tabular}{lcccccc}', r'\toprule',
          r'BF & $H_s$ [m] & $T_p$ [s] & $\lambda_p$ [m] & $c_p$ [m/s] & $c_g$ [m/s] & 성분 수 \\',
          r'\midrule']
    for bf in sorted(SEA_STATE):
        Hs, Tp = SEA_STATE[bf]['Hs'], SEA_STATE[bf]['Tp']
        lam, cp, cg, _ = derive(Tp, Hs)
        L.append(f'{bf[-1]} & {Hs:.2f} & {Tp:.1f} & {lam:.1f} & {cp:.1f} & {cg:.1f} & '
                 f'{NCOMP.get(bf, "---")} \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(ROOT, 'thesis', 'seastate_tables_2.tex'),
         'w').write('\n'.join(L) + '\n')
    print(f'✅{n} 해상상태 · 코드-코퍼스 대조 통과 → thesis/seastate_tables_{{1,2}}.tex')
    for bf in sorted(SEA_STATE):
        Hs, Tp = SEA_STATE[bf]['Hs'], SEA_STATE[bf]['Tp']
        lam, cp, cg, eps = derive(Tp, Hs)
        print(f'  {bf}  Hs {Hs:.2f}  Tp {Tp:.1f}  lam {lam:6.1f}  '
              f'cp {cp:5.1f}  cg {cg:5.1f}  eps {eps:.3f}')


if __name__ == '__main__':
    sys.exit(main())
