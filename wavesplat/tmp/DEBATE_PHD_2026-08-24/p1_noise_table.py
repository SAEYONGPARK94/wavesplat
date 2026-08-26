#!/usr/bin/env python3
"""P-1 — 실측 12 레코드의 레코드별 noise/sigma 표를 재집계하고 LaTeX 표를 낸다. (런 0회)

## 왜 이 스크립트가 저장소에 있어야 하는가
외부 심사 소견서 `B2_external_ocean.md` §2.2 가 레코드별 SNR 표를 제시하고 처방 P-1 이
그 표를 **논문에 넣으라**고 요구한다. 2026-08-24 에 저장소는 *"인용 가능한 최종형 수치를 낸
스크립트가 저장소에 없다"* 는 결함을 한 번 잡았다(`tmp/B3_REFIT_2026-08-23/final6.py` 가
그 처방이었다). 논문에 실릴 표는 대화형 재집계가 아니라 **재현 가능한 스크립트**가 내야 한다.

## 무엇을 어디서 가져오는가 (손복사 상수 0개)
- 층 정의(대형/소형)  : `gates_b2.LARGE_BOX` / `SMALL_BOX`  ← 단일출처, 결과 전 선언
- 잡음 분자 0.02 m    : `scene_station()` 의 **함수 기본값을 introspection** 으로 읽는다.
                        `run_b2.py:27` 이 `scene_station(record, win=G.WIN)` 로 부르므로
                        (noise 미지정) 그 기본값이 실제로 배치에 먹은 값이다.
- Hs, sigma           : `tmp/B2_REAL_2026-08-23/results.jsonl` 의 행 그대로.
                        `scene_station.py:154` 가 `sg = nanstd(Zmem)` 를 기록하고
                        `:100` 이 `Hs = 4.0 * nanstd(Zmem)` 를 파생시킨다
                        ⟹ ★**Hs 는 sigma 의 파생량이고 독립 관측이 아니다.**

## 자기검사 (게이트)
G-1 레코드 12개인가
G-2 sigma 가 팔(SHIP/PROP)에 무관한가  — 장면 속성이므로 같아야 한다.
    (`gates_b2.py:29` 의 `'SHIP': {}` 빈 dict 팔 결함이 이 표에 닿지 않음을 보인다)
G-3 Hs == 4*sigma 인가 — 위 파생 관계를 자료에서 확인한다.
G-4 심사위원이 §2.2 에 적은 범위/배수와 일치하는가 (불일치는 실패가 아니라 **보고**한다).

실행:  python3 tmp/DEBATE_PHD_2026-08-24/p1_noise_table.py
"""
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))   # /home/vexit/wavesplat/wavesplat
B2 = os.path.join(REPO, 'tmp', 'B2_REAL_2026-08-23')
sys.path.insert(0, B2)

import gates_b2 as G                    # noqa: E402  단일출처 층 정의
from scene_station import scene_station  # noqa: E402  잡음 기본값의 출처

RESULTS = os.path.join(B2, 'results.jsonl')
OUT_TEX = os.path.join(HERE, 'P1_noise_ratio_table.tex')

# ── 심사위원이 §2.2 에 적은 주장 (검사 대상이지 측정값이 아니다) ────────────────
# 출처: tmp/DEBATE_PHD_2026-08-24/B2_external_ocean.md:118 (제목행) 및 :439 (처방 P-1)
EXAMINER_LO, EXAMINER_HI, EXAMINER_FOLD = 0.01, 0.22, 22


def load():
    """results.jsonl → {record: {cell: (Hs, sigma)}}"""
    by = {}
    with open(RESULTS) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by.setdefault(r['record'], {})[r['cell']] = (r['Hs'], r['sigma'])
    return by


def main():
    noise = inspect.signature(scene_station).parameters['noise'].default
    by = load()
    gates = {}

    # G-1 ────────────────────────────────────────────────────────────────
    gates['G-1 records==12'] = (len(by) == 12, f'{len(by)} records')

    # G-2 sigma 가 팔에 무관한가 ─────────────────────────────────────────
    mism = [k for k, v in by.items() if len(set(v.values())) != 1]
    gates['G-2 sigma arm-independent'] = (not mism, f'mismatched={mism or "none"}')

    # 층 배정 (단일출처) ─────────────────────────────────────────────────
    rows = []
    for rec, cells in by.items():
        Hs, sg = next(iter(cells.values()))
        if rec in G.LARGE_BOX:
            layer = '대형'
        elif rec in G.SMALL_BOX:
            layer = '소형'
        else:
            raise SystemExit(f'층 미배정 레코드: {rec}')
        rows.append(dict(record=rec, site=rec[:2], Hs=Hs, sigma=sg,
                         ratio=noise / sg, layer=layer))
    rows.sort(key=lambda r: r['ratio'])

    # G-3 Hs == 4*sigma ─────────────────────────────────────────────────
    worst = max(abs(r['Hs'] - 4.0 * r['sigma']) for r in rows)
    gates['G-3 Hs==4*sigma'] = (worst < 1e-9, f'max|Hs-4s|={worst:.3e}')

    lo, hi = rows[0]['ratio'], rows[-1]['ratio']
    fold_raw = hi / lo
    fold_rounded = round(hi, 2) / round(lo, 2)

    # G-4 심사위원 주장과 대조 ──────────────────────────────────────────
    range_ok = (round(lo, 2) == EXAMINER_LO) and (round(hi, 2) == EXAMINER_HI)
    fold_ok = abs(fold_raw - EXAMINER_FOLD) < 0.5
    gates['G-4a examiner range 0.01-0.22'] = (
        range_ok, f'rounded=[{round(lo, 2)}, {round(hi, 2)}]')
    gates['G-4b examiner fold 22x'] = (
        fold_ok, f'raw={fold_raw:.2f}x  rounded-arithmetic={fold_rounded:.2f}x')

    # ── 콘솔 ────────────────────────────────────────────────────────────
    print(f'noise numerator (scene_station default) = {noise} m')
    print(f'{"record":8s}{"site":6s}{"Hs[m]":>8s}{"sigma[m]":>10s}'
          f'{"noise/sigma":>13s}  layer')
    for r in rows:
        print(f'{r["record"]:8s}{r["site"]:6s}{r["Hs"]:8.2f}{r["sigma"]:10.3f}'
              f'{r["ratio"]:13.3f}  {r["layer"]}')
    print(f'\nrange {lo:.4f} – {hi:.4f}   fold(raw) = {fold_raw:.2f}x')
    print(f'2dp 표시 기준 range {round(lo, 2)} – {round(hi, 2)}'
          f'   fold = {fold_rounded:.2f}x')

    print('\n--- gates ---')
    for k, (ok, msg) in gates.items():
        print(f'{"PASS" if ok else "FAIL"}  {k:32s} {msg}')

    if not fold_ok:
        print(f'\n⚠ 불일치: 심사위원의 "{EXAMINER_FOLD}배" 는 **2dp 로 반올림한 뒤** 나눈 값이다'
              f'({round(hi, 2)}/{round(lo, 2)} = {fold_rounded:.0f}).'
              f' 반올림 전 실제 배수는 {fold_raw:.2f}배다.')
        print('  (sigma·Hs·noise/sigma 세 배수는 모두 같다 — noise 가 상수이고 Hs=4σ 이므로.)')

    write_tex(rows, noise, lo, hi, fold_raw)
    print(f'\nwrote {OUT_TEX}')


def write_tex(rows, noise, lo, hi, fold_raw):
    """booktabs LaTeX 표. ★모든 숫자는 위에서 계산된 값을 포맷한 것이다.

    ⚠유니코드 주의: 논문 preamble 의 `newunicodechar` 사상표에 그리스 문자 시그마가 **없다**
    (`grep -n 'newunicodechar{σ}' thesis/thesis_main.tex` ⟹ 0건).
    ⟹ 본문 시그마는 반드시 수학모드 `$\\sigma$` 로 쓴다. 날문자를 넣으면 pdflatex 이 죽는다.
    """
    body = '\n'.join(
        f'{r["record"]} & {r["site"]} & {r["Hs"]:.2f} & {r["sigma"]:.3f} & '
        f'{r["ratio"]:.3f} & {r["layer"]} \\\\'
        for r in rows)

    tex = rf"""% ============================================================================
% P-1 (외부 심사 처방) — 실측 12 레코드의 레코드별 관측 SNR
% ★자동 생성: tmp/DEBATE_PHD_2026-08-24/p1_noise_table.py
%   자료   : tmp/B2_REAL_2026-08-23/results.jsonl (24행, SHIP/PROP 팔에서 sigma 동일)
%   손복사 상수 0개 — 층은 gates_b2.py, 잡음 분자는 scene_station() 기본값에서 온다.
% ⛔이 파일을 손으로 고치지 마라. 스크립트를 고치고 다시 돌려라.
% ============================================================================
\begin{{table}}[htbp]
\centering
\small
\caption{{실측 12 레코드의 표면 통계와 \textbf{{가정된 관측 SNR}}.
$\sigma$ 는 기억창 수면 표준편차이고 유의파고는 $H_s = 4\sigma$ 로 파생된 값이므로
두 열은 독립 관측이 아니다. 분자 {noise}\,m 는 \textbf{{우리가 합성으로 더한
등방 가우스 잡음}}이며 실제 센서 반환 잡음이 아니다.
비 $\sigma_\text{{noise}}/\sigma$ 는 레코드에 따라
{lo:.3f} 에서 {hi:.3f} 까지 \textbf{{{fold_raw:.1f}배}} 갈린다.
층(대형/소형)은 결과 이전에 박스 크기로 선언된 기하 층이다.
$\Rightarrow$ ★\textbf{{선언된 기하 층이 관측 SNR 과 완전히 교락돼 있고,
소형 층 7개는 모두 같은 사이트(BS)다}} --- 어느 쪽이 원인인지 이 표본은 분리하지 못한다.}}
\label{{tab:real_noise_ratio}}
\begin{{tabular}}{{llrrrl}}
\toprule
레코드 & 사이트 & $H_s$\,[m] & $\sigma$\,[m] & $\sigma_\text{{noise}}/\sigma$ & 층 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    with open(OUT_TEX, 'w') as f:
        f.write(tex)


if __name__ == '__main__':
    main()
