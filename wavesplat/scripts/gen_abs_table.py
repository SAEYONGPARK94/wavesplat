#!/usr/bin/env python3
"""`paper/abs_table.tex` --- **절대 오차 표**(미터). ⛔런 0회: 동결 행의 재집계다.

⛔왜(2026-09-02, 5차 감사 도전 E · 권고 3b):

> *"No absolute-error table. Five absolute numbers exist in 47 pages of body.
>  … this venue expects nRMSE **and** RMSE in metres, by stratum, with dispersion."*

★변환은 정의에서 나온다: 합성 하네스의 `nRMSE = RMSE / Hs`
(`wavesplat/evaluation/metrics.py:32-50`, 축자 *"Normalized RMSE: RMSE / **Hs**"*).
⟹ `RMSE[m] = nRMSE x Hs`, 그리고 `Hs` 는 각 에피소드 HDF5 의 `config` 속성이다.

⛔**새 주장이 아니다.** 같은 측정을 **다른 단위로** 보이는 것이고, 순위나 부호를
바꾸지 않는다(층 안에서 `Hs` 가 상수이므로 층별 순위는 **불변**이다).
⚠단 **층을 가로지르는 비교는 단위에 따라 달라진다** --- BF4 와 BF5 의 `Hs` 가 다르다.
⟹ 표는 층별로만 읽고, 그 경고를 캡션에 싣는다.
"""
import json
import os
import statistics as st
import sys

import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP = os.path.join(ROOT, 'tmp')
SEARCH = ('tmp/HOSGATE_2026-08-13/data_hos', 'tmp/HOSGATE_2026-08-13/data_lin',
          'data_s8_crowded_2026-08-10', 'data')

# (표시명, 배치 디렉터리, 우리 팔, 비교자 팔)
BATCHES = [
    ('B-12  non-linear reference', 'B12_HOS_EXTEND_2026-08-23', 'SHIP-NOW', 'SPEC25'),
    ('B-16  wavenumber ablation', 'B16_KGRID_2026-08-24', None, None),
]


def hs_of(ep, cache={}):
    if ep in cache:
        return cache[ep]
    # ⚠행의 `episode` 는 `BF4_15comp_seed1581_uniform_ra_rf_ds` 처럼
    #   **`episode_` 접두어와 확장자가 없는** 형태다.
    base = os.path.basename(ep)
    if not base.endswith('.h5'):
        base += '.h5'
    if not base.startswith('episode_'):
        base = 'episode_' + base
    for d in SEARCH:
        p = os.path.join(ROOT, d, base)
        if os.path.exists(p):
            with h5py.File(p, 'r') as f:
                cache[ep] = float(f['config'].attrs['Hs'])
            return cache[ep]
    raise SystemExit(f'⛔에피소드를 찾지 못했다: {base}')


def load(rel):
    d = os.path.join(TMP, rel)
    rows = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.jsonl'):
            rows += [json.loads(x) for x in open(os.path.join(d, f))]
    return rows


def cells(rows):
    """(팔, 층) -> 에피소드별 [nRMSE median-over-seeds, RMSE_m]."""
    by = {}
    for r in rows:
        if 'avg_nRMSE' not in r or r['avg_nRMSE'] is None:
            continue
        key = (r['arm'], r['bf'])
        by.setdefault(key, {}).setdefault(r['episode'], []).append(r['avg_nRMSE'])
    out = {}
    for key, eps in by.items():
        n = [st.median(v) for v in eps.values()]
        m = [st.median(v) * hs_of(e) for e, v in eps.items()]
        out[key] = dict(n_ep=len(n),
                        nrmse_med=st.median(n), rmse_m_med=st.median(m),
                        nrmse_iqr=(st.quantiles(n, n=4)[0], st.quantiles(n, n=4)[2])
                        if len(n) >= 4 else None,
                        rmse_m_iqr=(st.quantiles(m, n=4)[0], st.quantiles(m, n=4)[2])
                        if len(m) >= 4 else None,
                        hs=hs_of(next(iter(eps))))
    return out


def main():
    lines = ['% ⛔자동 생성 — 손으로 고치지 마라.',
             '%    생성: python3 scripts/gen_abs_table.py   (런 0회, 동결 행 재집계)',
             '%    변환: RMSE[m] = nRMSE x Hs  (metrics.py:32-50)', '']
    lines += [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
              r'\caption{Absolute accuracy, in the units the measurement is made in and '
              r'in metres. The synthetic harness normalises by significant wave height, '
              r'so metres follow from $\text{RMSE} = \text{nRMSE}\times H_s$ with no new '
              r'measurement; this table is a re-aggregation of the same frozen rows that '
              r'produce Table~\ref{tab:earns}. Each cell is the median over episodes of '
              r'the per-episode median over estimator seeds, with the interquartile '
              r'range. \emph{Read within a sea state, not across:} $H_s$ differs between '
              r'the two, so the ordering in metres and the ordering in nRMSE need not '
              r'agree across rows.}',
              r'\label{tab:absolute}',
              r'\begin{tabular}{llcrrrr}', r'\toprule',
              r'Batch & Arm & sea state & $H_s$ [m] & $n$ & nRMSE & RMSE [m] \\',
              r'\midrule']
    pins = {}
    for name, rel, ours, cmp_ in BATCHES:
        c = cells(load(rel))
        first = True
        for (arm, bf), v in sorted(c.items()):
            lab = name if first else ''
            first = False
            iqr_n = (f" {{\\scriptsize[{v['nrmse_iqr'][0]:.3f},{v['nrmse_iqr'][1]:.3f}]}}"
                     if v['nrmse_iqr'] else '')
            iqr_m = (f" {{\\scriptsize[{v['rmse_m_iqr'][0]:.3f},{v['rmse_m_iqr'][1]:.3f}]}}"
                     if v['rmse_m_iqr'] else '')
            lines.append(f"{lab} & \\texttt{{{arm}}} & {bf} & {v['hs']:.1f} & "
                         f"{v['n_ep']} & {v['nrmse_med']:.3f}{iqr_n} & "
                         f"{v['rmse_m_med']:.3f}{iqr_m} \\\\")
            pins[f'{rel.split("_")[0]}|{arm}|{bf}'] = (round(v['nrmse_med'], 3),
                                                       round(v['rmse_m_med'], 3))
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    # ★공짜로 붙는 교차 검증: `SHIP-NOW` 는 두 배치에서 **같은 팔.같은 코퍼스**다
    #   ⟹ 값이 어긋나면 어느 한쪽의 집계가 틀린 것이다. 침묵시키지 말고 죽인다.
    for bf in ('BF4', 'BF5'):
        a = pins.get(f'B12|SHIP-NOW|{bf}')
        b = pins.get(f'B16|SHIP-NOW|{bf}')
        if a and b and a != b:
            raise SystemExit(f'⛔교차 검증 실패 {bf}: B-12 {a} vs B-16 {b} — '
                             '같은 팔.같은 코퍼스인데 값이 다르다')
    out = os.path.join(ROOT, 'paper', 'abs_table.tex')
    open(out, 'w').write('\n'.join(lines) + '\n')
    # ★사이드카: 이 표가 **인쇄하는 모든 수**의 목록.
    #   ⛔인용금지 래칫이 이것을 읽어 *"이 수는 동결 행의 재집계에서 나왔지 손복사가
    #   아니다"* 를 확인한다. 표는 tex 와 **조판된 PDF 양쪽**에 나타나므로 값 단위로
    #   면제해야 한다(파일 단위 면제로는 PDF 를 덮지 못한다).
    import re as _re
    vals = sorted({m.group(0) for ln in lines
                   for m in _re.finditer(r'\d+\.\d+', ln)})
    side = os.path.join(ROOT, 'paper', 'abs_table.values.json')
    json.dump({'source': 'scripts/gen_abs_table.py',
               'note': ('동결 행 재집계 산출값. 인용금지 래칫이 값 단위 면제에 쓴다. '
                        '⛔면제는 "이 파일이 재생성으로 바이트 동일" 일 때만 유효하다.'),
               'values': vals}, open(side, 'w'), indent=1, ensure_ascii=False)
    for k, v in sorted(pins.items()):
        print(f'  {k:28s} nRMSE {v[0]:.3f}   RMSE {v[1]:.3f} m')
    print(f'\n✅{len(pins)} 셀 → {out}')


if __name__ == '__main__':
    sys.exit(main())
