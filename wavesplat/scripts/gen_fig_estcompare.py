#!/usr/bin/env python3
"""추정 비교 그림 둘 — ⛔런 0회(동결 verdict + 동결 행 재집계).

## 왜 이 둘인가
논문에는 표로만 있고 그림으로는 없는 것이 둘 있다:

**(A) 헤드라인의 짝 구조.** 표는 *"43/48, 중앙 −40.2%"* 만 보여 준다. 짝 산점도는
**분포**와 **음성 대조**를 동시에 보인다 --- 에피소드별 제로필드가 어디 있는지, 그리고
지는 다섯 셀이 어떤 셀인지.

**(B) 거리에 따른 오차.** `M-10` 의 셸 분해(`obs:0_5` … `obs:150_inf`)는 이 논문에서
가장 결정 관련이 높은 양인데 **본문에 한 셸(BAND_REAL)만** 인쇄된다.
파도장 추정을 쓰려는 사람이 실제로 묻는 것은 *"센서에서 얼마나 멀리까지 되는가"* 다.

## 규약
⛔손으로 적은 수 0개. 동결 verdict 과 **재집계가 어긋나면 죽는다**.
⛔음성 대조(제로필드)를 **서술이 아니라 선으로** 그린다 --- 이 저장소의 `F1` 부류 차단.
사이드카 = `paper/figures/fig_estcompare.values.json`(인용금지 래칫이 읽는다).
"""
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP = os.path.join(ROOT, 'tmp')
OUT = os.path.join(ROOT, 'paper', 'figures')
INK, ACC, MUTE, WARN = '#111', '#0b5394', '#888', '#a33'


def rows(rel):
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(TMP, rel))):
        out += [json.loads(x) for x in open(f)]
    return out


def med_by_ep(rs, arm, field='avg_nRMSE'):
    by = {}
    for r in rs:
        if r.get('arm') != arm or r.get(field) is None:
            continue
        by.setdefault(r['episode'], []).append(float(r[field]))
    return {k: st.median(v) for k, v in by.items()}


# ─────────────────────────────────────────────────────────────────────────────
def panel_a(ax, vals):
    """B-12 헤드라인의 짝 산점도 + 제로필드."""
    rs = rows('B12_HOS_EXTEND_2026-08-23/results*.jsonl')
    ship, spec = med_by_ep(rs, 'SHIP-NOW'), med_by_ep(rs, 'SPEC25')
    zero = med_by_ep(rs, 'SPEC25', 'zero_nRMSE')
    ks = sorted(set(ship) & set(spec))
    v = json.load(open(os.path.join(TMP, 'B12_HOS_EXTEND_2026-08-23',
                                    'verdict_b12.json')))
    p = v['ship_vs_spec']
    better = sum(1 for k in ks if ship[k] < spec[k])
    assert len(ks) == p['n'] and better == p['better'], (
        f'⛔재집계 불일치: {better}/{len(ks)} vs verdict {p["better"]}/{p["n"]}')
    cz = sum(1 for k in ks if spec[k] < zero[k])
    assert cz == v['spec_vs_zero']['better'], '⛔음성 대조 재집계 불일치'

    win = [k for k in ks if ship[k] < spec[k]]
    lose = [k for k in ks if ship[k] >= spec[k]]
    hi = max(max(spec.values()), max(zero.values())) * 1.08
    ax.plot([0, hi], [0, hi], ':', color=MUTE, lw=0.9, zorder=1)
    ax.plot([zero[k] for k in ks], [zero[k] for k in ks], 's', ms=3.4,
            mfc='none', mec=MUTE, mew=0.9, zorder=2, label='zero field (both axes)')
    ax.plot([spec[k] for k in win], [ship[k] for k in win], 'o', ms=4.6,
            color=ACC, alpha=.85, zorder=3, label=f'ours better ({len(win)})')
    ax.plot([spec[k] for k in lose], [ship[k] for k in lose], 'o', ms=4.6,
            color=WARN, alpha=.9, zorder=4, label=f'ours worse ({len(lose)})')
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect('equal')
    ax.set_xlabel('grid baseline, nRMSE', fontsize=7.5)
    ax.set_ylabel('ours (deployed), nRMSE', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.2, frameon=False, loc='upper left')
    ax.set_title('(a) one point per episode, non-linear reference (B-12)',
                 fontsize=8, loc='left')
    # ★음성 대조를 **그림 안에** 적는다 --- 서술로만 두지 않는다.
    ax.text(0.97, 0.06,
            f'baseline beats a zero field: {cz}/{len(ks)}\n'
            f'below the diagonal = ours more accurate',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.2,
            color=INK)
    vals.update({'b12_better': better, 'b12_n': len(ks), 'b12_cmp_zero': cz})


def panel_b(ax, vals):
    """M-10 셸 분해 — 거리에 따른 오차, 세 팔."""
    rs = rows('M10_GEOMETRY_2026-08-30/results*.jsonl')
    v = json.load(open(os.path.join(TMP, 'M10_GEOMETRY_2026-08-30',
                                    'verdict_m10.json')))
    assert sum(1 for g in v['gates'] if g[1]) == len(v['gates']), '⛔M-10 게이트 미통과'
    shells = ['obs:0_5', 'obs:5_10', 'obs:10_20', 'obs:20_30', 'obs:30_50',
              'obs:50_75', 'obs:75_100', 'obs:100_150', 'obs:150_inf']

    def curve(arm, key='shell_rmse'):
        out = []
        for sh in shells:
            by = {}
            for r in rs:
                if r.get('arm') != arm:
                    continue
                d = r.get(key) or {}
                if sh in d and d[sh] is not None:
                    by.setdefault(r['episode'], []).append(float(d[sh]))
            per = [st.median(x) for x in by.values()]
            out.append((st.median(per) if per else None, len(per)))
        return out

    ship, spec = curve('SHIP'), curve('SPEC25')
    zero = curve('SHIP', 'zero_shell_batch')
    x = list(range(len(shells)))
    # ★근거리 칸은 동결 verdict 의 abs_median 과 대조한다(어긋나면 죽는다).
    ab = v['primary']['abs_median']
    assert abs(ship[0][0] - ab['SHIP']) < 5e-3, f'⛔근거리 재집계 불일치 {ship[0][0]} vs {ab["SHIP"]}'

    ax.plot(x, [z[0] for z in zero], 's--', ms=3.6, color=MUTE, lw=1.0,
            label='zero field (no skill)')
    ax.plot(x, [s[0] for s in spec], 'o--', ms=4.2, mfc='none', mec=WARN,
            color=WARN, lw=1.0, label='grid baseline')
    ax.plot(x, [s[0] for s in ship], 'o-', ms=4.6, color=ACC, lw=1.6,
            label='ours (deployed)')
    ax.set_xticks(x)
    ax.set_xticklabels(['0–5', '5–10', '10–20', '20–30', '30–50', '50–75',
                        '75–100', '100–150', '>150'], fontsize=6.2, rotation=30)
    ax.set_xlabel('distance to the nearest observation [m]', fontsize=7.5)
    ax.set_ylabel('RMSE / $H_s$   (median over episodes)', fontsize=7.5)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.set_ylim(0, max(z[0] for z in zero) * 1.12)
    ax.legend(fontsize=6.4, frameon=False, loc='lower right')
    ax.set_title('(b) where the gain lives: error against distance (M-10)',
                 fontsize=8, loc='left')
    ax.axvspan(-0.4, 0.4, color=ACC, alpha=0.07, zorder=0)
    ax.text(0.02, 0.055, 'the band our measured\nrecords actually occupy',
            transform=ax.transAxes, fontsize=6.0, color=ACC, va='bottom')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    vals.update({f'm10_{k}': round(s[0], 4)
                 for k, s in zip(['near_ship'], [ship[0]])})
    vals['m10_far_ship'] = round(ship[-1][0], 4)
    vals['m10_far_zero'] = round(zero[-1][0], 4)


def main():
    vals = {}
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.5))
    panel_a(ax[0], vals)
    panel_b(ax[1], vals)
    fig.tight_layout(pad=0.7)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'fig_estcompare.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    json.dump({'source': 'scripts/gen_fig_estcompare.py',
               'note': '동결 verdict + 동결 행 재집계. 인용금지 래칫의 값 단위 면제용.',
               'values': vals},
              open(os.path.join(OUT, 'fig_estcompare.values.json'), 'w'),
              indent=1, ensure_ascii=False)
    print('wrote', out)
    for k, v in sorted(vals.items()):
        print(f'  {k} = {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
