#!/usr/bin/env python3
"""★`FINAL_REAL_TIMEAXIS.md` 의 **재현 가능한 단일 출처** (2026-08-24 신설).

## 왜 이 파일이 뒤늦게 생겼나
`FINAL_REAL_TIMEAXIS.md`(08-23)의 세 문장 (a)(b)(c) 는 **인용 가능한 최종형**인데,
그 표를 낸 스크립트가 **저장소에 없었다** --- 재집계가 대화형으로 수행됐다.
이 저장소가 반복해서 대가를 치른 부류(*"동결 분석기 미실행"*)와 같은 형태다:
**판정은 남고 그 판정을 낸 코드가 없으면 재현이 사람의 기억에 걸린다.**
⟹ 이 파일이 그 표를 **자료에서 다시 만든다.**  숫자를 손으로 옮겨 적지 않는다.

## 규약 (FINAL_REAL_TIMEAXIS.md 가 선언한 것 그대로)
1. **참 제로필드 기준** --- 레코드·창·lead 별 (`B2_REAL_2026-08-23/zero_ref.py` 가 단일 출처;
   여기서는 그 함수가 이미 채운 캐시 `true_zero_perlead.json` 을 읽는다).
   ⛔상수 `1.0` 을 쓰지 않는다(그 상수가 실측 아크의 계수를 전부 틀리게 했다).
2. **레코드 수준 접기** --- 창 6개(`B-3` {1,3,5} + `B-3c` {0,2,4})의 **중앙**을 취해
   **레코드를 표본 단위**로 삼는다(n=12). 단측 부호검정.
3. 구간 통계 = 그 구간 lead 들의 **평균**(`rv4/final.py` 의 `fc` 와 같은 정의).

⚠**lead 단위 분해는 탐색적이다** --- 사전등록된 것은 구간(1--3 / 4--14)이다.
그림은 그 분해를 그리므로 캡션에 그 사실을 적는다.

사용:  python3 tmp/B3_REFIT_2026-08-23/final6.py [--fig]
"""
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'thesis', 'figures_ws',
                                       'fig10_real_leadcurve.pdf'))

# ★문서 대조용 값(=DATA 손복사)이지만 **기준이 아니라 회귀 검사**다.
#   출처: FINAL_REAL_TIMEAXIS.md 표(같은 jsonl 에서 나왔다). 불일치하면 이 스크립트가 실패한다.
DOC_CHECK = {
    (1, 3, 'FROZEN'): (11, -19.92), (1, 3, 'REFIT2'): (11, -23.91),
    (4, 14, 'FROZEN'): (5, +1.76), (4, 14, 'REFIT2'): (11, -18.42),
    (1, 14, 'FROZEN'): (5, +0.58), (1, 14, 'REFIT2'): (11, -18.33),
}
ARMS = ('FROZEN', 'REFIT2')


def sign_p(w, n):
    """단측 부호검정 (w 승 이상이 우연히 나올 확률)."""
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def load():
    rows = []
    for f in ('results_b3.jsonl', 'results_b3c.jsonl'):
        rows += [json.loads(l) for l in open(os.path.join(HERE, f))]
    ok = [r for r in rows if 'error' not in r]
    zero = json.load(open(os.path.join(HERE, 'true_zero_perlead.json')))
    per = {}
    for r in ok:
        per.setdefault(r['cell'], {})[(r['record'], r['win'])] = r['per_lead']
    keys = sorted(per['FROZEN'])
    leads = sorted(float(x) for x in per['FROZEN'][keys[0]])
    recs = sorted({k[0] for k in keys})
    assert len(keys) == 72 and len(recs) == 12, f'표본이 12x6 이 아니다: {len(recs)}x{len(keys)}'
    for arm in ARMS:
        assert sorted(per[arm]) == keys, f'{arm} 의 셀 집합이 FROZEN 과 다르다'
    return per, zero, keys, leads, recs


def band(getter, keys, leads, lo, hi):
    return {k: st.mean([getter(k, l) for l in leads if lo <= l <= hi]) for k in keys}


def fold(d, keys, recs):
    """창 6개의 중앙 -> 레코드 (레코드가 표본 단위)."""
    return {rc: st.median([d[k] for k in keys if k[0] == rc]) for rc in recs}


def compare(a, z, recs):
    w = sum(1 for rc in recs if a[rc] < z[rc])
    rel = st.median([(a[rc] - z[rc]) / z[rc] * 100 for rc in recs])
    return {'better': w, 'n': len(recs), 'p_one_sided': sign_p(w, len(recs)),
            'rel_median_pct': rel}


def main():
    per, zero, keys, leads, recs = load()
    zget = lambda k, l: zero[f'{k[0]}|{k[1]}'][str(l)]

    # --- 1) 구간 표 (사전등록된 읽기) ---
    table = {}
    print('=== 참 제로필드 대비, 레코드 수준(n=12), 단측 부호검정 ===')
    for lo, hi in ((1, 3), (4, 14), (1, 14)):
        zr = fold(band(zget, keys, leads, lo, hi), keys, recs)
        for arm in ARMS:
            a = fold(band(lambda k, l: per[arm][k][str(l)], keys, leads, lo, hi), keys, recs)
            c = compare(a, zr, recs)
            table[f'{lo}-{hi}|{arm}'] = c
            print(f"  lead {lo:2d}-{hi:<2d} {arm:7s} {c['better']:2d}/12  "
                  f"p={c['p_one_sided']:.5f}  rel_med {c['rel_median_pct']:+.2f}%")
            exp = DOC_CHECK[(lo, hi, arm)]
            assert c['better'] == exp[0] and abs(c['rel_median_pct'] - exp[1]) < 0.005, (
                f'FINAL_REAL_TIMEAXIS.md 와 불일치: {lo}-{hi} {arm} '
                f"{c['better']}/{c['rel_median_pct']:+.2f} vs 문서 {exp}")
    print('  ✅ FINAL_REAL_TIMEAXIS.md 표 6행 전건 재현')

    # --- 2) lead 별 분해 (탐색적, 그림용) ---
    curve = {arm: [] for arm in ARMS}
    wins = {arm: [] for arm in ARMS}
    for l in leads:
        zr = fold({k: zget(k, l) for k in keys}, keys, recs)
        for arm in ARMS:
            a = fold({k: per[arm][k][str(l)] for k in keys}, keys, recs)
            curve[arm].append(st.median([a[rc] / zr[rc] for rc in recs]))
            wins[arm].append(sum(1 for rc in recs if a[rc] < zr[rc]))

    v = {'batch': 'FINAL6', 'source': ['results_b3.jsonl', 'results_b3c.jsonl'],
         'zero_ref': 'true_zero_perlead.json (B2_REAL_2026-08-23/zero_ref.py)',
         'n_records': len(recs), 'n_windows_per_record': 6,
         'band_table': table, 'leads': leads,
         'per_lead_ratio_median': curve, 'per_lead_wins': wins,
         'doc_check': 'PASS'}
    open(os.path.join(HERE, 'verdict_final6.json'), 'w').write(
        json.dumps(v, indent=1, ensure_ascii=False))
    print(f"\nlead 별 승수(탐색적) FROZEN: {wins['FROZEN']}")
    print(f"lead 별 승수(탐색적) REFIT2: {wins['REFIT2']}")

    if '--fig' in sys.argv:
        figure(leads, curve, wins, table)


def figure(leads, curve, wins, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {'FROZEN': dict(color='#B03A2E', marker='o', label='frozen fit (shipped default)'),
             'REFIT2': dict(color='#1A5276', marker='s', label='refit every 2 s')}
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                           gridspec_kw={'height_ratios': [2.0, 1.0], 'hspace': 0.12})

    ax[0].axhspan(1.0, 1.35, color='0.92', zorder=0)
    ax[0].axhline(1.0, color='0.35', lw=1.0, ls='--', zorder=1)
    ax[0].text(0.0, 1.02, 'no skill (true zero field)', ha='left', va='bottom',
               fontsize=8, color='0.35')
    for arm in ARMS:
        ax[0].plot(leads, curve[arm], lw=1.8, ms=4.5, **style[arm], zorder=3)
    ax[0].axvspan(0.5, 3.5, color='#F4D03F', alpha=0.18, zorder=0)
    ax[0].set_ylabel('normalized error / zero-field error\n(median over 12 records)', fontsize=9)
    ax[0].set_ylim(0.55, 1.35)
    ax[0].legend(loc='lower right', fontsize=8, frameon=False)
    ax[0].set_title('Forward-time skill of the deployed real-data path\n'
                    '(12 real records $\\times$ 6 windows, fixed stereo stations)',
                    fontsize=9.5, linespacing=1.3)

    for arm in ARMS:
        ax[1].plot(leads, wins[arm], lw=1.6, ms=4.5, **style[arm], zorder=3)
    ax[1].axhline(11, color='0.35', lw=1.0, ls='--')
    ax[1].text(4.2, 9.4, 'family-wise threshold  11/12 ($p=0.0032$)', ha='left',
               va='bottom', fontsize=8, color='0.35')
    ax[1].axhline(6, color='0.75', lw=0.8)
    ax[1].axvspan(0.5, 3.5, color='#F4D03F', alpha=0.18, zorder=0)
    ax[1].set_ylabel('records better\nthan zero field', fontsize=9)
    ax[1].set_xlabel('forecast lead time (s)', fontsize=10)
    ax[1].set_ylim(0, 12.6)
    ax[1].set_yticks([0, 3, 6, 9, 12])
    ax[1].set_xlim(-0.3, 14.3)
    ax[1].set_xticks(range(0, 15, 2))
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
        a.grid(axis='y', color='0.9', lw=0.6)
        a.set_axisbelow(True)

    b13 = table['1-3|FROZEN']
    b414f, b414r = table['4-14|FROZEN'], table['4-14|REFIT2']
    cap = ("pre-registered bands, record-level sign test (n=12) vs the true zero field:  "
           f"lead 1-3 s frozen {b13['better']}/12, p={b13['p_one_sided']:.4f} "
           f"({b13['rel_median_pct']:+.1f}%)\n"
           f"lead 4-14 s frozen {b414f['better']}/12, p={b414f['p_one_sided']:.2f} "
           f"({b414f['rel_median_pct']:+.1f}%)   ---   refit@2s {b414r['better']}/12, "
           f"p={b414r['p_one_sided']:.4f} ({b414r['rel_median_pct']:+.1f}%).  "
           "Per-lead decomposition is exploratory.")
    fig.text(0.5, 0.012, cap, ha='center', va='bottom', fontsize=7.0, color='0.3',
             linespacing=1.5)
    fig.subplots_adjust(left=0.135, right=0.98, top=0.90, bottom=0.175)
    fig.savefig(FIG_OUT)
    fig.savefig(FIG_OUT.replace('.pdf', '.png'), dpi=170)
    print(f'\n★그림: {FIG_OUT}')


if __name__ == '__main__':
    main()
