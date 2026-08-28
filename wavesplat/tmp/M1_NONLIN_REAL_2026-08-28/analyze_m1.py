"""M1 동결 분석기 — ⛔결과보다 **먼저** 쓰였다. 결과를 보고 고치지 않는다.

    python3 tmp/M1_NONLIN_REAL_2026-08-28/analyze_m1.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gates_m1 as G                                           # noqa: E402


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m1_*.jsonl'))):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        print('⛔유효 0행 — verdict 을 쓰지 않는다 (0행에서 판정 금지).', flush=True)
        sys.exit(1)

    agg = G.aggregate(rows)
    labels, arms = {}, {}
    for a in G.GT_ARMS:
        d = agg[a]
        lab, msg = G.arm_label(d['prim'], d['neg'])
        labels[a] = lab
        arms[a] = dict(label=lab, label_msg=msg, reading=G.READING[lab],
                       ship_vs_spec=d['prim'], spec_vs_zero=d['neg'], ship_vs_zero=d['sz'],
                       medians={k: (sorted(v.values())[len(v) // 2] if v else None)
                                for k, v in (('ship', d['ship']), ('spec', d['spec']),
                                             ('zero', d['zero']))})
    key, pair_txt = G.pair_reading(labels)

    # 짝 대비 (탐색적) — 같은 (ncomp, seed) 에서 HOS 대 LIN 의 SHIP 상대차
    paired_rel = []
    for k in sorted(agg['LIN']['ship']):
        h = k.replace('LIN|', 'HOS|')
        if h in agg['HOS']['ship']:
            lo, hi = agg['LIN']['ship'][k], agg['HOS']['ship'][h]
            paired_rel.append((k, lo, hi, (hi - lo) / lo * 100))
    import statistics as st
    pr = dict(n=len(paired_rel),
              hos_worse=sum(1 for _, lo, hi, _ in paired_rel if hi > lo),
              rel_median_pct=(st.median([r for *_, r in paired_rel]) if paired_rel else None),
              note='탐색적 — 사전등록의 PRIMARY 가 아니다')

    gates = (G.gates_runfree()
             + G.gates_data(manifest=G._load_manifest())
             + G.gates_run(rows)
             + G.gates_pc1(json.load(open(os.path.join(HERE, 'pc1_m1.json')))
                           if os.path.exists(os.path.join(HERE, 'pc1_m1.json')) else None))
    verdict = dict(batch='M1_NONLIN_REAL_2026-08-28',
                   question='실측 해상상태(SS-REAL)에서 참값이 비선형(HOS M=3)이어도 헤드라인이 사는가',
                   n_rows=len(rows), n_ok=len(ok),
                   thresholds=G.threshold_table(),
                   arms=arms, pair_key=list(key), pair_reading=pair_txt,
                   paired_contrast=pr,
                   label=f'NONLIN-REAL :: HOS = {labels["HOS"]} · LIN = {labels["LIN"]}',
                   gates=[{'name': n, 'pass': bool(o), 'msg': m} for n, o, m in gates],
                   gates_pass=sum(1 for _, o, _ in gates if o), gates_total=len(gates))
    json.dump(verdict, open(os.path.join(HERE, 'verdict_m1.json'), 'w'),
              ensure_ascii=False, indent=1)

    for a in G.GT_ARMS:
        d = arms[a]
        print(f'\n=== GT 팔 {a} ===')
        print(f'  라벨      : {d["label"]}  ({d["label_msg"]})')
        print(f'  SHIP<SPEC : {d["ship_vs_spec"]["better"]}/{d["ship_vs_spec"]["n"]} '
              f'rel {d["ship_vs_spec"]["rel_median_pct"]:.2f}% p={d["ship_vs_spec"]["p_one_sided"]:.3e}')
        print(f'  SPEC<ZERO : {d["spec_vs_zero"]["better"]}/{d["spec_vs_zero"]["n"]} '
              f'(음성대조 문턱 {G.NEG_MIN})')
        print(f'  SHIP<ZERO : {d["ship_vs_zero"]["better"]}/{d["ship_vs_zero"]["n"]}')
        print(f'  중앙      : {d["medians"]}')
    print(f'\n★짝 읽기 {key}: {pair_txt}')
    print(f'★짝 대비(탐색적): HOS 가 나쁜 짝 {pr["hos_worse"]}/{pr["n"]} · '
          f'상대 중앙 {pr["rel_median_pct"]}')
    print(f'\n게이트 {verdict["gates_pass"]}/{verdict["gates_total"]}')
    for g in verdict['gates']:
        if not g['pass']:
            print(f'  ⛔FAIL {g["name"]}: {g["msg"]}')


if __name__ == '__main__':
    main()
