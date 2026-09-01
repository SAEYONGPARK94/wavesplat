#!/usr/bin/env python3
"""`S-ENS` 분석기 → `verdict_sens.json`. ⛔런 0회 — 동결 결과만 읽는다.

★`PREREG §4.6` 강제: 앵커 계열 차단 게이트가 FAIL 하면 **PRIMARY 계수를 열지 않는다.**
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22')))
import gates_sens as G  # noqa: E402

ANCHOR_GATES = ['G-ANCHOR-M1', 'G-ENSK']


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results_sens_*.jsonl'))):
        for ln in open(f):
            if ln.strip():
                rows.append(json.loads(ln))
    ok = [r for r in rows if 'error' not in r]
    Gs = G.gates(rows)
    blocked = G.blocked_by(Gs)
    n_ok = sum(1 for _, o, _ in Gs if o)

    out = dict(batch=G.BATCH, n_rows=len(rows), n_ok_rows=len(ok),
               gates=[[n, bool(o), d] for n, o, d in Gs],
               gates_pass=f'{n_ok}/{len(Gs)}', blocked_by=blocked,
               thresholds=G.thresholds(), mde=G.mde(),
               disclosure=G.DISCLOSURE, prereg='PREREG.md')

    anchor_fail = [g for g in blocked if g in ANCHOR_GATES]
    if anchor_fail:
        out.update(label=None, primary_label=None,
                   anchor_break=dict(failed=anchor_fail,
                                     disposition='PREREG §4.6 — 계수를 열지 않았다.'),
                   reading='앵커 불일치 — 판정 없음.')
    elif blocked:
        out.update(label=None, primary_label=None,
                   reading='차단 게이트 FAIL — 판정 없음. ⛔게이트 수정·재실행 금지.')
    else:
        A = G.analyze(ok)
        out.update({k: A[k] for k in ('contrasts', 'holm', 'labels', 'grades', 'reading')})
        out['label'] = out['primary_label'] = A['labels'].get('PRIMARY')

    json.dump(out, open(os.path.join(HERE, 'verdict_sens.json'), 'w'),
              ensure_ascii=False, indent=1)

    print(f'행 {len(rows)} (오류 {len(rows)-len(ok)}) · 게이트 {out["gates_pass"]}')
    for n, o, d in Gs:
        print(f'  [{"OK " if o else "FAIL"}] {n}   {d[:66]}')
    if out['label'] is None:
        print(f'\n⛔판정 없음 — blocked_by={blocked}')
        return 1
    t, m = out['thresholds'], out['mde']
    print(f'\n문턱 NOMINAL={t["NOMINAL"]} Holm={t["HOLM_MIN_WINS"]} '
          f'NOT_EARN={t["NOT_EARN"]} · MDE q80={m["q80"]}')
    for k in G.CLAIMS:
        v = out['contrasts'][k]
        print(f'\n{k} ({G.CLAIMS[k]["stratum"]}): LEARN-M3 < LEARN-M1')
        print(f'   {v["better"]}/{v["n"]}  rel 중앙 {v["rel_median_pct"]:+.2f}%  '
              f'p={v["p_one_sided"]:.4g}  Holm 생존={out["holm"].get(k)}')
        print(f'   → {out["labels"][k]}   [{out["grades"][k]}]')
    print('\n── 공시 ──')
    for d in out['disclosure']:
        print(f'  · {d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
