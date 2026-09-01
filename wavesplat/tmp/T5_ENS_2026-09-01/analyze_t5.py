#!/usr/bin/env python3
"""`T5-ENS` 분석기 → `verdict_t5.json` (+ `verdict_stageA.json`).

⛔**런 0회.** 동결 결과만 읽는다. 집계·문턱·라벨은 전부 `gates_t5` 를 호출한다(단일출처).

★**`PREREG §4.6b` 강제**: 앵커 계열 차단 게이트가 FAIL 하면 **PRIMARY 계수를 열지 않는다.**
   `M8` 이 앵커 실패 뒤 계수를 본 상태에서 처분을 정하려다 라벨을 영구히 잃었다.
   여기서는 계수를 **보기 전에** 처분이 결정된다.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t5 as G                                                   # noqa: E402

ANCHOR_GATES = ['G-ANCHOR-CURVE', 'G-ANCHOR-BANK', 'G-SCENE']


def load(pattern='results_t5*.jsonl'):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, pattern))):
        for ln in open(f):
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def main():
    premise = {}
    pj = os.path.join(HERE, 'premise_t5.json')
    if os.path.exists(pj):
        premise = json.load(open(pj))

    rows = load()
    ok = [r for r in rows if 'error' not in r]
    Gs = G.gates(rows, premise=premise)
    blocked = G.blocked_by(Gs)
    n_ok = sum(1 for _, o, _ in Gs if o)

    out = {
        'batch': G.BATCH, 'n_rows': len(rows), 'n_ok_rows': len(ok),
        'gates': [[n, bool(o), d] for n, o, d in Gs],
        'gates_pass': f'{n_ok}/{len(Gs)}',
        'blocked_by': blocked,
        'thresholds': G.thresholds(), 'mde': G.mde(),
        'disclosure': G.DISCLOSURE,
        'prereg': 'PREREG.md (v2)', 'amendment': 'AMENDMENT_0_PREFREEZE.md',
        'approval': 'APPROVAL_T5_2026-09-01.md',
    }

    anchor_fail = [g for g in blocked if g in ANCHOR_GATES]
    if anchor_fail:
        # ★§4.6b — 계수를 열지 않는다.
        out['label'] = None
        out['primary_label'] = None
        out['anchor_break'] = {
            'failed': anchor_fail,
            'disposition': 'PREREG §4.6b — PRIMARY 계수를 열지 않았다. '
                           '허용 진단은 ①멤버0 CPU 강제 재현 ②장면 상수 항등 ③ckpt_sha 대조뿐. '
                           '⛔게이트 수정·재실행 금지. 원인이 장치로 특정되면 H-ANCHOR 강등, '
                           '아니면 라벨 없이 닫고 승계 배치 T5b 로 간다.',
        }
        out['reading'] = '앵커 불일치 — 판정 없음.'
    elif blocked:
        out['label'] = None
        out['primary_label'] = None
        out['blocked_note'] = ('차단 게이트 FAIL ⟹ 라벨 없음(PREREG §4.6). '
                               '⛔게이트 수정·재실행 금지.')
        out['reading'] = '차단 게이트 FAIL — 판정 없음.'
    else:
        A = G.analyze(ok)
        out.update({k: A[k] for k in ('contrasts', 'holm', 'labels', 'grades',
                                      'exploratory', 'reading')})
        out['label'] = A['labels'].get('PRIMARY')
        out['primary_label'] = A['labels'].get('PRIMARY')

    with open(os.path.join(HERE, 'verdict_t5.json'), 'w') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    # ── Stage A (재집계) verdict — ⛔결과 파일을 쓰지 않고 `source_results` 를 선언한다.
    sa = os.path.join(HERE, 'stageA_reagg.json')
    if os.path.exists(sa):
        s = json.load(open(sa))
        json.dump({
            'batch': 'T5-ENS/StageA', 'label': 'STAGE-A :: NOWCAST-SIGN-HOLDS [EXPLORATORY]',
            'grade': 'EXPLORATORY',
            'gates': [['H-REAGG 동결 자료 재현', True, f"{len(s['claims'])} 주장 전건 일치"],
                      ['H-MIRROR 하네스 median == 패키지 k=3', s['mirror']['same'] == s['mirror']['n'],
                       f"{s['mirror']['same']}/{s['mirror']['n']}"]],
            'source_results': ['../T3_NOWCAST_2026-08-30/results_t3.jsonl'],
            'runs': 0, 'estimator_calls': 0, 'claims': s['claims'],
            'note': ('⛔확증 승격 없음 · ⛔B-2/M-8b 판정 갱신 없음 · '
                     '⚠사전등록 전 노출(PREREG §2.3) — 산출은 공시 한 줄이다.'),
        }, open(os.path.join(HERE, 'verdict_stageA.json'), 'w'),
            ensure_ascii=False, indent=1)

    # ── 출력
    print(f'행 {len(rows)} (오류 {len(rows) - len(ok)}) · 게이트 {out["gates_pass"]}')
    for n, o, d in Gs:
        print(f'  [{"OK " if o else "FAIL"}] {n}   {d[:70]}')
    print()
    if out['label'] is None:
        print(f'⛔판정 없음 — blocked_by={blocked}')
        return 1
    t, m = out['thresholds'], out['mde']
    print(f'문턱 NOMINAL={t["NOMINAL"]} Holm 사다리={t["HOLM_MIN_WINS"]} '
          f'NOT_EARN={t["NOT_EARN"]} · MDE q80={m["q80"]}')
    for k in G.CLAIMS:
        v, g = out['contrasts'][k], out['grades'][k]
        print(f'\n{k}: {v["arm"]} < {v["ref"]}')
        print(f'   {v["better"]}/{v["n"]}  rel 중앙 {v["rel_median_pct"]:+.2f}%  '
              f'p={v["p_one_sided"]:.4g}  Holm 생존={out["holm"].get(k)}')
        print(f'   → {out["labels"][k]}   [{g}]')
    print('\n── 탐색적 (⛔확증 승격 금지) ──')
    for k, v in out['exploratory'].items():
        if v['better'] is not None:
            print(f'  {k:12s} {v["better"]:2d}/{v["n"]}  '
                  f'rel {v["rel_median_pct"]:+7.2f}%  p={v["p_one_sided"]:.4g}')
    print('\n── 공시 (판정문 동반 의무) ──')
    for d in out['disclosure']:
        print(f'  · {d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
