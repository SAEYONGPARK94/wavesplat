"""`R7-FLOOR` 러너 — ⛔**추정기 런 0회**. 동결 두 배치의 행을 맺기만 한다.

⛔이 파일은 `wavesplat` 추정기를 **임포트하지 않는다** — `G-NEWRUN` 이 소스로 강제한다.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_r7 as G                                                 # noqa: E402

OUT = os.path.join(HERE, 'results_r7.jsonl')
META = os.path.join(HERE, 'meta_r7.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')

SRC = [os.path.join(G.R1B, 'results_r1b.jsonl'),
       os.path.join(G.R1B, 'probes_r1b.jsonl'),
       os.path.join(G.R2B, 'results_r2b.jsonl')]


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def check_frozen():
    if not os.path.exists(FROZEN):
        return False, 'FROZEN_md5.txt 없음'
    want = {}
    for line in open(FROZEN, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        h, _, rel = line.partition('  ')
        want[rel.strip()] = h.strip()
    bad = []
    for p in SRC:
        rel = os.path.relpath(p, os.path.join(HERE, '..', '..'))
        if rel not in want:
            bad.append(f'미동결:{rel}')
        elif want[rel] != md5(p):
            bad.append(f'불일치:{rel}')
    return (not bad), ('; '.join(bad) if bad else f'{len(SRC)}/{len(SRC)} 일치')


def check_no_estimator():
    """★`G-NEWRUN` — 이 배치의 코드가 추정기를 임포트하지 않음을 **소스로** 확인."""
    pat = re.compile(r'^\s*(from|import)\s+(wavesplat|wavesim)\b', re.M)
    hits = []
    for fn in ('run_r7.py', 'analyze_r7.py', 'gates_r7.py'):
        s = open(os.path.join(HERE, fn), encoding='utf-8').read()
        if pat.search(s):
            hits.append(fn)
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main():
    r1 = [json.loads(l) for l in open(SRC[0], encoding='utf-8') if l.strip()]
    pr = [json.loads(l) for l in open(SRC[1], encoding='utf-8') if l.strip()]
    r2 = [json.loads(l) for l in open(SRC[2], encoding='utf-8') if l.strip()]

    now = {}
    for x in r1:
        if x.get('pass_') != 'judge' or 'error' in x:
            continue
        now.setdefault(x['record'], {})[x['cell']] = x
    probe = {p['record']: p for p in pr}
    noise = {x['record']: x for x in r2 if 'error' not in x}

    rows = []
    for rec in G.RECORDS:
        try:
            a, b, c = now[rec], probe[rec], noise[rec]
            sg = float(b['sg']); z = float(b['zero_new'])
            hf = float(c['hf_excess_rms_mm'])
            fl = hf / 1000.0 / sg
            ship = float(a['SHIP']['nowcast_new']); prop = float(a['PROP']['nowcast_new'])
            e = G.EXPOSED[rec]
            ex = (abs(sg - e['sg']) <= G.EXPOSE_TOL and abs(hf - e['hf_mm']) <= 0.06
                  and abs(z - e['zero']) <= G.EXPOSE_TOL
                  and abs(ship - e['ship']) <= G.EXPOSE_TOL
                  and abs(prop - e['prop']) <= G.EXPOSE_TOL)
            rows.append(dict(record=rec, pass_='judge', sg=sg,
                             hf_excess_rms_mm=hf, floor=fl, zero_new=z,
                             ship_nowcast=ship, prop_nowcast=prop,
                             ratio_ship=ship / z, ratio_prop=prop / z,
                             Tp_gauge=float(c['Tp_gauge']),
                             Hm0_gauge=float(c['Hm0_gauge']),
                             Hm0_stereo=float(c['Hm0_stereo']),
                             expose_ok=bool(ex)))
        except Exception as exc:                                     # noqa: BLE001
            rows.append(dict(record=rec, error=f'{type(exc).__name__}: {exc}'))

    with open(OUT, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    fok, fmsg = check_frozen()
    nok, nmsg = check_no_estimator()
    json.dump(dict(frozen_ok=fok, frozen_msg=fmsg,
                   no_estimator=nok, no_estimator_msg=nmsg,
                   src_md5={os.path.basename(p): md5(p)[:12] for p in SRC}),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for r in rows:
        if 'error' in r:
            print(f"  {r['record']:6s} ⛔{r['error']}")
        else:
            print(f"  {r['record']:6s} floor={r['floor']:.3f} ratio_ship={r['ratio_ship']:.3f} "
                  f"expose={r['expose_ok']}")
    print(f'  FROZEN: {fmsg}\n  NEWRUN: {nmsg}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
