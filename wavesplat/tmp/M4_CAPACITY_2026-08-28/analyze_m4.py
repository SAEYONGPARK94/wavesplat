"""M4 분석기 — 판정은 **여기서만** 나온다. 규칙은 `gates_m4.py` 단일출처.

    python3 tmp/M4_CAPACITY_2026-08-28/analyze_m4.py    # -> verdict_m4.json

⛔라벨을 손으로 붙이지 않는다. ⛔0행이면 판정하지 않고 **FAIL 로 끝난다**.
합성 기준선(`SHIP-K`) 행은 **`B-22` 의 `LEARN` 동결 행을 수입**한다(재실행 0회).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B22_PRIOR_ABLATION_2026-08-27'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'), os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import gates_m4 as G       # noqa: E402

OUT = os.path.join(HERE, 'verdict_m4.json')


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results_m4*.jsonl'))):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # ★합성 기준선 = B-22 `LEARN` 동결 행 수입 (arm 이름만 이 배치 규약으로 바꾼다)
    imported = 0
    for r in G._b22_learn_rows():
        rows.append(dict(stratum='syn', arm=G.SYN_IMPORTED_BASE, unit=r['episode'],
                         est_seed=r['est_seed'], avg_nRMSE=r['avg_nRMSE'],
                         zero_nRMSE=r.get('zero_nRMSE'),
                         n_gaussians=r.get('n_gaussians'),
                         K_expected=r.get('n_pool_expected'),
                         atoms_md5=r.get('atoms_md5'),
                         n_atoms_injected=r.get('n_atoms_injected'),
                         propose_calls=r.get('propose_calls'),
                         si_l1_readback=False, cuda_available=False,
                         imported_from='B22:LEARN'))
        imported += 1
    return rows, imported


def main():
    rows, imported = load_rows()
    ok = [r for r in rows if 'error' not in r]
    own = [r for r in ok if not r.get('imported_from')]
    if not own:
        print('FAIL — 이 배치의 결과 행이 0개다. 판정하지 않는다.')
        json.dump(dict(label='NO-RESULTS', gates=[['G-12 0행 가드', False, '0']],
                       note='결과 0행 — 분석기는 판정을 거부했다.'),
                  open(OUT, 'w'), indent=1, ensure_ascii=False)
        sys.exit(1)

    gates = [list(x) for x in G.gates_runfree()] + [list(x) for x in G.gates_run(ok)]

    zero = {rec: G.real_zero_ref(rec) for rec in G.RECORDS}
    real = dict(field=G.FIELD_REAL, zero_ref=zero)
    fam = []
    for a in G.LADDER:
        if a == G.BASE:
            continue
        pr = G.paired(ok, a, G.BASE, 'unit', G.FIELD_REAL)
        real[a] = pr
        fam.append((a, min(pr['p_one_sided'] or 1.0, pr['p_other_tail'] or 1.0) * 2))
    for a, b in G.TWOBYTWO:
        pr = G.paired(ok, b, a, 'unit', G.FIELD_REAL)
        real[f'{b}|{a}'] = pr
        fam.append((f'{b}|{a}', min(pr['p_one_sided'] or 1.0,
                                    pr['p_other_tail'] or 1.0) * 2))
    real['skill'] = {a: G.skill(ok, a, 'unit', G.FIELD_REAL, zero)
                     for a in G.LADDER}
    real['strata'] = {
        'small_box': {a: G.paired(ok, a, G.BASE, 'unit', G.FIELD_REAL,
                                  subset=G.SMALL_BOX) for a in G.LADDER
                      if a != G.BASE},
        'large_box': {a: G.paired(ok, a, G.BASE, 'unit', G.FIELD_REAL,
                                  subset=G.LARGE_BOX) for a in G.LADDER
                      if a != G.BASE}}
    real['holm_family'] = G.holm(fam)
    prim = real[G.PRIMARY_ARM]
    lab, why = G.label_real(prim)

    syn = {'field': G.FIELD_SYN, 'imported_base_rows': imported}
    for a in G.SYN_ARMS:
        if a == G.BASE:
            continue
        syn[a] = G.paired(ok, a, G.BASE, 'unit', G.FIELD_SYN)
    slab, swhy = (G.label_syn(syn[G.PRIMARY_ARM]) if G.PRIMARY_ARM in syn
                  else ('SYN-NOT-RUN', '합성 층 행 0개'))

    # ★AMENDMENT 1 — 개정이 PRIMARY 를 움직이지 않았음을 **논증이 아니라 비트로**
    gates.append(list(G.gate_amend_invariance(lab, prim, slab, syn.get(G.PRIMARY_ARM))))

    # 용량-반응표 (사다리 K 의 실제 중앙값)
    dose = {}
    for a in G.LADDER:
        ks = [r['K_expected'] for r in ok if r['arm'] == a and r['stratum'] == 'real'
              and r.get('K_expected') is not None]
        ns = [r['n_gaussians'] for r in ok if r['arm'] == a and r['stratum'] == 'real'
              and r.get('n_gaussians') is not None]
        import statistics as st
        dose[a] = dict(K_median=(st.median(ks) if ks else None),
                       n_gaussians_median=(st.median(ns) if ns else None),
                       n=len(ks))

    verdict = dict(
        label=lab, why=why, label_secondary=slab, why_secondary=swhy,
        direction=G.DIRECTION, reading=G.READING[lab],
        disclosure=G.DISCLOSURE, prediction_recorded_before=G.PREDICTION,
        primary=dict(prim),
        thresholds=dict(real=G.REAL_T, syn=G.SYN_T,
                        alpha=G.ALPHA, family_alpha=G.FAMILY_ALPHA),
        dose_response_real=dose, real=real, syn=syn,
        gates=gates,
        n_rows=dict(total=len(rows), own=len(own), imported=imported,
                    errors=len(rows) - len(ok)))
    json.dump(verdict, open(OUT, 'w'), indent=1, ensure_ascii=False, default=str)
    print(f'PRIMARY  {lab}  — {why}')
    print(f'SECONDARY(syn)  {slab}  — {swhy}')
    nfail = sum(1 for _, okv, _ in gates if not okv)
    print(f'게이트 {len(gates) - nfail}/{len(gates)} PASS')
    for gid, okv, msg in gates:
        if not okv:
            print(f'  [FAIL] {gid}: {msg[:110]}')
    print('->', OUT)
    sys.exit(1 if nfail else 0)


if __name__ == '__main__':
    main()
