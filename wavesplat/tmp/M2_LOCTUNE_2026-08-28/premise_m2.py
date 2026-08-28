"""M2 전제 — ★**런 0회**. PREREG §0 의 모든 수를 낸다(⛔손복사 0).

    python3 tmp/M2_LOCTUNE_2026-08-28/premise_m2.py    ->  premise_m2.json
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_m2 as G                                            # noqa: E402


def main():
    fr = G.frozen_rows()
    sw = G.sweep_corpus_lambda()
    depth = G.B26.gen_params()['depth']
    out = {
        'P1_derivation': {
            'ladder_from_run_s17': list(G.ladder()),
            'loc_incumbent_from_gates_b11': G.LOC_INCUMBENT,
            'sweep_corpus': sw,
            'lam_p_by_corner': {c: G.lam_p(G.B26.CORNERS[c]['Tp'], depth)
                                for c in G.B26.CORNER_ORDER},
            'loc_over_lam_sweep': G.LOC_INCUMBENT / sw['median'],
            'loc_over_lam_corner': {c: G.LOC_INCUMBENT / G.lam_p(
                G.B26.CORNERS[c]['Tp'], depth) for c in G.B26.CORNER_ORDER},
            'mismatch_factor': {c: (G.LOC_INCUMBENT / G.lam_p(
                G.B26.CORNERS[c]['Tp'], depth)) / (G.LOC_INCUMBENT / sw['median'])
                for c in G.B26.CORNER_ORDER},
            'loc_star': {c: G.loc_star(c) for c in G.B26.CORNER_ORDER},
            'candidates': {c: list(G.candidates(c)) for c in G.B26.CORNER_ORDER},
            # 참조 선택 민감도: BF4 단독을 기준으로 써도 후보 집합이 같은가
            'loc_star_bf4_ref': {
                c: G.LOC_INCUMBENT * G.lam_p(G.B26.CORNERS[c]['Tp'], depth)
                / G.lam_p(G.B26.CORNERS['SS-BF4']['Tp'], depth)
                for c in G.B26.CORNER_ORDER},
        },
        'P2_grid': {},
        'P3_frozen_b26': {},
        'P4_s17': G.s17_rules(),
        'P5_mde': {},
    }
    dx, dy = G._grid_dxdy()
    out['P2_grid'] = {'nx': G.MESH_GRID[0], 'ny': G.MESH_GRID[1], 'dx': dx, 'dy': dy,
                      'gc_support_dx': {a: 2 * G.ARM_LOC[a] / dx for a in G.ARM_ORDER},
                      'gc_support_dy': {a: 2 * G.ARM_LOC[a] / dy for a in G.ARM_ORDER}}

    for c in G.CORNERS_M2:
        sub = [r for r in fr if r.get('corner') == c]
        ship = G.med_by_episode(sub, G.SHIP_ARM)
        spec = G.med_by_episode(sub, G.INCUMBENT_ARM)
        zero = G.med_by_episode(sub, G.INCUMBENT_ARM, field='zero_nRMSE')
        raw = [r for r in sub if r.get('arm') == G.INCUMBENT_ARM]
        s = G.arm_stats(ship, spec, zero, raw)
        e = G.eligible(G.INCUMBENT_ARM, s)
        ks = sorted(set(spec) & set(zero))
        # R-1b' 를 통과시키려면 기준선이 몇 % 좋아져야 하는가 (이분탐색)
        lo, hi = 0.5, 1.0
        for _ in range(60):
            m = (lo + hi) / 2
            if st.median([(m * spec[k] - zero[k]) / zero[k] for k in ks]) <= -0.05:
                lo = m
            else:
                hi = m
        out['P3_frozen_b26'][c] = dict(
            ship_median=st.median(list(ship.values())),
            spec_median=st.median(list(spec.values())),
            zero_median=st.median(list(zero.values())),
            W=s['W'], rel=s['rel'], p=s['p'],
            spec_lt_zero=s['a_lt_zero'], skill=s['skill'],
            zero_loss=s['zero_loss'], zero_loss_raw=s['zero_loss_raw'],
            elig_b26=e['b26'], elig_s17=e['s17'],
            f_to_pass_r1b=lo, pct_to_pass_r1b=(1 - lo) * 100,
            ship_vs_zero=G.paired(ship, zero))
        out['P5_mde'][c] = G.mde(ship, spec)

    p = os.path.join(HERE, 'premise_m2.json')
    open(p, 'w').write(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(f'\n-> {p}')


if __name__ == '__main__':
    main()
