"""B-14 분석기 — ★**결과가 나오기 전에 작성·커밋된다.** 라벨은 `gates_b14` 가 낸다.

읽는 것: `results_b14.jsonl` · (양성대조용) `../B2_REAL_2026-08-23/results_b2b.jsonl`
쓰는 것: `verdict_b14.json`
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b14 as G                                # noqa: E402

ROWS = os.path.join(HERE, 'results_b14.jsonl')
B2B = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23', 'results_b2b.jsonl'))


def load():
    rows = [json.loads(l) for l in open(ROWS)]
    return rows, [r for r in rows if 'error' not in r]


def pos_control(ok):
    """G-1 양성대조 — `IID σ=0.02` 가 `B-2b` 의 같은 셀과 비트 동일한가."""
    if not os.path.exists(B2B):
        return None, 'B-2b 결과 없음'
    ref = {}
    for l in open(B2B):
        r = json.loads(l)
        if 'error' not in r and abs(r['noise'] - 0.02) < 1e-12:
            ref[r['record']] = r['nowcast']
    mine = {r['record']: r['nowcast'] for r in ok
            if r['axis'] == 'A' and r['obs_model'] == 'IID' and r['noise'] == 0.02}
    common = sorted(set(ref) & set(mine))
    if not common:
        return None, '겹치는 레코드 없음'
    d = [abs(mine[k] - ref[k]) for k in common]
    n_exact = sum(1 for x in d if x <= 1e-9)
    return (n_exact == len(common)), f'{n_exact}/{len(common)} 비트 동일 (최대 Δ={max(d):.3e})'


def main():
    rows, ok = load()
    v = {'batch': 'B-14', 'n_rows': len(rows), 'n_ok': len(ok)}

    gts = G.gates(rows)
    pc, pc_txt = pos_control(ok)
    gts.insert(0, ('G-1 양성대조: IID σ=0.02 = B-2b 재현', bool(pc), pc_txt))
    v['gates'] = [[g[0], bool(g[1]), str(g[2])] for g in gts]
    v['gates_pass'] = sum(1 for g in gts if g[1])
    v['gates_total'] = len(gts)

    # ── H1: σ→0 단조성 회복 ────────────────────────────────────────────────
    h1 = {}
    for mdl in G.OBS_MODELS:
        by = {}
        for r in ok:
            if r['axis'] == 'A' and r['obs_model'] == mdl:
                by.setdefault(r['record'], {})[r['noise']] = r['nowcast']
        recs = [k for k, d in by.items() if 0.0 in d and 0.02 in d]
        wins = sum(1 for k in recs if by[k][0.0] < by[k][0.02])
        interior = sum(1 for k in recs
                       if 0.005 in by[k] and by[k][0.005] < min(by[k][0.0], by[k][0.02]))
        rel = st.median([(by[k][0.0] - by[k][0.02]) / by[k][0.02] * 100 for k in recs]) \
            if recs else float('nan')
        h1[mdl] = {'n': len(recs), 'zero_better': wins,
                   'p_one_sided': G.sign_test_one_sided(wins, len(recs)) if recs else None,
                   'interior_min': interior, 'rel_median_pct': rel,
                   'median_nrmse': {str(nz): st.median([r['nowcast'] for r in ok
                                                       if r['axis'] == 'A'
                                                       and r['obs_model'] == mdl
                                                       and r['noise'] == nz] or [float('nan')])
                                    for nz in G.NOISE_LEVELS}}
    best = max(h1, key=lambda m: h1[m]['zero_better']) if h1 else None
    lab1 = G.label_h1(best, h1[best]['zero_better']) if best else ('NO-DATA', '')
    v['h1'] = h1
    v['h1_label'], v['h1_text'] = lab1

    # ── H2: 조밀·무잡음에서 표현이 무기술 기준을 넘는가 ───────────────────
    h2 = {}
    for m_ in G.DENSITIES:
        cell = [r for r in ok if r['axis'] == 'B' and r['m_scan'] == m_]
        h2[str(m_)] = {'n': len(cell),
                       'beats_zero': sum(1 for r in cell if r['beats_zero']),
                       'median_nrmse': st.median([r['nowcast'] for r in cell])
                       if cell else None,
                       'median_zero': st.median([r['zero_ref'] for r in cell])
                       if cell else None,
                       'median_obs': st.median([r['n_obs_used'] for r in cell])
                       if cell else None}
    top = str(max(G.DENSITIES))
    lab2 = G.label_h2(h2[top]['beats_zero']) if h2[top]['n'] else ('NO-DATA', '')
    v['h2'] = h2
    v['h2_label'], v['h2_text'] = lab2
    v['h2_p_one_sided'] = (G.sign_test_one_sided(h2[top]['beats_zero'], h2[top]['n'])
                           if h2[top]['n'] else None)

    open(os.path.join(HERE, 'verdict_b14.json'), 'w').write(
        json.dumps(v, indent=1, ensure_ascii=False))

    print('=== 게이트 ===')
    for n_, q_, d_ in gts:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print(f'\n=== H1 · σ→0 이 σ=0.02 보다 나은 레코드 수 (n={G.N}) ===')
    for mdl, d in h1.items():
        print(f"  {mdl:6s} {d['zero_better']:2d}/{d['n']}  p={d['p_one_sided']:.4f}  "
              f"내부최소 {d['interior_min']:2d}  상대중앙 {d['rel_median_pct']:+7.2f}%  "
              f"중앙 nRMSE " + ' / '.join(f"{k}:{x:.4f}" for k, x in d['median_nrmse'].items()))
    print(f"  ⟹ {v['h1_label']} — {v['h1_text']}")
    print('\n=== H2 · 무잡음에서 밀도를 올리면 (참 제로필드 대비) ===')
    for m_, d in h2.items():
        print(f"  m={m_:>3}  관측 {d['median_obs']}  이김 {d['beats_zero']}/{d['n']}  "
              f"중앙 nRMSE {d['median_nrmse']:.4f} vs 제로 {d['median_zero']:.4f}")
    print(f"  ⟹ {v['h2_label']} — {v['h2_text']}  (p={v['h2_p_one_sided']})")


if __name__ == '__main__':
    main()
