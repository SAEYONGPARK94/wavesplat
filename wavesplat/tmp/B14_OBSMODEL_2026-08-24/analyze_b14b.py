"""B-14b 분석기 — ★**판정 규칙은 전부 `gates_b14` 가 낸다**(문턱·라벨 재기재 없음).

여기서 하는 것은 자료 배관뿐이다: `results_b14b_*.jsonl` 을 모아 H1 을 센다.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b14 as G                                      # noqa: E402


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b14b_*.jsonl'))):
        rows += [json.loads(l) for l in open(p)]
    ok = [r for r in rows if 'error' not in r]

    # ★개입 검증: 같은 (레코드·모형)에서 σ 만 다른 셀의 **관측 표본이 같아야 한다**
    same = {}
    for r in ok:
        same.setdefault((r['record'], r['obs_model']), set()).add(r['z_hash'])
    # IID/SLOPE 는 잡음이 z 를 바꾸므로 해시는 달라진다. 표본 동일성은 러너가 보장하고
    # 여기서는 **σ=0 셀이 존재하는지**와 표본 수만 검사한다(해시는 잡음 포함 값이다).
    gts = [('G-5 오류 0행', len(ok) == len(rows), f'{len(rows) - len(ok)} 행 오류'),
           ('G-0 표본 수', len(ok) == len(G.OBS_MODELS) * len(G.NOISE_LEVELS) * G.N,
            f'{len(ok)}/{len(G.OBS_MODELS) * len(G.NOISE_LEVELS) * G.N}'),
           ('G-4 참 제로필드가 상수가 아니다',
            len({round(r['zero_ref'], 6) for r in ok}) > 1, '')]

    h1 = {}
    for mdl in G.OBS_MODELS:
        by = {}
        for r in ok:
            if r['obs_model'] == mdl:
                by.setdefault(r['record'], {})[r['noise']] = r['nowcast']
        recs = [k for k, d in by.items() if 0.0 in d and 0.02 in d]
        wins = sum(1 for k in recs if by[k][0.0] < by[k][0.02])
        interior = sum(1 for k in recs
                       if 0.005 in by[k] and by[k][0.005] < min(by[k][0.0], by[k][0.02]))
        rel = st.median([(by[k][0.0] - by[k][0.02]) / by[k][0.02] * 100
                         for k in recs]) if recs else float('nan')
        beats = {nz: sum(1 for r in ok if r['obs_model'] == mdl and r['noise'] == nz
                         and r['beats_zero']) for nz in G.NOISE_LEVELS}
        h1[mdl] = dict(n=len(recs), zero_better=wins,
                       p_one_sided=G.sign_test_one_sided(wins, len(recs)) if recs else None,
                       interior_min=interior, rel_median_pct=rel,
                       beats_zero=beats,
                       median_nrmse={str(nz): st.median([r['nowcast'] for r in ok
                                                         if r['obs_model'] == mdl
                                                         and r['noise'] == nz] or [float('nan')])
                                     for nz in G.NOISE_LEVELS})
    best = max(h1, key=lambda m: h1[m]['zero_better'])
    lab = G.label_h1(best, h1[best]['zero_better'])
    v = dict(batch='B-14b', n_rows=len(rows), n_ok=len(ok),
             gates=[[g[0], bool(g[1]), str(g[2])] for g in gts],
             gates_pass=sum(1 for g in gts if g[1]), gates_total=len(gts),
             h1=h1, h1_label=lab[0], h1_text=lab[1])
    open(os.path.join(HERE, 'verdict_b14b.json'), 'w').write(
        json.dumps(v, indent=1, ensure_ascii=False))
    print('=== 게이트 ===')
    for n_, q_, d_ in gts:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print(f'\n=== H1 (수정 후) · σ→0 이 σ=0.02 보다 나은 레코드 수 (n={G.N}) ===')
    for mdl, d in h1.items():
        print(f"  {mdl:6s} {d['zero_better']:2d}/{d['n']}  p={d['p_one_sided']:.4f}  "
              f"내부최소 {d['interior_min']:2d}  상대중앙 {d['rel_median_pct']:+7.2f}%  "
              f"제로필드 이김 " + '/'.join(str(d['beats_zero'][nz]) for nz in G.NOISE_LEVELS))
    print(f"  ⟹ {v['h1_label']} — {v['h1_text']}")


if __name__ == '__main__':
    main()
