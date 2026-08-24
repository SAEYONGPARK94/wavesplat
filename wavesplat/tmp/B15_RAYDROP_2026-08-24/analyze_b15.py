"""B-15 분석기 — ★결과 전에 작성·커밋된다. 라벨은 `gates_b15` 가 낸다."""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b15 as G                                     # noqa: E402


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b15_*.jsonl'))):
        rows += [json.loads(l) for l in open(p)]
    return rows, [r for r in rows if 'error' not in r]


def med_by_episode(ok, arm):
    """추정기/필터 시드 median-of-k → 에피소드 단위 값 (B-11 규약)."""
    by = {}
    for r in ok:
        if r['arm'] == arm:
            by.setdefault(r['episode'], []).append(r)
    out = {}
    for ep, rs in by.items():
        out[ep] = dict(nrmse=st.median([x['avg_nRMSE'] for x in rs]),
                       zero=st.median([x['zero_nRMSE'] for x in rs
                                       if x.get('zero_nRMSE') is not None] or [float('nan')]),
                       n_seed=len(rs), raydrop=rs[0]['raydrop'], pair=rs[0]['pair'],
                       inc=rs[0].get('mean_inc_deg'), pts=rs[0].get('pts_per_frame'),
                       gt_hash=rs[0].get('gt_hash'))
    return out


def gates(rows, ok, ship, spec):
    out = []
    out.append(('G-5 오류 0행', len(ok) == len(rows), f'{len(rows) - len(ok)} 행 오류'))
    need = len(G.RAYDROPS) * G.N_PAIR
    out.append(('G-0 표본 수(에피소드)', len(ship) == need and len(spec) == need,
                f'SHIP {len(ship)}/{need} · SPEC {len(spec)}/{need}'))
    # G-1 짝 동일성: 같은 pair 의 uniform/slope 가 **같은 파랑장**인가
    pairs, bad = {}, []
    for ep, d in ship.items():
        pairs.setdefault(d['pair'], {})[d['raydrop']] = d['gt_hash']
    for pr, d in pairs.items():
        if len(d) == 2 and abs(d['uniform'] - d['slope']) > 1e-6:
            bad.append(pr)
    out.append(('G-1 짝의 참 파랑장 동일', not bad and len(pairs) == G.N_PAIR,
                f'{len(pairs)} 짝 · 불일치 {len(bad)}'))
    # G-2 개입 발화: 반환점의 평균 입사각이 raydrop 별로 다르다
    inc = {}
    for ep, d in ship.items():
        inc.setdefault(d['raydrop'], []).append(d['inc'])
    fired = (len(inc) == 2 and
             abs(st.median(inc['uniform']) - st.median(inc['slope'])) > 0.1)
    out.append(('G-2 개입 발화(반환 입사각이 다르다)', fired,
                ' · '.join(f'{k} {st.median(v):.2f}°' for k, v in sorted(inc.items()))))
    # G-4 배포 상수 기록
    keys = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh', 'lambda_l1')
    have = all(all(k in r for k in keys) for r in ok if r['arm'] == 'SHIP-NOW')
    out.append(('G-4 배포 상수 행에 기록', have, ' · '.join(keys)))
    # G-6 제로필드 기준이 있고 상수가 아니다
    zs = {round(d['zero'], 6) for d in spec.values()}
    out.append(('G-6 제로필드 기준이 상수가 아니다', len(zs) > 1, f'{len(zs)} 종'))
    return out


def main():
    rows, ok = load()
    ship, spec = med_by_episode(ok, 'SHIP-NOW'), med_by_episode(ok, 'SPEC25')
    gts = gates(rows, ok, ship, spec)

    res = {}
    for rd in G.RAYDROPS:
        eps = sorted([e for e, d in ship.items() if d['raydrop'] == rd])
        common = [e for e in eps if e in spec]
        w = sum(1 for e in common if ship[e]['nrmse'] < spec[e]['nrmse'])
        neg = sum(1 for e in common if spec[e]['nrmse'] < spec[e]['zero'])
        rel = (st.median([(ship[e]['nrmse'] - spec[e]['nrmse']) / spec[e]['nrmse'] * 100
                          for e in common]) if common else None)
        res[rd] = dict(n=len(common), ship_beats_spec=w,
                       p_one_sided=G.sign_test_one_sided(w, len(common)) if common else None,
                       spec_beats_zero=neg, rel_median_pct=rel,
                       median_ship=st.median([ship[e]['nrmse'] for e in common]) if common else None,
                       median_spec=st.median([spec[e]['nrmse'] for e in common]) if common else None,
                       mean_inc_deg=st.median([ship[e]['inc'] for e in common]) if common else None)
    neg_ok = res['slope']['spec_beats_zero'] >= G.NEG_MIN if res['slope']['n'] else False
    lab = G.label_primary(res['slope']['ship_beats_spec'], neg_ok)
    lab2 = G.label_secondary(res['uniform']['ship_beats_spec'], res['slope']['ship_beats_spec'])

    v = dict(batch='B-15', n_rows=len(rows), n_ok=len(ok),
             gates=[[g[0], bool(g[1]), str(g[2])] for g in gts],
             gates_pass=sum(1 for g in gts if g[1]), gates_total=len(gts),
             by_raydrop=res, neg_control_ok=neg_ok,
             label=lab[0], label_text=lab[1],
             secondary_label=lab2[0], secondary_text=lab2[1])
    open(os.path.join(HERE, 'verdict_b15.json'), 'w').write(
        json.dumps(v, indent=1, ensure_ascii=False))

    print('=== 게이트 ===')
    for n_, q_, d_ in gts:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print(f'\n=== raydrop 별 (짝 {G.N_PAIR}, median-of-3) ===')
    for rd, d in res.items():
        print(f"  {rd:8s} SHIP<SPEC {d['ship_beats_spec']:2d}/{d['n']}  p={d['p_one_sided']:.4g}  "
              f"상대중앙 {d['rel_median_pct']:+7.2f}%  |  SPEC<ZERO {d['spec_beats_zero']:2d}/{d['n']}  "
              f"|  입사각 {d['mean_inc_deg']:.2f}°")
    print(f"\nPRIMARY   {v['label']} — {v['label_text']}")
    print(f"SECONDARY {v['secondary_label']} — {v['secondary_text']}")


if __name__ == '__main__':
    main()
