"""B-16 분석기 — ★결과 전에 작성·커밋된다. 라벨은 `gates_b16` 이 낸다."""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b16 as G                                       # noqa: E402


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b16_*.jsonl'))):
        rows += [json.loads(l) for l in open(p)]
    ok = [r for r in rows if 'error' not in r]

    med = {}
    for r in ok:
        med.setdefault((r['arm'], r['episode']), []).append(r)
    val = {k: st.median([x['avg_nRMSE'] for x in v]) for k, v in med.items()}
    eps = sorted({e for (_, e) in val})

    gts = [('G-5 오류 0행', len(ok) == len(rows), f'{len(rows) - len(ok)} 행 오류')]
    need = len(G.ARM_ORDER) * G.N_EP
    gts.append(('G-0 표본 수(에피소드×팔)', len(val) == need, f'{len(val)}/{need}'))
    fired = {a: sum(1 for r in ok if r['arm'] == a and r['k_snap_fired'])
             for a in G.ARM_ORDER}
    tot = {a: sum(1 for r in ok if r['arm'] == a) for a in G.ARM_ORDER}
    gts.append(('G-1 개입 발화(스냅 팔 전건)',
                fired.get('KGRID1') == tot.get('KGRID1') and
                fired.get('KGRID2') == tot.get('KGRID2') and fired.get('SHIP-NOW') == 0,
                ' · '.join(f'{a} {fired[a]}/{tot[a]}' for a in G.ARM_ORDER)))
    dks = {a: sorted({r['k_snap_dk'] for r in ok if r['arm'] == a}) for a in G.ARM_ORDER}
    gts.append(('G-2 dk 되읽기', dks.get('KGRID1') == [G.DK1] and dks.get('KGRID2') == [G.DK2],
                str(dks)))
    keys = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh', 'lambda_l1')
    gts.append(('G-3 배포 상수 기록', all(all(k in r for k in keys) for r in ok),
                ' · '.join(keys)))

    def cmp(a, b):
        common = [e for e in eps if (a, e) in val and (b, e) in val]
        w = sum(1 for e in common if val[(a, e)] < val[(b, e)])
        rel = st.median([(val[(a, e)] - val[(b, e)]) / val[(b, e)] * 100 for e in common]) \
            if common else None
        return dict(n=len(common), better=w,
                    p_one_sided=G.sign_test_one_sided(w, len(common)) if common else None,
                    rel_median_pct=rel,
                    median_a=st.median([val[(a, e)] for e in common]) if common else None,
                    median_b=st.median([val[(b, e)] for e in common]) if common else None)

    p1 = cmp('SHIP-NOW', 'KGRID1')
    p2 = cmp('KGRID1', 'KGRID2')
    lab = G.label_primary(p1['better'])
    lab2 = G.label_dose(p2['better'])
    cells = {a: (st.median([r['k_snap_cells'] for r in ok
                            if r['arm'] == a and r['k_snap_cells'] is not None])
                 if any(r['arm'] == a and r['k_snap_cells'] is not None for r in ok) else None)
             for a in G.ARM_ORDER}

    v = dict(batch='B-16', n_rows=len(rows), n_ok=len(ok),
             gates=[[g[0], bool(g[1]), str(g[2])] for g in gts],
             gates_pass=sum(1 for g in gts if g[1]), gates_total=len(gts),
             primary=p1, dose=p2, median_unique_cells=cells,
             label=lab[0], label_text=lab[1],
             dose_label=lab2[0], dose_text=lab2[1])
    open(os.path.join(HERE, 'verdict_b16.json'), 'w').write(
        json.dumps(v, indent=1, ensure_ascii=False))

    print('=== 게이트 ===')
    for n_, q_, d_ in gts:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print(f"\nPRIMARY  SHIP-NOW < KGRID1 : {p1['better']}/{p1['n']}  p={p1['p_one_sided']:.4g}  "
          f"상대중앙 {p1['rel_median_pct']:+.2f}%  ({p1['median_a']:.4f} vs {p1['median_b']:.4f})")
    print(f"DOSE     KGRID1  < KGRID2 : {p2['better']}/{p2['n']}  p={p2['p_one_sided']:.4g}  "
          f"상대중앙 {p2['rel_median_pct']:+.2f}%")
    print(f"원자 중앙 격자칸 수: {cells}")
    print(f"\n{v['label']} — {v['label_text']}\n{v['dose_label']} — {v['dose_text']}")


if __name__ == '__main__':
    main()
