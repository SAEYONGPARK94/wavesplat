"""B-22 분석기 — ★**결과 0행에서 작성·동결된다.** 라벨도 게이트도 `gates_b22` 만이 낸다.

`results_b22*.jsonl` 을 전부 모아 `verdict_b22.json`(`label` + `gates`)을 쓴다.
⛔결과 0행이면 verdict 을 **쓰지 않고 exit 1** 한다(`B-19` M-4: 돌지 않은 배치가 산출물
검사기를 통과하면 안 된다).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b22 as G                                        # noqa: E402


def _med(ok, arm, key):
    v = [r[key] for r in ok if r['arm'] == arm and r.get(key) is not None]
    return st.median(v) if v else None


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b22*.jsonl'))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        print(f'⛔결과 0행 (읽은 행 {len(rows)}) ⟹ verdict 을 쓰지 않고 종료한다.',
              file=sys.stderr)
        sys.exit(1)

    agg = G.aggregate(rows)
    gts = G.gates(rows, agg)
    val, good, zero, n_used = agg['val'], agg['good'], agg['zero'], agg['n_used']
    cmpx = G.comparisons(val, good)
    negs = {a: G.cmp_zero(val, zero, good, a) for a in G.ARM_ORDER}
    nominal, confirm, not_earn = G.thresholds(n_used) if n_used else (None, None, None)
    lab, txt, title = G.label_primary(cmpx, n_used, negs)
    lab2, txt2 = G.label_secondary(cmpx, n_used)
    try:
        mesh_counts, mesh_per = G.mesh_census(G.episodes())
    except Exception as exc:                                 # noqa: BLE001
        mesh_counts, mesh_per = {'error': repr(exc)}, {}

    v = dict(batch='B-22', n_rows=len(rows), n_ok=len(ok),
             n_episodes=len(agg['eps_all']), n_used=n_used,
             excluded=sorted(set(agg['eps_all']) - set(good)),
             arms=G.ARM_ORDER, arm_defs={a: G.ARMS[a] for a in G.ARM_ORDER},
             deploy_constants_seen=(lambda: G.deploy_constants())(),
             thresholds=dict(n=n_used, nominal=nominal, confirm=confirm,
                             not_earn=not_earn, family_alpha=G.FAMILY_ALPHA),
             gates=[[a, b, c] for a, b, c in gts],
             gates_pass=sum(1 for a, b, c in gts if b), gates_total=len(gts),
             comparisons=cmpx, neg_vs_zero=negs,
             median_zero_nRMSE=(st.median(list(zero.values())) if zero else None),
             median_nRMSE={a: _med(ok, a, 'avg_nRMSE') for a in G.ARM_ORDER},
             median_n_atoms={a: _med(ok, a, 'n_gaussians') for a in G.ARM_ORDER},
             median_n_selected={a: _med(ok, a, 'auto_n_selected') for a in G.ARM_ORDER},
             median_init_time_s={a: _med(ok, a, 'init_time_s') for a in G.ARM_ORDER},
             median_wall_s={a: _med(ok, a, 'wall_s') for a in G.ARM_ORDER},
             mesh_verdict_counts=mesh_counts, mesh_verdict_per_episode=mesh_per,
             readback=dict(derived_expected=G.DERIVED_EXPECTED,
                           deploy_pin=G.DEPLOY_EXPECTED),
             label=lab, label_text=txt, title_decision=title,
             secondary_label=lab2, secondary_text=txt2)
    with open(os.path.join(HERE, 'verdict_b22.json'), 'w') as fh:
        fh.write(json.dumps(v, indent=1, ensure_ascii=False, default=str))

    print('=== 게이트 ===')
    for name, okk, det in gts:
        print(f'  [{"PASS" if okk else "FAIL"}] {name}: {det}')
    print(f'\n유효 표본 n_used={n_used}/{len(agg["eps_all"])}  '
          f'문턱 NOMINAL {nominal} · CONFIRM {confirm} · NOT_EARN {not_earn}')
    for k in ('L1a', 'L1b', 'L2', 'S2a', 'S2b', 'VP_LEARN', 'VP_ANA',
              'SHIP_vs_ANA', 'SHIP_vs_ANA_N'):
        c = cmpx[k]
        if c['n']:
            print(f"  {k:14s} {c['better']}/{c['n']}  p={c['p_one_sided']:.4g}  "
                  f"상대중앙 {c['rel_median_pct']:+.2f}%")
    print('  음성대조(vs 참 제로필드): '
          + ' · '.join(f"{a} {negs[a]['better']}/{negs[a]['n']}" for a in G.ARM_ORDER))
    print(f"\n{v['label']} — {v['label_text']}\n★제목 판정: {v['title_decision']}"
          f"\n{v['secondary_label']} — {v['secondary_text']}")


if __name__ == '__main__':
    main()
