"""B-19 분석기 — ★**결과 0행에서 작성·동결·커밋된다.** 라벨도 게이트도 `gates_b19` 만이 낸다.

`results_b19*.jsonl` 을 glob 으로 전부 모아 `verdict_b19.json`(`label` + `gates`)을 쓴다.
`tmp/_audit/check_batch_artifacts.py` 가 그 두 키를 읽는다.

## 2026-08-26 수리
- **게이트가 여기 있었다.** 초판은 G-0…G-12·N-1·N-2 를 **이 파일 안**에서 만들었다
  (`gates_b19.py` 에 `def gates` 가 0개) ⟹ 규율 #7(단일출처) 이탈이고, 러너가 런 전에
  게이트를 자가진단할 방법이 원리적으로 없었다. 이제 **`gates_b19.gates(rows)` 를 호출만**
  하고, 집계(`aggregate`)·비교(`cmp_pair`/`cmp_zero`)도 같은 모듈의 것을 쓴다.
  ⟹ 이 파일은 **자료 배관과 출력뿐**이다.
- ★**결과 0행이면 verdict 을 쓰지 않고 exit 1** 한다(M-4). 초판은 0행에서도
  `SAMPLE-LOSS-UNDECIDED` verdict 파일을 만들어, **돌지 않은 배치가 산출물 검사기에
  통과하는** 상태를 만들 수 있었다.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b19 as G                                        # noqa: E402


def _med_by_arm(ok, key):
    return {a: (st.median([r[key] for r in ok if r['arm'] == a and r.get(key) is not None])
                if any(r['arm'] == a and r.get(key) is not None for r in ok) else None)
            for a in G.ARM_ORDER}


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b19*.jsonl'))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    ok = [r for r in rows if 'error' not in r]

    # ★M-4: 결과가 없으면 판정 파일을 만들지 않는다. (돌지 않은 배치가 검사기를 통과하면 안 된다)
    if not ok:
        print(f'⛔결과 0행 (읽은 행 {len(rows)}) ⟹ verdict 을 쓰지 않고 종료한다.',
              file=sys.stderr)
        sys.exit(1)

    agg = G.aggregate(rows)
    gts = G.gates(rows, agg)                    # ★게이트 단일출처 = gates_b19
    val, good, zero, n_used = agg['val'], agg['good'], agg['zero'], agg['n_used']
    eps_all = agg['eps_all']

    nominal, confirm, not_earn = G.thresholds(n_used) if n_used else (None, None, None)
    neg1 = G.cmp_zero(val, zero, good, 'SHIP-NOW')
    neg2 = G.cmp_zero(val, zero, good, 'KGRID1')
    neg3 = G.cmp_zero(val, zero, good, 'KGRID2')      # 보고용(판정에 쓰지 않는다)
    neg1_ok = bool(nominal is not None and neg1['n'] and neg1['better'] >= nominal)
    neg2_ok = bool(nominal is not None and neg2['n'] and neg2['better'] >= nominal)

    prim = G.cmp_pair(val, good, 'SHIP-NOW', 'KGRID1')
    dose = G.cmp_pair(val, good, 'KGRID1', 'KGRID2')
    lab = G.label_primary(prim['better'], n_used, neg1_ok, neg2_ok, prim['n_tie'])
    lab2 = G.label_dose(dose['better'], n_used, dose['n_tie'])

    try:
        mesh_counts, mesh_per = G.mesh_census()
    except Exception as exc:                             # noqa: BLE001
        mesh_counts, mesh_per = {'error': repr(exc)}, {}

    v = dict(batch='B-19', n_rows=len(rows), n_ok=len(ok),
             n_episodes=len(eps_all), n_used=n_used,
             excluded=sorted(set(eps_all) - set(good)),
             thresholds=dict(n=n_used, nominal=nominal, confirm=confirm,
                             not_earn=not_earn, family_alpha=G.FAMILY_ALPHA),
             dk=dict(DK1=G.DK1, DK2=G.DK2),
             gates=[[a, b, c] for a, b, c in gts],
             gates_pass=sum(1 for a, b, c in gts if b), gates_total=len(gts),
             primary=prim, dose=dose,
             neg_ship_vs_zero=neg1, neg_kgrid1_vs_zero=neg2, neg_kgrid2_vs_zero=neg3,
             median_zero_nRMSE=(st.median(list(zero.values())) if zero else None),
             median_unique_cells=_med_by_arm(ok, 'k_snap_cells'),
             median_n_atoms=_med_by_arm(ok, 'n_gaussians'),
             median_k_shift=_med_by_arm(ok, 'k_snap_shift_median'),
             # ★표본의 비선형성 실현률 (m-9) — ⛔"HOS M=3" 로 단일 서술하지 마라
             mesh_verdict_counts=mesh_counts, mesh_verdict_per_episode=mesh_per,
             # ★되읽기 증거의 출처를 verdict 에 명시한다 (⛔cfg 항등식이 아니다)
             readback=dict(derived_expected=G.DERIVED_EXPECTED,
                           counter_expected=G.COUNTER_EXPECTED,
                           route_expected=G.ROUTE_EXPECTED,
                           no_derived_evidence=sorted(G.NO_DERIVED_EVIDENCE)),
             label=lab[0], label_text=lab[1],
             dose_label=lab2[0], dose_text=lab2[1])
    with open(os.path.join(HERE, 'verdict_b19.json'), 'w') as fh:
        fh.write(json.dumps(v, indent=1, ensure_ascii=False, default=str))

    print('=== 게이트 ===')
    for name, okk, det in gts:
        print(f'  [{"PASS" if okk else "FAIL"}] {name}: {det}')
    print(f'\n유효 표본 n_used={n_used}/{len(eps_all)}  문턱 NOMINAL {nominal} · '
          f'CONFIRM {confirm} · NOT_EARN {not_earn}')
    print(f'표본 비선형성 실현률(mesh_verdict): {mesh_counts}')
    if prim['n']:
        print(f"PRIMARY  SHIP-NOW < KGRID1 : {prim['better']}/{prim['n']} "
              f"(동점 {prim['n_tie']})  p={prim['p_one_sided']:.4g}  "
              f"상대중앙 {prim['rel_median_pct']:+.2f}%  "
              f"({prim['median_a']:.4f} vs {prim['median_b']:.4f})")
        print(f"DOSE     KGRID1  < KGRID2 : {dose['better']}/{dose['n']} "
              f"(동점 {dose['n_tie']})  p={dose['p_one_sided']:.4g}  "
              f"상대중앙 {dose['rel_median_pct']:+.2f}%")
        print(f"N-1 SHIP<ZERO {neg1['better']}/{neg1['n']} · "
              f"N-2 KGRID1<ZERO {neg2['better']}/{neg2['n']} · "
              f"(보고) KGRID2<ZERO {neg3['better']}/{neg3['n']} · "
              f"ZERO 중앙 {v['median_zero_nRMSE']}")
        print(f"원자 중앙 {v['median_n_atoms']} · 고유 격자칸 중앙 "
              f"{v['median_unique_cells']} · |Δk| 중앙 {v['median_k_shift']}")
    print(f"\n{v['label']} — {v['label_text']}\n{v['dose_label']} — {v['dose_text']}")


if __name__ == '__main__':
    main()
