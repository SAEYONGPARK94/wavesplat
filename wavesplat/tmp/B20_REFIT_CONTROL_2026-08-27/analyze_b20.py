"""B-20 분석기 — ★**결과 0행에서 작성·동결된다.** 라벨도 게이트도 `gates_b20` 만이 낸다.

`results_b20*.jsonl` 을 glob 으로 모아 `verdict_b20.json`(`label` + `gates`)을 쓴다.
`tmp/_audit/check_batch_artifacts.py` 가 그 두 키를 읽는다.

이 파일에는 **판정 논리가 0줄**이다 — 자료 배관과 출력뿐이다
(`gates()`·`aggregate()`·`cmp_pair`·`cmp_zero`·`decompose`·`magnitudes`·`label_a`·`label_b`
전부 `gates_b20`).  ★결과 0행이면 verdict 을 쓰지 않고 `exit 1` 한다(M-4 부류 방지).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b20 as G                                          # noqa: E402


def _med_by_arm(ok, key):
    return {a: (st.median([r[key] for r in ok if r['arm'] == a and r.get(key) is not None])
                if any(r['arm'] == a and r.get(key) is not None for r in ok) else None)
            for a in G.ARM_ORDER}


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b20*.jsonl'))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        print(f'⛔결과 0행 (읽은 행 {len(rows)}) ⟹ verdict 을 쓰지 않고 종료한다.',
              file=sys.stderr)
        sys.exit(1)

    agg = G.aggregate(rows)
    gts = G.gates(rows, agg)
    val, good, zero, n_used = agg['val'], agg['good'], agg['zero'], agg['n_used']
    eps_all = agg['eps_all']

    nominal, confirm, not_earn = G.thresholds(n_used) if n_used else (None, None, None)
    neg1 = G.cmp_zero(val, zero, good, 'SHIP-NOW')
    neg2 = G.cmp_zero(val, zero, good, 'NOOP-REFIT')
    neg3 = G.cmp_zero(val, zero, good, 'KGRID1')          # 보고용(판정에 쓰지 않는다)
    neg1_ok = bool(nominal is not None and neg1['n'] and neg1['better'] >= nominal)
    neg2_ok = bool(nominal is not None and neg2['n'] and neg2['better'] >= nominal)
    g15_ok = G.gate_ok(gts, 'G-15 ')

    A = G.cmp_pair(val, good, 'SHIP-NOW', 'NOOP-REFIT')   # PRIMARY-A: 재적합 단독
    B = G.cmp_pair(val, good, 'NOOP-REFIT', 'KGRID1')     # PRIMARY-B: 교락 없는 절제
    T = G.cmp_pair(val, good, 'SHIP-NOW', 'KGRID1')       # 총효과 (재현 확인 — 주장 아님)
    mags = G.magnitudes(val, good)
    dec = G.decompose(val, good)

    lab = G.label_a(A['better'], n_used, A['n_tie'], neg1_ok, g15_ok,
                    bool(mags['magnitude_unmoved']))
    labb = G.label_b(B['better'], n_used, B['n_tie'], neg1_ok, neg2_ok, g15_ok)

    try:
        mesh_counts, mesh_per = G.mesh_census()
    except Exception as exc:                                   # noqa: BLE001
        mesh_counts, mesh_per = {'error': repr(exc)}, {}

    v = dict(batch='B-20', n_rows=len(rows), n_ok=len(ok),
             n_episodes=len(eps_all), n_used=n_used,
             excluded=sorted(set(eps_all) - set(good)),
             thresholds=dict(n=n_used, nominal=nominal, confirm=confirm,
                             not_earn=not_earn, family_alpha=G.FAMILY_ALPHA,
                             note='A 의 검출은 NOMINAL(α=0.05, 무보정) · '
                                  'A 의 HELPS 가지와 B 는 CONFIRM(가족단위)'),
             dk=dict(DK0=G.DK0, DK1=G.DK1, phase_bound=G.PHASE_BOUND,
                     derivation=G.derive_dk0()),
             gates=[[a, b, c] for a, b, c in gts],
             gates_pass=sum(1 for a, b, c in gts if b), gates_total=len(gts),
             primary_a=A, primary_b=B, total_effect_reproduction=T,
             magnitudes=mags, log_decomposition=dec,
             neg_ship_vs_zero=neg1, neg_noop_vs_zero=neg2, neg_kgrid1_vs_zero=neg3,
             median_zero_nRMSE=(st.median(list(zero.values())) if zero else None),
             median_n_atoms=_med_by_arm(ok, 'n_gaussians'),
             median_unique_cells=_med_by_arm(ok, 'k_snap_cells'),
             median_k_shift=_med_by_arm(ok, 'k_snap_shift_median'),
             median_k_dpsi_max=_med_by_arm(ok, 'k_dpsi_max'),
             min_k_bit_identical_frac=_med_by_arm(ok, 'k_bit_identical_frac'),
             median_w_abs_change_max=_med_by_arm(ok, 'w_abs_change_max'),
             median_phi_abs_change_max=_med_by_arm(ok, 'phi_abs_change_max'),
             mesh_verdict_counts=mesh_counts, mesh_verdict_per_episode=mesh_per,
             readback=dict(derived_expected=G.DERIVED_EXPECTED,
                           counter_expected=G.COUNTER_EXPECTED,
                           route_expected=G.ROUTE_EXPECTED,
                           no_derived_evidence=sorted(G.NO_DERIVED_EVIDENCE)),
             label=lab[0], label_text=lab[1],
             label_b=labb[0], label_b_text=labb[1])
    with open(os.path.join(HERE, 'verdict_b20.json'), 'w') as fh:
        fh.write(json.dumps(v, indent=1, ensure_ascii=False, default=str))

    print('=== 게이트 ===')
    for name, okk, det in gts:
        print(f'  [{"PASS" if okk else "FAIL"}] {name}: {det}')
    print(f'\n유효 표본 n_used={n_used}/{len(eps_all)}  문턱 NOMINAL {nominal} · '
          f'CONFIRM {confirm} · NOT_EARN {not_earn}')
    print(f'표본 비선형성 실현률(mesh_verdict): {mesh_counts}')
    if A['n']:
        for tag, c in (('A  SHIP-NOW   < NOOP-REFIT', A),
                       ('B  NOOP-REFIT < KGRID1    ', B),
                       ('T  SHIP-NOW   < KGRID1  (재현)', T)):
            print(f"{tag}: {c['better']}/{c['n']} (동점 {c['n_tie']})  "
                  f"p={c['p_one_sided']:.4g}  상대중앙 {c['rel_median_pct']:+.3f}%  "
                  f"({c['median_a']:.5f} vs {c['median_b']:.5f})")
        print(f"크기 결정: MAG_MAIN {mags['mag_main']:+.3f}% · MAG_CLEAN "
              f"{mags['mag_clean']:+.3f}% · |이동| {mags['shift_pp']:.3f} pp vs "
              f"PP_BAND {mags['pp_band']:.3f} pp ⟹ "
              f"{'밴드 안(옛 크기 유지)' if mags['magnitude_unmoved'] else '밴드 밖(새 크기로 교체)'}")
        if dec:
            print(f"로그 분해: d_total {dec['median_d_total']:+.4f} = d_refit "
                  f"{dec['median_d_refit']:+.4f} + d_quant {dec['median_d_quant']:+.4f} "
                  f"(가법 오차 {dec['additivity_max_abs_err']:.2e}) · 담지자 몫 "
                  f"{dec['share_of_medians']} · 에피소드별 몫 중앙 {dec['median_of_shares']} "
                  f"(가드 제외 {dec['n_guard_dropped']})")
        print(f"N-1 SHIP<ZERO {neg1['better']}/{neg1['n']} · "
              f"N-2 NOOP<ZERO {neg2['better']}/{neg2['n']} · "
              f"(보고) KGRID1<ZERO {neg3['better']}/{neg3['n']} · "
              f"ZERO 중앙 {v['median_zero_nRMSE']}")
        print(f"원자 중앙 {v['median_n_atoms']} · |Δk| 중앙 {v['median_k_shift']} · "
              f"Δψ_max 중앙 {v['median_k_dpsi_max']} · "
              f"파수 비트동일 비율 {v['min_k_bit_identical_frac']}")
        print(f"재적합 크기: |Δw|max 중앙 {v['median_w_abs_change_max']} · "
              f"|Δφ|max 중앙 {v['median_phi_abs_change_max']}")
    print(f"\nA: {v['label']} — {v['label_text']}")
    print(f"B: {v['label_b']} — {v['label_b_text']}")


if __name__ == '__main__':
    main()
