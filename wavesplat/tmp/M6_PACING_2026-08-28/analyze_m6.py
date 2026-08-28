"""M-6 분석기 (결과 전 동결) — results_m6.jsonl → verdict_m6.json.

판정 규약은 전부 gates_m6.py 에 있다. 여기는 그 규약을 **기계적으로** 적용만 한다:
  1. 게이트 (NC-1 실패 ⟹ 배치 라벨 VACUOUS-PACING-LEAK 로 즉시 종료)
  2. 배속별 짝 부호검정 (ASYNC5_P{r} vs FROZEN_P1, FIELD=fc_mean) + Holm(8)
     "생존" = Holm 통과 ∧ w ≥ NOMINAL(24)/36
  3. 기제 용량-반응: 배속별 first_apply_lead 중앙 · applied_pre_wait 분포 · 경보 발화율
     단조성 = 적용>0 런이 있는 배속들에서 first_apply_lead 중앙이 r 에 비감소
  4. r_dead_hat = (max_lead − TAU) / t_fit_전런중앙   ← ★실측 유도(손복사 금지)
"""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_m6 as G

ROWS = [json.loads(l) for l in open(os.path.join(HERE, 'results_m6.jsonl'))]


def main():
    ok = [r for r in ROWS if 'error' not in r]
    gate_rows = G.gates(ROWS)
    base = {G.key(r): r for r in ok if r['cell'] == G.PAIR_BASE}
    per_pace, pvals = {}, {}
    for p in G.PACES:
        arm = f'ASYNC5_P{p:g}'
        cell = [r for r in ok if r['cell'] == arm]
        pairs = [(r[G.FIELD], base[G.key(r)][G.FIELD]) for r in cell
                 if G.key(r) in base and r[G.FIELD] is not None]
        w = sum(1 for a, f in pairs if a < f)
        rel = st.median([(a - f) / f * 100.0 for a, f in pairs]) if pairs else float('nan')
        pv = G.sign_test_one_sided(w, len(pairs)) if pairs else 1.0
        fal = [r['first_apply_lead'] for r in cell if r.get('first_apply_lead') is not None]
        per_pace[f'{p:g}'] = dict(
            n=len(pairs), better=w, rel_median_pct=rel, p_one_sided=pv,
            late_better=sum(1 for r in cell if G.key(r) in base
                            and r[G.LATE_FIELD] is not None
                            and r[G.LATE_FIELD] < base[G.key(r)][G.LATE_FIELD]),
            first_apply_lead_median=(st.median(fal) if fal else None),
            n_applied_pre_wait_zero=sum(1 for r in cell if r['applied_pre_wait'] == 0),
            n_noop_warning=sum(1 for r in cell if r['noop_warning_fired']),
            skipped=sorted({r['async_skipped'] for r in cell}))
        pvals[f'{p:g}'] = pv
    hol = G.holm(pvals)
    alive_by_pace = {p: bool(hol[f'{p:g}'] and per_pace[f'{p:g}']['better'] >= G.NOMINAL)
                     for p in G.PACES}
    # 기제 단조성 + r_dead_hat (실측 유도)
    fals = [(p, per_pace[f'{p:g}']['first_apply_lead_median']) for p in G.PACES
            if per_pace[f'{p:g}']['first_apply_lead_median'] is not None]
    mech_monotone = all(b[1] >= a[1] for a, b in zip(fals, fals[1:])) and len(fals) >= 3
    tf = [r['t_fit_median_s'] for r in ok
          if r['cell'].startswith('ASYNC5_') and r.get('t_fit_median_s')]
    t_fit_med = st.median(tf) if tf else float('nan')
    max_lead = max(float(k) for r in ok for k in r['per_lead'])
    rdead_hat = (max_lead - G.TAU) / t_fit_med if tf else float('nan')
    lab, expl = G.label(gate_rows, alive_by_pace, mech_monotone, rdead_hat)
    verdict = dict(
        label=lab, explanation=expl,
        gates=[{'name': n, 'ok': bool(o), 'note': str(x)} for n, o, x in gate_rows],
        per_pace=per_pace, holm=hol,
        alive_by_pace={f'{k:g}': v for k, v in alive_by_pace.items()},
        t_fit_median_s=t_fit_med, max_lead=max_lead, rdead_hat=rdead_hat,
        field=G.FIELD, pair_base=G.PAIR_BASE,
        nominal=G.NOMINAL, confirm=G.CONFIRM, n_rows=len(ROWS))
    out = os.path.join(HERE, 'verdict_m6.json')
    json.dump(verdict, open(out, 'w'), indent=1, ensure_ascii=False)
    print(lab); print(expl)
    for n, o, x in gate_rows:
        print(('PASS' if o else '⛔FAIL'), n, x)
    for k, v in per_pace.items():
        print(f"pace {k}: {v['better']}/{v['n']} {v['rel_median_pct']:+.2f}% "
              f"p={v['p_one_sided']:.2e} first_apply={v['first_apply_lead_median']} "
              f"noop0={v['n_applied_pre_wait_zero']} warn={v['n_noop_warning']}")


if __name__ == '__main__':
    main()
