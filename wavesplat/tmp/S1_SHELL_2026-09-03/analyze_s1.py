"""`S1-SHELL` 분석기 — ⛔**결과 전에 작성·동결**."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_s1 as G                                                 # noqa: E402
RES = os.path.join(HERE, 'results_s1.jsonl'); META = os.path.join(HERE, 'meta_s1.json')
OUT = os.path.join(HERE, 'verdict_s1.json')


def _med(a):
    a = sorted(x for x in a if x == x)
    return float('nan') if not a else (a[len(a)//2] if len(a) % 2 else 0.5*(a[len(a)//2-1]+a[len(a)//2]))


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    ok = [r for r in rows if 'error' not in r]
    thr = G.min_w(G.N_EPISODES)

    def wins(sub, key):
        return sum(1 for r in sub if r[f'{key}_ens'] < r[f'{key}_base'])

    out = {}
    for st in ('crowded', 'sparse'):
        sub = [r for r in ok if r['stratum'] == st]
        out[st] = dict(
            n=len(sub),
            far=wins(sub, 'far'), near=wins(sub, 'near'),
            global_=wins(sub, 'glob'), near_field=wins(sub, 'nearfield'),
            far_med_base=_med([r['far_base'] for r in sub]),
            far_med_ens=_med([r['far_ens'] for r in sub]),
            near_med_base=_med([r['near_base'] for r in sub]),
            near_med_ens=_med([r['near_ens'] for r in sub]),
            per_shell={s: sum(1 for r in sub if r['shell_ens'][s] < r['shell_base'][s])
                       for s in G.SHELLS})
    c = out['crowded']
    gate_rows = G.gates(rows, meta); blocked = G.blocked_by(gate_rows)
    lab, msg = (('BLOCKED', '차단: ' + ','.join(blocked)) if blocked
                else G.label(c['far'], c['near'], c['n'], thr))
    lab = 'S1-SHELL :: ' + lab
    v = dict(batch='S1-SHELL', kind='REANALYSIS-OF-FROZEN-DATA', new_runs=0,
             grade='CONFIRMATORY', n_rows=len(rows), threshold=thr,
             threshold_p=G.sign_test_one_sided(thr, G.N_EPISODES),
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked,
             primary=dict(stat='M3 < M1 per shell group (crowded)',
                          far=c['far'], near=c['near'], n=c['n'],
                          far_p=G.sign_test_one_sided(c['far'], c['n']),
                          near_p=G.sign_test_one_sided(c['near'], c['n'])),
             strata=out,
             prediction=dict(written_before_results='ENSEMBLE-BUYS-FAR-SELLS-NEAR',
                             stated_probability=0.45,
                             note='오늘 예측 2연속 과대평가를 감안해 직관 0.6 -> 0.45'),
             label=lab, label_msg=msg,
             disclosures=[
                 '⛔S-ENS 판정 갱신 아님 — 전역 승리(혼잡 55/64)는 그대로다.',
                 '⛔배포 권고 아님 — S-ENS PREREG §5 의 남은 조건 셋은 그대로다.',
                 '⛔D3 의 NEAR 8/36 과 직접 비교 금지(다른 배치·팔·자).',
                 '⛔기제 지목 금지 — 교환이 나와도 "왜" 는 이 자료로 분리 불가.',
                 '⚠시드 하나(1626) · ⚠셸 결합은 **동일 가중**(표본수 가중 자료 없음).',
                 '⛔실측으로 옮기지 마라 — 합성 코퍼스의 성질이다.',
             ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}\n  gates {v["gates_pass"]}  문턱 {thr} (p={v["threshold_p"]:.5f})')
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:46s} {d}")
    for st in ('crowded', 'sparse'):
        d = out[st]
        print(f"  {st:8s} n={d['n']}  전역 {d['global_']}  근 {d['near']}  먼 {d['far']}  "
              f"근거리열 {d['near_field']}")
        print(f"    근 중앙 {d['near_med_base']:.4f}→{d['near_med_ens']:.4f} · "
              f"먼 중앙 {d['far_med_base']:.4f}→{d['far_med_ens']:.4f}")
        print('    셸별 승:', ' '.join(f"{s}:{d['per_shell'][s]}" for s in G.SHELLS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
