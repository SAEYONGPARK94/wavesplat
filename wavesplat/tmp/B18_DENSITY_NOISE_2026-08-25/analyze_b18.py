"""B-18 분석기 — ★**결과가 나오기 전에 작성·커밋된다.** 판정 라벨은 `gates_b18` 만이 낸다.

    python3 tmp/B18_DENSITY_NOISE_2026-08-25/analyze_b18.py

여기서 하는 것은 **자료 배관뿐**이다 — 문턱·라벨·팔·게이트는 재기재하지 않는다.
읽는 것: `results_b18_*.jsonl` (glob) · 쓰는 것: `verdict_b18.json`
⚠`verdict.label` 은 **단일 문자열**(2요인 모양 라벨)이다. 부차 판정은 별도 키로 싣는다
(`B-14` 가 축 둘을 한 verdict 에 넣어 검사기 열이 `--` 로 남은 것을 되풀이하지 않는다).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b18 as G                                     # noqa: E402


def _med(xs):
    return st.median(xs) if xs else None


def main():
    files = sorted(glob.glob(os.path.join(HERE, 'results_b18_*.jsonl')))
    rows = []
    for p in files:
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ok = [r for r in rows if 'error' not in r]

    # ⛔**결과 0행에서 verdict 을 쓰지 않는다.** 빈 판정 파일이 남으면 배치 산출물 검사기가
    # `verdict` 열을 OK 로 읽어 *"분석됐다"* 로 보인다(2026-08-26 감사 M-4).
    if not ok:
        print(f'B-18: 유효 행 0 (전체 {len(rows)}) — verdict 을 쓰지 않는다.', file=sys.stderr)
        for n_, q_, d_ in G.gates(rows):
            print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}', file=sys.stderr)
        return 1

    gts = G.gates(rows)

    # ── 셀 색인: (arm, m_scan, noise) -> {record: row} ──────────────────────
    cell = {}
    for r in ok:
        cell.setdefault((r['arm'], r['m_scan'], r['noise']), {})[r['record']] = r

    def wins(key):
        d = cell.get(key, {})
        return sum(1 for r in d.values() if r['beats_zero']), len(d)

    def nr(key):
        return {k: v[G.FIELD] for k, v in cell.get(key, {}).items()}

    # ── PRIMARY: 네 모서리의 "회복"(참 제로필드를 이긴 레코드 수) ────────────
    corner_key = {c: (G.GRID, m, nz) for c, (m, nz) in G.CORNERS.items()}
    primary, w = {}, {}
    for c in G.CORNER_ORDER:
        k = corner_key[c]
        wi, n = wins(k)
        w[c] = wi
        vals = nr(k)
        primary[c] = dict(m_scan=k[1], noise=k[2], n=n, beats_zero=wi,
                          p_one_sided=(G.sign_test_one_sided(wi, n) if n else None),
                          median_nrmse=_med(list(vals.values())))

    have_all = all(primary[c]['n'] == G.N for c in G.CORNER_ORDER)
    label, label_text = (G.label_shape(w) if have_all
                         else ('NO-DATA', f'모서리 표본 미완: '
                                          f'{ {c: primary[c]["n"] for c in G.CORNER_ORDER} }'))

    # ── SECONDARY: 짝 검정 (기준값이 소거되므로 ZERO_REF 사고에 면역) ────────
    A, B, C, D = (nr(corner_key[c]) for c in G.CORNER_ORDER)
    common = sorted(set(A) & set(B) & set(C) & set(D))

    def paired(a, b, name):
        """`a` 가 `b` 보다 작은(= 더 나은) 레코드 수."""
        ks = sorted(set(a) & set(b))
        wi = sum(1 for k in ks if a[k] < b[k])
        lab = G.label_dose(wi, name) if len(ks) == G.N else (f'{name}-NO-DATA', f'n={len(ks)}')
        return dict(n=len(ks), better=wi,
                    p_one_sided=(G.sign_test_one_sided(wi, len(ks)) if ks else None),
                    rel_median_pct=(_med([(a[k] - b[k]) / b[k] * 100 for k in ks])
                                    if ks else None),
                    label=lab[0], text=lab[1])

    s1 = paired(B, A, 'DENSITY-DOSE@shipnoise')     # 밀도만 올렸을 때
    s2 = paired(C, A, 'NOISE-DOSE@shipdensity')     # 잡음만 낮췄을 때
    s3 = paired(D, A, 'BOTH-DOSE')                  # 둘 다

    # 상호작용 (가법 대비 I = (A−B) − (C−D); I<0 = 상승/synergy)
    inter = [(A[k] - B[k]) - (C[k] - D[k]) for k in common]
    w_syn = sum(1 for x in inter if x < 0)
    ilab = (G.label_interaction(w_syn) if len(common) == G.N
            else ('INTERACTION-NO-DATA', f'n={len(common)}'))
    interaction = dict(n=len(common), synergy_count=w_syn,
                       p_two_sided=(G.sign_test_two_sided(w_syn, len(common))
                                    if common else None),
                       median_contrast=_med(inter), label=ilab[0], text=ilab[1])

    # ── 서술 표 (사전 선언, 검정 아님): 밀도 × 잡음 전 셀 ────────────────────
    table = {}
    for m_ in G.DENSITIES:
        for nz in G.NOISE_LEVELS:
            wi, n = wins((G.GRID, m_, nz))
            vals = nr((G.GRID, m_, nz))
            table[f'm{m_}|s{nz}'] = dict(n=n, beats_zero=wi, median_nrmse=_med(list(vals.values())),
                                         n_obs=(G.DENSITY_CAP_PER_FRAME * m_))
    pc = {}
    for nz in G.NOISE_LEVELS:
        wi, n = wins((G.PC, G.PC_M_SCAN, nz))
        pc[f's{nz}'] = dict(n=n, beats_zero=wi,
                            median_nrmse=_med(list(nr((G.PC, G.PC_M_SCAN, nz)).values())),
                            n_obs=G.PC_CAP)

    # ── ★σ=0 열 재현 대조 (서술 — 검정 아님) ────────────────────────────────
    #   `B-14` 축 B 가 **같은 셀**을 이미 쟀다(사전 노출, PREREG §12).
    #   ⚠2026-08-24 난수 수리로 점 집합이 달라 **값의 재현은 보장되지 않는다** — `G-1c` 참조.
    prior = G.prior_exposure_axisB()
    sigma0 = {}
    for m_ in G.DENSITIES:
        wi, n = wins((G.GRID, m_, G.NZ_LO))
        pb, pn, pobs = prior.get(m_, (None, 0, None))
        sigma0[f'm{m_}'] = dict(n_obs=pobs, b18_beats=wi, b18_n=n,
                                b14_axisB_beats=pb, b14_axisB_n=pn)

    v = dict(batch='B-18', n_rows=len(rows), n_ok=len(ok),
             shards=[os.path.basename(f) for f in files],
             gates=[[g[0], bool(g[1]), str(g[2])] for g in gts],
             gates_pass=sum(1 for g in gts if g[1]), gates_total=len(gts),
             label=label, label_text=label_text,
             primary=primary, corner_wins=w,
             secondary=dict(density_dose=s1, noise_dose=s2, both_dose=s3,
                            interaction=interaction),
             grid_table=table, positive_control_cells=pc,
             sigma0_vs_b14_axisB=sigma0,
             labels_practically_unreachable=list(G.LABELS_PRACTICALLY_UNREACHABLE),
             thresholds=dict(nominal=G.NOMINAL, confirm=G.CONFIRM,
                             family_alpha=G.FAMILY_ALPHA, n=G.N))
    open(os.path.join(HERE, 'verdict_b18.json'), 'w').write(
        json.dumps(v, indent=1, ensure_ascii=False))

    # ── 출력 ────────────────────────────────────────────────────────────────
    print(f'=== 게이트 ({v["gates_pass"]}/{v["gates_total"]}) ===')
    for n_, q_, d_ in gts:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print(f'\n=== 서술 표 · 참 제로필드를 이긴 레코드 수 / 중앙 nRMSE (n={G.N}) ===')
    print('  관측     ' + ' '.join(f'σ={nz:<6}' for nz in G.NOISE_LEVELS))
    for m_ in G.DENSITIES:
        cells_ = [table[f'm{m_}|s{nz}'] for nz in G.NOISE_LEVELS]
        print(f'  {G.DENSITY_CAP_PER_FRAME * m_:>6d}   ' +
              ' '.join(f'{c["beats_zero"]:>2d}/{c["n"]:<2d} '
                       f'{(c["median_nrmse"] if c["median_nrmse"] is not None else float("nan")):.4f}'
                       for c in cells_))
    print(f'  {G.PC_CAP:>6d}*  ' +
          ' '.join(f'{pc[f"s{nz}"]["beats_zero"]:>2d}/{pc[f"s{nz}"]["n"]:<2d} '
                   f'{(pc[f"s{nz}"]["median_nrmse"] if pc[f"s{nz}"]["median_nrmse"] is not None else float("nan")):.4f}'
                   for nz in G.NOISE_LEVELS) + '   (* 양성대조 팔 = 배포 예산)')
    print(f'\n=== PRIMARY · 모서리 (문턱 {G.NOMINAL} 명목 / {G.CONFIRM} 가족단위) ===')
    for c in G.CORNER_ORDER:
        d = primary[c]
        p_ = d['p_one_sided']
        print(f"  {c:9s} m={d['m_scan']:<4} σ={d['noise']:<6} "
              f"회복 {d['beats_zero']:2d}/{d['n']:<2d} "
              f"p={(f'{p_:.4g}' if p_ is not None else 'na'):>9s} "
              f"중앙 nRMSE {(d['median_nrmse'] if d['median_nrmse'] is not None else float('nan')):.4f}")
    print(f'\n★ {label} — {label_text}')
    print('\n=== SECONDARY (짝 검정 · 탐색적) ===')
    for nm, d in (('밀도만  B<A', s1), ('잡음만  C<A', s2), ('둘 다   D<A', s3)):
        rp = d['rel_median_pct']
        print(f"  {nm}: {d['better']:2d}/{d['n']:<2d} p={d['p_one_sided']} "
              f"상대중앙 {(rp if rp is not None else float('nan')):+7.2f}%  -> {d['label']}")
    print(f"  상호작용 I=(A−B)−(C−D): I<0 {interaction['synergy_count']}/{interaction['n']} "
          f"양측 p={interaction['p_two_sided']} -> {interaction['label']} ({interaction['text']})")
    print('\n=== σ=0 열 재현 대조 (서술 · ⚠rng 수리로 비트 동일 불가) ===')
    for m_ in G.DENSITIES:
        d = sigma0[f'm{m_}']
        # ⛔`:>2d` 를 쓰지 않는다 — 참조 파일이 없으면 값이 None 이고 포맷이 죽는다
        # (verdict 은 이미 쓴 뒤라 *"판정은 있는데 출력이 없다"* 가 된다).
        print(f"  m={m_:<4} n_obs={str(d['n_obs']):<6} "
              f"B-18 {d['b18_beats']}/{d['b18_n']} "
              f"vs B-14 축B {d['b14_axisB_beats']}/{d['b14_axisB_n']}"
              + ('   ⚠참조 없음' if d['b14_axisB_n'] == 0 else ''))
    print(f"  ⚠사실상 발화하지 않는 라벨: {', '.join(G.LABELS_PRACTICALLY_UNREACHABLE)}")


if __name__ == '__main__':
    sys.exit(main() or 0)
