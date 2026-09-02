"""`R5-REALGEOM` 분석기 — ⛔런 0회. 규칙은 `gates_r5.py`(→`gates_b17` 임포트) 단일출처."""
import glob, json, os, statistics as st, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r5 as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_r5_*.jsonl'); OUT = os.path.join(HERE, 'verdict_r5.json')


def sgn(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def paired(a, b, keys):
    """`a` 가 `b` 보다 나은 레코드 수(짝 부호검정)."""
    ks = [k for k in keys if k in a and k in b]
    w = sum(1 for k in ks if a[k]['nowcast'] < b[k]['nowcast'])
    rel = [(a[k]['nowcast'] - b[k]['nowcast']) / b[k]['nowcast'] * 100 for k in ks]
    return dict(n=len(ks), better=w, p_one_sided=sgn(w, len(ks)) if ks else None,
                rel_median_pct=(st.median(rel) if rel else None),
                worse_records=[k for k in ks if a[k]['nowcast'] >= b[k]['nowcast']])


def main():
    rows = [json.loads(l) for p in sorted(glob.glob(ROWS)) for l in open(p) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    frz = {}
    for p in sorted(glob.glob(G.B17_GLOB)):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if 'error' not in r:
                    frz[(r['cell'], r['record'])] = r
    const = {k[1]: v for k, v in frz.items() if k[0] == G.REF_ARM}
    anc = {r['record']: r for r in ok if r['cell'] == G.ANCHOR_ARM}
    trt = {r['record']: r for r in ok if r['cell'] == G.TREAT_ARM}
    _, geo_sha = G.measured_geometry()
    gts = G.gates(rows, const, geo_sha); blocked = G.blocked_by(gts)

    lw = sum(1 for k in G.LARGE_BOX if k in trt and trt[k]['nowcast'] < trt[k]['zero_ref'])
    sw = sum(1 for k in G.SMALL_BOX if k in trt and trt[k]['nowcast'] < trt[k]['zero_ref'])
    lab, msg = (G.stratum_label(lw, sw) if not blocked else (None, 'BLOCKED'))
    lw0 = sum(1 for k in G.LARGE_BOX if k in anc and anc[k]['nowcast'] < anc[k]['zero_ref'])
    sw0 = sum(1 for k in G.SMALL_BOX if k in anc and anc[k]['nowcast'] < anc[k]['zero_ref'])

    v = dict(batch='R5-REALGEOM', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R5-REALGEOM :: {lab}' if lab else None), label_msg=msg,
             primary=dict(arm=G.TREAT_ARM, large=f'{lw}/{len(G.LARGE_BOX)}',
                          small=f'{sw}/{len(G.SMALL_BOX)}',
                          gap=lw / len(G.LARGE_BOX) - sw / len(G.SMALL_BOX),
                          anchor_large=f'{lw0}/{len(G.LARGE_BOX)}',
                          anchor_small=f'{sw0}/{len(G.SMALL_BOX)}',
                          note='⚠대형 5는 두 팔이 구조적으로 비트 동일 — PRIMARY 는 소형에만 걸린다'),
             secondary=dict(
                 meas_vs_const=paired(trt, const, G.RECORDS),
                 meas_vs_const_small=paired(trt, const, G.SMALL_BOX),
                 meas_vs_anchor=paired(trt, anc, G.RECORDS),
                 meas_vs_anchor_small=paired(trt, anc, G.SMALL_BOX)),
             sigma_shift={k: dict(
                 r_med=[anc[k]['sigma_stats'].get('r_med'), trt[k]['sigma_stats'].get('r_med')],
                 cos_med=[anc[k]['sigma_stats'].get('cos_med'), trt[k]['sigma_stats'].get('cos_med')],
                 sigma_rms=[anc[k]['sigma_stats'].get('rms'), trt[k]['sigma_stats'].get('rms')],
                 clamp=[anc[k]['sigma_stats'].get('frac_cos_clamped'),
                        trt[k]['sigma_stats'].get('frac_cos_clamped')])
                 for k in G.SMALL_BOX if k in anc and k in trt},
             prediction=dict(written_before_results='GEOM-STRATUM-SURVIVES · REALNOISE-NULL 유지',
                             flip_probability_stated=0.25,
                             outcome=('예측 적중' if lab == 'GEOM-STRATUM-SURVIVES' else '예측 빗나감')),
             disclosure=[
                 '⛔blind 아님 — B-17 의 라벨·계수와 R3b 의 측정 기하를 전부 알고 동결했다.',
                 '⚠스모크에서 BS11a 한 셀을 보았다(1.0260 → 1.2742).',
                 '⛔CONFIRMATORY 아님 — EXPLORATORY.',
                 '★대형 박스 5 는 측정 기하가 없어 두 팔이 구조적으로 동일 ⟹ PRIMARY 는 소형에만 걸린다.',
                 '⛔"12 레코드를 측정 기하로 쟀다" 라 쓰지 마라 — 측정된 것은 BS 7 뿐이다.',
                 '⛔배포 기본값 변경 아님 · B-17 판정 갱신 아님 · 사이트 일반화 금지 · 기제 지목 금지.',
                 '⚠AA·YS·LJ 원점은 여전히 쐐기적합/격자중심이고 높이는 선언값이다.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY 대형 {lw}/{len(G.LARGE_BOX)} · 소형 {sw}/{len(G.SMALL_BOX)} "
          f"· gap {v['primary']['gap']:+.3f}   (앵커: {lw0}/{sw0})")
    for k, d in v['secondary'].items():
        print(f"  SEC {k:22s} {d['better']}/{d['n']} p={d['p_one_sided']} rel={d['rel_median_pct']}")
    print(f"예측: {v['prediction']['outcome']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
