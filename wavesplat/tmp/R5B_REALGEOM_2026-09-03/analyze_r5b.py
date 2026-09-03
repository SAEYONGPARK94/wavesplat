"""`R5b-REALGEOM` 분석기 — ⛔런 0회. 규칙은 `gates_r5b.py` 단일출처."""
import glob, json, os, statistics as st, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r5b as G                                               # noqa: E402
OUT = os.path.join(HERE, 'verdict_r5b.json')


def sgn(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def load(pat, cell=None):
    out = {}
    for p in sorted(glob.glob(pat)):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if 'error' in r:
                    continue
                if cell is None or r['cell'] == cell:
                    out[(r['cell'], r['record'])] = r
    return out


def paired(a, b, keys):
    ks = [k for k in keys if k in a and k in b]
    w = sum(1 for k in ks if a[k]['nowcast'] < b[k]['nowcast'])
    rel = [(a[k]['nowcast'] - b[k]['nowcast']) / b[k]['nowcast'] * 100 for k in ks]
    return dict(n=len(ks), better=w, p_one_sided=(sgn(w, len(ks)) if ks else None),
                rel_median_pct=(st.median(rel) if rel else None))


def main():
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(HERE, 'results_r5b_*.jsonl')))
            for l in open(p) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    r5 = {k[1]: v for k, v in load(G.R5_ROWS).items() if k[0] == 'RANGEANG-MEAS'}
    anc = {k[1]: v for k, v in load(G.R5_ROWS).items() if k[0] == G.ANCHOR_ARM}
    b17 = load(G.B17_GLOB)
    const = {k[1]: v for k, v in b17.items() if k[0] == G.REF_ARM}
    anc_b17 = {k[1]: v for k, v in b17.items() if k[0] == G.ANCHOR_ARM}
    anchor_ok = all(anc[k]['curve'] == anc_b17[k]['curve'] for k in anc if k in anc_b17)
    geo, sha = G.measured_geometry()
    gts = G.gates(rows, geo, sha, r5, const, anchor_ok); blocked = G.blocked_by(gts)

    # 대형은 **앵커에서 승계**한다(구조적 동일, R5 G-IDENT5 가 확립)
    lw = sum(1 for k in G.LARGE_BOX if k in anc and anc[k]['nowcast'] < anc[k]['zero_ref'])
    res = {}
    for arm in G.ARM_ORDER:
        a = {r['record']: r for r in ok if r['cell'] == arm}
        sw = sum(1 for k in G.SMALL_BOX if k in a and a[k]['nowcast'] < a[k]['zero_ref'])
        lab, msg = (G.stratum_label(lw, sw) if not blocked else (None, 'BLOCKED'))
        res[arm] = dict(large_inherited=f'{lw}/{len(G.LARGE_BOX)}',
                        small=f'{sw}/{len(G.SMALL_BOX)}',
                        gap=lw / len(G.LARGE_BOX) - sw / len(G.SMALL_BOX),
                        label=lab, label_msg=msg,
                        vs_const=paired(a, const, G.SMALL_BOX),
                        vs_reflected=paired(a, r5, G.SMALL_BOX),
                        cos_med={k: [round(r5[k]['sigma_stats']['cos_med'], 4),
                                     round(a[k]['sigma_stats']['cos_med'], 4)]
                                 for k in G.SMALL_BOX if k in a and k in r5},
                        sigma_rms={k: [round(r5[k]['sigma_stats']['rms'], 4),
                                       round(a[k]['sigma_stats']['rms'], 4)]
                                   for k in G.SMALL_BOX if k in a and k in r5})
    prim = res[G.FIX_ARM]
    v = dict(batch='R5b-REALGEOM', n_rows=len(rows),
             gates=[[a_, bool(b_), c_] for a_, b_, c_ in gts],
             gates_pass=f'{sum(1 for _, b_, _ in gts if b_)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f"R5b-REALGEOM :: {prim['label']}" if prim['label'] else None),
             label_msg=prim['label_msg'], primary=prim, arms=res,
             prediction=dict(written_before_results='SURVIVES-FIXED (소형 2–3/7)',
                             flip_probability_stated=0.35,
                             mm_arm='⛔예측하지 않는다 — 수준 고정 시 방향이 반대일 수 있다',
                             outcome=('예측 적중' if prim['label'] == 'GEOM-STRATUM-SURVIVES-FIXED'
                                      else '예측 빗나감')),
             supersedes='R5_REALGEOM_2026-09-03 (⛔축 반사 결함, AMENDMENT_1_POSTRESULT.md)',
             disclosure=[
                 '⛔blind 아님 — R5 의 반사 팔 결과와 감사의 cos_med 정정값을 알고 동결했다.',
                 '⛔CONFIRMATORY 아님 — EXPLORATORY.',
                 '⛔B-17 판정 갱신 아님 · 배포 기본값 변경 아님.',
                 '⛔측정 기하는 BS 7 뿐 — "12 레코드를 측정 기하로 쟀다" 금지.',
                 '★대형 5 는 앵커에서 **승계**했다(구조적 동일) ⟹ PRIMARY 는 소형에만 걸린다.',
                 '⚠잡음 시드 1개(42) · 창 1개 · 1 사이트 — 사이트 일반화 금지 · 기제 지목 금지.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a_, b_, c_ in gts:
        if not b_: print(f'  FAIL {a_}  {c_}')
    for arm in G.ARM_ORDER:
        d = res[arm]
        print(f"  {arm:22s} 대형{d['large_inherited']}(승계) 소형{d['small']} gap {d['gap']:+.3f} "
              f"→ {d['label']}")
        print(f"      vs CONST {d['vs_const']['better']}/{d['vs_const']['n']} "
              f"p={d['vs_const']['p_one_sided']} · vs 반사팔 {d['vs_reflected']['better']}/"
              f"{d['vs_reflected']['n']} rel={d['vs_reflected']['rel_median_pct']}")
    print(f"예측: {v['prediction']['outcome']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
