"""`H3-FRF` 분석기 — ⛔런 0회. 규칙은 `gates_h3.py` 단일출처."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_h3 as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_h3.jsonl'); CELLS = os.path.join(HERE, 'cells_h3.npz')
OUT = os.path.join(HERE, 'verdict_h3.json')


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    z = np.load(CELLS)
    rate, P, d, th = z['rate'], z['P'], z['d'], z['theta']
    rng = np.random.default_rng(G.PERM_SEED)

    # ★전역 스케일 하나 (PREREG §1) — 셀별·파일별 정합 금지
    s = float(np.median(rate) / np.median(P))
    mre = float(np.median(np.abs(s * P - rate) / rate))

    # ★G-NEG — 모형을 셀 간 치환하면 더 나빠야 한다
    shuf = []
    for _ in range(G.N_PERM):
        Ps = rng.permutation(P)
        ss = float(np.median(rate) / np.median(Ps))
        shuf.append(float(np.median(np.abs(ss * Ps - rate) / rate)))
    mre_shuf_p5 = float(np.percentile(shuf, 5))

    # ★G-POS — 측정 반환율에 구조가 있는가 (입사각 십분위 퍼짐 vs 섞은 귀무)
    q = np.quantile(th, np.linspace(0, 1, 11)); q[0] -= 1e-9; q[-1] += 1e-9
    b = np.digitize(th, q) - 1
    dec = np.array([rate[b == i].mean() for i in range(10) if (b == i).sum() > 10])
    spread = float(dec.max() - dec.min())
    null = []
    for _ in range(G.N_PERM):
        rs = rng.permutation(rate)
        dn = np.array([rs[b == i].mean() for i in range(10) if (b == i).sum() > 10])
        null.append(float(dn.max() - dn.min()))
    pos_null_p95 = float(np.percentile(null, 95))

    from scipy.stats import spearmanr
    summary = dict(mre=mre, scale=s, mre_shuf_p5=mre_shuf_p5,
                   pos_spread=spread, pos_null_p95=pos_null_p95)
    gts = G.gates(rows, summary); blocked = G.blocked_by(gts)
    lab, msg = (G.label(mre) if not blocked else (None, 'BLOCKED'))

    def dec_curve(x, lo, hi, nb=8):
        e = np.linspace(lo, hi, nb + 1)
        return [[float(e[i]), float(e[i+1]), int(((x >= e[i]) & (x < e[i+1])).sum()),
                 float(rate[(x >= e[i]) & (x < e[i+1])].mean())
                 if ((x >= e[i]) & (x < e[i+1])).sum() > 10 else None] for i in range(nb)]

    v = dict(batch='H3-FRF', n_rows=len(rows), n_cells=int(len(rate)),
             gates=[[a, bool(b_), c] for a, b_, c in gts],
             gates_pass=f'{sum(1 for _, b_, _ in gts if b_)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'H3-FRF :: {lab}' if lab else None), label_msg=msg,
             primary=dict(MRE=mre, threshold=G.MRE_THRESH, global_scale=s,
                          threshold_source='thesis_main.tex:9711-9712 (제5장 H3 자신의 수)'),
             controls=dict(pos_spread=spread, pos_null_p95=pos_null_p95,
                           mre_shuffled_p5=mre_shuf_p5, n_perm=G.N_PERM),
             secondary=dict(spearman=float(spearmanr(P, rate).statistic),
                            rate_vs_range=dec_curve(d, float(d.min()), float(np.quantile(d, .99))),
                            rate_vs_theta=dec_curve(th, float(th.min()), float(th.max())),
                            valid_frac=[min(r['valid_frac'] for r in rows if 'error' not in r),
                                        max(r['valid_frac'] for r in rows if 'error' not in r)]),
             prediction=dict(written_before_results='OBSMODEL-NOT-FAITHFUL',
                             stated_probability=0.85,
                             outcome=('예측 적중' if lab == 'OBSMODEL-NOT-FAITHFUL'
                                      else '예측 빗나감')),
             disclosure=[
                 '⛔정확도 주장 아님 — 추정기 런 0회. 어느 배치의 계수도 갱신하지 않는다.',
                 '⛔⛔레짐 교락: 이 자료는 **쇄파대**(거품·퇴적물 지배)이고 모형은 **심해 정반사**를 '
                 '겨눈다 = **역방향 전이** ⟹ 허용 진술은 "이 레짐에서 이 모형은 반환율을 예측하지 '
                 '못한다" 까지다. ⛔"모형이 틀렸다" 로 넓히지 마라.',
                 '⛔측정 반환율은 **상한**이다 — 한 번도 반환하지 않은 빔은 점군에 없다.',
                 '⛔H3 의 "노이즈 분산" 절반은 이 배치가 재지 않는다 — raydrop 만 잰다.',
                 '⛔"수조를 대체했다" 로 쓰지 마라 — 대체하는 것은 H3 의 raydrop 절반이고 '
                 'G-8(개구)·SNR 은 별개다.',
                 '⚠8 세션이 사실상 2일 · 한 사이트 ⟹ 사이트·계절 일반화 금지.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b_, c in gts:
        if not b_: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY MRE = {mre:.4f} (문턱 {G.MRE_THRESH}) · 전역 스케일 {s:.5g} · 셀 {len(rate)}")
    print(f"양성대조 퍼짐 {spread:.4f} vs 귀무p95 {pos_null_p95:.4f}")
    print(f"음성대조 섞음 MRE p5 {mre_shuf_p5:.4f} vs 실제 {mre:.4f}")
    print(f"Spearman {v['secondary']['spearman']:.4f}")
    print(f"예측: {v['prediction']['outcome']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
