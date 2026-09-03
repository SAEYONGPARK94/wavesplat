"""`H3c-LAW` 러너 — ⛔**신규 계측 0회 · 추정기 런 0회**. `H3` 동결 셀 재분석.

Model A (배포)  : rate ≈ s · A          — 곱셈 이득 베르누이
Model C (문턱)  : rate ≈ exp(−T / A)    — 스페클 + 문턱 검출 (M=1)
두 모형 모두 **중앙값 정합**으로 모수 하나를 정한다(⛔최소화 금지).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_h3c as G                                                # noqa: E402

OUT = os.path.join(HERE, 'results_h3c.jsonl')
META = os.path.join(HERE, 'meta_h3c.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')


def mre(pred, rate):
    return float(np.median(np.abs(pred - rate) / rate))


def fit_A(P, rate):
    """중앙값 정합 — `H3` 와 **한 글자도 다르지 않다**."""
    return float(np.median(rate) / np.median(P))


def fit_C(P, rate):
    """`median(exp(−T/P)) = median(rate)` 를 이분법으로 푼다(단조 감소)."""
    target = float(np.median(rate))
    lo, hi = 1e-12, 1.0
    while float(np.median(np.exp(-hi / P))) > target:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.median(np.exp(-mid / P))) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_frozen():
    if not os.path.exists(FROZEN):
        return False, 'FROZEN_md5.txt 없음'
    want = {}
    for line in open(FROZEN, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        h, _, rel = line.partition('  ')
        want[rel.strip()] = h.strip()
    bad = []
    for f in ('cells_h3.npz', 'verdict_h3.json'):
        rel = f'tmp/H3_FRF_2026-09-03/{f}'
        cur = hashlib.md5(open(os.path.join(ROOT, rel), 'rb').read()).hexdigest()
        if rel not in want:
            bad.append(f'미동결:{rel}')
        elif want[rel] != cur:
            bad.append(f'불일치:{rel}')
    return (not bad), ('; '.join(bad) if bad else '2/2 일치')


def main():
    from wavesim.raydrop_threshold import received_power
    import wavesim.raydrop_threshold as RT

    z = np.load(os.path.join(G.H3, 'cells_h3.npz'))
    rate, P, d, th = z['rate'], z['P'], z['d'], z['theta']
    v3 = json.load(open(os.path.join(G.H3, 'verdict_h3.json'), encoding='utf-8'))

    # ★G-IMPORT-A — 모듈의 식(엡실론 포함)과 **비트 동일**한가 = 임포트했다는 직접 증명
    from wavesim.raydrop_uniform import fresnel_reflectance
    thr = np.radians(th)
    dn = d / G.D_REF
    A_formula = fresnel_reflectance(thr) * np.cos(thr) / (dn ** 2 + G.EPS_MODULE)
    A_re = received_power(thr, d, d_ref=G.D_REF)
    import_a_ok = bool(np.array_equal(A_re, A_formula))
    # ★G-IMPORT-B — H3 의 P 와의 차이가 **엡실론이 만들 수 있는 상한** 이하인가.
    #   P_h3 = X/dn^2 · A_re = X/(dn^2+eps) ⟹ 상대차 = eps/(dn^2+eps) ⟹ 상한은 자료가 정한다.
    eps_bound = float(np.max(G.EPS_MODULE / (dn ** 2 + G.EPS_MODULE)))
    import_relerr = float(np.max(np.abs(A_re - P) / np.maximum(np.abs(P), 1e-30)))
    import_b_ok = bool(import_relerr <= eps_bound * (1 + 1e-6))

    s = fit_A(P, rate)
    mre_a = mre(s * P, rate)
    T = fit_C(P, rate)
    pred_c = np.exp(-T / P)
    mre_c = mre(pred_c, rate)

    # ★G-REPRO — H3 를 재현하는가
    repro_relerr = max(abs(mre_a - v3['primary']['MRE']) / v3['primary']['MRE'],
                       abs(s - v3['primary']['global_scale']) / v3['primary']['global_scale'])

    # ★G-FIT — 두 모형 모두 중앙값이 맞았는가
    fit_a = float(np.median(s * P) / np.median(rate))
    fit_c = float(np.median(pred_c) / np.median(rate))

    # G-NEG — 같은 시드·반복수로 두 모형 각각의 치환 귀무
    rng = np.random.default_rng(G.PERM_SEED)
    sh_a, sh_c = [], []
    for _ in range(G.N_PERM):
        Ps = rng.permutation(P)
        sh_a.append(mre(fit_A(Ps, rate) * Ps, rate))
        sh_c.append(mre(np.exp(-fit_C(Ps, rate) / Ps), rate))
    p5_a, p5_c = float(np.percentile(sh_a, 5)), float(np.percentile(sh_c, 5))

    # SECONDARY (보고 전용) — M 스윕. ⛔판정에 쓰지 않는다.
    from scipy.special import gammaincc
    m_sweep = {}
    for M in (1, 2, 4, 8):
        lo, hi, target = 1e-12, 1.0, float(np.median(rate))
        f = lambda t: float(np.median(gammaincc(M, M * t / P)))          # noqa: E731
        while f(hi) > target and hi < 1e12:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        Tm = 0.5 * (lo + hi)
        m_sweep[str(M)] = dict(T=Tm, mre=mre(gammaincc(M, M * Tm / P), rate))

    fok, fmsg = check_frozen()
    excl_ok = G.norm_ws(G.EXCL_B) in G.norm_ws(RT.__doc__)   # ★공백 정규화
    row = dict(model_A=dict(kind='bernoulli-gain', param_name='s', param=s, MRE=mre_a,
                            shuf_p5=p5_a, has_signal=bool(mre_a < p5_a)),
               model_C=dict(kind='threshold-speckle M=1', param_name='T', param=T,
                            MRE=mre_c, shuf_p5=p5_c, has_signal=bool(mre_c < p5_c)),
               m_sweep=m_sweep, n_cells=int(rate.size),
               import_relerr=import_relerr, eps_bound=eps_bound,
               import_a_bit_identical=import_a_ok,
               repro_relerr=float(repro_relerr),
               fit_a=fit_a, fit_c=fit_c, excl_quote_found=bool(excl_ok))
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    meta = dict(frozen_ok=fok, frozen_msg=fmsg,
                repro_ok=bool(repro_relerr < G.REPRO_TOL), repro_relerr=float(repro_relerr),
                import_a_ok=import_a_ok,
                import_a_msg=('bit-identical' if import_a_ok else '⛔모듈 식과 불일치'),
                import_b_ok=import_b_ok, import_relerr=import_relerr,
                eps_bound=eps_bound,
                fit_ok=bool(abs(fit_a - 1) < G.FIT_TOL and abs(fit_c - 1) < G.FIT_TOL),
                fit_a=fit_a, fit_c=fit_c,
                neg_ok=True, neg_msg=f'A p5={p5_a:.4f} · C p5={p5_c:.4f} (n={G.N_PERM})',
                excl_ok=bool(excl_ok), excl_msg=G.EXCL_B, n_cells=int(rate.size))
    json.dump(meta, open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"  Model A  s={s:.6g}  MRE={mre_a:.4f}  shufp5={p5_a:.4f}")
    print(f"  Model C  T={T:.6g}  MRE={mre_c:.4f}  shufp5={p5_c:.4f}")
    print(f"  FROZEN {fmsg} · REPRO {repro_relerr:.2e}")
    print(f"  IMPORT-A bit={import_a_ok} · IMPORT-B relerr {import_relerr:.3e} "
          f"≤ bound {eps_bound:.3e} → {import_b_ok}")
    print(f"  EXCL quote(정규화) = {excl_ok}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
