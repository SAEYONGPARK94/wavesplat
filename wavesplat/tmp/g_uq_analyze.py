"""Agent G — ANALYSIS for the CRB-anchored calibrated-UQ gate (ROUTE 1).
Prereg: tmp/AGENT_G_PREREG_crb_calibrated_uq_2026-07-23.md  (frozen before numbers).

Reads the raw npz from g_uq_collect.py. Leave-one-unit-out: fit ONE global scale
alpha per anchor on TRAINING units only, evaluate on the held-out unit.

Bars (all runtime-computable unless marked ORACLE):
  const      1                                        (global constant bar)
  crb        sigma_CRB                                 <-- the pre-registered claim
  emp        sigma_emp (ridge posterior)               (Fucile-2018 LEPrE / Rodgers-OE analogue)
  dist       nearest-obs distance
  const_r    resid_unit                                (runtime per-unit level, NO shape)
  crb_r      resid_unit * sigma_CRB/mean_unit(sigma_CRB)   (runtime level x CRB shape)
  emp_r      resid_unit * sigma_emp/mean_unit(sigma_emp)
  persea     1, alpha refit on held-out unit           ORACLE (needs GT)
  crboracle  sigma_CRB, alpha refit on held-out unit   ORACLE (needs GT)
Exploratory (2 global params, NOT pre-registered): crb_pow = alpha * sigma_CRB**beta.

Usage: python3 tmp/g_uq_analyze.py results/direction3/g_uq_raw_S.npz
"""
import sys
import json
import numpy as np
from scipy import stats

FLOOR = 1e-12
BARS = ['const', 'crb', 'emp', 'dist', 'const_r', 'crb_r', 'emp_r']


def anchors_of(d):
    n = len(d['err'])
    u = d['unit']
    crb = np.maximum(d['crb'].astype(np.float64), FLOOR)
    emp = np.maximum(d['emp'].astype(np.float64), FLOOR)
    dist = np.maximum(d['dist'].astype(np.float64), FLOOR)
    res = np.maximum(d['resid'].astype(np.float64), FLOOR) if 'resid' in d else np.ones(n)
    crb_n = np.empty(n); emp_n = np.empty(n)
    for uu in np.unique(u):                      # within-unit normalisation (runtime-legal)
        m = u == uu
        crb_n[m] = crb[m] / max(crb[m].mean(), FLOOR)
        emp_n[m] = emp[m] / max(emp[m].mean(), FLOOR)
    return dict(const=np.ones(n), crb=crb, emp=emp, dist=dist,
                const_r=res, crb_r=res * crb_n, emp_r=res * emp_n)


def alpha_mle(e, s):
    return float(np.sqrt(np.mean((e / s) ** 2)))


def fit_pow(e, s, betas=np.arange(0.1, 1.61, 0.05)):
    """Exploratory 2-param: sigma = alpha * s**beta; pick beta minimising train NLL."""
    best = None
    for b in betas:
        sb = s ** b
        a = alpha_mle(e, sb)
        sig = a * sb
        nll = np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * (e / sig) ** 2)
        if best is None or nll < best[0]:
            best = (nll, a, float(b))
    return best[1], best[2]


def metrics(e, sig):
    z = e / sig
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2))
    pit = stats.norm.cdf(z)
    q = np.quantile(sig, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(sig, q[1:-1]), 0, 9)
    ms, mr = [], []
    for j in range(10):
        m = b == j
        if m.sum() > 5:
            ms.append(np.mean(sig[m])); mr.append(np.sqrt(np.mean(e[m] ** 2)))
    ms, mr = np.array(ms), np.array(mr)
    ok = len(ms) > 3 and np.all(ms > 0) and np.all(mr > 0) and ms.std() > 0
    return dict(nll=nll,
                cov68=float(np.mean(np.abs(z) < 1.0)),
                cov95=float(np.mean(np.abs(z) < 1.959964)),
                zstd=float(np.std(z)),
                ks=float(stats.kstest(pit, 'uniform').statistic),
                ss_rho=float(stats.spearmanr(ms, mr).correlation) if ok else np.nan,
                ss_slope=float(np.polyfit(np.log(ms), np.log(mr), 1)[0]) if ok else np.nan)


def loso(d, units, permute_crb=False, rng=None):
    A = anchors_of(d)
    e = d['err'].astype(np.float64)
    u = d['unit']
    if permute_crb:
        A = dict(A)
        for key in ('crb', 'crb_r'):
            cp = A[key].copy()
            for uu in units:
                m = u == uu
                cp[m] = rng.permutation(cp[m])
            A[key] = cp
    rows = []
    for held in units:
        te = u == held; tr = ~te
        r = dict(unit=held, n=int(te.sum()))
        for a in BARS:
            al = alpha_mle(e[tr], A[a][tr])
            for k, v in metrics(e[te], al * A[a][te]).items():
                r[f'{a}_{k}'] = v
            r[f'{a}_alpha'] = al
        # exploratory 2-param power calibration on the CRB anchor
        ap, bp = fit_pow(e[tr], A['crb'][tr])
        for k, v in metrics(e[te], ap * A['crb'][te] ** bp).items():
            r[f'crbpow_{k}'] = v
        r['crbpow_alpha'] = ap; r['crbpow_beta'] = bp
        # ORACLE references (need held-out ground truth)
        for nm, anc in (('persea', A['const']), ('crboracle', A['crb'])):
            al_o = alpha_mle(e[te], anc[te])
            for k, v in metrics(e[te], al_o * anc[te]).items():
                r[f'{nm}_{k}'] = v
            r[f'{nm}_alpha'] = al_o
        rows.append(r)
    return rows


def boot_med(v, B=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([np.median(rng.choice(v, len(v))) for _ in range(B)])
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(bs.std())


def report(rows, label, out=None):
    allbars = BARS + ['crbpow', 'persea', 'crboracle']
    print(f"\n{'='*86}\n{label}   (n_units={len(rows)})\n{'='*86}")
    print(f"{'bar':<12s} {'NLL':>9s} {'cov68':>7s} {'cov95':>7s} {'z-std':>7s} "
          f"{'PIT-KS':>7s} {'ss-rho':>7s} {'ss-slp':>7s} {'alpha':>9s}")
    summ = {}
    for a in allbars:
        med = {k: float(np.nanmedian([r[f'{a}_{k}'] for r in rows]))
               for k in ('nll', 'cov68', 'cov95', 'zstd', 'ks', 'ss_rho', 'ss_slope', 'alpha')}
        summ[a] = med
        tag = a + (' *OR' if a in ('persea', 'crboracle') else '')
        print(f"{tag:<12s} {med['nll']:>9.3f} {med['cov68']:>7.3f} {med['cov95']:>7.3f} "
              f"{med['zstd']:>7.3f} {med['ks']:>7.3f} {med['ss_rho']:>7.3f} "
              f"{med['ss_slope']:>7.3f} {med['alpha']:>9.3f}")
    print("\n  paired dNLL, median [95% CI] / win-frac / SE   (positive = the CRB-side bar is BETTER)")
    deltas = {}
    pairs = [('const', 'crb'), ('emp', 'crb'), ('dist', 'crb'),
             ('const_r', 'crb_r'), ('emp_r', 'crb_r'),
             ('const', 'crb_r'), ('const_r', 'crb'),
             ('const', 'crbpow'), ('emp', 'crbpow'),
             ('persea', 'crb'), ('crboracle', 'crb')]
    for a, b in pairs:
        dv = [r[f'{a}_nll'] - r[f'{b}_nll'] for r in rows]
        m, lo, hi, se = boot_med(dv)
        wf = float(np.mean(np.array(dv) > 0))
        under = (se > abs(m) / 2) if m == m else True
        deltas[f'{a}-{b}'] = dict(median=m, ci=[lo, hi], win_frac=wf, se=se, underpowered=bool(under))
        print(f"   dNLL({a:<8s} - {b:<8s}) = {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
              f"win {wf:.2f}  SE {se:.4f}{'   <-- UNDERPOWERED' if under else ''}")
    ac = np.array([r['crboracle_alpha'] for r in rows])
    print(f"\n  per-unit CRB alpha (transfer diagnostic): median {np.median(ac):.2f}  "
          f"IQR[{np.percentile(ac,25):.2f},{np.percentile(ac,75):.2f}]  "
          f"min {ac.min():.2f} max {ac.max():.2f}  max/min = {ac.max()/max(ac.min(),1e-9):.2f}x")
    bp = np.array([r['crbpow_beta'] for r in rows])
    print(f"  exploratory power-calib beta: median {np.median(bp):.2f} "
          f"[{bp.min():.2f},{bp.max():.2f}]  (1.0 = pure scale)")
    summ['alpha_spread'] = dict(median=float(np.median(ac)), min=float(ac.min()),
                                max=float(ac.max()), ratio=float(ac.max() / max(ac.min(), 1e-9)))
    summ['beta'] = dict(median=float(np.median(bp)), min=float(bp.min()), max=float(bp.max()))
    if out:
        json.dump(dict(rows=rows, summary=summ, deltas=deltas), open(out, 'w'), indent=1, default=float)
        print(f"  saved -> {out}")
    return summ, deltas


if __name__ == '__main__':
    path = sys.argv[1]
    d = dict(np.load(path, allow_pickle=True))
    units = sorted(set(d['unit'].tolist()))
    print(f"loaded {path}: rows={len(d['err'])} units={len(units)}")
    tag = path.split('g_uq_raw_')[-1].replace('.npz', '')
    report(loso(d, units), f'ARM {tag} — LOSO calibration gate (ROUTE 1)',
           out=path.replace('g_uq_raw_', 'g_uq_gate_').replace('.npz', '.json'))

    kp_u = {u: float(np.median(d['kp'][d['unit'] == u])) for u in units}
    med_kp = np.median(list(kp_u.values()))
    lo_u = [u for u in units if kp_u[u] <= med_kp]
    hi_u = [u for u in units if kp_u[u] > med_kp]
    for nm, uu in (('UNDER-DET (kp<=med)', lo_u), ('WELL-DET (kp>med)', hi_u)):
        if len(uu) >= 3:
            m = np.isin(d['unit'], uu)
            sub = {k: (v[m] if hasattr(v, '__len__') and len(v) == len(d['err']) else v)
                   for k, v in d.items()}
            report(loso(sub, uu), f'ARM {tag} — {nm}')

    rng = np.random.default_rng(0)
    rows_p = loso(d, units, permute_crb=True, rng=rng)
    for a, b in (('const', 'crb'), ('const_r', 'crb_r')):
        dv = [r[f'{a}_nll'] - r[f'{b}_nll'] for r in rows_p]
        m, lo, hi, se = boot_med(dv)
        print(f"  PERMUTATION CONTROL (sigma_CRB shuffled within unit): "
              f"dNLL({a}-{b}) = {m:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
