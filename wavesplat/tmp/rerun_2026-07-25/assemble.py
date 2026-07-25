"""Assemble finemesh carried-over tables from the batch JSONLs. Prints old-vs-new for each."""
import json
import os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R24 = os.path.join(HERE, '..', 'rerun_2026-07-24')

TABLE1_EPS = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
]
SPECTRAL_EPS = [
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
]


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(r)
    return out


# ---- WS from ep100 (median-of-5 panel) ----
def ws_panel_by_ep():
    recs = load_jsonl(os.path.join(R24, 'ep100_runs.jsonl'))
    keys = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2', 'Hs_error', 'elapsed_ms']
    by = {}
    for r in recs:
        if 'error' in r:
            continue
        by.setdefault(r['episode'], []).append(r)
    out = {}
    for ep, rs in by.items():
        out[ep] = {k: float(np.median([x[k] for x in rs if k in x and np.isfinite(x.get(k, np.nan))])) for k in keys}
        out[ep]['n_seeds'] = len(rs)
    return out


# ---- WS RMSE_unobs + spectral from batchB (median-of-5) ----
def ws_batchB_by_ep():
    recs = load_jsonl(os.path.join(HERE, 'batchB_ws.jsonl'))
    by = {}
    for r in recs:
        if 'error' in r:
            continue
        by.setdefault(r['episode'], []).append(r)
    return by


# ---- baselines from batchA ----
def batchA_by_method_ep():
    recs = load_jsonl(os.path.join(HERE, 'batchA_baselines.jsonl'))
    out = {}
    for r in recs:
        if 'error' in r:
            continue
        out.setdefault(r['method'], {})[r['episode']] = r
    return out


def col(d, key):
    return np.array([v[key] for v in d.values() if key in v and np.isfinite(v.get(key, np.nan))])


PANEL_MAP = {'nRMSE': 'nRMSE', 'VR': 'Variance_Ratio', 'rho': 'Pearson_rho',
             'SS': 'Skill_Score', 'Dir_R2': 'Directional_R2', 'Hs_err': 'Hs_error'}


def agg_baseline(epdict, eps):
    r = {}
    for short, full in PANEL_MAP.items():
        vals = [epdict[e][full] for e in eps if e in epdict and np.isfinite(epdict[e].get(full, np.nan))]
        r[short] = float(np.mean(vals)) if vals else float('nan')
    for short, full in [('RMSE_unobs', 'RMSE_unobserved'), ('ms', 'ms')]:
        vals = [epdict[e][full] for e in eps if e in epdict and np.isfinite(epdict[e].get(full, np.nan))]
        r[short] = float(np.mean(vals)) if vals else float('nan')
    r['n'] = sum(1 for e in eps if e in epdict)
    return r


def agg_ws(wspanel, eps):
    r = {}
    m = {'nRMSE': 'nRMSE', 'VR': 'Variance_Ratio', 'rho': 'Pearson_rho', 'SS': 'Skill_Score',
         'Dir_R2': 'Directional_R2', 'Hs_err': 'Hs_error', 'ms': 'elapsed_ms'}
    for short, full in m.items():
        vals = [wspanel[e][full] for e in eps if e in wspanel]
        r[short] = float(np.mean(vals)) if vals else float('nan')
    r['n'] = sum(1 for e in eps if e in wspanel)
    return r


def fmt(r, cols):
    return '  '.join(f'{c}={r.get(c, float("nan")):+.4f}' if isinstance(r.get(c), float) else f'{c}={r.get(c)}' for c in cols)


def main():
    wsp = ws_panel_by_ep()
    A = batchA_by_method_ep()
    all100 = sorted(wsp.keys())
    cols = ['nRMSE', 'VR', 'rho', 'SS', 'Dir_R2', 'Hs_err', 'ms']

    print('=' * 90)
    print('TABLE tab:strong_baseline  (FINEMESH)')
    print('=' * 90)
    print('-- Full 100-episode --')
    print(f'  WaveSplat (med-5)  n={agg_ws(wsp, all100)["n"]:3d}  ' + fmt(agg_ws(wsp, all100), cols))
    if 'LETKF' in A:
        print(f'  LETKF(loc50)       n={agg_baseline(A["LETKF"], all100)["n"]:3d}  ' + fmt(agg_baseline(A['LETKF'], all100), cols))
    print('-- 12-episode subset --')
    print(f'  WaveSplat (med-5)  ' + fmt(agg_ws(wsp, TABLE1_EPS), cols))
    for m in ['LETKF', 'ETKF', 'GPR-1', 'GPR-disp']:
        if m in A:
            print(f'  {m:<17s}  ' + fmt(agg_baseline(A[m], TABLE1_EPS), cols))

    print()
    print('=' * 90)
    print('TABLE tab:fivemetric  RMSE_unobs column (FINEMESH)   [OLD: WS 0.262 / ETKF 0.348 / GPR-1 0.294]')
    print('=' * 90)
    # WS RMSE_unobs from batchB median-of-5
    B = ws_batchB_by_ep()
    if B:
        per = {}
        for ep, rs in B.items():
            vals = [x['avg_RMSE_unobserved'] for x in rs if np.isfinite(x.get('avg_RMSE_unobserved', np.nan))]
            if vals:
                per[ep] = float(np.median(vals))
        ws_ru = float(np.mean(list(per.values()))) if per else float('nan')
        # also cross-check WS panel from batchB
        nr = {}
        for ep, rs in B.items():
            v = [x['nRMSE'] for x in rs if np.isfinite(x.get('nRMSE', np.nan))]
            if v:
                nr[ep] = float(np.median(v))
        print(f'  WaveSplat  RMSE_unobs={ws_ru:.4f}  (n_ep={len(per)}, seeds/ep~{np.median([len(v) for v in B.values()]):.0f})  '
              f'[xcheck nRMSE={np.mean(list(nr.values())):.4f} over {len(nr)} ep]')
    else:
        print('  WaveSplat  RMSE_unobs=PENDING (batchB not done)')
    for m in ['ETKF', 'GPR-1']:
        if m in A:
            vals = col(A[m], 'RMSE_unobserved')
            nrm = col(A[m], 'nRMSE')
            print(f'  {m:<9s}  RMSE_unobs={np.mean(vals):.4f}  (n={len(vals)})  [xcheck nRMSE={np.mean(nrm):.4f}]')

    print()
    print('=' * 90)
    print('TABLE Wilcoxon (FINEMESH)   [OLD: WS 0.219/ETKF 0.321/GPR3 0.253; WSvETKF W=194 p=1.1e-15 d=1.16; WSvGPR3 W=1508 p=4.7e-4 d=0.42]')
    print('=' * 90)
    ws_med = {e: wsp[e]['nRMSE'] for e in wsp}
    et = {e: A['ETKF'][e]['nRMSE'] for e in A.get('ETKF', {})}
    g3recs = load_jsonl(os.path.join(HERE, 'batchA2_gpr3.jsonl'))
    g3 = {r['episode']: r['nRMSE'] for r in g3recs if 'error' not in r}
    for name, base in [('WS vs ETKF', et), ('WS vs GPR-3', g3)]:
        common = sorted(set(ws_med) & set(base))
        if len(common) >= 50:
            a = np.array([ws_med[e] for e in common]); b = np.array([base[e] for e in common])
            w, p = wilcoxon(a, b)
            d = (a - b).mean() / (a - b).std(ddof=1)
            diff = b - a
            rng = np.random.default_rng(42)
            boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f'  {name:<12s} n={len(common)} WS={a.mean():.4f} base={b.mean():.4f} '
                  f'meandiff={diff.mean():+.4f} W={w:.0f} p={p:.3g} |d|={abs(d):.3f} CI[{lo:.3f},{hi:.3f}]')
        else:
            print(f'  {name:<12s} PENDING (n_common={len(common)})')

    print()
    print('=' * 90)
    print('TABLE tab:spectral_accuracy (FINEMESH, median-of-5)   [OLD by comp: 1:100/100/.005/.1  2:50/96.9/.001/.2  5:60/93.7/.001/.1  15:50/66.7/.595/26.2  30:38/49.1/.714/26.1]')
    print('=' * 90)
    if B:
        # per-episode median-of-5 of each spec metric, then mean over episodes per complexity
        specmetrics = ['detection_rate', 'energy_detection_rate', 'k_mag_rel_error_mean', 'direction_error_deg_mean']
        per_ep = {}
        for ep in SPECTRAL_EPS:
            rs = B.get(ep, [])
            if not rs:
                continue
            med = {}
            for sm in specmetrics:
                vals = [x['spec'][sm] for x in rs if 'spec' in x and sm in x['spec']
                        and np.isfinite(x['spec'].get(sm, np.nan))]
                med[sm] = float(np.median(vals)) if vals else float('nan')
            med['n_seeds'] = len(rs)
            per_ep[ep] = med
        import re
        for nc in [1, 2, 5, 15, 30]:
            eps_nc = [e for e in SPECTRAL_EPS if f'_{nc}comp_' in e and e in per_ep]
            if not eps_nc:
                print(f'  {nc:2d}comp: PENDING')
                continue
            det = np.mean([per_ep[e]['detection_rate'] for e in eps_nc])
            edet = np.mean([per_ep[e]['energy_detection_rate'] for e in eps_nc])
            ke = np.nanmean([per_ep[e]['k_mag_rel_error_mean'] for e in eps_nc])
            de = np.nanmean([per_ep[e]['direction_error_deg_mean'] for e in eps_nc])
            print(f'  {nc:2d}comp n_ep={len(eps_nc)} det={det:.1%} edet={edet:.1%} |k|={ke:.3f} dir={de:.1f}')
    else:
        print('  PENDING (batchB not done)')


if __name__ == '__main__':
    main()
