"""D2 rebuttal probe — was the mixture class ever instantiated, and where does
the -0.017 come from (commitment deficit vs precision deficit)?

A (refit, si=100..): ESS/mode-span of the v2 posterior + POINT-COMMIT arm
   (never scored in e42_blind_recovery) + smult=1 refit (is the mis-scaled
   likelihood leaving precision on the table?).
B (no fit, all scored seas): precision->value curve: oracle-family grids with
   controlled (d1, d2) errors at the blind fit's achieved precision.
Read-only wrt existing scripts/results. Output: tmp/e42_rebuttal_probe_D2.json
"""
import sys, json, time
sys.path.insert(0, 'tmp'); sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, mu_hat_of, ang_err_deg
from e41_regime_prize_ladder import lobe_grid, tp_of_k, TP2_RATIO_NODES, D2_OFF_NODES, W_MARG
from e42_detector_v2 import JointEngine, DIRS, A_GRID, W_GRID, TOPK, MIN_SEP_DEG
from d3_sa_achievability_gap import dispersion_lam
from e29_gennet_C import DEPTH

REC = json.load(open('results/direction3/e42_blind_recovery.json'))
ROWS = {r['si']: r for r in REC['rows']}
TAU = REC['tau']


def lattice_fit(S, smult):
    """Replicates fit_joint's frozen lattice, returning loge arrays too."""
    eng = JointEngine(S, smult=smult)
    keys = [(round(float(a), 3), round(float(m), 5)) for m in DIRS for a in A_GRID]
    eng.build_lobes(keys)
    lut = {k: eng._lobe[k] for k in keys}
    cand1 = [(m, a) for m in DIRS for a in A_GRID]
    idx1 = np.array([lut[(round(float(a), 3), round(float(m), 5))] for (m, a) in cand1])
    loge1 = eng.logev_pairs(idx1, -np.ones(len(idx1), int), np.zeros(len(idx1)))
    order = np.argsort(-loge1)[:TOPK]
    hyp, II, JJ, WW = [], [], [], []
    for oi in order:
        m1, a1 = cand1[oi]; i1 = idx1[oi]
        for m2 in DIRS:
            if abs(np.rad2deg(np.angle(np.exp(1j*(m2-m1))))) < MIN_SEP_DEG:
                continue
            for a2 in A_GRID:
                i2 = lut[(round(float(a2), 3), round(float(m2), 5))]
                for w in W_GRID:
                    hyp.append((m1, a1, m2, a2, float(w)))
                    II.append(i1); JJ.append(i2); WW.append(w)
    loge2 = eng.logev_pairs(II, JJ, WW)
    return eng, hyp, np.array(II), np.array(JJ), np.array(WW), loge1, loge2


def ess(w):
    w = np.asarray(w, float); w = w / w.sum()
    return float(1.0 / (w ** 2).sum())


def probeA(sis, smults=(3.0, 1.0)):
    out = []
    t0 = time.time()
    for si in sis:
        S = build_sea_crosstp(si)
        row = dict(si=si, r2=S['r2'])
        for sm in smults:
            eng, hyp, II, JJ, WW, loge1, loge2 = lattice_fit(S, sm)
            w2 = np.exp(loge2 - loge2.max()); w2 /= w2.sum()
            b2 = int(np.argmax(loge2))
            m1f, a1f, m2f, a2f, wf = hyp[b2]
            # match fitted lobes to truth (same convention as match_errors)
            e_direct = ang_err_deg(m1f, S['d1']) + ang_err_deg(m2f, S['d2'])
            e_swap = ang_err_deg(m1f, S['d2']) + ang_err_deg(m2f, S['d1'])
            if e_direct <= e_swap:
                d1e, d2e = ang_err_deg(m1f, S['d1']), ang_err_deg(m2f, S['d2'])
            else:
                d1e, d2e = ang_err_deg(m2f, S['d1']), ang_err_deg(m1f, S['d2'])
            # mode span of the top-48 (KEEP_HYP) posterior mass
            keep = np.argsort(-w2)[:48]
            wk = w2[keep] / w2[keep].sum()
            d2s = np.array([hyp[j][2] for j in keep])
            far = float(wk[np.abs(np.angle(np.exp(1j*(d2s - m2f)))) > np.deg2rad(30)].sum())
            tag = f"s{int(sm)}"
            row[f'ess2_{tag}'] = ess(w2)            # over full 2-lobe lattice
            row[f'essk_{tag}'] = ess(wk)            # over retained top-48
            row[f'far_{tag}'] = far                 # retained mass >30deg from argmax d2
            row[f'd1e_{tag}'] = d1e
            row[f'd2e_{tag}'] = d2e
            row[f'dlogE_{tag}'] = float(loge2.max() - loge1.max())
            if sm == 3.0:
                # POINT-COMMIT arm (missing from the scored run)
                g_pt = eng.grid_of(II[b2], JJ[b2], WW[b2])
                row['commit_pt'] = solve_with_grid(S, g_pt)
                row['fired'] = bool(row['dlogE_s3'] >= TAU)
        r0 = ROWS[si]
        row.update(cur=r0['cur_grid'], m2=r0['m2'], bma=r0['fit_bma'],
                   bmaF=r0['fit_bmaF'], oracle=r0['oracle'])
        out.append(row)
        print(f"si{si} ESS2 s3 {row['ess2_s3']:7.1f} s1 {row['ess2_s1']:6.1f} | "
              f"far s3 {row['far_s3']:.2f} s1 {row['far_s1']:.2f} | "
              f"d2e s3 {row['d2e_s3']:5.1f} s1 {row['d2e_s1']:5.1f} | "
              f"d1e s3 {row['d1e_s3']:5.1f} s1 {row['d1e_s1']:5.1f} | "
              f"commit {row['commit_pt']:+.3f} bma {row['bma']:+.3f} "
              f"cur {row['cur']:+.3f} orc {row['oracle']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return out


def oracle_jit_grid(S, d1_off, d2_off, rng):
    """Oracle-family grid with controlled direction errors (random signs)."""
    s1 = rng.choice([-1, 1]); s2 = rng.choice([-1, 1])
    d1 = S['d1'] + s1 * np.deg2rad(d1_off)
    d2 = S['d2'] + s2 * np.deg2rad(d2_off)
    L1 = lobe_grid(S['kp'], [(7.0, 3.3, 1.0)], [(d1, 30.0, 1.0)])
    L2 = lobe_grid(S['kp'], [(S['Tp2'], 3.3, 1.0)], [(d2, 30.0, 1.0)])
    return (1 - S['r2']) * L1 + S['r2'] * L2


def probeB(sis):
    """Precision->value: what do (d1,d2) errors at the blind fit's achieved
    level cost, holding ring/w/model structure ORACLE?"""
    rng = np.random.default_rng(20260717)
    out = []
    t0 = time.time()
    for si in sis:
        S = build_sea_crosstp(si)
        r0 = ROWS[si]
        row = dict(si=si, cur=r0['cur_grid'], oracle=r0['oracle'])
        for (d1o, d2o, tag) in [(0, 15, 'j0_15'), (0, 30, 'j0_30'),
                                (10, 15, 'j10_15'), (20, 30, 'j20_30')]:
            row[tag] = solve_with_grid(S, oracle_jit_grid(S, d1o, d2o, rng))
        out.append(row)
        if (si - sis[0]) % 16 == 0:
            print(f"  jit si{si} [{time.time()-t0:.0f}s]", flush=True)
    return out


def summ(rows, key, base='cur'):
    d = np.array([r[key] - r[base] for r in rows], float)
    d = d[np.isfinite(d)]
    return f"{d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f} (win {(d>0).sum()}/{len(d)})"


if __name__ == '__main__':
    n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    sis_fit = list(range(100, 100 + n_fit))
    print("=== PROBE A: ESS / mode-span / point-commit / smult=1 ===")
    A = probeA(sis_fit)
    print("\n=== PROBE B: precision->value curve (all 64 scored seas) ===")
    B = probeB(list(range(100, 164)))
    print("\n--- A summary ---")
    for k in ['ess2_s3', 'ess2_s1', 'far_s3', 'far_s1', 'd2e_s3', 'd2e_s1',
              'd1e_s3', 'd1e_s1']:
        v = np.array([r[k] for r in A], float)
        print(f"  {k:9s}: median {np.median(v):8.2f}  mean {v.mean():8.2f}")
    print(f"  commit_pt - cur : {summ(A, 'commit_pt')}")
    print(f"  commit_pt - bma : {summ(A, 'commit_pt', 'bma')}")
    print(f"  bmaF      - cur : {summ(A, 'bmaF')}")
    print(f"  oracle    - cur : {summ(A, 'oracle')}")
    print("\n--- B summary (vs cur, n=64) ---")
    for tag in ['j0_15', 'j0_30', 'j10_15', 'j20_30']:
        print(f"  oracle_jit {tag:7s}: {summ(B, tag)}")
    print(f"  oracle           : {summ(B, 'oracle')}")
    json.dump(dict(A=A, B=B), open('tmp/e42_rebuttal_probe_D2.json', 'w'),
              indent=1, default=float)
    print("saved -> tmp/e42_rebuttal_probe_D2.json")
