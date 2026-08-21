"""A2 scoring under the FROZEN convention of PREREG.md (2026-08-06). Reads
tmp/A2_realproto_2026-08-06/cells.json + tmp/wp_real_2026-07-25/wp_real.json,
writes analysis.txt. No convention changes after seeing results."""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon, binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
D = 'tmp/A2_realproto_2026-08-06'
CELLS = ['FF', 'BF', 'FB', 'BB', 'BB400']
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}
L = []


def p(s=''):
    print(s); L.append(s)


def wil(x):
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return float('nan'), float('nan')
    r = wilcoxon(x)
    n = len(x)
    # rank-biserial for signed-rank: r = 1 - 2*W_minus/(n(n+1)/2)  (sign convention: +=x>0)
    from scipy.stats import rankdata
    nz = x[x != 0]
    rk = rankdata(np.abs(nz))
    wp_, wm_ = rk[nz > 0].sum(), rk[nz < 0].sum()
    rb = (wp_ - wm_) / (wp_ + wm_)
    return float(r.pvalue), float(rb)


def report(name, x, tag='learned better'):
    x = np.asarray(x, float)
    pv, rb = wil(x)
    npos = int((x > 0).sum()); n = len(x)
    bp = binomtest(npos, n, 0.5).pvalue
    p(f"  {name:34s} median {np.median(x):+.4f}  mean {x.mean():+.4f}  "
      f"{tag} {npos}/{n}  wilcoxon p={pv:.4g}  sign p={bp:.4g}  rb={rb:+.3f}")
    return dict(median=float(np.median(x)), mean=float(x.mean()), pos=npos, n=n,
                wilcoxon_p=pv, sign_p=float(bp), rank_biserial=rb)


def mde_bootstrap(x, B=10000, seed=7):
    """Smallest location shift s with >=80% power for two-sided Wilcoxon p<0.05,
    resampling the observed paired deltas (centered)."""
    rng = np.random.default_rng(seed)
    xc = np.asarray(x, float) - np.median(x)
    n = len(xc)
    for s in np.arange(0.0, 0.60, 0.005):
        idx = rng.integers(0, n, size=(400, n))
        samp = xc[idx] + s
        ok = 0
        for row in samp:
            try:
                if wilcoxon(row).pvalue < 0.05:
                    ok += 1
            except ValueError:
                pass
        if ok / 400 >= 0.80:
            return float(s)
    return float('nan')


def main():
    C = json.load(open(f'{D}/cells.json'))
    W = {(r['record'], r['win']): r for r in
         json.load(open('tmp/wp_real_2026-07-25/wp_real.json'))['results']}
    R = {(r['record'], r['win']): r for r in C['results']}
    keys = [(rec, w) for rec in RECS for w in range(6)]
    S1 = [k for k in keys if k not in COLLAPSE]
    out = {}

    p('=' * 96)
    p('A2 — real-data comparison protocol hygiene.  Frozen convention: PREREG.md (2026-08-06)')
    p(f"hand seeds {C['hand_seeds']} (shipping median-of-3), 30 windows, wall {C['wall_sec']}s")
    p('=' * 96)

    # --- 0. reproduction control -------------------------------------------------
    p('\n[0] REPRODUCTION CONTROL — FF_exact vs wp_real.json:hand_now')
    dd = [abs(R[k]['FF_exact']['nowcast'] - W[k]['hand_now']) for k in keys]
    p(f"  max |delta| over 30 windows = {max(dd):.3e}   "
      f"({'PASS' if max(dd) < 1e-6 else 'FAIL — protocol invalid'})")
    out['repro_max_abs_delta'] = float(max(dd))

    # --- 1. nulls ---------------------------------------------------------------
    p('\n[1] FLAT-SEA NULL (obligatory)  null_zero = ||gt0||/sg,  null_mean = ||gt0-mu||/sg')
    nz = np.array([R[k]['null_zero'] for k in keys])
    nm = np.array([R[k]['null_mean'] for k in keys])
    p(f"  null_zero over 30 windows: median {np.median(nz):.4f} range [{nz.min():.4f},{nz.max():.4f}]")
    p(f"  null_mean over 30 windows: median {np.median(nm):.4f} range [{nm.min():.4f},{nm.max():.4f}]")
    below = {}
    for c in CELLS:
        below[c] = [k for k in keys if R[k]['cells'][c]['median'] >= R[k]['null_zero']]
    below['learned'] = [k for k in keys if W[k]['learned_now'] >= R[k]['null_zero']]
    for c in CELLS + ['learned']:
        b1 = [k for k in below[c] if k in S1]
        p(f"  {c:7s} fails null: {len(below[c]):2d}/30  (of which in S1: {len(b1)})  "
          + ' '.join(f'{a}_w{b}' for a, b in below[c]))
    bad = set().union(*below.values())
    S2 = [k for k in S1 if k not in bad]
    p(f"  S2 (all arms of all cells beat null_zero, within S1) : n={len(S2)}")
    out['S2'] = [list(k) for k in S2]

    # --- 2. per-cell paired comparison vs learned -------------------------------
    p('\n[2] PRIMARY — paired Delta = nRMSE(hand,cell) - nRMSE(learned); + = learned better')
    p(f'    decision set S1 = 27 normal windows (wp_real frozen exclusion)')
    deltas = {}
    for c in CELLS:
        d = np.array([R[k]['cells'][c]['median'] - W[k]['learned_now'] for k in S1])
        deltas[c] = d
        hm = np.median([R[k]['cells'][c]['median'] for k in S1])
        p(f"  --- cell {c}  (hand median {hm:.4f} | learned median "
          f"{np.median([W[k]['learned_now'] for k in S1]):.4f})")
        out[f'S1_{c}'] = report(f'Delta_{c} (S1,n=27)', d)
    p('\n  wp_real published arm cross-check (learned vs wp_real hand_now, S1):')
    dwp = np.array([W[k]['hand_now'] - W[k]['learned_now'] for k in S1])
    out['S1_wp_real'] = report('Delta_wp_real (S1,n=27)', dwp)

    p('\n  same on S2 (null-passing subset):')
    for c in CELLS:
        d = np.array([R[k]['cells'][c]['median'] - W[k]['learned_now'] for k in S2])
        out[f'S2_{c}'] = report(f'Delta_{c} (S2,n={len(S2)})', d)

    p('\n  collapse windows (excluded from decision, reported separately):')
    for k in sorted(COLLAPSE):
        p(f"    {k[0]}_w{k[1]:d}  null0 {R[k]['null_zero']:.3f} | "
          + ' '.join(f"{c} {R[k]['cells'][c]['median']:.3f}" for c in CELLS)
          + f" | learned {W[k]['learned_now']:.3f}")

    # --- 3. knob separation -----------------------------------------------------
    p('\n[3] KNOB SEPARATION (paired per-window, then median). + = WORSE (higher nRMSE)')
    p('    NOTE: FF is a BUNDLED cell (both GT knobs ON simultaneously).')
    g = lambda c: np.array([R[k]['cells'][c]['median'] for k in S1])
    FF, BF, FB, BB, B4 = (g(c) for c in CELLS)
    out['E_kp_sigGT'] = report('E_kp | sigma=GT  (BF-FF)', BF - FF, 'worse')
    out['E_kp_sigobs'] = report('E_kp | sigma=obs (BB-FB)', BB - FB, 'worse')
    out['E_sig_kpGT'] = report('E_sigma | kp=GT   (FB-FF)', FB - FF, 'worse')
    out['E_sig_kpNone'] = report('E_sigma | kp=None (BB-BF)', BB - BF, 'worse')
    out['interaction'] = report('Interaction (BB-FB)-(BF-FF)', (BB - FB) - (BF - FF), 'super-add')
    out['bundle_BB_FF'] = report('BUNDLE  (BB-FF)', BB - FF, 'worse')
    out['additive_pred'] = report('additive pred (BF-FF)+(FB-FF)', (BF - FF) + (FB - FF), 'worse')
    out['obs_budget'] = report('obs budget 600->400 (BB400-BB)', B4 - BB, 'worse')

    # --- 4. per-record stratification -------------------------------------------
    p('\n[4] PER-RECORD (descriptive only; two-sided sign-test floor p: n=5 -> 0.0625, n=6 -> 0.03125)')
    p(f"  {'rec':7s} {'n':>2s} " + ' '.join(f'{c:>17s}' for c in ['FF', 'BB', 'BB400']))
    for rec in RECS:
        ks = [k for k in S1 if k[0] == rec]
        row = f"  {rec:7s} {len(ks):2d} "
        for c in ['FF', 'BB', 'BB400']:
            d = np.array([R[k]['cells'][c]['median'] - W[k]['learned_now'] for k in ks])
            row += f" med{np.median(d):+.4f}({int((d > 0).sum())}/{len(ks)})"
        p(row)

    # --- 5. power ---------------------------------------------------------------
    p('\n[5] POWER / MDE')
    for n in (27, len(S2)):
        need = next(w for w in range(n // 2, n + 1) if binomtest(w, n, 0.5).pvalue < 0.05)
        p(f"  n={n}: two-sided sign test needs >= {need}/{n} wins for p<0.05 "
          f"(floor p = {binomtest(n, n, 0.5).pvalue:.3g})")
    m = mde_bootstrap(deltas['BB'])
    out['MDE_BB'] = m
    p(f"  MDE (80% power, two-sided Wilcoxon p<0.05, bootstrap of BB paired deltas): "
      f"{m:+.4f} nRMSE")
    p(f"  observed |median Delta_BB| = {abs(np.median(deltas['BB'])):.4f}")

    # --- 6. gap closure ---------------------------------------------------------
    p('\n[6] GAP CLOSURE (secondary, not a decision rule)')
    gFF, gBB = -np.median(deltas['FF']), -np.median(deltas['BB'])
    p(f"  median gap (learned-hand): FF {gFF:+.4f} -> BB {gBB:+.4f}  "
      f"closure {(1 - gBB / gFF) * 100:+.1f}%")
    g4 = -np.median(deltas['BB400'])
    p(f"  incl. obs budget:          FF {gFF:+.4f} -> BB400 {g4:+.4f}  "
      f"closure {(1 - g4 / gFF) * 100:+.1f}%")
    out['gap_FF'] = float(gFF); out['gap_BB'] = float(gBB); out['gap_BB400'] = float(g4)

    # --- 7. verdict -------------------------------------------------------------
    p('\n[7] VERDICT under frozen rule (PREREG section 4), decided on cell BB / S1')
    md = np.median(deltas['BB']); pv = out['S1_BB']['wilcoxon_p']
    v = 'FLIP' if (md > 0 and pv < 0.05) else ('BOUND' if md > 0 else 'HOLD')
    p(f"  median Delta_BB = {md:+.4f}, wilcoxon p = {pv:.4g}  ==>  {v}")
    md4 = np.median(deltas['BB400']); pv4 = out['S1_BB400']['wilcoxon_p']
    v4 = 'FLIP' if (md4 > 0 and pv4 < 0.05) else ('BOUND' if md4 > 0 else 'HOLD')
    p(f"  (BB400 secondary: median {md4:+.4f}, p {pv4:.4g} ==> {v4})")
    out['verdict_BB'] = v; out['verdict_BB400'] = v4

    open(f'{D}/analysis.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(f'{D}/analysis_summary.json', 'w'), indent=1, default=float)
    print(f"\nwrote {D}/analysis.txt")


if __name__ == '__main__':
    main()
