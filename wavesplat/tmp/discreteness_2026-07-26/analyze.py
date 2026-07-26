"""Assemble the discreteness experiment per tmp/discreteness_2026-07-26/PREREG.md §4.

Frozen decision cells (evaluated in order):
  0 INVALID   : R_A > -0.20  (control fails to replicate)
  1 CONFIRM   : R_A <= -0.20 AND (R_C > -0.10 OR p_C >= 0.05)
  2 REFUTE    : R_C <= -0.20 AND p_C < 0.05
  3 AMBIGUOUS : anything else
R = median over the cell's episodes of (cond_i - ws_best_i)/ws_best_i.
p = one-sided Wilcoxon signed-rank on the paired absolute deltas, H1 cond < ws_best.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

D = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.dirname(D)))
WS = f'{D}/ws_runs.jsonl'
COND = f'{D}/cond_runs.jsonl'
SEC = f'{D}/secondary.jsonl'
REAL_BF = 'tmp/figs_opus_real_2026-07-26/bf_ratio_real.json'
CELLS = ['a', 'b', 'c']
CELLNAME = {'a': 'A discrete control (M=1, sig_sub=0)',
            'b': 'B mild spread (M=31, sig_sub=10deg)',
            'c': 'C strong spread (M=31, sig_sub=28deg + omega jitter)'}
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (10, 11, 12, 13, 14) for r in ('uniform', 'slope')]


def load():
    ws = defaultdict(lambda: {'init': [], 'track': [], 'ng': []})
    cond = defaultdict(list)
    errs = []
    for line in open(WS):
        d = json.loads(line)
        if d['err']:
            errs.append(('ws', d['cell'], d['episode'], d['seed'], d['err'])); continue
        ws[(d['cell'], d['episode'])]['init'].append(d['init_nrmse'])
        ws[(d['cell'], d['episode'])]['track'].append(d['avg_nRMSE'])
        ws[(d['cell'], d['episode'])]['ng'].append(d.get('n_gaussians'))
    for line in open(COND):
        d = json.loads(line)
        if d['err']:
            errs.append(('cond', d['cell'], d['episode'], d['seed'], d['err'])); continue
        cond[(d['cell'], d['episode'])].append(d['cond_polish_nrmse_Hs'])
    sec = {}
    if os.path.exists(SEC):
        for line in open(SEC):
            d = json.loads(line)
            sec[(d['cell'], d['episode'])] = d
    return ws, cond, sec, errs


def cell_rows(cell, ws, cond):
    rows = []
    for ep in EPS:
        wi = ws[(cell, ep)]['init']; wt = ws[(cell, ep)]['track']; cm = cond[(cell, ep)]
        if len(wi) != 5 or len(cm) != 5:
            print(f"  INCOMPLETE {cell} {ep}: ws={len(wi)} cond={len(cm)}")
        mi, mt, mc = np.median(wi), np.median(wt), np.median(cm)
        best = float(min(mi, mt))
        rows.append(dict(cell=cell, ep=ep, arm='slope' if 'slope' in ep else 'uniform',
                         ws_init=float(mi), ws_init_sd=float(np.std(wi)),
                         ws_track=float(mt), ws_track_sd=float(np.std(wt)),
                         ws_best=best, cond=float(mc), cond_sd=float(np.std(cm)),
                         d_abs=float(mc - best), d_rel=float((mc - best) / best),
                         ng=float(np.median([g for g in ws[(cell, ep)]['ng'] if g]))
                         if any(ws[(cell, ep)]['ng']) else float('nan')))
    return rows


def stat(rows):
    c = np.array([r['cond'] for r in rows]); b = np.array([r['ws_best'] for r in rows])
    d = c - b; rel = np.array([r['d_rel'] for r in rows])
    try:
        _, p = wilcoxon(c, b, alternative='less', zero_method='wilcox')
    except Exception:
        p = float('nan')
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    wp = ranks[nz > 0].sum(); wm = ranks[nz < 0].sum(); tot = wp + wm
    return dict(n=len(d), cond_mean=float(c.mean()), cond_med=float(np.median(c)),
                ws_mean=float(b.mean()), ws_med=float(np.median(b)),
                med_abs=float(np.median(d)), mean_abs=float(d.mean()),
                R=float(np.median(rel)), R_mean=float(rel.mean()),
                wins=int((d < 0).sum()), p=float(p),
                rrb=float((wm - wp) / tot) if tot else float('nan'))


def main():
    ws, cond, sec, errs = load()
    print("=== data integrity ===")
    print(f"  errored runs: {len(errs)}")
    for e in errs[:10]:
        print("   ", e)

    allrows, S = {}, {}
    for c in CELLS:
        allrows[c] = cell_rows(c, ws, cond)
        S[c] = stat(allrows[c])

    print("\n=== per-cell arm numbers (median-of-5 both arms) ===")
    hdr = (f"{'cell':6s}{'n':>3s}{'cond mean':>11s}{'cond med':>10s}{'ws_best mean':>14s}"
           f"{'ws_best med':>13s}{'medAbsD':>10s}{'R(med rel)':>12s}{'wins':>6s}{'p':>11s}")
    print(hdr)
    for c in CELLS:
        s = S[c]
        print(f"{c:6s}{s['n']:3d}{s['cond_mean']:11.4f}{s['cond_med']:10.4f}"
              f"{s['ws_mean']:14.4f}{s['ws_med']:13.4f}{s['med_abs']:+10.4f}"
              f"{s['R']*100:+11.1f}%{s['wins']:>4d}/{s['n']}{s['p']:11.2e}")

    RA, RC, pC = S['a']['R'], S['c']['R'], S['c']['p']
    if RA > -0.20:
        verdict = 'INVALID'
        why = (f"control cell A failed to replicate (R_A = {RA*100:+.1f}% > -20%); "
               "generation changed too much -- not evidence in either direction")
    elif (RC > -0.10) or (pC >= 0.05):
        verdict = 'CONFIRM'
        why = (f"R_A = {RA*100:+.1f}% (replicates) and strong-spread cell "
               f"R_C = {RC*100:+.1f}%, p_C = {pC:.3g} "
               f"({'shrank below half' if RC > -0.10 else 'lost significance'})")
    elif (RC <= -0.20) and (pC < 0.05):
        verdict = 'REFUTE'
        why = (f"R_C = {RC*100:+.1f}% <= -20% and p_C = {pC:.3g} < 0.05 -- the advantage "
               "survives the continuum")
    else:
        verdict = 'AMBIGUOUS'
        why = (f"R_C = {RC*100:+.1f}%, p_C = {pC:.3g} -- between the frozen cells; "
               "no narrative rescue")
    print(f"\n=== PRE-REGISTERED VERDICT: {verdict} ===")
    print(f"  {why}")
    print(f"  monotonicity read-out (not a decision variable): "
          f"R_A {S['a']['R']*100:+.1f}% -> R_B {S['b']['R']*100:+.1f}% "
          f"-> R_C {S['c']['R']*100:+.1f}%")
    print(f"  reference (wp_crowded BF4/15comp, seeds 0-4): R = -64.6%")

    print("\n=== full paired tables ===")
    for c in CELLS:
        print(f"\n-- cell {c}: {CELLNAME[c]}")
        print(f"{'episode':40s}{'cond(sd)':>15s}{'WSinit(sd)':>15s}{'WStrk(sd)':>15s}"
              f"{'ws_best':>9s}{'dAbs':>8s}{'dRel':>9s}{'N_g':>6s}")
        for r in allrows[c]:
            print(f"{r['ep']:40s}{r['cond']:9.3f}({r['cond_sd']:.3f})"
                  f"{r['ws_init']:9.3f}({r['ws_init_sd']:.3f})"
                  f"{r['ws_track']:9.3f}({r['ws_track_sd']:.3f})"
                  f"{r['ws_best']:9.3f}{r['d_abs']:+8.3f}{r['d_rel']*100:+8.1f}%"
                  f"{r['ng']:6.0f}")

    print("\n=== arm split within cell (uniform / slope) ===")
    for c in CELLS:
        for arm in ('uniform', 'slope'):
            sub = [r for r in allrows[c] if r['arm'] == arm]
            s = stat(sub)
            print(f"  {c} {arm:8s} n={s['n']} cond {s['cond_mean']:.3f} "
                  f"ws_best {s['ws_mean']:.3f} R {s['R']*100:+.1f}% wins {s['wins']}/{s['n']}")

    # ── SENSITIVITY: realized-Hs renormalization (NOT the frozen primary) ────
    # Both arms normalize by the stored nominal Hs=1.2. The realized window variance scatters
    # (cell A has deterministic component amplitudes, sd 0.004; the spread cells are properly
    # Gaussian, sd 0.13/0.04), so the zero-predictor floor differs slightly per episode. R is a
    # RATIO of the two arms on the same episode, so it is EXACTLY invariant to any per-episode
    # common rescaling -- this block confirms that and re-runs the Wilcoxon on rescaled deltas,
    # whose signed ranks are not invariant. Reported as a robustness check only.
    print("\n=== SENSITIVITY: renormalize both arms by realized window Hs (not the primary) ===")
    hsr = {}
    for c in CELLS:
        mp = f'{D}/data/cell_{c}/_manifest.json'
        if os.path.exists(mp):
            for ep, v in json.load(open(mp)).items():
                if v.get('status') == 'ok':
                    hsr[(c, ep)] = v['Hs_realized']
    for c in CELLS:
        rows2 = []
        for r in allrows[c]:
            f = 1.2 / hsr.get((c, r['ep']), 1.2)
            rows2.append(dict(r, cond=r['cond'] * f, ws_best=r['ws_best'] * f,
                              d_rel=r['d_rel']))
        s2 = stat(rows2)
        print(f"  cell {c}: cond {s2['cond_mean']:.4f}  ws_best {s2['ws_mean']:.4f}  "
              f"R {s2['R']*100:+.1f}% (invariant)  p {s2['p']:.2e}  "
              f"[raw p {S[c]['p']:.2e}]")

    # ── secondary ───────────────────────────────────────────────────────────
    print("\n=== SECONDARY: n_selected (learned proposal cardinality, median-of-5) ===")
    nsel = {}
    for c in CELLS:
        v = [sec[(c, ep)]['n_selected_med'] for ep in EPS
             if (c, ep) in sec and sec[(c, ep)]['n_selected_med'] is not None]
        nsel[c] = v
        if v:
            allv = [x for ep in EPS if (c, ep) in sec and sec[(c, ep)]['n_selected']
                    for x in sec[(c, ep)]['n_selected']]
            print(f"  cell {c}: median {np.median(v):.1f}  mean {np.mean(v):.2f}  "
                  f"range [{min(v):.0f}, {max(v):.0f}]  "
                  f"per-seed pooled median {np.median(allv):.1f}  "
                  f"frac at cap(30) {np.mean(np.array(allv) >= 30):.2f}  "
                  f"frac >= router thresh 8: {np.mean(np.array(allv) >= 8):.2f}")
    if nsel['a'] and nsel['c']:
        u, pu = mannwhitneyu(nsel['a'], nsel['c'], alternative='two-sided')
        print(f"  A vs C Mann-Whitney p = {pu:.3g}")

    print("\n=== SECONDARY: bf_peak_ratio vs the real-window reference ===")
    ref = json.load(open(REAL_BF))['rows']
    rv = np.array([r['bf_ratio'] for r in ref], float)
    print(f"  REAL ({len(rv)} windows): median {np.median(rv):.3f}  "
          f"IQR [{np.percentile(rv,25):.3f}, {np.percentile(rv,75):.3f}]  "
          f"range [{rv.min():.3f}, {rv.max():.3f}]")
    lo, hi = 0.27, 1.00
    print(f"  real-window band used for the migration read-out: [{lo}, {hi}] "
          f"({np.mean((rv>=lo)&(rv<=hi))*100:.0f}% of real windows inside)")
    for c in CELLS:
        v = np.array([sec[(c, ep)]['bf_peak_ratio'] for ep in EPS
                      if (c, ep) in sec and sec[(c, ep)]['bf_peak_ratio'] is not None])
        if len(v):
            print(f"  cell {c} (n={len(v)}): median {np.median(v):.3f}  "
                  f"IQR [{np.percentile(v,25):.3f}, {np.percentile(v,75):.3f}]  "
                  f"range [{v.min():.3f}, {v.max():.3f}]  "
                  f"in-band {np.mean((v>=lo)&(v<=hi))*100:.0f}%")

    # ── generation provenance ───────────────────────────────────────────────
    print("\n=== generation gates (mesh adequacy / Hs / spread) ===")
    flags = []
    for c in CELLS:
        mp = f'{D}/data/cell_{c}/_manifest.json'
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        vs = [v for v in m.values() if v.get('status') == 'ok']
        verd = set(v['mesh_verdict'] for v in vs)
        print(f"  cell {c}: {len(vs)}/10 ok  verdicts={verd}  "
              f"atoms={set(v['n_atoms'] for v in vs)}  "
              f"var_under max {max(v['var_under'] for v in vs):.4f}  "
              f"Hs_realized {np.mean([v['Hs_realized'] for v in vs]):.3f}"
              f"+-{np.std([v['Hs_realized'] for v in vs]):.3f}  "
              f"dir_spread {np.mean([v['dir_spread_deg'] for v in vs]):.1f}"
              f"+-{np.std([v['dir_spread_deg'] for v in vs]):.1f} deg")
        if verd != {'ADEQUATE'}:
            flags.append(f"cell {c} mesh verdicts {verd} != ADEQUATE")
        if len(vs) != 10:
            flags.append(f"cell {c} only {len(vs)}/10 episodes generated")
        hsv = [v['Hs_realized'] for v in vs]
        if max(abs(h - 1.2) / 1.2 for h in hsv) > 0.10:
            flags.append(
                f"G-roundtrip: cell {c} realized window Hs spans "
                f"[{min(hsv):.3f}, {max(hsv):.3f}] vs nominal 1.200 (>10% band). "
                "Component-level Hs is EXACT (1.2000+-0.00000) in every cell; this is "
                "finite-window realization scatter, which the discrete cell A does not have "
                "because its component amplitudes are deterministic. R is a same-episode "
                "ratio and is exactly invariant to it (see SENSITIVITY block)")
    if errs:
        flags.append(f"{len(errs)} errored arm runs")
    print("\n=== validity flags ===")
    print("  none" if not flags else "\n".join("  " + f for f in flags))

    json.dump(dict(verdict=verdict, why=why, cells=S, rows=allrows,
                   R_A=RA, R_B=S['b']['R'], R_C=RC, p_A=S['a']['p'],
                   p_B=S['b']['p'], p_C=pC, flags=flags),
              open(f'{D}/results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
