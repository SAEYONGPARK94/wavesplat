"""Aperture arm analysis — frozen cells of PREREG_APERTURE_ADDENDUM.md §4, evaluated in order:

 0 APERTURE-LIMITED : mean analytic skill in C-big < 0.15
 1 DISCRETENESS NOT BINDING : solvable AND R_Cbig <= -20% AND p_Cbig < 0.05
 2 DISCRETENESS IS WHAT LEARNING EXPLOITS : solvable AND (R_Cbig > -10% OR p_Cbig >= 0.05)
                                            AND R_Abig <= -20% AND p_Abig < 0.05
 3 AMBIGUOUS : solvable but neither

skill = 1 - nRMSE/floor, floor = zero-predictor nRMSE at the same 3 nowcast frames.
"""
import json, os, sys
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import wilcoxon

D = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.dirname(D)))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'scripts')

WS = f'{D}/ws_runs_aperture.jsonl'
COND = f'{D}/cond_runs_aperture.jsonl'
SEC = f'{D}/secondary_aperture.jsonl'
REAL_BF = 'tmp/figs_opus_real_2026-07-26/bf_ratio_real.json'
CELLS = ['abig', 'cbig']
NAME = {'abig': 'A-big  discrete control @ D_eff 475 m',
        'cbig': 'C-big  strong-spread continuum @ D_eff 475 m'}
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (20, 21, 22, 23, 24) for r in ('uniform', 'slope')]


def floors():
    """Zero-predictor nRMSE at the 3 nowcast frames, per episode."""
    cache = f'{D}/noskill_floor_aperture.json'
    if os.path.exists(cache):
        return json.load(open(cache))
    from wavesplat.data_loader import EpisodeDataLoader
    from d23_main_results import K_TP
    out = {}
    for c in CELLS:
        for ep in EPS:
            L = EpisodeDataLoader(f'{D}/data/cell_{c}/{ep}')
            nb = int(round(K_TP * L.Tp_true / 0.1))
            sel = np.linspace(nb + 3, len(L) - 2, 3).astype(int)
            out[f'{c}/{ep}'] = float(np.mean(
                [np.sqrt(np.mean(L[int(i)].gt_field.numpy() ** 2)) / max(L.Hs_true, 1e-6)
                 for i in sel]))
    json.dump(out, open(cache, 'w'), indent=1)
    return out


def load():
    ws = defaultdict(lambda: {'init': [], 'track': [], 'ng': []})
    cond = defaultdict(list); errs = []
    meta = {}
    for line in open(WS):
        d = json.loads(line)
        if d['err']:
            errs.append(('ws', d['cell'], d['episode'], d['seed'], d['err'])); continue
        k = (d['cell'], d['episode'])
        ws[k]['init'].append(d['init_nrmse']); ws[k]['track'].append(d['avg_nRMSE'])
        ws[k]['ng'].append(d.get('n_gaussians'))
        meta[k] = (d.get('init_frames'), d.get('init_span_s'), d.get('init_n_obs'))
    for line in open(COND):
        d = json.loads(line)
        if d['err']:
            errs.append(('cond', d['cell'], d['episode'], d['seed'], d['err'])); continue
        cond[(d['cell'], d['episode'])].append(d['cond_polish_nrmse_Hs'])
    sec = {}
    if os.path.exists(SEC):
        for line in open(SEC):
            d = json.loads(line); sec[(d['cell'], d['episode'])] = d
    return ws, cond, sec, errs, meta


def rows_of(cell, ws, cond, fl):
    rows = []
    for ep in EPS:
        wi, wt, cm = ws[(cell, ep)]['init'], ws[(cell, ep)]['track'], cond[(cell, ep)]
        if len(wi) != 5 or len(cm) != 5:
            print(f"  INCOMPLETE {cell} {ep}: ws={len(wi)} cond={len(cm)}")
        mi, mt, mc = np.median(wi), np.median(wt), np.median(cm)
        best = float(min(mi, mt)); f = fl[f'{cell}/{ep}']
        rows.append(dict(cell=cell, ep=ep, ws_init=float(mi), ws_init_sd=float(np.std(wi)),
                         ws_track=float(mt), ws_track_sd=float(np.std(wt)),
                         ws_best=best, cond=float(mc), cond_sd=float(np.std(cm)),
                         floor=f, skill_ws=1 - best / f, skill_cond=1 - float(mc) / f,
                         d_abs=float(mc - best), d_rel=float((mc - best) / best),
                         ng=float(np.median([g for g in ws[(cell, ep)]['ng'] if g]))))
    return rows


def stat(rows):
    c = np.array([r['cond'] for r in rows]); b = np.array([r['ws_best'] for r in rows])
    d = c - b
    try:
        _, p = wilcoxon(c, b, alternative='less', zero_method='wilcox')
    except Exception:
        p = float('nan')
    return dict(n=len(d), cond_mean=float(c.mean()), cond_med=float(np.median(c)),
                ws_mean=float(b.mean()), ws_med=float(np.median(b)),
                med_abs=float(np.median(d)), R=float(np.median([r['d_rel'] for r in rows])),
                wins=int((d < 0).sum()), p=float(p),
                skill_ws=float(np.mean([r['skill_ws'] for r in rows])),
                skill_cond=float(np.mean([r['skill_cond'] for r in rows])),
                skill_ws_med=float(np.median([r['skill_ws'] for r in rows])),
                floor=float(np.mean([r['floor'] for r in rows])),
                n_ws_noskill=int(sum(1 for r in rows if r['ws_best'] >= r['floor'])))


def main():
    fl = floors()
    ws, cond, sec, errs, meta = load()
    print("=== data integrity ===")
    print(f"  errored runs: {len(errs)}")
    for e in errs[:8]:
        print("   ", e)
    mv = set(v for v in meta.values())
    print(f"  aperture window actually used (frames, span_s, n_obs): {sorted(mv)[:3]}")

    R, S = {}, {}
    for c in CELLS:
        R[c] = rows_of(c, ws, cond, fl); S[c] = stat(R[c])

    print("\n=== SOLVABILITY (frozen gate: mean analytic skill >= 0.15 in C-big) ===")
    print(f"{'cell':6s}{'floor':>9s}{'ws_best':>9s}{'cond':>9s}"
          f"{'skill_ws':>10s}{'skill_cond':>12s}{'ws no-skill eps':>17s}")
    for c in CELLS:
        s = S[c]
        print(f"{c:6s}{s['floor']:9.4f}{s['ws_mean']:9.4f}{s['cond_mean']:9.4f}"
              f"{s['skill_ws']:10.3f}{s['skill_cond']:12.3f}{s['n_ws_noskill']:>13d}/10")
    solvable = S['cbig']['skill_ws'] >= 0.15
    print(f"  C-big analytic skill = {S['cbig']['skill_ws']:.3f} "
          f"=> {'SOLVABLE' if solvable else 'NOT SOLVABLE'}")
    print("  [parent run, original aperture: cell C analytic skill 0.015, cell A 0.543]")

    print("\n=== advantage (median-of-5 both arms, n=10 per cell) ===")
    print(f"{'cell':6s}{'cond mean':>11s}{'ws_best mean':>14s}{'medAbsD':>10s}"
          f"{'R':>9s}{'wins':>7s}{'p':>11s}")
    for c in CELLS:
        s = S[c]
        print(f"{c:6s}{s['cond_mean']:11.4f}{s['ws_mean']:14.4f}{s['med_abs']:+10.4f}"
              f"{s['R']*100:+8.1f}%{s['wins']:>5d}/10{s['p']:11.2e}")

    RA, pA = S['abig']['R'], S['abig']['p']
    RC, pC = S['cbig']['R'], S['cbig']['p']
    if not solvable:
        verdict = 'APERTURE-LIMITED'
        why = (f"mean analytic skill in C-big = {S['cbig']['skill_ws']:.3f} < 0.15: continuum "
               "cells are still information-limited at this aperture. Retraining cannot recover "
               "skill at this sensor configuration; expand observation window / footprint first. "
               "No discreteness verdict may be issued from this run.")
    elif RC <= -0.20 and pC < 0.05:
        verdict = '(1) DISCRETENESS NOT BINDING'
        why = (f"C-big solvable (skill {S['cbig']['skill_ws']:.3f}) and the learned advantage "
               f"REAPPEARS: R_Cbig = {RC*100:+.1f}%, p = {pC:.3g}. Discreteness was not the "
               "binding constraint at adequate aperture; there is recoverable continuum skill to "
               "chase => GO for the mixed retrain.")
    elif (RC > -0.10 or pC >= 0.05) and RA <= -0.20 and pA < 0.05:
        verdict = '(2) DISCRETENESS IS WHAT LEARNING EXPLOITS'
        why = (f"C-big solvable (skill {S['cbig']['skill_ws']:.3f}) yet the advantage stays "
               f"collapsed (R_Cbig = {RC*100:+.1f}%, p = {pC:.3g}) while the aperture-matched "
               f"discrete control retains it (R_Abig = {RA*100:+.1f}%, p = {pA:.3g}). Retraining "
               "must specifically add continuum competence => GO with curriculum emphasis.")
    else:
        verdict = 'AMBIGUOUS'
        why = (f"C-big solvable (skill {S['cbig']['skill_ws']:.3f}) but neither frozen cell "
               f"obtains: R_Abig = {RA*100:+.1f}% (p {pA:.3g}), R_Cbig = {RC*100:+.1f}% "
               f"(p {pC:.3g}). No narrative rescue.")
    print(f"\n=== PRE-REGISTERED VERDICT: {verdict} ===")
    print(f"  {why}")

    print("\n=== full paired tables ===")
    for c in CELLS:
        print(f"\n-- {NAME[c]}")
        print(f"{'episode':40s}{'cond(sd)':>15s}{'WSinit(sd)':>15s}{'WStrk(sd)':>15s}"
              f"{'ws_best':>9s}{'floor':>8s}{'sk_ws':>8s}{'dRel':>9s}{'N_g':>6s}")
        for r in R[c]:
            print(f"{r['ep']:40s}{r['cond']:9.3f}({r['cond_sd']:.3f})"
                  f"{r['ws_init']:9.3f}({r['ws_init_sd']:.3f})"
                  f"{r['ws_track']:9.3f}({r['ws_track_sd']:.3f})"
                  f"{r['ws_best']:9.3f}{r['floor']:8.3f}{r['skill_ws']:8.3f}"
                  f"{r['d_rel']*100:+8.1f}%{r['ng']:6.0f}")

    if sec:
        print("\n=== SECONDARY: n_selected / bf_peak_ratio (aperture cells) ===")
        ref = np.array([x['bf_ratio'] for x in json.load(open(REAL_BF))['rows']], float)
        print(f"  REAL windows: median {np.median(ref):.3f} "
              f"IQR [{np.percentile(ref,25):.3f}, {np.percentile(ref,75):.3f}]")
        for c in CELLS:
            ns = [sec[(c, ep)]['n_selected_med'] for ep in EPS if (c, ep) in sec]
            bf = [sec[(c, ep)]['bf_peak_ratio'] for ep in EPS if (c, ep) in sec]
            if ns:
                print(f"  {c}: n_selected median {np.median(ns):.1f} "
                      f"range [{min(ns):.0f}, {max(ns):.0f}] | "
                      f"bf_peak_ratio median {np.median(bf):.3f} "
                      f"IQR [{np.percentile(bf,25):.3f}, {np.percentile(bf,75):.3f}]")

    print("\n=== generation gates ===")
    flags = []
    for c in CELLS:
        mp = f'{D}/data/cell_{c}/_manifest.json'
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp)); vs = [v for v in m.values() if v.get('status') == 'ok']
        verd = set(v['mesh_verdict'] for v in vs)
        print(f"  {c}: {len(vs)}/10 ok verdicts={verd} atoms={set(v['n_atoms'] for v in vs)} "
              f"var_under max {max(v['var_under'] for v in vs):.4f} "
              f"frames={set(v['n_frames'] for v in vs)} "
              f"Hs_real {np.mean([v['Hs_realized'] for v in vs]):.3f}"
              f"+-{np.std([v['Hs_realized'] for v in vs]):.3f} "
              f"spread {np.mean([v['dir_spread_deg'] for v in vs]):.1f} deg")
        if verd != {'ADEQUATE'}:
            flags.append(f"{c}: mesh verdicts {verd}")
        if len(vs) != 10:
            flags.append(f"{c}: {len(vs)}/10 episodes")
    if errs:
        flags.append(f"{len(errs)} errored arm runs")
    print("\n=== validity flags ===")
    print("  none" if not flags else "\n".join("  " + f for f in flags))

    json.dump(dict(verdict=verdict, why=why, solvable=bool(solvable), cells=S, rows=R,
                   R_Abig=RA, R_Cbig=RC, p_Abig=pA, p_Cbig=pC, flags=flags),
              open(f'{D}/results_aperture.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
