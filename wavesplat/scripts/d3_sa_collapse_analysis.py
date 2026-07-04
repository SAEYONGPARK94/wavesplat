"""
d3_sa_collapse_analysis.py — U-law vs aperture-law discrimination across records.

Input: results/direction3/sa_multirecord.json (d3_sa_multirecord.py).
Point = (record, window, condition): x in {U/c_p, D_eff/lam_p, D_eff3/lam_p},
y = motion gain = corr_bl(B) - corr_bl(A) on common_unobs (RELAX and RIDGE).

Discrimination (the part single records cannot do):
  1. Leave-one-RECORD-out law test: fit a monotone (isotonic) curve gain(x) on all
     other records, predict the held-out record, mean |error|. A law that
     generalizes across records (different T_coh/Tp) wins. D2-null spread is the
     noise floor reference.
  2. Pre-registered qualitative checks (aperture-law predictions, 2026-07-04):
     - AA15a (T_coh/Tp ~ 4.9): gain > 0 already at U/c_p = 0.2 (U-law: harm)
     - BS11b (T_coh/Tp ~ 0.32): no gain even at U/c_p = 1.0 (U-law: gain)
     - LJ18n negative control: no gain at ANY speed (D_eff/lam_true <= ~0.2);
       x renormalized with lam_ref/cp_ref from Grid2 (patch-limited lam_p artifact)
     - AA15a lfrac variants: same U, capped track -> gain follows D_eff, not U
  3. Pooled zero-crossing of gain(D_eff/lam) vs the independent 0.7 lam
     operability threshold.

Usage: python3 scripts/d3_sa_collapse_analysis.py [--json PATH] [--smoke]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))


# ------------------------------------------------------------------ data
def load_points(store, which='mean'):
    """One point per (record, window, moving condition). which: 'mean'|'mean_ridge'."""
    pts = []
    for key, rec in store.items():
        lam_ref, cp_ref = rec.get('lam_ref'), rec.get('cp_ref')
        for wi, w in enumerate(rec['windows']):
            A = w['conditions']['A']
            D2 = w['conditions']['D2']
            null = (D2[which]['common_unobs']['corr_bl']
                    - A[which]['common_unobs']['corr_bl'])
            for name, e in w['conditions'].items():
                if not name.startswith('B_'):
                    continue
                gain = (e[which]['common_unobs']['corr_bl']
                        - A[which]['common_unobs']['corr_bl'])
                # dimensionless x: record-level ref overrides window-level values
                # where the window spectral estimate is patch-limited (LJ18n)
                lam = lam_ref if lam_ref else w['lam_p']
                cp = cp_ref if cp_ref else w['cp']
                pts.append(dict(
                    record=key, window=wi, cond=name, gain=gain, null=null,
                    u_cp=e['U'] / cp, deff=e['D_eff_m'] / lam,
                    deff3=e['D_eff3_m'] / lam,
                    tcoh_tp=w['t_coh_s'] / w['Tp'], lfrac=e.get('lfrac', 1.0),
                    n_obs=float(np.mean([r['n_obs'] for r in e['reps']])),
                    # confound covariates (H2 spread / H3 alignment / H5 patch
                    # size / obs density per coherent lam^2 of swept corridor)
                    spread=w['spread_deg'],
                    track_angle=w['track_angle_to_peak_deg'],
                    patch_lam=w['L_diag'] / w['lam_p'],
                    obs_per_lam2=(400.0 / max(e.get('coverage_lam2', 1e-9), 1e-9)
                                  if lam_ref is None else np.nan),
                    A_abs=A[which]['common_unobs']['corr_bl']))
    return pts


# ------------------------------------------------------------------ isotonic
def isotonic_fit(x, y):
    """Nondecreasing least-squares fit (PAV). Returns step function (xs, ys)."""
    o = np.argsort(x)
    xs, ys = np.asarray(x, float)[o], np.asarray(y, float)[o]
    # pool adjacent violators
    vals = list(ys)
    wts = [1.0] * len(ys)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
            wts[i] += wts[i + 1]
            vals[i] = v
            del vals[i + 1], wts[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    # expand back to sample positions
    fit = np.empty(len(ys))
    j = 0
    acc = 0
    for v, wt in zip(vals, wts):
        n = int(round(wt))
        fit[acc:acc + n] = v
        acc += n
    return xs, fit


def iso_predict(xs, fit, xq):
    return np.interp(np.asarray(xq, float), xs, fit,
                     left=fit[0] if len(fit) else np.nan,
                     right=fit[-1] if len(fit) else np.nan)


def group_velocity(kp, depth, g=9.81):
    """c_g = dw/dk for the linear dispersion w^2 = g k tanh(kd)."""
    kd = kp * depth
    om = np.sqrt(g * kp * np.tanh(kd))
    return float(g * (np.tanh(kd) + kd / np.cosh(kd) ** 2) / (2 * om))


def load_abs_points(store, which='mean'):
    """Absolute-performance points: every condition (A included) contributes
    y = common_unobs corr_bl. Separates the aperture law from the
    variable-baseline confound in the differential gain (gain = B - A inherits
    A's window-to-window fluctuation with a minus sign).

    Two aperture definitions per point:
      deff  — B-only sweep aperture min(U t_coh, L_seg) + fp (A sits at fp/lam)
      deffg — full transport aperture min(U t_coh, L_seg) + c_g t_coh + fp:
              the original coherence law D ~ L + (U + c_g) min(T, T_coh); a
              FIXED sensor already integrates waves advected past it at c_g,
              so records with long t_coh/Tp give A a supra-threshold aperture
              (candidate explanation for strong fixed baselines on AA/BS13)."""
    pts = []
    for key, rec in store.items():
        lam_ref = rec.get('lam_ref')
        cp_ref = rec.get('cp_ref')
        fp = rec['fp_diam']
        for wi, w in enumerate(rec['windows']):
            lam = lam_ref if lam_ref else w['lam_p']
            cp = cp_ref if cp_ref else w['cp']
            # c_g from the window peak; for LJ18n the window kp is a patch
            # artifact, use the reference wave via lam_ref (kp = 2 pi / lam)
            kp = 2 * np.pi / lam if lam_ref else w['kp']
            cg = group_velocity(kp, rec['depth'])
            tcoh = min(w['t_coh_s'], rec['T_win'])
            drift = cg * tcoh
            for name, e in w['conditions'].items():
                if name == 'D2':
                    continue
                y = e[which]['common_unobs']['corr_bl']
                if name == 'A':
                    pts.append(dict(record=key, window=wi, cond='A', y=y,
                                    u_cp=0.0, deff=fp / lam,
                                    deffg=(fp + drift) / lam, lfrac=1.0))
                elif name.startswith('B_'):
                    sweep = e['D_eff_m'] - fp          # min(U t_coh, L_seg)
                    pts.append(dict(record=key, window=wi, cond=name, y=y,
                                    u_cp=e['U'] / cp, deff=e['D_eff_m'] / lam,
                                    deffg=(sweep + drift + fp) / lam,
                                    lfrac=e.get('lfrac', 1.0)))
    return pts


def loro_abs_test(pts, xkey, centered=False):
    """LORO on absolute performance; centered=True removes each record's mean y
    first (law then only explains within-record structure)."""
    pts = [dict(p) for p in pts if p['lfrac'] == 1.0]
    if centered:
        for r in {p['record'] for p in pts}:
            sub = [p for p in pts if p['record'] == r]
            mu = float(np.mean([p['y'] for p in sub]))
            for p in sub:
                p['y'] -= mu
    recs = sorted({p['record'] for p in pts})
    errs = {}
    for held in recs:
        tr = [p for p in pts if p['record'] != held]
        te = [p for p in pts if p['record'] == held]
        xs, fit = isotonic_fit([p[xkey] for p in tr], [p['y'] for p in tr])
        pred = iso_predict(xs, fit, [p[xkey] for p in te])
        errs[held] = float(np.mean(np.abs(pred - np.array([p['y'] for p in te]))))
    return float(np.mean(list(errs.values()))), errs


def loro_law_test(pts, xkey):
    """Leave-one-record-out mean |prediction error| for law x=xkey."""
    recs = sorted({p['record'] for p in pts})
    errs, per_rec = [], {}
    for held in recs:
        tr = [p for p in pts if p['record'] != held]
        te = [p for p in pts if p['record'] == held]
        if not tr or not te:
            continue
        xs, fit = isotonic_fit([p[xkey] for p in tr], [p['gain'] for p in tr])
        pred = iso_predict(xs, fit, [p[xkey] for p in te])
        e = float(np.mean(np.abs(pred - np.array([p['gain'] for p in te]))))
        per_rec[held] = e
        errs.append(e)
    return float(np.mean(errs)), per_rec


def zero_crossing(pts, xkey):
    xs, fit = isotonic_fit([p[xkey] for p in pts], [p['gain'] for p in pts])
    pos = np.where(fit > 0)[0]
    if len(pos) == 0:
        return None
    i = pos[0]
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[i - 1], xs[i], fit[i - 1], fit[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))


# ------------------------------------------------------------------ checks
def qualitative_checks(pts):
    out = {}

    def sel(rec, cond_pred, lo, hi, xkey='u_cp'):
        v = [p['gain'] for p in pts if p['record'] == rec
             and lo <= p[xkey] <= hi and cond_pred(p)]
        return (float(np.mean(v)), len(v)) if v else (None, 0)

    main = lambda p: p['lfrac'] == 1.0
    out['AA15a_lowU_gain'] = sel('AA15a', main, 0.15, 0.25)      # D-law: > 0
    out['AA15a_r01_gain'] = sel('AA15a', main, 0.05, 0.12)       # threshold-ish
    out['BS11b_highU_gain'] = sel('BS11b', main, 0.9, 1.1)       # D-law: <= 0
    out['BS11a_highU_gain'] = sel('BS11a', main, 0.9, 1.1)       # anchor: ~ crossing
    lj = [p for p in pts if p['record'] == 'LJ18n']
    out['LJ18n_max_gain'] = (float(max(p['gain'] for p in lj)) if lj else None,
                             len(lj))
    out['LJ18n_mean_gain'] = ((float(np.mean([p['gain'] for p in lj])), len(lj))
                              if lj else (None, 0))
    # lfrac decoupling on AA15a: paired same-window same-ratio main vs capped
    pairs = []
    for p in pts:
        if p['record'] == 'AA15a' and p['lfrac'] < 1.0:
            m = [q for q in pts if q['record'] == 'AA15a' and q['lfrac'] == 1.0
                 and q['window'] == p['window']
                 and abs(q['u_cp'] - p['u_cp']) < 0.02]
            if m:
                pairs.append(dict(window=p['window'], u_cp=p['u_cp'],
                                  deff_main=m[0]['deff'], deff_cap=p['deff'],
                                  gain_main=m[0]['gain'], gain_cap=p['gain']))
    out['AA15a_lfrac_pairs'] = pairs
    if pairs:
        out['AA15a_lfrac_drop'] = float(np.mean(
            [q['gain_main'] - q['gain_cap'] for q in pairs]))
    return out


def confound_diagnostics(pts):
    """Why might the gain differ across records at matched D_eff/lam?
    Correlate the gain of supra-threshold points (D_eff > 0.7 lam, main track)
    with candidate drivers, pooled over records (Pearson r + per-record means)."""
    sup = [p for p in pts if p['lfrac'] == 1.0 and p['deff'] > 0.7
           and np.isfinite(p.get('obs_per_lam2', np.nan))]
    out = {'n_supra': len(sup)}
    if len(sup) < 8:
        return out
    g = np.array([p['gain'] for p in sup])
    for name in ('spread', 'track_angle', 'patch_lam', 'obs_per_lam2', 'A_abs',
                 'tcoh_tp'):
        x = np.array([p[name] for p in sup], float)
        okm = np.isfinite(x)
        r = float(np.corrcoef(x[okm], g[okm])[0, 1]) if okm.sum() > 3 else np.nan
        out[f'corr_gain_vs_{name}'] = r
    per_rec = {}
    for rec in sorted({p['record'] for p in sup}):
        sub = [p for p in sup if p['record'] == rec]
        per_rec[rec] = dict(
            gain=float(np.mean([p['gain'] for p in sub])),
            spread=float(np.mean([p['spread'] for p in sub])),
            track_angle=float(np.mean([p['track_angle'] for p in sub])),
            patch_lam=float(np.mean([p['patch_lam'] for p in sub])),
            A_abs=float(np.mean([p['A_abs'] for p in sub])), n=len(sub))
    out['per_record_supra'] = per_rec
    return out


# ------------------------------------------------------------------ plot
def make_plot(pts_rx, pts_rd, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    recs = sorted({p['record'] for p in pts_rx})
    cmap = plt.get_cmap('tab10')
    colors = {r: cmap(i % 10) for i, r in enumerate(recs)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey='row')
    for row, (pts, tag) in enumerate([(pts_rx, 'RELAX'), (pts_rd, 'RIDGE')]):
        for col, (xkey, xlab) in enumerate([('u_cp', 'U / c_p'),
                                            ('deff', 'D_eff / lam_p')]):
            ax = axes[row][col]
            for r in recs:
                sub = [p for p in pts if p['record'] == r and p['lfrac'] == 1.0]
                x = [p[xkey] for p in sub]
                y = [p['gain'] for p in sub]
                mk = 'x' if r == 'LJ18n' else 'o'
                ax.scatter(x, y, s=18, alpha=0.55, color=colors[r], marker=mk,
                           label=r if row == 0 and col == 0 else None)
            cap = [p for p in pts if p['lfrac'] < 1.0]
            if cap:
                ax.scatter([p[xkey] for p in cap], [p['gain'] for p in cap],
                           s=40, facecolors='none', edgecolors='k', marker='s',
                           label='track-capped' if row == 0 and col == 0 else None)
            allp = [p for p in pts if p['lfrac'] == 1.0]
            xs, fit = isotonic_fit([p[xkey] for p in allp],
                                   [p['gain'] for p in allp])
            ax.plot(xs, fit, 'k-', lw=2, alpha=0.7)
            nul = [p['null'] for p in allp]
            ax.axhspan(np.mean(nul) - np.std(nul), np.mean(nul) + np.std(nul),
                       color='gray', alpha=0.15)
            ax.axhline(0, color='k', lw=0.5)
            if xkey == 'deff':
                ax.axvline(0.7, color='crimson', ls='--', lw=1.2)
                ax.text(0.72, ax.get_ylim()[0] * 0.9, '0.7 lam', color='crimson',
                        fontsize=8)
            ax.set_xlabel(xlab)
            if col == 0:
                ax.set_ylabel(f'{tag}: gain in common-unobs corr_bl (B - A)')
            ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle('Multi-record synthetic-aperture motion gain: '
                 'speed law vs aperture law (real stereo records)')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"plot -> {path}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    stem = 'sa_multirecord_smoke' if args.smoke else 'sa_multirecord'
    path = args.json or f'results/direction3/{stem}.json'
    store = json.load(open(path))
    print(f"records in {path}: {list(store)}")

    result = dict(source=path, records=list(store))
    for which, tag in [('mean', 'RELAX'), ('mean_ridge', 'RIDGE')]:
        pts = load_points(store, which)
        main_pts = [p for p in pts if p['lfrac'] == 1.0]
        print(f"\n===== {tag} ({len(pts)} points, {len(main_pts)} main) =====")
        laws = {}
        for xkey in ('u_cp', 'deff', 'deff3'):
            err, per_rec = loro_law_test(main_pts, xkey)
            laws[xkey] = dict(mean_abs_err=err, per_record=per_rec)
            print(f"  LORO law {xkey:6s}: mean|err|={err:.3f} | " +
                  " ".join(f"{k}:{v:.2f}" for k, v in sorted(per_rec.items())))
        best = min(laws, key=lambda k: laws[k]['mean_abs_err'])
        zc = zero_crossing(main_pts, 'deff')
        nul = [p['null'] for p in main_pts]
        print(f"  best law: {best} | gain(D_eff) zero-crossing: "
              f"{zc if zc else 'none'} (0.7 lam reference) | "
              f"D2 null: {np.mean(nul):+.3f} +- {np.std(nul):.3f}")
        checks = qualitative_checks(pts)
        for k, v in checks.items():
            if k != 'AA15a_lfrac_pairs':
                print(f"  check {k}: {v}")
        abs_pts = load_abs_points(store, which)
        abs_res = {}
        for cen in (False, True):
            for xkey in ('u_cp', 'deff', 'deffg'):
                err, per = loro_abs_test(abs_pts, xkey, centered=cen)
                abs_res[f"{xkey}{'_cen' if cen else ''}"] = err
        print("  ABS-mode LORO (y=corr_bl, A included at D_eff=fp): " +
              " ".join(f"{k}={v:.3f}" for k, v in abs_res.items()))
        # does the c_g-drift aperture explain the fixed-baseline strength?
        A_pts = [p for p in abs_pts if p['cond'] == 'A']
        xa = np.array([p['deffg'] for p in A_pts])
        ya = np.array([p['y'] for p in A_pts])
        print(f"  A-only: corr(A corr_bl, D_eff_A(cg)/lam) = "
              f"{np.corrcoef(xa, ya)[0, 1]:+.2f} over {len(A_pts)} windows; "
              f"per-record A aperture/lam: " +
              " ".join(f"{r}:{np.mean([p['deffg'] for p in A_pts if p['record'] == r]):.2f}"
                       for r in sorted({p['record'] for p in A_pts})))
        conf = confound_diagnostics(pts)
        for k, v in conf.items():
            if k.startswith('corr_'):
                print(f"  confound {k}: {v:+.2f}")
        if 'per_record_supra' in conf:
            for r, d in conf['per_record_supra'].items():
                print(f"    supra-0.7lam {r}: gain={d['gain']:+.3f} "
                      f"spread={d['spread']:.0f} trk_ang={d['track_angle']:.0f} "
                      f"patch={d['patch_lam']:.1f}lam A_abs={d['A_abs']:+.2f} "
                      f"n={d['n']}")
        result[tag] = dict(laws=laws, best_law=best, zero_crossing_deff=zc,
                           null_mean=float(np.mean(nul)),
                           null_sd=float(np.std(nul)), checks=checks,
                           confounds=conf, abs_loro=abs_res)

    out_json = path.replace('.json', '_analysis.json')
    json.dump(result, open(out_json, 'w'), indent=1, default=float)
    print(f"\nsaved {out_json}")
    try:
        make_plot(load_points(store, 'mean'), load_points(store, 'mean_ridge'),
                  path.replace('.json', '_collapse.png'))
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == '__main__':
    main()
