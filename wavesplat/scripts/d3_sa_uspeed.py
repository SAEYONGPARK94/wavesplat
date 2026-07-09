"""
d3_sa_uspeed.py — Speed-dependence of the synthetic-aperture motion gain on REAL waves.

Follow-up to d3_sa_gonogo.py, which showed (BS_2011, Hs 0.317 m, lam_p ~14 m, cp 4.67 m/s):
U=1 HURTS (-0.165 RELAX / -0.052 RIDGE unobs corr gain), U=2 neutral, U=4 clear GO.
Question: is the gain a SPEED LAW? If yes, in which dimensionless variable does it
generalize — U/c_p, or the achieved effective aperture D_eff/lam_p? If neither,
narrow the useful-(U, heading) window for real experiments.

Design (extends the verified fairness protocol of d3_sa_gonogo):
  * U set per window as ratio * c_p(window) -> exact U/c_p grid.
  * CENTERED-SHRUNK track: achieved segment L_seg = min(U*T, L_diag), CENTERED on the
    patch center along the best-aligned diagonal (oriented along resolved propagation).
    => U -> 0 converges to fixed condition A by construction, so the low-U regime
    measures genuine slow-motion effects, not a corner-offset artifact. For
    U*T >= L_diag (ratio >~ 0.11 here) geometry is identical to d3_sa_gonogo.
  * Matched budget: identical n_obs, footprint, frame times for all conditions.
  * Scoring masks: the U-INDEPENDENT full-diagonal design corridor for all main
    conditions (common_unobs identical across U -> apples-to-apples). Achieved
    corridors are subsets, so common_unobs is never observed by any condition.
    Additionally each moving condition scores its own ACHIEVED corridor
    ('swept_ach') paired with A's field on the same mask ('swept_ach_Aref').
  * RELAX (iterative_relax_cv) + RIDGE fixed-dictionary control per condition,
    n_rep seeds, D2 fixed re-seed null — all as in d3_sa_gonogo.

Effective aperture (documented choice):
  * T_coh: temporal coherence time of the field itself — the lag where the Hilbert
    envelope of the spatially-averaged temporal autocorrelation rho(tau) (400 valid
    pixels, unbiased covariance) drops below 1/e. Envelope is normalized at
    lag = 1 frame, not lag 0, to skip the frame-uncorrelated stereo-noise spike.
    If no crossing within T/2 the value is capped (flagged). A fixed 3 s proxy
    (the record's known forecast-skill window 0.5-3 s) is carried in parallel as
    D_eff3 so conclusions don't hinge on the T_coh estimator.
  * D_eff = min(U*T_coh, L_seg) + fp_diam: spatial extent the footprint sweeps
    within one coherence time, capped by the achieved track, plus footprint size.

Collapse test (the decisive part): on ONE record, U/c_p and D_eff/lam_p are
monotonically linked along the main curve, so a collapse cannot be claimed from the
main sweep alone. Decoupling probe: track-length-capped variants (same U,
L_seg capped at frac*L_diag => smaller D_eff at same U/c_p). If their gain matches
the main curve at equal D_eff (not equal U), the aperture law wins; if it matches
at equal U, speed itself is the variable.

Heading check (secondary): propagation SIGN resolved from the cross-spectrum phase
between lagged frames (spatial FFT alone is +-180 ambiguous). Equal-length chords
through the patch center along vs perpendicular to propagation at ratio_best
(H_align vs H_beam), plus exact single-traversal follow vs head at U = L_chord/T
(H_follow vs H_head, same chord = same masks). Caveat: the patch (~1.1 lam_p)
cannot sustain following-sea motion at U ~ c_p (each sweep lasts ~1.4 Tp), so the
U*cos(theta) ~ c_p degeneracy is only probed obliquely; stated in the output.

Usage:
  python3 scripts/d3_sa_uspeed.py --smoke     # coarse ratios, 2 windows, 2 reps (~4 min)
  python3 scripts/d3_sa_uspeed.py             # full: 12 ratios x 8 windows x 4 reps (~30 min)
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_gonogo import (WassRecord, diagonal_track, build_obs, region_masks,
                          score_regions, ridge_field, characterize, mean_scores,
                          G, DEFAULT_RECORD)
from realdata_analysis import lowpass_field
from estimator_improved import iterative_relax_cv

FULL_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.15]
SMOKE_RATIOS = [0.1, 0.2, 0.43, 0.85]
# Decoupling requires frac*L_diag < U*T_coh (else min() ignores the cap and D_eff is
# unchanged — verified on BS_2011 where T_coh ~ 1.6 s => only high ratios + frac<=0.25
# move D_eff; 0.85:0.5 moves D_eff3 (3 s proxy) but not D_eff).
FULL_LFRAC = ['0.6:0.25', '0.85:0.5', '0.85:0.25', '1.15:0.5', '1.15:0.25']
SMOKE_LFRAC = ['0.85:0.25']


# ------------------------------------------------------------------ coherence time
def _analytic_envelope(x):
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


def estimate_tcoh(Zwin, dt, rng, n_pix=400):
    """1/e time of the Hilbert envelope of the spatially-averaged temporal
    autocorrelation. Normalized at lag=1 frame (skips the stereo-noise spike at
    lag 0, which is uncorrelated frame-to-frame). Returns (t_coh_s, capped)."""
    nf = Zwin.shape[0]
    flat = Zwin.reshape(nf, -1)
    ok = np.where(np.isfinite(flat).all(axis=0))[0]
    sel = rng.choice(ok, size=min(n_pix, len(ok)), replace=False)
    S = flat[:, sel].astype(np.float64)
    S -= S.mean(axis=0)
    var = np.maximum((S * S).mean(axis=0), 1e-12)
    nlag = nf // 2
    rho = np.empty(nlag)
    rho[0] = 1.0
    for lag in range(1, nlag):
        rho[lag] = float(np.mean((S[:-lag] * S[lag:]).mean(axis=0) / var))
    sym = np.concatenate([rho[::-1], rho[1:]])          # even extension for Hilbert
    env = _analytic_envelope(sym)[nlag - 1:]
    envn = env / max(env[1], 1e-9)
    thr = np.exp(-1.0)
    below = np.where(envn[1:] < thr)[0]
    if len(below) == 0:
        return float(nlag * dt), True
    i = int(below[0]) + 1
    e0, e1 = envn[i - 1], envn[i]
    frac = float((e0 - thr) / max(e0 - e1, 1e-12))
    return float((i - 1 + frac) * dt), False


# ------------------------------------------------------------------ propagation sign
def resolve_prop_sign(Zwin, dx, dy, kp, pdir_deg, dt):
    """Resolve the +-180 ambiguity of the spatial-FFT peak direction using the
    cross-spectrum phase between frames lagged by ~0.35 s. Component at bin +k
    evolves as exp(-i*omega*tau) => weighted mean phase < 0 iff propagation is
    along pdir. Returns (resolved_dir_deg, mean_phase, expected_|phase|)."""
    nf, n0, n1 = Zwin.shape
    kx = 2 * np.pi * np.fft.fftfreq(n1, dx)              # axis1 = x
    ky = 2 * np.pi * np.fft.fftfreq(n0, dy)              # axis0 = y
    KX, KY = np.meshgrid(kx, ky)
    p = np.array([np.cos(np.radians(pdir_deg)), np.sin(np.radians(pdir_deg))])
    near = np.hypot(KX - kp * p[0], KY - kp * p[1]) < 0.35 * kp
    lag = max(1, int(round(0.35 / dt)))
    acc = 0j
    for f0i in (0, nf // 3, 2 * nf // 3):
        if f0i + lag >= nf:
            continue
        F0 = np.fft.fft2(np.nan_to_num(Zwin[f0i]))
        F1 = np.fft.fft2(np.nan_to_num(Zwin[f0i + lag]))
        cross = F1[near] * np.conj(F0[near])
        w = np.abs(F0[near]) ** 2
        acc += np.sum(w * np.exp(1j * np.angle(cross)))
    mean_phi = float(np.angle(acc))
    om_p = np.sqrt(G * kp)                               # deep-ish; only for sanity print
    expected = float(om_p * lag * dt)
    resolved = pdir_deg if mean_phi < 0 else pdir_deg + 180.0
    return float((resolved + 180.0) % 360.0 - 180.0), mean_phi, expected


# ------------------------------------------------------------------ track geometry
def centered_segment(Pa, Pb, L_seg):
    """Sub-segment of Pa->Pb with length L_seg, centered at the segment midpoint."""
    L = float(np.linalg.norm(Pb - Pa))
    u = (Pb - Pa) / L
    m = 0.5 * (Pa + Pb)
    return m - u * (L_seg / 2.0), m + u * (L_seg / 2.0)


def chord_half(gx, gy, r_fp, ang_deg):
    """Half-length of the chord through the patch center at angle ang_deg,
    inset by r_fp from the patch edges."""
    hx = (gx.max() - gx.min()) / 2.0 - r_fp
    hy = (gy.max() - gy.min()) / 2.0 - r_fp
    c, s = abs(np.cos(np.radians(ang_deg))), abs(np.sin(np.radians(ang_deg)))
    tx = hx / c if c > 1e-9 else np.inf
    ty = hy / s if s > 1e-9 else np.inf
    return float(min(tx, ty))


def corridor_mask(rec, r_fp, P0, P1):
    """Valid pixels within r_fp of segment P0->P1."""
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    seg = P1 - P0
    L2 = max(float(seg @ seg), 1e-12)
    tpar = np.clip(((Xg - P0[0]) * seg[0] + (Yg - P0[1]) * seg[1]) / L2, 0, 1)
    d = np.hypot(Xg - (P0[0] + tpar * seg[0]), Yg - (P0[1] + tpar * seg[1]))
    return rec.valid & (d <= r_fp)


# ------------------------------------------------------------------ experiment
def moving_meta(U, cp, L_seg, lfrac, lam_p, t_coh, args):
    d_eff = min(U * t_coh, L_seg) + args.fp_diam
    d_eff3 = min(U * 3.0, L_seg) + args.fp_diam
    r_fp = args.fp_diam / 2.0
    return dict(
        U=float(U), U_over_cp=float(U / cp), lfrac=float(lfrac),
        track_len_m=float(U * args.T), track_len_lam=float(U * args.T / lam_p),
        L_seg_m=float(L_seg), L_seg_lam=float(L_seg / lam_p),
        n_sweeps=float(U * args.T / L_seg),
        extent_lam=float((L_seg + args.fp_diam) / lam_p),
        coverage_lam2=float((L_seg * args.fp_diam + np.pi * r_fp ** 2) / lam_p ** 2),
        D_eff_m=float(d_eff), D_eff_lam=float(d_eff / lam_p),
        D_eff3_m=float(d_eff3), D_eff3_lam=float(d_eff3 / lam_p),
        t_coh_s=float(t_coh))


def run_window(rec, i0, args, ratios, lfracs, rng_seed):
    nf = int(round(args.T / rec.dt))
    Zwin = rec.window(i0, nf)
    Hs_w, lam_p, kp, pdir, spread, Tp, cp = characterize(Zwin, rec, args.depth)
    hs_lo, hs_hi = getattr(args, 'hs_lo', 0.05), getattr(args, 'hs_hi', 5.0)
    assert hs_lo < Hs_w < hs_hi, \
        f"Hs={Hs_w:.3f} m outside ({hs_lo}, {hs_hi}) — int16 scaling wrong?"
    gt = Zwin[-1]
    gt_f = np.nan_to_num(gt)
    k_cut = 1.5 * kp
    gt_bl = lowpass_field(gt_f, rec.dx, rec.dy, k_cut)
    t_off = float(rec.t[i0])
    t_star = float(rec.t[i0 + nf - 1]) - t_off
    target = {'gx': rec.gx, 'gy': rec.gy, 't': t_star, 'gt': gt_f, 'Hs': Hs_w}
    r_fp = args.fp_diam / 2.0

    # coherence time + propagation sign (window-level, deterministic)
    t_coh, t_coh_capped = estimate_tcoh(Zwin, rec.dt, np.random.default_rng(rng_seed))
    prop_dir, mean_phi, exp_phi = resolve_prop_sign(Zwin, rec.dx, rec.dy, kp, pdir, rec.dt)
    pvec = np.array([np.cos(np.radians(prop_dir)), np.sin(np.radians(prop_dir))])

    # main track: best-aligned diagonal, oriented along propagation.
    # args.track_fn (optional) overrides the chord choice — used by
    # d3_sa_multirecord for records with large invalid regions, where the blind
    # diagonal would cross NaN holes and break the matched obs budget.
    track_fn = getattr(args, 'track_fn', None)
    if track_fn is not None:
        P0d, P1d, track_angle = track_fn(rec, r_fp, pdir)
    else:
        P0d, P1d, track_angle = diagonal_track(rec.gx, rec.gy, r_fp, pdir)
    if float((P1d - P0d) @ pvec) < 0:
        P0d, P1d = P1d, P0d
    L_diag = float(np.linalg.norm(P1d - P0d))

    mask_sets = {'main': region_masks(rec, r_fp, args.margin, P0d, P1d)}

    # ---------------- conditions
    conds = {'A': dict(mode='fixed', mask_set='main'),
             'D2': dict(mode='fixed', mask_set='main')}
    for r in ratios:
        U = r * cp
        L_seg = min(U * args.T, L_diag)
        seg = centered_segment(P0d, P1d, L_seg)
        conds[f'B_r{r:g}'] = dict(mode='moving', mask_set='main', U=U, ratio=r,
                                  seg=seg, L_seg=L_seg, lfrac=1.0)
    for (r, frac) in lfracs:
        U = r * cp
        L_seg = min(U * args.T, frac * L_diag)
        seg = centered_segment(P0d, P1d, L_seg)
        conds[f'B_r{r:g}_L{frac:g}'] = dict(mode='moving', mask_set='main', U=U,
                                            ratio=r, seg=seg, L_seg=L_seg, lfrac=frac)
    ratio_single = None
    if args.heading:
        h_eq = min(chord_half(rec.gx, rec.gy, r_fp, prop_dir),
                   chord_half(rec.gx, rec.gy, r_fp, prop_dir + 90.0))
        L_h = 2.0 * h_eq
        c = np.array([(rec.gx[0] + rec.gx[-1]) / 2, (rec.gy[0] + rec.gy[-1]) / 2])
        qvec = np.array([np.cos(np.radians(prop_dir + 90)), np.sin(np.radians(prop_dir + 90))])
        U_best = args.ratio_best * cp
        for name, uvec in (('H_align', pvec), ('H_beam', qvec)):
            L_seg = min(U_best * args.T, L_h)
            seg = (c - uvec * L_seg / 2, c + uvec * L_seg / 2)
            conds[name] = dict(mode='moving', mask_set=name, U=U_best,
                               ratio=args.ratio_best, seg=seg, L_seg=L_seg, lfrac=1.0)
            mask_sets[name] = region_masks(rec, r_fp, args.margin, seg[0], seg[1])
        U_s = L_h / args.T                                # exactly one traversal
        ratio_single = float(U_s / cp)
        segF = (c - pvec * h_eq, c + pvec * h_eq)
        conds['H_follow'] = dict(mode='moving', mask_set='H_follow', U=U_s,
                                 ratio=ratio_single, seg=segF, L_seg=L_h, lfrac=1.0)
        conds['H_head'] = dict(mode='moving', mask_set='H_follow', U=U_s,
                               ratio=ratio_single, seg=(segF[1], segF[0]),
                               L_seg=L_h, lfrac=1.0)
        mask_sets['H_follow'] = region_masks(rec, r_fp, args.margin, segF[0], segF[1])

    # per-moving-condition achieved-corridor mask ('swept_ach'), excluding A's disk
    fixed_fp = mask_sets['main']['fixed_fp']
    for name, cf in conds.items():
        if cf['mode'] == 'moving':
            cf['swept_ach'] = corridor_mask(rec, r_fp, cf['seg'][0], cf['seg'][1]) & ~fixed_fp

    res = dict(i0=int(i0), t0=float(rec.t[i0]), Hs_win=Hs_w, Tp=Tp, cp=cp, lam_p=lam_p,
               kp=kp, peak_dir=pdir, prop_dir=prop_dir, prop_mean_phase=mean_phi,
               prop_expected_phase=exp_phi, spread_deg=spread, k_cut=k_cut,
               track_angle_to_peak_deg=track_angle, L_diag=L_diag,
               t_coh_s=t_coh, t_coh_capped=bool(t_coh_capped),
               ratio_single=ratio_single,
               region_px={k: int(v.sum()) for k, v in mask_sets['main'].items()},
               conditions={})
    print(f"    t_coh={t_coh:.2f}s{'(cap)' if t_coh_capped else ''} "
          f"prop_dir={prop_dir:.0f}deg (phase {mean_phi:+.2f}, expect |{exp_phi:.2f}|) "
          f"L_diag={L_diag:.1f}m cp={cp:.2f}m/s")

    rep_store = {name: [] for name in conds}
    names = list(conds.keys())
    for rep in range(args.n_rep):
        A_fields = None
        for ci, name in enumerate(names):
            cf = conds[name]
            seed = rng_seed + 997 * ci + 13 * rep
            rng = np.random.default_rng(seed)
            tic = time.time()
            if cf['mode'] == 'fixed':
                P, z, _, _ = build_obs(Zwin, rec, i0, args.nobs, r_fp, 'fixed', rng)
            else:
                P, z, _, _ = build_obs(Zwin, rec, i0, args.nobs, r_fp, 'moving', rng,
                                       U=cf['U'], track=cf['seg'])
            P[:, 2] -= t_off
            est_name = getattr(args, 'estimator', 'relax')
            if est_name in ('mmse', 'mmse_net'):
                Xq, Yq = np.meshgrid(rec.gx, rec.gy)
                Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                                      np.full(Xq.size, t_star)])
                if est_name == 'mmse':
                    from d3_sa_nomp import mmse_bank_estimate
                    fld = mmse_bank_estimate(P, z, args.depth, Pq,
                                             sigma0=0.05 * Hs_w,
                                             kp_hint=kp,
                                             seed=seed).reshape(Xq.shape)
                else:
                    from d3_prior_net_eval import get_net, net_gamma_mmse
                    fld = net_gamma_mmse(get_net(), P, z, args.depth, Pq,
                                         0.05 * Hs_w, seed=seed,
                                         use_em=False,
                                         kp_hint=kp).reshape(Xq.shape)
                N, klim = 0, 0.0
            else:
                fld, N, klim = iterative_relax_cv(P.copy(), z.copy(),
                                                  args.depth, target,
                                                  n_outer=args.n_outer,
                                                  seed=seed)
            fld_bl = lowpass_field(fld, rec.dx, rec.dy, k_cut)
            frd = ridge_field(P, z, args.depth, kp, rec.gx, rec.gy, t_star)
            frd_bl = lowpass_field(frd, rec.dx, rec.dy, k_cut)
            base = mask_sets[cf['mask_set']]
            extra = {'swept_ach': cf['swept_ach']} if cf['mode'] == 'moving' else {}
            sc = score_regions(fld, gt_f, Hs_w, {**base, **extra}, fld_bl, gt_bl)
            scr = score_regions(frd, gt_f, Hs_w, {**base, **extra}, frd_bl, gt_bl)
            if name == 'A':
                A_fields = (fld, fld_bl, frd, frd_bl)
            if cf['mode'] == 'moving':
                # paired A-reference on this condition's own masks (same rep)
                am = dict(extra)
                if cf['mask_set'] != 'main':
                    am.update({k: base[k] for k in ('common_unobs', 'swept')})
                am = {k + '_Aref': v for k, v in am.items()}
                fA, fAb, rA, rAb = A_fields
                sc.update(score_regions(fA, gt_f, Hs_w, am, fAb, gt_bl))
                scr.update(score_regions(rA, gt_f, Hs_w, am, rAb, gt_bl))
            rep_store[name].append(dict(seed=seed, n_obs=int(len(z)), N_comp=int(N),
                                        klim=float(klim),
                                        runtime_s=round(time.time() - tic, 1),
                                        scores=sc, scores_ridge=scr))

    for name in names:
        cf = conds[name]
        reps = rep_store[name]
        entry = dict(reps=reps, mean=mean_scores(reps),
                     mean_ridge=mean_scores([dict(scores=r['scores_ridge']) for r in reps]))
        if cf['mode'] == 'moving':
            entry.update(moving_meta(cf['U'], cp, cf['L_seg'], cf['lfrac'], lam_p,
                                     t_coh, args))
            if name.startswith('B_r'):
                entry['ratio'] = float(cf['ratio'])
        res['conditions'][name] = entry
        u, ur = entry['mean']['common_unobs'], entry['mean_ridge']['common_unobs']
        tag = ''
        if cf['mode'] == 'moving':
            tag = (f" U/cp={entry['U_over_cp']:.2f} Deff={entry['D_eff_lam']:.2f}lam"
                   f" swp={entry['n_sweeps']:.1f}")
        print(f"    {name:14s} RELAX unobs bl {u['corr_bl']:+.3f} | "
              f"RIDGE {ur['corr_bl']:+.3f}{tag} "
              f"[{sum(r['runtime_s'] for r in reps):.0f}s]")
    del Zwin
    return res


# ------------------------------------------------------------------ analysis
def _win_vals(windows, cond, which, region, key):
    out = []
    for w in windows:
        e = w['conditions'].get(cond)
        if e and key in e[which].get(region, {}):
            out.append(e[which][region][key])
        else:
            out.append(np.nan)
    return np.asarray(out, float)


def _gain(windows, cond, which, region='common_unobs', key='corr_bl', ref='A'):
    return (_win_vals(windows, cond, which, region, key)
            - _win_vals(windows, ref, which, region, key))


def _crossings(xs, ys):
    out = []
    for i in range(len(ys) - 1):
        if np.isnan(ys[i]) or np.isnan(ys[i + 1]) or ys[i] * ys[i + 1] >= 0:
            continue
        t = ys[i] / (ys[i] - ys[i + 1])
        out.append(float(xs[i] + t * (xs[i + 1] - xs[i])))
    return out


def analyze(windows, ratios, lfracs, args):
    summary = {}
    # null: paired per-window D2 - A
    null = {}
    for which in ('mean', 'mean_ridge'):
        d = _gain(windows, 'D2', which)
        null[which] = dict(mean=float(np.nanmean(d)), sd=float(np.nanstd(d)))
    summary['null'] = null

    curve = []
    for r in ratios:
        c = f'B_r{r:g}'
        gr, gd = _gain(windows, c, 'mean'), _gain(windows, c, 'mean_ridge')
        sw = (_win_vals(windows, c, 'mean', 'swept_ach', 'corr_bl')
              - _win_vals(windows, c, 'mean', 'swept_ach_Aref', 'corr_bl'))
        swd = (_win_vals(windows, c, 'mean_ridge', 'swept_ach', 'corr_bl')
               - _win_vals(windows, c, 'mean_ridge', 'swept_ach_Aref', 'corr_bl'))
        meta = [w['conditions'][c] for w in windows if c in w['conditions']]
        curve.append(dict(
            ratio=r,
            gain_relax=float(np.nanmean(gr)), sd_relax=float(np.nanstd(gr)),
            gain_ridge=float(np.nanmean(gd)), sd_ridge=float(np.nanstd(gd)),
            swept_gain_relax=float(np.nanmean(sw)),
            swept_gain_ridge=float(np.nanmean(swd)),
            D_eff_lam=float(np.mean([m['D_eff_lam'] for m in meta])),
            D_eff3_lam=float(np.mean([m['D_eff3_lam'] for m in meta])),
            track_len_lam=float(np.mean([m['track_len_lam'] for m in meta])),
            L_seg_lam=float(np.mean([m['L_seg_lam'] for m in meta])),
            coverage_lam2=float(np.mean([m['coverage_lam2'] for m in meta])),
            n_sweeps=float(np.mean([m['n_sweeps'] for m in meta]))))
    summary['curve'] = curve

    nw = max(sum(1 for _ in windows), 1)
    print(f"\n{'=' * 96}\nGAIN CURVE (paired B-A per window, mean over {nw} windows; "
          f"common_unobs corr_bl)")
    print(f"null D2-A: RELAX {null['mean']['mean']:+.3f} (sd {null['mean']['sd']:.3f}) | "
          f"RIDGE {null['mean_ridge']['mean']:+.3f} (sd {null['mean_ridge']['sd']:.3f})")
    hdr = (f"{'U/cp':>5} {'trk/lam':>8} {'Deff/lam':>8} {'Deff3':>6} {'swp':>5} | "
           f"{'RELAX':>7} {'sd':>5} | {'RIDGE':>7} {'sd':>5} | {'sw_RLX':>7} {'sw_RDG':>7} | verdict")
    print(hdr)
    for c in curve:
        thr_r = 2 * max(abs(null['mean']['mean']), null['mean']['sd'] / np.sqrt(nw), 1e-3)
        thr_d = 2 * max(abs(null['mean_ridge']['mean']),
                        null['mean_ridge']['sd'] / np.sqrt(nw), 1e-3)
        vr = 'GO' if c['gain_relax'] > thr_r else ('HURT' if c['gain_relax'] < -thr_r else 'null')
        vd = 'GO' if c['gain_ridge'] > thr_d else ('HURT' if c['gain_ridge'] < -thr_d else 'null')
        print(f"{c['ratio']:5.2f} {c['track_len_lam']:8.2f} {c['D_eff_lam']:8.2f} "
              f"{c['D_eff3_lam']:6.2f} {c['n_sweeps']:5.1f} | "
              f"{c['gain_relax']:+7.3f} {c['sd_relax']:5.3f} | "
              f"{c['gain_ridge']:+7.3f} {c['sd_ridge']:5.3f} | "
              f"{c['swept_gain_relax']:+7.3f} {c['swept_gain_ridge']:+7.3f} | "
              f"RLX:{vr} RDG:{vd}")

    xs = np.array([c['ratio'] for c in curve])
    deff = np.array([c['D_eff_lam'] for c in curve])
    for which, key in (('RELAX', 'gain_relax'), ('RIDGE', 'gain_ridge')):
        ys = np.array([c[key] for c in curve])
        cr_u = _crossings(xs, ys)
        cr_d = _crossings(deff, ys)
        summary[f'crossings_Ucp_{which}'] = cr_u
        summary[f'crossings_Deff_{which}'] = cr_d
        print(f"zero-crossings [{which}]: U/cp = {[round(v, 3) for v in cr_u]} | "
              f"D_eff/lam = {[round(v, 3) for v in cr_d]}  (0.7-lam operability ref)")

    # ---------------- collapse test: same U, capped track => decoupled D_eff
    if lfracs:
        print(f"\nCOLLAPSE TEST (track-length-capped variants: same U/cp, smaller D_eff)")
        print(f"{'variant':>16} {'U/cp':>5} {'Deff/lam':>8} {'Deff3':>6} | {'obs':>7} "
              f"{'pred_U':>7} {'pred_D':>7} {'pred_D3':>7} | {'|e_U|':>6} {'|e_D|':>6} "
              f"{'|e_D3|':>6}  (RIDGE; RELAX in json)")
        deff3 = np.array([c['D_eff3_lam'] for c in curve])
        order, order3 = np.argsort(deff), np.argsort(deff3)
        rows = []
        for (r, frac) in lfracs:
            c = f'B_r{r:g}_L{frac:g}'
            if c not in windows[0]['conditions']:
                continue
            row = dict(name=c, ratio=r, lfrac=frac)
            meta = [w['conditions'][c] for w in windows]
            dv = float(np.mean([m['D_eff_lam'] for m in meta]))
            dv3 = float(np.mean([m['D_eff3_lam'] for m in meta]))
            row['D_eff_lam'], row['D_eff3_lam'] = dv, dv3
            for which, key in (('relax', 'gain_relax'), ('ridge', 'gain_ridge')):
                obs = float(np.nanmean(_gain(windows, c,
                                             'mean' if which == 'relax' else 'mean_ridge')))
                ys = np.array([cc[key] for cc in curve])
                pred_u = float(ys[list(xs).index(r)]) if r in list(xs) else float(np.interp(r, xs, ys))
                row[f'obs_{which}'] = obs
                row[f'pred_U_{which}'] = pred_u
                row[f'pred_D_{which}'] = float(np.interp(dv, deff[order], ys[order]))
                row[f'pred_D3_{which}'] = float(np.interp(dv3, deff3[order3], ys[order3]))
            rows.append(row)
            print(f"{c:>16} {r:5.2f} {dv:8.2f} {dv3:6.2f} | {row['obs_ridge']:+7.3f} "
                  f"{row['pred_U_ridge']:+7.3f} {row['pred_D_ridge']:+7.3f} "
                  f"{row['pred_D3_ridge']:+7.3f} | "
                  f"{abs(row['obs_ridge'] - row['pred_U_ridge']):6.3f} "
                  f"{abs(row['obs_ridge'] - row['pred_D_ridge']):6.3f} "
                  f"{abs(row['obs_ridge'] - row['pred_D3_ridge']):6.3f}")
        summary['collapse'] = rows
        if rows:
            for which in ('relax', 'ridge'):
                eu = np.mean([abs(r[f'obs_{which}'] - r[f'pred_U_{which}']) for r in rows])
                ed = np.mean([abs(r[f'obs_{which}'] - r[f'pred_D_{which}']) for r in rows])
                ed3 = np.mean([abs(r[f'obs_{which}'] - r[f'pred_D3_{which}']) for r in rows])
                summary[f'collapse_err_U_{which}'] = float(eu)
                summary[f'collapse_err_D_{which}'] = float(ed)
                summary[f'collapse_err_D3_{which}'] = float(ed3)
                best = min((eu, 'U/cp (speed law)'), (ed, 'D_eff/lam (aperture law, T_coh)'),
                           (ed3, 'D_eff3/lam (aperture law, 3 s proxy)'))
                print(f"  [{which.upper()}] mean|err|: U-law {eu:.3f} | D-law {ed:.3f} | "
                      f"D3-law {ed3:.3f} -> best predictor: {best[1]}")
            print("  note: variants whose frac*L_diag > U*T_coh do NOT move D_eff "
                  "(cap inactive) — they instead test L_seg/coverage at fixed U and D_eff.")

    # ---------------- heading
    if args.heading and 'H_align' in windows[0]['conditions']:
        print(f"\nHEADING CHECK (own-corridor masks, paired vs A of same rep)")
        for c in ('H_align', 'H_beam', 'H_follow', 'H_head'):
            g_r = (_win_vals(windows, c, 'mean', 'common_unobs', 'corr_bl')
                   - _win_vals(windows, c, 'mean', 'common_unobs_Aref', 'corr_bl'))
            g_d = (_win_vals(windows, c, 'mean_ridge', 'common_unobs', 'corr_bl')
                   - _win_vals(windows, c, 'mean_ridge', 'common_unobs_Aref', 'corr_bl'))
            meta = windows[0]['conditions'][c]
            summary[f'heading_{c}'] = dict(gain_relax=float(np.nanmean(g_r)),
                                           sd_relax=float(np.nanstd(g_r)),
                                           gain_ridge=float(np.nanmean(g_d)),
                                           sd_ridge=float(np.nanstd(g_d)),
                                           U_over_cp=meta['U_over_cp'])
            print(f"  {c:9s} U/cp={meta['U_over_cp']:.2f}: RELAX {np.nanmean(g_r):+.3f} "
                  f"(sd {np.nanstd(g_r):.3f}) | RIDGE {np.nanmean(g_d):+.3f} "
                  f"(sd {np.nanstd(g_d):.3f})")
        print("  caveat: patch ~1.1 lam_p cannot sustain following-sea motion at U~cp "
              "(one sweep ~1.4 Tp); U*cos(theta)~cp degeneracy only probed obliquely.")
    return summary


def make_plot(summary, path, n_win, n_rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve = summary['curve']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, xkey, xlab in ((axes[0], 'ratio', 'U / c_p'),
                           (axes[1], 'D_eff_lam', 'D_eff / lam_p')):
        xs = [c[xkey] for c in curve]
        for key, sk, lab, mk in (('gain_relax', 'sd_relax', 'RELAX', 'o'),
                                 ('gain_ridge', 'sd_ridge', 'RIDGE (fixed dict)', 's')):
            ax.errorbar(xs, [c[key] for c in curve], yerr=[c[sk] for c in curve],
                        marker=mk, capsize=3, label=lab)
        for which, mkr in (('relax', 'o'), ('ridge', 's')):
            pts = summary.get('collapse', [])
            xv = [p[xkey if xkey == 'ratio' else 'D_eff_lam'] for p in pts]
            ax.scatter(xv, [p[f'obs_{which}'] for p in pts], facecolors='none',
                       edgecolors='k', marker=mkr, zorder=5,
                       label='L-capped' if which == 'relax' else None)
        nm = summary['null']['mean_ridge']
        band = 2 * max(abs(nm['mean']), nm['sd'] / max(np.sqrt(n_win), 1))
        ax.axhspan(-band, band, color='0.85', zorder=0)
        ax.axhline(0, color='k', lw=0.6)
        if xkey == 'D_eff_lam':
            ax.axvline(0.7, color='r', ls=':', lw=1, label='0.7 lam operability')
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('common_unobs corr_bl gain (B - A)')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'SA motion gain vs speed — BS_2011 real stereo '
                 f'({n_win} windows x {n_rep} reps)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default=DEFAULT_RECORD)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--T', type=float, default=30.0)
    ap.add_argument('--nobs', type=int, default=400)
    ap.add_argument('--fp_diam', type=float, default=2.5)
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--depth', type=float, default=30.0)
    ap.add_argument('--n_outer', type=int, default=2)
    ap.add_argument('--n_windows', type=int, default=None)
    ap.add_argument('--n_rep', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ratios', type=float, nargs='+', default=None,
                    help='U/c_p grid (U set per window as ratio*c_p)')
    ap.add_argument('--lfrac', type=str, nargs='*', default=None,
                    help='collapse-check variants "ratio:frac", e.g. 0.85:0.5')
    ap.add_argument('--heading', dest='heading', action='store_true', default=True)
    ap.add_argument('--no-heading', dest='heading', action='store_false')
    ap.add_argument('--ratio_best', type=float, default=0.85)
    args = ap.parse_args()

    if args.smoke:
        ratios = args.ratios or SMOKE_RATIOS
        lfrac_s = args.lfrac if args.lfrac is not None else SMOKE_LFRAC
        n_win = args.n_windows or 2
        args.n_rep = args.n_rep or 2
    else:
        ratios = args.ratios or FULL_RATIOS
        lfrac_s = args.lfrac if args.lfrac is not None else FULL_LFRAC
        n_win = args.n_windows or 8
        args.n_rep = args.n_rep or 4
    lfracs = [tuple(float(v) for v in s.split(':')) for s in lfrac_s]

    rec = WassRecord(args.record)
    nf = int(round(args.T / rec.dt))
    starts = np.linspace(200, rec.T_all - nf - 10, n_win).astype(int)
    print(f"RECORD {os.path.basename(args.record)}: {rec.T_all} frames @ "
          f"{1 / rec.dt:.1f} Hz, patch {rec.gx.max() - rec.gx.min():.1f} x "
          f"{rec.gy.max() - rec.gy.min():.1f} m, depth={args.depth:.0f} m")
    n_cond = 2 + len(ratios) + len(lfracs) + (4 if args.heading else 0)
    print(f"ratios={ratios} lfrac={lfracs} heading={args.heading} | "
          f"{n_cond} conds x {n_win} windows x {args.n_rep} reps "
          f"~= {n_cond * n_win * args.n_rep * 2.3 / 60:.0f} min")

    windows = []
    for w, i0 in enumerate(starts):
        print(f"  window {w + 1}/{n_win} (i0={i0}, t0={rec.t[i0]:.0f}s)")
        windows.append(run_window(rec, int(i0), args, ratios, lfracs,
                                  rng_seed=args.seed + 10 * w))

    summary = analyze(windows, ratios, lfracs, args)
    out = dict(record=os.path.basename(args.record), depth=args.depth, T_win=args.T,
               n_obs=args.nobs, fp_diam=args.fp_diam, margin=args.margin,
               ratios=ratios, lfracs=[list(v) for v in lfracs],
               heading=bool(args.heading), ratio_best=args.ratio_best,
               seed=args.seed, smoke=bool(args.smoke), n_rep=args.n_rep,
               deff_def=('min(U*T_coh, L_seg)+fp_diam; T_coh = 1/e crossing of Hilbert '
                         'envelope of spatially-averaged temporal autocorr, normalized '
                         'at lag=1 frame; D_eff3 uses fixed 3 s proxy'),
               summary=summary, windows=windows)
    os.makedirs('results/direction3', exist_ok=True)
    stem = 'sa_uspeed_smoke' if args.smoke else 'sa_uspeed'
    path = f'results/direction3/{stem}.json'
    json.dump(out, open(path, 'w'), indent=1, default=float)
    print(f"saved {path}")
    try:
        make_plot(summary, f'results/direction3/{stem}.png', len(windows), args.n_rep)
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == '__main__':
    main()
