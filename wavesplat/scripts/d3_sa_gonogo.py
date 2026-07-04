"""
d3_sa_gonogo.py — Synthetic-aperture GO/NO-GO on REAL waves (virtual ship-track resampling).

Question: does a MOVING virtual LiDAR footprint beat a FIXED one at MATCHED budget on
real stereo wave data? The Bergamasco/Guimaraes WASS record is a dense spatiotemporal
surface eta(t,x,y); we sample a small footprint (~0.2 lam_p) along a virtual track and
compare FIXED (A) vs MOVING (B, speed U) vs a fixed re-seed control (D2, the null),
scoring the RELAX nowcast against the real field as ground truth.

Design decisions (fairness):
  * Matched budget: identical n_obs, identical footprint diameter, identical frame
    times (linspace over the window) for A/B/D2; only the footprint CENTER differs.
  * No synthetic noise added (real stereo noise is already in the data, same for all).
  * Track = patch DIAGONAL (the longest axis, ~1.4 lam_p for BS), chosen as the
    diagonal best aligned with the spectral peak direction. U*T exceeds the diagonal,
    so the footprint ping-pongs; achieved spatial extent saturates at the diagonal.
  * Scoring split into regions (same masks for all conditions of a window):
      fixed_fp      = disk around A's center (both observed it, B only in transit)
      swept         = B's corridor minus fixed_fp (coverage gain — trivially B's)
      common_unobs  = margin away from BOTH A's disk and B's corridor
                      (genuine info-gain — the real GO/NO-GO region)
      full          = whole valid patch
  * Raw and band-limited (|k| < 1.5 kp) scores — the latter removes stereo noise floor.
  * Estimator identical for all conditions: iterative_relax_cv (adaptive klim,
    CV model order, off-grid refine); target['gt'] is never used by the estimator.

Usage:
  python3 scripts/d3_sa_gonogo.py --smoke                # few windows x {A, B(U=2), D2}
  python3 scripts/d3_sa_gonogo.py                        # full sweep: windows x U in {1,2,4} + D2
  python3 scripts/d3_sa_gonogo.py --record <path> --depth 30
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

import h5py

from realdata_analysis import _unpack, directional_spread, lowpass_field
from estimator_improved import iterative_relax_cv
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field

G = 9.81
DEFAULT_RECORD = 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc'


# ------------------------------------------------------------------ lazy record
class WassRecord:
    """Lazy access to a WASS Surfaces .nc: coordinates once, Z one time-slice at a time."""

    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.zd = self.f['Z']                                # (T, n0, n1) int16 packed
        self.T_all, self.n0, self.n1 = self.zd.shape
        X2 = _unpack(self.f['_nc4_non_coord_X'], np.s_[:])   # varies along axis0
        Y2 = _unpack(self.f['Y'], np.s_[:])                  # varies along axis1
        # same convention as realdata_analysis.load_surface_nc:
        # "x" = coordinate along axis1 (cols), "y" = coordinate along axis0 (rows)
        self.gx = np.asarray(Y2[0, :], float)                # (n1,)
        self.gy = np.asarray(X2[:, 0], float)                # (n0,)
        self.t = self.f['time'][:].astype(float)
        self.dt = float(np.median(np.diff(self.t)))
        mask = _unpack(self.f['mask_Z'], np.s_[:])
        self.valid = mask > 0                                # (n0, n1)
        self.dx = float(np.median(np.diff(self.gx)))
        self.dy = float(np.median(np.diff(self.gy)))

    def window(self, i0, n_frames):
        Z = _unpack(self.zd, np.s_[i0:i0 + n_frames])        # (nf, n0, n1) float32 [m]
        Z[:, ~self.valid] = np.nan
        return Z


# ------------------------------------------------------------------ geometry
def diagonal_track(gx, gy, r_fp, peak_dir_deg):
    """Endpoints of the inset diagonal best aligned with the peak wave axis."""
    x0, x1 = gx.min() + r_fp, gx.max() - r_fp
    y0, y1 = gy.min() + r_fp, gy.max() - r_fp
    d1 = (np.array([x0, y0]), np.array([x1, y1]))            # SW -> NE
    d2 = (np.array([x0, y1]), np.array([x1, y0]))            # NW -> SE
    p = np.array([np.cos(np.radians(peak_dir_deg)), np.sin(np.radians(peak_dir_deg))])
    best = None
    for P0, P1 in (d1, d2):
        u = (P1 - P0) / np.linalg.norm(P1 - P0)
        a = abs(float(u @ p))                                # fold handles +-180 ambiguity
        if best is None or a > best[0]:
            best = (a, P0, P1)
    return best[1], best[2], float(np.degrees(np.arccos(np.clip(best[0], 0, 1))))


def track_position(t_rel, P0, P1, U):
    """Ping-pong along segment P0->P1 at speed U (triangle-wave fold)."""
    L = float(np.linalg.norm(P1 - P0))
    s = np.mod(U * t_rel, 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    return P0[None, :] + (d[:, None] / L) * (P1 - P0)[None, :], L


# ------------------------------------------------------------------ sampling
def sample_footprint(Zwin, rec, centers, frame_idx, t_abs, r_fp, rng):
    """Draw one obs per (frame, center): a random valid grid point within r_fp of center.
    Returns P (n,3) [x, y, t_abs] and z (n,)."""
    rpx_x = int(np.floor(r_fp / rec.dx))
    rpx_y = int(np.floor(r_fp / rec.dy))
    ii, jj = np.meshgrid(np.arange(-rpx_y, rpx_y + 1), np.arange(-rpx_x, rpx_x + 1),
                         indexing='ij')
    inside = (ii * rec.dy) ** 2 + (jj * rec.dx) ** 2 <= r_fp ** 2
    off = np.column_stack([ii[inside], jj[inside]])          # (n_off, 2) row/col offsets
    xs, ys, ts, zs = [], [], [], []
    for fi, c, ta in zip(frame_idx, centers, t_abs):
        jc = int(round((c[0] - rec.gx[0]) / rec.dx))
        ic = int(round((c[1] - rec.gy[0]) / rec.dy))
        for _ in range(20):                                  # NaN-rejection retries
            oi, oj = off[rng.integers(len(off))]
            i, j = ic + oi, jc + oj
            if 0 <= i < rec.n0 and 0 <= j < rec.n1 and np.isfinite(Zwin[fi, i, j]):
                xs.append(rec.gx[j]); ys.append(rec.gy[i]); ts.append(ta)
                zs.append(float(Zwin[fi, i, j]))
                break
    P = np.column_stack([xs, ys, ts]).astype(np.float32)
    return P, np.asarray(zs, np.float32)


def build_obs(Zwin, rec, i0, n_obs, r_fp, mode, rng, U=None, track=None):
    """mode='fixed' or 'moving'. Frame times: linspace over the window (matched budget)."""
    nf = Zwin.shape[0]
    frame_idx = np.round(np.linspace(0, nf - 1, n_obs)).astype(int)
    t_rel = frame_idx * rec.dt
    t_abs = rec.t[i0 + frame_idx]                            # actual timestamps
    if mode == 'fixed':
        c = np.array([(rec.gx[0] + rec.gx[-1]) / 2, (rec.gy[0] + rec.gy[-1]) / 2])
        centers = np.repeat(c[None, :], n_obs, axis=0)
        L = 0.0
    else:
        centers, L = track_position(t_rel, track[0], track[1], U)
    P, z = sample_footprint(Zwin, rec, centers, frame_idx, t_abs, r_fp, rng)
    return P, z, centers, L


# ------------------------------------------------------------------ scoring
def region_masks(rec, r_fp, margin, P0, P1):
    """Masks over the (n0, n1) grid; identical for all conditions of a window.
    B corridor = analytic distance to the design segment P0->P1 (U-independent:
    every tested U traverses the full diagonal at least once for T=30 s)."""
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)                     # (n0, n1)
    cA = np.array([(rec.gx[0] + rec.gx[-1]) / 2, (rec.gy[0] + rec.gy[-1]) / 2])
    dA = np.hypot(Xg - cA[0], Yg - cA[1])
    seg = P1 - P0
    L2 = float(seg @ seg)
    tpar = np.clip(((Xg - P0[0]) * seg[0] + (Yg - P0[1]) * seg[1]) / L2, 0, 1)
    dB = np.hypot(Xg - (P0[0] + tpar * seg[0]), Yg - (P0[1] + tpar * seg[1]))
    v = rec.valid
    m = {}
    m['fixed_fp'] = v & (dA <= r_fp)
    m['swept'] = v & (dB <= r_fp) & ~m['fixed_fp']
    m['common_unobs'] = v & (dA > r_fp + margin) & (dB > r_fp + margin)
    m['full'] = v
    return m


def score_regions(fld, gt, Hs, masks, fld_bl, gt_bl):
    out = {}
    for name, mk in masks.items():
        n = int(mk.sum())
        if n < 20:
            out[name] = dict(n_px=n)
            continue
        f, g = fld[mk], gt[mk]
        fb, gb = fld_bl[mk], gt_bl[mk]
        out[name] = dict(
            n_px=n,
            nrmse=float(np.sqrt(np.mean((f - g) ** 2)) / Hs),
            corr=float(np.corrcoef(f, g)[0, 1]),
            nrmse_bl=float(np.sqrt(np.mean((fb - gb) ** 2)) / Hs),
            corr_bl=float(np.corrcoef(fb, gb)[0, 1]),
            nrmse_zero=float(np.sqrt(np.mean(g ** 2)) / Hs),
        )
    return out


# ------------------------------------------------------------------ ridge control
def ridge_field(P, z, depth, kp, gx, gy, t_star,
                rings=(0.6, 0.8, 1.0, 1.2, 1.4), n_theta=24):
    """Deterministic fixed-dictionary control: ridge fit of dispersion-shell
    components (identical dictionary for every condition). Removes the greedy
    RELAX detection instability, so A-vs-B differences here measure the
    INFORMATION content of the sampling geometry, not estimator luck."""
    kr = kp * np.asarray(rings)
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    KX = (kr[:, None] * np.cos(th)[None, :]).ravel()
    KY = (kr[:, None] * np.sin(th)[None, :]).ravel()
    obs = np.column_stack([P, z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(KX, KY, obs, depth, lambda_reg=0.5)
    return render_wave_field(KX, KY, A, phi, depth, gx, gy, t_star)


# ------------------------------------------------------------------ experiment
def characterize(Zwin, rec, depth):
    """Hs, spatial peak (lam_p, kp, dir, spread); Tp from kp via dispersion (a 30 s
    window is too short for a reliable Welch peak — spatial kp is robust)."""
    Hs = 4.0 * float(np.nanstd(Zwin))
    gt = Zwin[-1]
    spread, lam_p, pdir, kp = directional_spread(np.nan_to_num(gt), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    Tp = float(2 * np.pi / om_p)
    cp = float(om_p / kp)                                    # phase speed at the peak
    return Hs, lam_p, kp, pdir, spread, Tp, cp


def mean_scores(reps):
    """Average score dicts over reps (region -> metric -> mean over reps)."""
    out = {}
    for region in reps[0]['scores']:
        keys = [k for k in reps[0]['scores'][region] if k != 'n_px']
        out[region] = {k: float(np.mean([r['scores'][region][k] for r in reps
                                         if k in r['scores'][region]])) for k in keys}
        out[region]['n_px'] = reps[0]['scores'][region]['n_px']
    return out


def run_window(rec, i0, args, U_list, rng_seed):
    nf = int(round(args.T / rec.dt))
    Zwin = rec.window(i0, nf)
    Hs_w, lam_p, kp, pdir, spread, Tp, cp = characterize(Zwin, rec, args.depth)
    assert 0.05 < Hs_w < 5.0, f"Hs={Hs_w:.3f} m not physical — int16 scaling wrong?"
    gt = Zwin[-1]
    gt_f = np.nan_to_num(gt)
    k_cut = 1.5 * kp
    gt_bl = lowpass_field(gt_f, rec.dx, rec.dy, k_cut)
    # window-relative times (keeps float32 phases small in the torch refine)
    t_off = float(rec.t[i0])
    t_star = float(rec.t[i0 + nf - 1]) - t_off
    target = {'gx': rec.gx, 'gy': rec.gy, 't': t_star, 'gt': gt_f, 'Hs': Hs_w}
    r_fp = args.fp_diam / 2.0
    P0, P1, track_angle = diagonal_track(rec.gx, rec.gy, r_fp, pdir)

    conds = {'A': dict(mode='fixed'), 'D2': dict(mode='fixed')}
    for U in U_list:
        conds[f'B_U{U:g}'] = dict(mode='moving', U=U)

    L_path = float(np.linalg.norm(P1 - P0))
    masks = region_masks(rec, r_fp, args.margin, P0, P1)

    res = dict(i0=int(i0), t0=float(rec.t[i0]), Hs_win=Hs_w, Tp=Tp, cp=cp, lam_p=lam_p,
               kp=kp, peak_dir=pdir, spread_deg=spread, k_cut=k_cut,
               track_angle_to_peak_deg=track_angle, L_path=L_path,
               region_px={k: int(v.sum()) for k, v in masks.items()},
               conditions={})

    for ci, (name, cf) in enumerate(conds.items()):
        reps = []
        for rep in range(args.n_rep):
            # D2 = null control: same protocol as A under an independent seed stream
            seed = rng_seed + 997 * ci + 13 * rep
            rng = np.random.default_rng(seed)
            tic = time.time()
            if cf['mode'] == 'fixed':
                P, z, _, _ = build_obs(Zwin, rec, i0, args.nobs, r_fp, 'fixed', rng)
                extent = args.fp_diam
            else:
                P, z, _, L = build_obs(Zwin, rec, i0, args.nobs, r_fp, 'moving', rng,
                                       U=cf['U'], track=(P0, P1))
                extent = min(cf['U'] * args.T, L) + args.fp_diam  # achieved spatial span
            P[:, 2] -= t_off
            fld, N, klim = iterative_relax_cv(P.copy(), z.copy(), args.depth, target,
                                              n_outer=args.n_outer, seed=seed)
            fld_bl = lowpass_field(fld, rec.dx, rec.dy, k_cut)
            sc = score_regions(fld, gt_f, Hs_w, masks, fld_bl, gt_bl)
            # deterministic fixed-dictionary control (same dictionary for A/B/D2)
            frd = ridge_field(P, z, args.depth, kp, rec.gx, rec.gy, t_star)
            frd_bl = lowpass_field(frd, rec.dx, rec.dy, k_cut)
            sc_r = score_regions(frd, gt_f, Hs_w, masks, frd_bl, gt_bl)
            reps.append(dict(seed=seed, n_obs=int(len(z)), N_comp=int(N),
                             klim=float(klim), runtime_s=round(time.time() - tic, 1),
                             scores=sc, scores_ridge=sc_r))
        entry = dict(reps=reps, mean=mean_scores(reps),
                     mean_ridge=mean_scores([dict(scores=r['scores_ridge']) for r in reps]),
                     extent_m=float(extent), extent_lam=float(extent / lam_p))
        if cf['mode'] == 'moving':
            entry.update(U=cf['U'], U_over_cp=float(cf['U'] / cp),
                         track_len_m=float(cf['U'] * args.T),
                         track_len_lam=float(cf['U'] * args.T / lam_p),
                         n_sweeps=float(cf['U'] * args.T / L_path))
        res['conditions'][name] = entry
        u = entry['mean']['common_unobs']
        ur = entry['mean_ridge']['common_unobs']
        ub = [r['scores']['common_unobs']['corr_bl'] for r in reps]
        print(f"    {name:6s} N={np.mean([r['N_comp'] for r in reps]):4.1f} "
              f"ext={extent / lam_p:.2f}lam | RELAX unobs bl {u['corr_bl']:.3f} "
              f"(sd {np.std(ub):.2f}) swept bl {entry['mean']['swept']['corr_bl']:.3f} | "
              f"RIDGE unobs bl {ur['corr_bl']:.3f} swept bl "
              f"{entry['mean_ridge']['swept']['corr_bl']:.3f} "
              f"[{sum(r['runtime_s'] for r in reps):.0f}s/{args.n_rep}reps]")
    del Zwin
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default=DEFAULT_RECORD)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--T', type=float, default=30.0, help='window length [s]')
    ap.add_argument('--nobs', type=int, default=400)
    ap.add_argument('--fp_diam', type=float, default=2.5, help='footprint diameter [m]')
    ap.add_argument('--margin', type=float, default=1.5, help='unobs margin beyond r_fp [m]')
    ap.add_argument('--U', type=float, nargs='+', default=None, help='speeds [m/s]')
    ap.add_argument('--depth', type=float, default=30.0, help='Black Sea platform ~30 m')
    ap.add_argument('--n_outer', type=int, default=2)
    ap.add_argument('--n_windows', type=int, default=None)
    ap.add_argument('--n_rep', type=int, default=None,
                    help='reps per condition (sampling+estimator seeds averaged)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rec = WassRecord(args.record)
    nf = int(round(args.T / rec.dt))
    if args.smoke:
        U_list = args.U or [2.0]
        n_win = args.n_windows or 3
        args.n_rep = args.n_rep or 3
    else:
        U_list = args.U or [1.0, 2.0, 4.0]
        n_win = args.n_windows or 8
        args.n_rep = args.n_rep or 4
    starts = np.linspace(200, rec.T_all - nf - 10, n_win).astype(int)

    print(f"RECORD {os.path.basename(args.record)}: {rec.T_all} frames @ {1 / rec.dt:.1f} Hz, "
          f"patch {rec.gx.max() - rec.gx.min():.1f} x {rec.gy.max() - rec.gy.min():.1f} m, "
          f"dx={rec.dx:.3f} m, depth={args.depth:.0f} m")
    print(f"windows={list(starts)}  T={args.T:.0f}s  n_obs={args.nobs}  "
          f"fp={args.fp_diam:.1f}m  U={U_list}")

    windows = []
    for w, i0 in enumerate(starts):
        print(f"  window {w + 1}/{n_win} (i0={i0}, t0={rec.t[i0]:.0f}s)")
        windows.append(run_window(rec, int(i0), args, U_list, rng_seed=args.seed + 10 * w))

    # ---------------- summary
    def collect(cond, region, key, which='mean'):
        vals = [w['conditions'][cond][which][region][key] for w in windows
                if cond in w['conditions'] and key in w['conditions'][cond][which].get(region, {})]
        return float(np.mean(vals)) if vals else float('nan')

    summary = {}
    conds = list(windows[0]['conditions'].keys())
    b_keys = [c for c in conds if c.startswith('B_')]
    print(f"\n{'=' * 74}\nSUMMARY (mean over {len(windows)} windows x {args.n_rep} reps)")
    for which, tag in (('mean', 'RELAX'), ('mean_ridge', 'RIDGE (fixed dictionary — pure info content)')):
        print(f"--- {tag} ---")
        print(f"{'cond':8s} {'full bl':>8} | {'UNOBS corr':>10} {'UNOBS bl':>9} "
              f"{'UNOBS nRMSE_bl':>14} | {'swept bl':>9} {'fp bl':>7}")
        summary[which] = {}
        for c in conds:
            row = dict(
                full_corr=collect(c, 'full', 'corr', which),
                full_corr_bl=collect(c, 'full', 'corr_bl', which),
                unobs_corr=collect(c, 'common_unobs', 'corr', which),
                unobs_corr_bl=collect(c, 'common_unobs', 'corr_bl', which),
                unobs_nrmse_bl=collect(c, 'common_unobs', 'nrmse_bl', which),
                swept_corr_bl=collect(c, 'swept', 'corr_bl', which),
                fixed_fp_corr_bl=collect(c, 'fixed_fp', 'corr_bl', which))
            summary[which][c] = row
            print(f"{c:8s} {row['full_corr_bl']:8.3f} | "
                  f"{row['unobs_corr']:10.3f} {row['unobs_corr_bl']:9.3f} "
                  f"{row['unobs_nrmse_bl']:14.3f} | {row['swept_corr_bl']:9.3f} "
                  f"{row['fixed_fp_corr_bl']:7.3f}")
        null_gap = abs(summary[which]['A']['unobs_corr_bl']
                       - summary[which]['D2']['unobs_corr_bl'])
        summary[which]['null_gap_unobs_corr_bl'] = null_gap
        print(f"  null (|A - D2| UNOBS corr_bl): {null_gap:.3f}")
        for b in b_keys:
            d = summary[which][b]['unobs_corr_bl'] - summary[which]['A']['unobs_corr_bl']
            print(f"  GO/NO-GO [{tag.split()[0]}] {b} - A on common_unobs corr_bl: {d:+.3f} "
                  f"({'beyond null' if abs(d) > 2 * max(null_gap, 1e-3) else 'within null'})")

    out = dict(record=os.path.basename(args.record), depth=args.depth, T_win=args.T,
               n_obs=args.nobs, fp_diam=args.fp_diam, margin=args.margin,
               U_list=U_list, seed=args.seed, smoke=bool(args.smoke), n_rep=args.n_rep,
               summary=summary, windows=windows)
    os.makedirs('results/direction3', exist_ok=True)
    path = 'results/direction3/sa_gonogo_smoke.json' if args.smoke \
        else 'results/direction3/sa_gonogo.json'
    json.dump(out, open(path, 'w'), indent=2)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
