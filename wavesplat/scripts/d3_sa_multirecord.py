"""
d3_sa_multirecord.py — Multi-record D_eff/lambda collapse: U-law vs aperture-law
discrimination on REAL waves (thesis-pivot GO/NO-GO).

Why: on a single record T_coh is fixed, so U/c_p and D_eff/lam_p are collinear along
the main sweep and the two laws cannot be discriminated (d3_sa_uspeed honest caveat).
Across records T_coh/Tp spans 0.3 - 4.9 (15x, measured 2026-07-04), so at the SAME
U/c_p different records achieve very different D_eff/lam_p:
    U * T_coh / lam_p = (U/c_p) * (T_coh/Tp)          [c_p = lam_p/Tp at the peak]
Decisive disagreements (aperture-law predictions, fp/lam ~ 0.18):
  * AA15a (T_coh/Tp ~ 4.9): gain should turn POSITIVE already at U/c_p ~ 0.1-0.2,
    where BS_2011 (0.5) showed low-U HARM. U-law predicts harm on both.
  * BS11b (T_coh/Tp ~ 0.32): D_eff <= 0.5 lam even at U/c_p = 1.0 -> NO gain at any
    tested speed. U-law predicts gain at high U/c_p.
  * LJ18n (La Jument Grid1, patch 0.19 lam_true): D_eff capped at ~0.2 lam_true by
    the patch no matter the speed (even U/c_p_true ~ 0.9) -> negative control.
Protocol: run_window() of d3_sa_uspeed verbatim (matched budget, centered-shrunk
track, U-independent common_unobs scoring, RELAX + RIDGE, D2 null) with per-record
SCALED parameters: fp_diam ~ 0.18 lam_p, T_win ~ 10 Tp, margin = 0.6 fp_diam, so all
records are dimensionlessly matched to the verified BS_2011 anchor setup.

Record-specific handling (audited 2026-07-04, tmp/record_characterization.json):
  * LJ_2018 Grid2 has dead pixels NOT flagged by mask_Z (constant fill -13.94,
    temporal std = 0) -> robust valid mask = mask_Z & (temporal std > 0.1 median)
    & (finite fraction > 0.9), probed on 160 frames. Applied to every record (no-op
    where data are clean).
  * LJ_2018 Grid1 (40 m patch under lam ~ 300 m storm swell): the spatial-peak
    lam_p/c_p from characterize() are PATCH-LIMITED artifacts. The run uses them
    only to set U = ratio * c_p_char (giving realistic 0.8-16 m/s speeds); the
    ANALYSIS renormalizes with lam_ref/c_p_ref taken from Grid2 (same sea).
  * AA_2014 / YS_2017 have large invalid regions (valid 0.43/0.57). Center disk is
    100% valid (A unaffected); moving tracks lose some samples to NaN rejection
    (~ conservative bias AGAINST the motion gain). Achieved n_obs is logged per rep.
  * Spatial stride 2 on both LJ grids (dx 0.1/0.5 m vs lam ~ 300 m: massively
    oversampled; keeps window memory < 400 MB).

Usage:
  python3 scripts/d3_sa_multirecord.py --smoke                 # AA14, 2 win x 2 rep
  python3 scripts/d3_sa_multirecord.py                         # all records (hours)
  python3 scripts/d3_sa_multirecord.py --records BS11a AA15a   # subset (merges JSON)
"""
import os
import sys
import json
import time
import zlib
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from realdata_analysis import _unpack
from d3_sa_gonogo import WassRecord
from d3_sa_uspeed import run_window

DEFAULT_RATIOS = [0.1, 0.2, 0.43, 0.7, 1.0]

# fp = 0.18 * lam_p(record), T = ~10 Tp, from tmp/record_characterization.json.
# lfracs on AA15a: at U/c_p 0.2/0.43 the capped track (frac 0.25) pulls D_eff from
# >1 lam back to ~0.65 lam at the SAME U -> within-record decoupling probe on the
# one record where T_coh is long enough for the cap to bind.
RECORDS = {
    'BS11a': dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
                  depth=30.0, fp=2.5, T=30.0, nw=6, stride=1, hs=(0.05, 2.0),
                  note='anchor — same record/params as d3_sa_gonogo + d3_sa_uspeed'),
    'AA15a': dict(path='data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc',
                  depth=17.0, fp=9.3, T=58.0, nw=6, stride=1, hs=(0.5, 5.0),
                  lfracs=[(0.2, 0.25), (0.43, 0.25)],
                  note='decisive long-T_coh (T_coh/Tp~4.9): aperture-law predicts low-U gain'),
    'BS11b': dict(path='data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',
                  depth=30.0, fp=1.7, T=25.0, nw=6, stride=1, hs=(0.05, 2.0),
                  note='decisive short-T_coh (~0.32 Tp): aperture-law predicts NO gain any U'),
    'AA14':  dict(path='data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc',
                  depth=17.0, fp=6.3, T=47.0, nw=6, stride=1, hs=(0.3, 4.0),
                  note='valid 0.43 — center disk clean, tracks lose samples to NaN'),
    'BS13a': dict(path='data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc',
                  depth=30.0, fp=3.4, T=35.0, nw=6, stride=1, hs=(0.05, 2.5)),
    'BS13b': dict(path='data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc',
                  depth=30.0, fp=3.1, T=33.0, nw=6, stride=1, hs=(0.05, 2.5)),
    'AA15b': dict(path='data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc',
                  depth=17.0, fp=7.2, T=51.0, nw=6, stride=1, hs=(0.3, 4.0)),
    'YS17':  dict(path='data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',
                  depth=50.0, fp=10.5, T=61.0, nw=5, stride=1, hs=(0.5, 5.0),
                  note='valid 0.57; only 450 s record'),
    'LJ18n': dict(path='data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc',
                  depth=45.0, fp=7.3, T=120.0, nw=5, stride=2, hs=(3.0, 15.0),
                  ratios=[0.1, 0.2, 0.43, 1.0, 2.0],
                  lam_ref=301.4, cp_ref=18.59, Tp_ref=16.21,
                  note='NEGATIVE CONTROL: 40 m patch under lam~300 m swell; '
                       'characterize() lam_p/c_p are patch artifacts — analysis uses *_ref'),
    'LJ18g': dict(path='data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc',
                  depth=45.0, fp=54.0, T=130.0, nw=4, stride=2, hs=(5.0, 20.0),
                  note='storm swell at 1.3 lam patch; robust mask kills dead pixels'),
}


class MRecord(WassRecord):
    """WassRecord + optional spatial stride + robust valid mask (dead-pixel kill)."""

    def __init__(self, path, stride=1, robust=True):
        super().__init__(path)
        self.stride = int(stride)
        if self.stride > 1:
            s = self.stride
            self.gx, self.gy = self.gx[::s], self.gy[::s]
            self.valid = self.valid[::s, ::s]
            self.n0, self.n1 = self.valid.shape
            self.dx = float(np.median(np.diff(self.gx)))
            self.dy = float(np.median(np.diff(self.gy)))
        if robust:
            rb = self._robust_mask()
            killed = int((self.valid & ~rb).sum())
            self.valid = self.valid & rb
            if killed:
                print(f"  robust mask: killed {killed} px "
                      f"({killed / self.valid.size:.1%} of grid)")

    def _robust_mask(self, n_probe=160):
        """Kill dead pixels (temporal std ~ 0: un-flagged constant fill, LJ Grid2)
        AND noise-dominated pixels (temporal std > 2x median: long-range stereo
        junk, individual-pixel 'Hs' up to 3x the sea's). No-op on clean records."""
        idx = np.linspace(0, self.T_all - 1, n_probe).astype(int)
        s = self.stride
        frames = np.stack([
            _unpack(self.zd, np.s_[int(i)])[::s, ::s] for i in idx])
        tsd = np.nanstd(frames, axis=0)
        finite = np.isfinite(frames).mean(axis=0)
        med = float(np.nanmedian(tsd[tsd > 0]))
        return (finite > 0.9) & (tsd > 0.1 * med) & (tsd < 2.0 * med)

    def window(self, i0, n_frames):
        if self.stride > 1:
            Z = _unpack(self.zd, np.s_[i0:i0 + n_frames, ::self.stride, ::self.stride])
        else:
            Z = _unpack(self.zd, np.s_[i0:i0 + n_frames])
        Z[:, ~self.valid] = np.nan
        # static stereo-bias removal: per-pixel temporal mean over the window
        # (LJ Grid2 carries a smooth reconstruction-bias field, spatial std ~4 m
        # vs Hs ~11 m; waves are zero-mean over T_win ~ 10 Tp, so this is a
        # sensor calibration applied identically to GT and obs of all conditions)
        Z -= np.nanmean(Z, axis=0, keepdims=True)
        return Z


def valid_aware_track(rec, r_fp, pdir, vf_min=0.92):
    """Longest well-aligned chord whose corridor is >= vf_min valid.

    Candidates: the two inset diagonals (anchor protocol) + a fan of center
    chords (10-deg steps) with lateral offsets up to +-25% of the patch.
    Deterministic. Among candidates passing vf_min, maximize length x |alignment
    with the peak direction|; if none passes, take the max-valid-fraction one.
    Returns (P0, P1, angle_to_peak_deg) like diagonal_track."""
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    x0, x1 = rec.gx.min() + r_fp, rec.gx.max() - r_fp
    y0, y1 = rec.gy.min() + r_fp, rec.gy.max() - r_fp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pvec = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    cands = [(np.array([x0, y0]), np.array([x1, y1])),
             (np.array([x0, y1]), np.array([x1, y0]))]
    half = min((x1 - x0) / 2, (y1 - y0) / 2)
    for ang in np.arange(0.0, 180.0, 10.0):
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        nrm = np.array([-u[1], u[0]])
        for f in (-0.25, -0.125, 0.0, 0.125, 0.25):
            c = np.array([cx, cy]) + nrm * (f * 2 * half)
            tlo, thi = -np.inf, np.inf
            ok = True
            for lo, hi, ci, ui in ((x0, x1, c[0], u[0]), (y0, y1, c[1], u[1])):
                if abs(ui) < 1e-9:
                    if not (lo <= ci <= hi):
                        ok = False
                        break
                else:
                    ta, tb = (lo - ci) / ui, (hi - ci) / ui
                    tlo, thi = max(tlo, min(ta, tb)), min(thi, max(ta, tb))
            if ok and thi - tlo >= 4 * r_fp:
                cands.append((c + tlo * u, c + thi * u))
    best, best_fb = None, None
    for P0, P1 in cands:
        seg = P1 - P0
        L = float(np.linalg.norm(seg))
        tpar = np.clip(((Xg - P0[0]) * seg[0] + (Yg - P0[1]) * seg[1]) / (L * L), 0, 1)
        d = np.hypot(Xg - (P0[0] + tpar * seg[0]), Yg - (P0[1] + tpar * seg[1]))
        cor = d <= r_fp
        vf = float(rec.valid[cor].mean()) if cor.any() else 0.0
        a = abs(float((seg / L) @ pvec))
        ang_pk = float(np.degrees(np.arccos(np.clip(a, 0, 1))))
        if best_fb is None or vf > best_fb[0]:
            best_fb = (vf, P0, P1, ang_pk)
        if vf >= vf_min and (best is None or L * a > best[0]):
            best = (L * a, P0, P1, ang_pk, vf)
    if best is not None:
        L = float(np.linalg.norm(best[2] - best[1]))
        print(f"    valid_aware_track: L={L:.0f}m vf={best[4]:.2f} "
              f"angle_to_peak={best[3]:.0f}deg")
        return best[1], best[2], best[3]
    print(f"    valid_aware_track: NO chord reaches vf>={vf_min}, "
          f"falling back to vf={best_fb[0]:.2f}")
    return best_fb[1], best_fb[2], best_fb[3]


def run_record(key, cfg, args):
    ratios = args.ratios or cfg.get('ratios', DEFAULT_RATIOS)
    lfracs = cfg.get('lfracs', []) if not args.smoke else []
    if args.smoke:
        ratios = [0.2, 1.0]

    rec = MRecord(cfg['path'], stride=cfg.get('stride', 1))
    # valid-aware chord only where the blind diagonal would cross NaN holes;
    # clean records keep the exact anchor (gonogo/uspeed) diagonal protocol
    track_fn = valid_aware_track if rec.valid.mean() < 0.98 else None
    rc = argparse.Namespace(
        T=cfg['T'], nobs=args.nobs, fp_diam=cfg['fp'], margin=0.6 * cfg['fp'],
        depth=cfg['depth'], n_outer=args.n_outer, n_rep=args.n_rep,
        heading=False, ratio_best=0.85, hs_lo=cfg['hs'][0], hs_hi=cfg['hs'][1],
        track_fn=track_fn, estimator=getattr(args, 'estimator', 'relax'),
        sparse_kp=getattr(args, 'sparse_kp', False))
    nf = int(round(rc.T / rec.dt))
    n_win = args.n_windows or (2 if args.smoke else cfg['nw'])
    starts = np.linspace(200, rec.T_all - nf - 10, n_win).astype(int)
    print(f"[{key}] {os.path.basename(cfg['path'])}: {rec.T_all} fr @ {1/rec.dt:.0f} Hz, "
          f"patch {rec.gx.max()-rec.gx.min():.0f}x{rec.gy.max()-rec.gy.min():.0f} m "
          f"(stride {rec.stride}), fp={rc.fp_diam} m, T={rc.T} s, "
          f"ratios={ratios}, lfracs={lfracs}, {n_win} win x {rc.n_rep} rep")

    windows = []
    for w, i0 in enumerate(starts):
        print(f"  window {w + 1}/{n_win} (i0={i0}, t0={rec.t[i0]:.0f}s)")
        rec_seed = zlib.crc32(key.encode()) % 100000     # deterministic per record
        windows.append(run_window(rec, int(i0), rc, ratios,
                                  [tuple(v) for v in lfracs],
                                  rng_seed=args.seed + rec_seed + 10 * w))
    rec.f.close()

    return dict(record=os.path.basename(cfg['path']), path=cfg['path'],
                depth=cfg['depth'], T_win=rc.T, n_obs=rc.nobs, fp_diam=rc.fp_diam,
                margin=rc.margin, stride=cfg.get('stride', 1), ratios=ratios,
                lfracs=[list(v) for v in lfracs], n_rep=rc.n_rep,
                lam_ref=cfg.get('lam_ref'), cp_ref=cfg.get('cp_ref'),
                Tp_ref=cfg.get('Tp_ref'), note=cfg.get('note', ''),
                seed=args.seed, smoke=bool(args.smoke), windows=windows)


def quick_gain_table(key, out):
    """Per-record print: mean common_unobs corr_bl gain (B - A) over windows."""
    rows = {}
    for w in out['windows']:
        A = w['conditions']['A']
        for name, e in w['conditions'].items():
            if not name.startswith('B_'):
                continue
            g = (e['mean']['common_unobs']['corr_bl']
                 - A['mean']['common_unobs']['corr_bl'])
            gr = (e['mean_ridge']['common_unobs']['corr_bl']
                  - A['mean_ridge']['common_unobs']['corr_bl'])
            nobs = np.mean([r['n_obs'] for r in e['reps']])
            rows.setdefault(name, []).append((g, gr, e['D_eff_lam'],
                                              e['U_over_cp'], nobs))
    print(f"  [{key}] gain summary (RELAX / RIDGE on common_unobs corr_bl):")
    for name in sorted(rows, key=lambda n: np.mean([v[3] for v in rows[n]])):
        v = np.array(rows[name])
        print(f"    {name:14s} U/cp={v[:,3].mean():.2f} Deff={v[:,2].mean():.2f}lam "
              f"gain {v[:,0].mean():+.3f} / {v[:,1].mean():+.3f} "
              f"(n={len(v)}, n_obs={v[:,4].mean():.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', nargs='+', default=None,
                    help=f"subset of {list(RECORDS)} (default: all)")
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--nobs', type=int, default=400)
    ap.add_argument('--n_outer', type=int, default=2)
    ap.add_argument('--n_windows', type=int, default=None)
    ap.add_argument('--n_rep', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ratios', type=float, nargs='+', default=None)
    ap.add_argument('--estimator', choices=['relax', 'mmse', 'mmse_net'],
                    default='relax',
                    help="mmse: spectrum-weighted linear MMSE over the fine "
                         "continuous bank (2026-07-05 synthetic winner); "
                         "mmse_net: amortized prior net v3 gamma (G3 "
                         "transfer gate)")
    ap.add_argument('--sparse_kp', action='store_true',
                    help="audit E-12d/g fix: MMSE arms get kp & Hs from the "
                         "sparse obs only (kp_hint=None, Hs=4*std(z)), not from "
                         "the dense stereo GT — the fair 'MMSE > RELAX' re-run")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.n_rep = args.n_rep or (2 if args.smoke else 3)

    keys = args.records or (['AA14'] if args.smoke else list(RECORDS))
    bad = [k for k in keys if k not in RECORDS]
    assert not bad, f"unknown records {bad}; known: {list(RECORDS)}"

    stem = 'sa_multirecord_smoke' if args.smoke else 'sa_multirecord'
    path = args.out or f'results/direction3/{stem}.json'
    os.makedirs('results/direction3', exist_ok=True)
    store = {}
    if os.path.exists(path):
        store = json.load(open(path))
        print(f"merging into existing {path} (has {list(store)})")

    for key in keys:
        tic = time.time()
        out = run_record(key, RECORDS[key], args)
        out['runtime_min'] = round((time.time() - tic) / 60.0, 1)
        store[key] = out
        json.dump(store, open(path, 'w'), indent=1, default=float)
        quick_gain_table(key, out)
        print(f"[{key}] done in {out['runtime_min']} min -> {path}\n")


if __name__ == '__main__':
    main()
