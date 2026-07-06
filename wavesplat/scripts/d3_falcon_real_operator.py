"""
d3_falcon_real_operator.py — GIVEN-SENSOR validation, step 2 (2026-07-06):
Falcon-like observation operator on REAL stereo waves.

A virtual ship transits the YS17 stereo patch (220 x 240 m — the only
record large enough for the real fan geometry; dt = 0.10 s = Falcon's
native 10 FPS) on an exogenous straight course INTO the waves (head
seas). Every frame the forward fan (FOV +-60 deg, range 20-100 m — the
grazing-incidence usable band of the 250m@10% spec) samples valid stereo
pixels; z gets N(0, 2 cm) Falcon range noise. CAUSAL estimation from the
trailing W seconds, encounter forecast at the ship's FUTURE positions
(control-relevant target, lead 0-20 s), scored against the stereo
surface itself. hand-MMSE only (net is OOD for this operator until
retrained); kp_hint from causal in-window characterize (the 2026-07-05
prescription: external kp beats blind spectral placement on real data).

Simplifications (documented, v1): raydrop = uniform random m_scan pixels
per frame inside the fan (no reflectivity/incidence model — the m_cap
subsample dominates anyway); ship course is straight at constant U
(exogenous trajectory, per the given-sensor doctrine).

Peak-direction 180-deg ambiguity of the single-frame FFT is resolved
temporally (advect the lowpassed frame by +-c_p * dt along the axis,
keep the sign that correlates with the later frame).

Usage: python3 scripts/d3_falcon_real_operator.py [--record YS17]
       [--U 2.0] [--w_mems 10 20 40] [--n_win 4]
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.ndimage import shift as ndshift

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from realdata_analysis import directional_spread
from d3_sa_gonogo import lowpass_field
from d3_sa_multirecord import MRecord
from d3_sa_nomp import mmse_bank_estimate
from d3_sa_achievability_gap import corr

G = 9.81
OUT = 'results/direction3/falcon_real_operator_{}.json'   # per record

RECORDS = {
    # only records whose patch can host a (possibly scaled) forward fan
    'YS17': dict(path='data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',
                 depth=50.0, stride=2, r_fan=(20.0, 100.0), U=2.0,
                 lead_max=20.0, note='220x240 m — real-scale Falcon fan'),
    'AA15a': dict(path='data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc',
                  depth=17.0, stride=1, r_fan=(8.0, 40.0), U=1.2,
                  lead_max=14.0, note='65x85 m — fan scaled ~0.4x'),
}


def bilinear(Z, gx, gy, xq, yq):
    """NaN-aware bilinear sample of one frame Z (ny, nx) at (xq, yq)."""
    ix = np.interp(xq, gx, np.arange(len(gx)))
    iy = np.interp(yq, gy, np.arange(len(gy)))
    i0x, i0y = int(np.floor(ix)), int(np.floor(iy))
    if not (0 <= i0x < len(gx) - 1 and 0 <= i0y < len(gy) - 1):
        return np.nan
    fx, fy = ix - i0x, iy - i0y
    q = Z[i0y:i0y + 2, i0x:i0x + 2]
    if not np.all(np.isfinite(q)):
        return np.nan
    return float(q[0, 0] * (1 - fx) * (1 - fy) + q[0, 1] * fx * (1 - fy)
                 + q[1, 0] * (1 - fx) * fy + q[1, 1] * fx * fy)


def resolve_direction(Zmem, rec, kp, pdir_axis, cp, dt_gap=1.0):
    """Return propagation direction [deg]: pdir_axis or its 180-flip,
    whichever advects frame -1-n onto frame -1 better."""
    n = max(1, int(round(dt_gap / rec.dt)))
    k_cut = 1.5 * kp
    f0 = lowpass_field(np.nan_to_num(Zmem[-1 - n]), rec.dx, rec.dy, k_cut)
    f1 = lowpass_field(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy, k_cut)
    best = None
    for sgn in (1.0, -1.0):
        th = np.radians(pdir_axis)
        dxy = sgn * cp * n * rec.dt * np.array([np.cos(th), np.sin(th)])
        # rows vary along gy (y), cols along gx (x)
        fs = ndshift(f0, (dxy[1] / rec.dy, dxy[0] / rec.dx), order=1,
                     mode='constant', cval=0.0)
        c = corr(fs, f1)
        if best is None or c > best[0]:
            best = (c, sgn)
    pdir = pdir_axis if best[1] > 0 else pdir_axis + 180.0
    return float((pdir + 180.0) % 360.0 - 180.0), float(best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default='YS17', choices=list(RECORDS))
    ap.add_argument('--U', type=float, default=None,
                    help='ship speed [m/s] (default: record preset)')
    ap.add_argument('--w_mems', type=float, nargs='+', default=[10., 20., 40.])
    ap.add_argument('--n_win', type=int, default=6)
    ap.add_argument('--m_scan', type=int, default=12,
                    help='fan pixels kept per frame (raydrop proxy)')
    ap.add_argument('--m_cap', type=int, default=600)
    ap.add_argument('--noise', type=float, default=0.02,
                    help='Falcon range noise 1-sigma [m]')
    a = ap.parse_args()
    cfg = RECORDS[a.record]
    U = a.U if a.U is not None else cfg['U']
    tag = a.record if a.U is None else f"{a.record}_U{U:g}"
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth = cfg['depth']
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    W_max = max(a.w_mems)
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)

    lo = nfw + 1
    hi = rec.T_all - n_lead - 2
    assert hi > lo, 'record too short'
    i0s = np.linspace(lo, hi, a.n_win).astype(int)

    out = dict(record=a.record, U=U, leads=leads.tolist(), m_scan=a.m_scan,
               m_cap=a.m_cap, noise=a.noise, r_fan=list(cfg['r_fan']),
               w_mems=list(a.w_mems), windows=[])
    for wi, i0 in enumerate(i0s):
        rng = np.random.default_rng(700 + wi)
        Zmem = rec.window(i0 - nfw + 1, nfw)              # causal frames
        Zfut = rec.window(i0, n_lead + 1)                 # scoring only
        t0 = float(rec.t[i0])
        Hs = 4.0 * float(np.nanstd(Zmem))
        spread, lam_p, pdir_ax, kp = directional_spread(
            np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
        om_p = np.sqrt(G * kp * np.tanh(kp * depth))
        Tp, cp = 2 * np.pi / om_p, om_p / kp
        pdir, c_adv = resolve_direction(Zmem, rec, kp, pdir_ax, cp)
        w = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
        h = -w                                            # head seas
        T_tr = W_max + cfg['lead_max']

        def ship_pos(t):
            return (np.array([cx, cy])
                    + U * (t - (t0 - W_max) - T_tr / 2) * h)

        # ---- forward-fan sampling, every frame of the memory window
        r0, r1 = cfg['r_fan']
        P_all, z_all = [], []
        for j in range(nfw):
            t_j = float(rec.t[i0 - nfw + 1 + j])
            sp = ship_pos(t_j)
            RX, RY = Xg - sp[0], Yg - sp[1]
            R = np.hypot(RX, RY)
            ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
            m = ((R >= r0) & (R <= r1) & (ca >= 0.5)      # +-60 deg fan
                 & np.isfinite(Zmem[j]))
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, min(a.m_scan, len(idx)), replace=False)
            iy, ix = np.unravel_index(pick, Zmem[j].shape)
            P_all.append(np.column_stack([
                rec.gx[ix], rec.gy[iy], np.full(len(pick), t_j)]))
            z_all.append(Zmem[j][iy, ix]
                         + rng.normal(0, a.noise, len(pick)))
        P_all = np.vstack(P_all)
        z_all = np.concatenate(z_all)

        # ---- encounter targets (ship future positions) + stereo GT
        tq = t0 + leads
        Pq_enc = np.array([[*ship_pos(t), t] for t in tq])
        gt_q = np.array([bilinear(Zfut[int(round((t - t0) / rec.dt))],
                                  rec.gx, rec.gy, px, py)
                         for (px, py, t) in Pq_enc])
        n_gt = int(np.isfinite(gt_q).sum())
        # ---- map-nowcast diagnostic: fan-region pixels at t0 (same solve)
        sp0 = ship_pos(t0)
        RX, RY = Xg - sp0[0], Yg - sp0[1]
        R = np.hypot(RX, RY)
        ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
        mmap = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zfut[0]))
        midx = np.flatnonzero(mmap)
        midx = rng.choice(midx, min(400, len(midx)), replace=False)
        my, mx = np.unravel_index(midx, mmap.shape)
        Pq_map = np.column_stack([rec.gx[mx], rec.gy[my],
                                  np.full(len(midx), t0)])
        gt_map = Zfut[0][my, mx]
        Pq = np.vstack([Pq_enc, Pq_map])
        row = dict(i0=int(i0), t0=t0, Hs=Hs, Tp=float(Tp), lam_p=float(lam_p),
                   kp=float(kp), pdir=pdir, adv_corr=c_adv, spread=float(spread),
                   n_gt_valid=n_gt, gt=gt_q.tolist(), by_W={})
        print(f"[win {wi}] t0={t0:.0f}s Hs={Hs:.2f} Tp={Tp:.1f}s "
              f"lam={lam_p:.0f}m pdir={pdir:.0f}deg(adv c={c_adv:.2f}) "
              f"gt_valid={n_gt}/{len(gt_q)}", flush=True)
        for W in a.w_mems:
            m = P_all[:, 2] >= t0 - W
            P, z = P_all[m], z_all[m]
            if len(z) > a.m_cap:
                ii = np.random.default_rng(0).choice(len(z), a.m_cap,
                                                     replace=False)
                P, z = P[ii], z[ii]
            fq = mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05 * Hs,
                                    kp_hint=kp, seed=int(i0))
            f_enc, f_map = fq[:len(tq)], fq[len(tq):]
            c_map = corr(f_map, gt_map)
            row['by_W'][str(W)] = dict(n_obs=int(len(z)), pred=f_enc.tolist(),
                                       map_corr=c_map)
            v = np.isfinite(gt_q)
            cc = corr(f_enc[v], gt_q[v]) if v.sum() > 3 else np.nan
            print(f"    W={W:4.0f}s n={len(z):4d} enc_corr={cc:+.2f} "
                  f"map_now_corr={c_map:+.2f}", flush=True)
        out['windows'].append(row)
        json.dump(out, open(OUT.format(tag), "w"), indent=1,
                  default=float)

    # ---- pooled summary: corr by lead band, per memory W
    print(f"\n===== {a.record} encounter forecast, pooled over "
          f"{len(out['windows'])} windows (U={U} m/s head seas) =====")
    print(f"{'W_mem':>6s} | corr by lead band: "
          + "  ".join(f"{lo:.0f}-{hi:.0f}s" for lo, hi in ((0, 10), (10, 20))))
    for W in a.w_mems:
        preds = np.array([w_['by_W'][str(W)]['pred'] for w_ in out['windows']])
        gts = np.array([w_['gt'] for w_ in out['windows']])
        line = f"{W:6.0f} |"
        for lo_, hi_ in ((0, 10), (10, 20)):
            m = (leads >= lo_) & (leads <= hi_)
            p_, g_ = preds[:, m].ravel(), gts[:, m].ravel()
            v = np.isfinite(g_)
            line += f"   {corr(p_[v], g_[v]):+.2f} (n={v.sum()})"
        print(line)
    print(f"saved {OUT.format(tag)}")


if __name__ == '__main__':
    main()
