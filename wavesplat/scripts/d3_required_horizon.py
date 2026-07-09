"""d3_required_horizon.py — re-score WaveSplat vs WavePrior at the horizon
the controller actually needs, on the quantity it actually consumes.

Two corrections to every comparison we have run so far:

1. HORIZON. The FF preview requirement is 0.1-0.5 s (optimum 0.1-0.3 s,
   tuned 0.2 s; STUDY_GUIDE_2026-07-06 §1.2). Our "requirement band" of
   0-2 s is already 4-10x longer than that, and the GIF comparison ran to
   20 s. Leads are scored here on a 0.1 s grid out to 1.0 s.

2. QUANTITY. The controller never touches the elevation. It consumes
   (hydrofoil/controller.py:135)

       alpha_wave = -slope - w / U

   the surface slope along the heading and the orbital vertical velocity.
   Scoring eta is scoring a quantity nobody uses. Both are obtained here
   by finite differences through the SAME estimator, so no extra machinery
   is needed: slope from a +-2 m stencil along the heading, w from a
   +-0.1 s stencil in time.

nRMSE is normalized per scene by the GT std of each quantity over a long
reference series (leads 0-10 s), not over the 1 s window itself: a 1 s
window of a 6 s wave barely varies, and self-normalizing there would
manufacture huge nRMSE out of nothing.

Usage: python3 scripts/d3_required_horizon.py --case all --reps N
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse

G = 9.81
NET_CKPT = 'results/direction3/prior_net_v3.pt'
OUT_JSON = 'results/direction3/required_horizon.json'

DEPTH = 50.0
SCAN_DT = 0.5
M_PER_SCAN = 12
NOISE = 0.02
SPINUP = 10.0
DT_HOS = 0.1

LEADS = np.round(np.arange(0.0, 1.001, 0.1), 3)     # required-horizon grid
REF_LEADS = np.round(np.arange(0.0, 10.001, 0.5), 3)  # normalizer only
H_X = 2.0            # m, slope stencil along heading
H_T = 0.1            # s, vertical-velocity stencil (matches real dt)

REQ_BANDS = [(0.1, 0.3, 'optimum 0.1-0.3s'), (0.0, 0.5, 'operational 0-0.5s'),
             (0.0, 1.0, 'margin 0-1.0s')]


def stencil(px, py, t, hvec):
    """5 query points around a track point: center, +-x along heading, +-t."""
    hx, hy = hvec
    return np.array([
        [px, py, t],
        [px + H_X * hx, py + H_X * hy, t],
        [px - H_X * hx, py - H_X * hy, t],
        [px, py, t + H_T],
        [px, py, t - H_T],
    ])


def derive(vals, U):
    """(eta, slope, w, alpha_wave) from a 5-point stencil block."""
    c, xp, xm, tp, tm = vals
    slope = (xp - xm) / (2 * H_X)
    w = (tp - tm) / (2 * H_T)
    return c, slope, w, -slope - w / U


# ----------------------------------------------------------------- scenes
def build_sim(kind, seed):
    U, Hs, Tp, W = 3.0, 2.0, 8.0, 20.0
    kp = (2 * np.pi / Tp) ** 2 / G
    rng = np.random.default_rng(400 + seed)

    if kind == 'linear':
        comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=200,
                                   depth=DEPTH, direction_spread_deg=25.0,
                                   seed=9000 + seed)
        kx = np.array([c.kx for c in comps])
        ky = np.array([c.ky for c in comps])
        A = np.array([c.A for c in comps])
        ph = np.array([c.phi for c in comps])
        om = omega_of(kx, ky, DEPTH)

        def eta(x, y, t):
            ps = kx[None] * np.atleast_1d(x)[:, None] \
                + ky[None] * np.atleast_1d(y)[:, None] \
                - om[None] * np.atleast_1d(t)[:, None] + ph[None]
            return (A[None] * np.cos(ps)).sum(1)

        def fresh_eta():
            return eta
        t0, cx, cy = W, 0.0, 0.0
    else:
        Lx, Ly, Nx = 500.0, 400.0, 128

        def _make():
            """A HOS simulator only marches forward. Any GT pass that needs a
            time earlier than the current state (the w stencil asks for
            t0-0.1 s) must start from a fresh instance, or get_eta_at
            silently returns the surface at the wrong time."""
            s = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=128,
                                       depth=DEPTH, M=3))
            s.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=0.0,
                                 direction_spread_deg=25.0, seed=9000 + seed)

            def _eta(x, y, t):
                if t < s.t - 1e-9:
                    raise RuntimeError(f'HOS asked to go back: {t} < {s.t}')
                while s.t < t - 1e-9:
                    s.step(min(DT_HOS, t - s.t))
                return s.get_eta_at(np.mod(np.atleast_1d(x), Lx),
                                    np.mod(np.atleast_1d(y), Ly))
            return _eta

        eta = _make()          # this instance is consumed by the fan scans
        fresh_eta = _make
        t0, cx, cy = SPINUP + W, Lx / 2, Ly / 2

    def ship_pos(t):
        return np.array([cx - U * (t - t0), cy])

    hvec = np.array([-1.0, 0.0])                # heading (ship travels -x)

    scan_ts = np.arange(t0 - W, t0 + 1e-9, SCAN_DT)
    P_all, z_all = [], []
    for t_s in scan_ts:
        sp = ship_pos(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, M_PER_SCAN)
        xs = sp[0] - rr * np.cos(aa)
        ys = sp[1] + rr * np.sin(aa)
        zz = eta(xs, ys, t_s)
        P_all.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
        z_all.append(zz + rng.normal(0, NOISE, M_PER_SCAN))
    P, z = np.vstack(P_all), np.concatenate(z_all)
    return dict(P=P, z=z, t0=t0, U=U, Hs=Hs, kp=kp, depth=DEPTH,
                ship_pos=ship_pos, hvec=hvec, fresh_eta=fresh_eta, seed=seed)


def build_real(record, win):
    from realdata_analysis import directional_spread
    from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
    from d3_sa_multirecord import MRecord

    cfg = RECORDS[record]
    U, depth = cfg['U'], cfg['depth']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    W_max, lead_max = 40.0, cfg['lead_max']
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(lead_max / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    # one frame BEFORE i0: the w stencil queries t0 - 0.1 s, which would
    # otherwise clamp to frame 0 and silently corrupt the vertical velocity.
    Zfut = rec.window(i0 - 1, n_lead + 2)
    K0 = 1                                  # index of t0 inside Zfut
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    _, _, pdir_ax, kp = directional_spread(np.nan_to_num(Zmem[-1]),
                                           rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    wv = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    hvec = -wv
    T_tr = W_max + lead_max

    def ship_pos(t):
        return np.array([cx, cy]) + U * (t - (t0 - W_max) - T_tr / 2) * hvec

    r0, r1 = cfg['r_fan']
    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        sp = ship_pos(t_j)
        RX, RY = Xg - sp[0], Yg - sp[1]
        R = np.hypot(RX, RY)
        ca = (RX * hvec[0] + RY * hvec[1]) / np.maximum(R, 1e-9)
        m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(12, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, 0.02, len(pick)))
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    m = P_all[:, 2] >= t0 - 20.0
    P, z = P_all[m], z_all[m]
    if len(z) > 600:
        ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
        P, z = P[ii], z[ii]

    def eta(x, y, t):
        """GT elevation by nearest-frame + bilinear (frames are 0.1 s apart,
        and every query time here lands on the frame grid by construction)."""
        out = []
        for xx, yy, tt in zip(np.atleast_1d(x), np.atleast_1d(y),
                              np.atleast_1d(t)):
            k = int(round((tt - t0) / rec.dt)) + K0
            if k < 0 or k >= Zfut.shape[0]:
                out.append(np.nan)
                continue
            out.append(bilinear(Zfut[k], rec.gx, rec.gy, xx, yy))
        return np.array(out)

    def fresh_eta():
        return eta

    return dict(P=P, z=z, t0=t0, U=U, Hs=Hs, kp=kp, depth=depth,
                ship_pos=ship_pos, hvec=hvec, fresh_eta=fresh_eta, seed=i0)


# ----------------------------------------------------------------- scoring
def track_queries(sc, leads):
    return np.vstack([stencil(*sc['ship_pos'](sc['t0'] + l_),
                              sc['t0'] + l_, sc['hvec']) for l_ in leads])


def gt_series(sc, lead_sets):
    """GT (eta, slope, w, alpha) per lead, for several lead grids at once.

    All queries go through ONE fresh evaluator in ascending time order, so a
    stateful (HOS) surface is marched forward exactly once and never asked to
    step backwards.
    """
    blocks = [track_queries(sc, lds) for lds in lead_sets]
    Q = np.vstack(blocks)
    eta = sc['fresh_eta']()
    vals = np.empty(len(Q))
    for i in np.argsort(Q[:, 2], kind='stable'):
        vals[i] = eta(Q[i, 0:1], Q[i, 1:2], Q[i, 2])[0]

    out, off = [], 0
    for lds in lead_sets:
        v = vals[off:off + 5 * len(lds)]
        out.append(np.array([derive(v[5 * i:5 * i + 5], sc['U'])
                             for i in range(len(lds))]))
        off += 5 * len(lds)
    return out


def pred_series(fq, leads, U):
    return np.array([derive(fq[5 * i:5 * i + 5], U) for i in range(len(leads))])


QNAMES = ['eta', 'slope', 'w', 'alpha_wave']


def score(gt, pred, norm, leads):
    out = {}
    for lo, hi, lbl in REQ_BANDS:
        m = (leads >= lo) & (leads <= hi)
        ent = {}
        for qi, qn in enumerate(QNAMES):
            g, p = gt[m, qi], pred[m, qi]
            ok = np.isfinite(g) & np.isfinite(p)
            if ok.sum() < 3 or norm[qi] <= 0:
                continue
            g, p = g[ok], p[ok]
            ent[qn] = {
                'nrmse': float(np.sqrt(np.mean((p - g) ** 2)) / norm[qi]),
                'corr': float(np.corrcoef(p, g)[0, 1])
                        if np.std(p) > 1e-12 and np.std(g) > 1e-12 else None,
            }
        out[lbl] = ent
    return out


def run_case(case, reps, record='YS17'):
    net = get_net(NET_CKPT)
    rows = []
    for r in range(reps):
        sc = build_sim(case, r) if case in ('linear', 'hos') \
            else build_real(record, r)
        # One ascending GT pass: the short required-horizon grid, and the long
        # reference series that sets the nRMSE normalizer.
        gt, ref = gt_series(sc, [LEADS, REF_LEADS])
        norm = np.nanstd(ref, axis=0)

        Pq = track_queries(sc, LEADS)
        s0 = 0.05 * sc['Hs']
        fq_ws = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq,
                                   sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'])
        fq_wp = net_gamma_mmse(net, sc['P'], sc['z'], sc['depth'], Pq,
                               sigma0=s0, seed=sc['seed'], kp_hint=sc['kp'])
        row = {'norm': {q: float(n) for q, n in zip(QNAMES, norm)},
               'WaveSplat': score(gt, pred_series(fq_ws, LEADS, sc['U']),
                                  norm, LEADS),
               'WavePrior': score(gt, pred_series(fq_wp, LEADS, sc['U']),
                                  norm, LEADS)}
        rows.append(row)
        a = row['WaveSplat']['operational 0-0.5s']['alpha_wave']['nrmse']
        b = row['WavePrior']['operational 0-0.5s']['alpha_wave']['nrmse']
        print(f"  [{case} rep {r}] alpha_wave 0-0.5s nRMSE: "
              f"WS {a:.3f} / WP {b:.3f}", flush=True)

    agg = {'case': case, 'reps': reps, 'leads': LEADS.tolist()}
    for name in ('WaveSplat', 'WavePrior'):
        agg[name] = {}
        for _, _, lbl in REQ_BANDS:
            agg[name][lbl] = {}
            for qn in QNAMES:
                for sub in ('nrmse', 'corr'):
                    v = [r[name][lbl][qn][sub] for r in rows
                         if qn in r[name][lbl]
                         and r[name][lbl][qn][sub] is not None]
                    if v:
                        agg[name][lbl][f'{qn}_{sub}'] = {
                            'mean': float(np.mean(v)),
                            'std': float(np.std(v))}
    # paired per-rep difference on the control-relevant headline
    d = [r['WaveSplat']['operational 0-0.5s']['alpha_wave']['nrmse']
         - r['WavePrior']['operational 0-0.5s']['alpha_wave']['nrmse']
         for r in rows]
    agg['paired_delta_alpha_0-0.5s'] = {
        'mean': float(np.mean(d)), 'std': float(np.std(d)),
        'per_rep': [float(x) for x in d],
        'wp_wins': int(sum(1 for x in d if x > 0)),
        'note': 'positive = WavePrior better'}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all',
                    choices=['linear', 'hos', 'real', 'all'])
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--record', default='YS17')
    a = ap.parse_args()
    cases = ['linear', 'hos', 'real'] if a.case == 'all' else [a.case]

    allm = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for case in cases:
        reps = a.reps if case != 'hos' else min(a.reps, 3)
        key = case if case != 'real' else f'real_{a.record}'
        print(f"[{key}] reps={reps}", flush=True)
        allm[key] = run_case(case, reps, a.record)
    with open(OUT_JSON, 'w') as f:
        json.dump(allm, f, indent=1)
    print(f"\n-> {OUT_JSON}")


if __name__ == '__main__':
    sys.exit(main())
