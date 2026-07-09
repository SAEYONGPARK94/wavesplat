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
REF_TIMES = np.round(np.arange(0.0, 20.001, 0.5), 3)  # normalizer only

# Stencils. GT slope and w are finite differences of a MEASURED surface, so
# they amplify its reconstruction noise as sigma/(sqrt(2) h). tmp/probe_gt_*
# fitted var = sigma^2/(2h^2) + var_true on the stereo records and found
# sigma ~ 0.03-0.08 m (YS17) and ~0.5 m (LJ18g), R^2 up to 0.98 — i.e. the
# white-noise law, not unresolved short waves. At the h_x=2 m / h_t=0.1 s we
# used first, the storm GT derivatives had SNR 0.63 (slope) and 0.39 (w):
# we were scoring estimators against noise. These wider stencils restore
# SNR ~1.3-2.5 (storm) and ~2.6-7 (YS17) at a central-difference bias of
# sinc(k h) ~ 5% (YS17) / 1% (LJ18g). GT and prediction get the SAME
# operator, so the comparison stays consistent; what changes is that the
# target is now a band-limited slope — the honest limit of what stereo GT
# supports.
H_X = 4.0            # m, slope stencil along heading
H_T = 0.4            # s, vertical-velocity stencil

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
    """(eta, slope, w, alpha_wave) from a 5-point stencil block.

    alpha_wave is undefined at rest: the foil's wave-induced angle of attack
    is atan(w/U), which the controller linearizes to w/U, and that diverges
    as U -> 0. Physically the foil is not foilborne below takeoff speed, so
    there is no feed-forward requirement there at all. We return NaN rather
    than a large number, so a stationary case reports eta/slope/w only.
    """
    c, xp, xm, tp, tm = vals
    slope = (xp - xm) / (2 * H_X)
    w = (tp - tm) / (2 * H_T)
    alpha = np.nan if U <= 1e-9 else -slope - w / U
    return c, slope, w, alpha


# ----------------------------------------------------------------- scenes
def build_sim(kind, seed, U=3.0, W=20.0):
    Hs, Tp = 2.0, 8.0
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
                ship_pos=ship_pos, hvec=hvec, fresh_eta=fresh_eta, seed=seed,
                center=(cx, cy))


def build_real(record, win, U=None, W_track=40.0, W_obs=20.0, lead_span=None):
    """U and the window lengths are properties of the SYNTHETIC ship track we
    lay over the stereo patch, not of the record. The patch is finite, so a
    design-speed track only fits once the track span shrinks with the
    (much shorter) required horizon.
    """
    from realdata_analysis import directional_spread
    from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
    from d3_sa_multirecord import MRecord

    cfg = RECORDS[record]
    depth = cfg['depth']
    U = cfg['U'] if U is None else float(U)
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    W_max = float(W_track)
    # Track geometry span (how far the synthetic ship runs) is independent of
    # how many future frames we need to hold: the normalizer samples 20 s at a
    # fixed point regardless of where the ship goes.
    lead_span_s = cfg['lead_max'] if lead_span is None else float(lead_span)
    frame_span_s = max(cfg['lead_max'], float(REF_TIMES[-1]) + 1.0)
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(frame_span_s / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    # one frame BEFORE i0: the w stencil queries t0 - 0.1 s, which would
    # otherwise clamp to frame 0 and silently corrupt the vertical velocity.
    # Pad backwards by the w stencil: t0 - H_T must exist as a frame, or the
    # vertical velocity at lead 0 silently clamps to the first frame.
    pad = int(np.ceil(H_T / rec.dt)) + 1
    Zfut = rec.window(i0 - pad, n_lead + pad + 1)
    K0 = pad                                # index of t0 inside Zfut
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    _, _, pdir_ax, kp = directional_spread(np.nan_to_num(Zmem[-1]),
                                           rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    wv = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    hvec = -wv
    T_tr = W_max + lead_span_s
    # Does the synthetic track (plus the forward fan) still fit in the patch?
    need = U * T_tr + cfg['r_fan'][1]
    have = min(rec.gx.max() - rec.gx.min(), rec.gy.max() - rec.gy.min())
    if need > have:
        print(f"    [warn] {record}: track {U*T_tr:.0f} m + fan "
              f"{cfg['r_fan'][1]:.0f} m = {need:.0f} m > patch {have:.0f} m "
              f"— scans will fall off the patch", flush=True)

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
    m = P_all[:, 2] >= t0 - float(W_obs)
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
                ship_pos=ship_pos, hvec=hvec, fresh_eta=fresh_eta, seed=i0,
                center=(cx, cy))


# ----------------------------------------------------------------- scoring
def track_queries(sc, leads):
    return np.vstack([stencil(*sc['ship_pos'](sc['t0'] + l_),
                              sc['t0'] + l_, sc['hvec']) for l_ in leads])


def fixed_queries(sc, times):
    """Stencils at a FIXED point (the patch centre) over a long time span.

    The nRMSE normalizer must not move with the ship speed, or a speed sweep
    changes the denominator along with the numerator and the curve means
    nothing. Sampling the field at one point gives a per-scene scale that
    depends on the sea (and, through w/U, on U) but not on the track.
    """
    cx, cy = sc['center']
    return np.vstack([stencil(cx, cy, sc['t0'] + t, sc['hvec'])
                      for t in times])


def eval_gt(sc, blocks):
    """Evaluate GT on several query blocks with ONE fresh evaluator, in
    ascending time order — a stateful (HOS) surface is marched forward once
    and never asked to step backwards."""
    Q = np.vstack(blocks)
    eta = sc['fresh_eta']()
    vals = np.empty(len(Q))
    for i in np.argsort(Q[:, 2], kind='stable'):
        vals[i] = eta(Q[i, 0:1], Q[i, 1:2], Q[i, 2])[0]
    out, off = [], 0
    for b in blocks:
        v = vals[off:off + len(b)]
        out.append(np.array([derive(v[5 * i:5 * i + 5], sc['U'])
                             for i in range(len(b) // 5)]))
        off += len(b)
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


def run_case(case, reps, record='YS17', U=None, W_track=40.0, W_obs=20.0,
             lead_span=None):
    net = get_net(NET_CKPT)
    rows = []
    for r in range(reps):
        if case in ('linear', 'hos'):
            sc = build_sim(case, r, U=(3.0 if U is None else U), W=W_obs)
        else:
            sc = build_real(record, r, U=U, W_track=W_track, W_obs=W_obs,
                            lead_span=lead_span)
        # One ascending GT pass: the required-horizon track, and the
        # fixed-point reference series that sets the nRMSE normalizer.
        gt, ref = eval_gt(sc, [track_queries(sc, LEADS),
                               fixed_queries(sc, REF_TIMES)])
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
        band = row['WaveSplat']['operational 0-0.5s']
        hq = 'alpha_wave' if 'alpha_wave' in band else 'slope'
        a = row['WaveSplat']['operational 0-0.5s'][hq]['nrmse']
        b = row['WavePrior']['operational 0-0.5s'][hq]['nrmse']
        print(f"  [{case} rep {r}] {hq} 0-0.5s nRMSE: "
              f"WS {a:.3f} / WP {b:.3f}", flush=True)

    agg = {'case': case, 'reps': reps, 'leads': LEADS.tolist(),
           'U': (3.0 if U is None else U) if case in ('linear', 'hos') else U,
           'W_obs': W_obs, 'W_track': W_track,
           'stencil': {'h_x_m': H_X, 'h_t_s': H_T},
           'normalizer': 'GT std at fixed patch centre over 0-20 s'}
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
    # paired per-rep differences, every quantity (alpha absent when U = 0)
    agg['paired_delta_0-0.5s'] = {'note': 'positive = WavePrior better'}
    for qn in QNAMES:
        d = [r['WaveSplat']['operational 0-0.5s'][qn]['nrmse']
             - r['WavePrior']['operational 0-0.5s'][qn]['nrmse']
             for r in rows if qn in r['WaveSplat']['operational 0-0.5s']]
        if not d:
            continue
        agg['paired_delta_0-0.5s'][qn] = {
            'mean': float(np.mean(d)), 'std': float(np.std(d)),
            'per_rep': [float(x) for x in d],
            'wp_wins': int(sum(1 for x in d if x > 0)),
            'n': len(d)}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all',
                    choices=['linear', 'hos', 'real', 'all'])
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--record', default='YS17')
    ap.add_argument('--U', type=float, nargs='+', default=[None],
                    help='ship speed(s) m/s; several = speed sweep. '
                         'Design point is 11.3 (22 kt).')
    ap.add_argument('--W_obs', type=float, default=20.0,
                    help='observation memory [s]')
    ap.add_argument('--W_track', type=float, default=40.0,
                    help='real: synthetic track span before t0 [s]')
    ap.add_argument('--lead_span', type=float, default=None,
                    help='real: track span after t0 [s] (default record max)')
    ap.add_argument('--tag', default='', help='suffix for the result key')
    a = ap.parse_args()
    cases = ['linear', 'hos', 'real'] if a.case == 'all' else [a.case]

    allm = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for case in cases:
        reps = a.reps if case != 'hos' else min(a.reps, 3)
        for U in a.U:
            key = case if case != 'real' else f'real_{a.record}'
            if U is not None:
                key += f'_U{U:g}'
            key += a.tag
            print(f"[{key}] reps={reps} U={U} W_obs={a.W_obs}", flush=True)
            allm[key] = run_case(case, reps, a.record, U=U, W_obs=a.W_obs,
                                 W_track=a.W_track, lead_span=a.lead_span)
    with open(OUT_JSON, 'w') as f:
        json.dump(allm, f, indent=1)
    print(f"\n-> {OUT_JSON}")


if __name__ == '__main__':
    sys.exit(main())
