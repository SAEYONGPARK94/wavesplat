"""d3_static_field.py — STATIONARY-platform wave-field estimation.

The thesis object is the ESTIMATOR, not the controller: the field must be
recovered in any state, including at rest. A stationary platform is the hard
case for a scanning LiDAR, because the synthetic aperture that ship motion
would sweep out collapses to a single fixed fan. What is left is: the waves
themselves advect through that fan, and the estimator must propagate what it
saw into the rest of the domain.

So the question this script answers is spatial, not temporal:

    inside the observed fan, and OUTSIDE it, how well is the field recovered,
    as a function of range and of forecast lead, with U = 0?

Both estimators see identical observations and solve the same physics MMSE on
the same bank; only the prior differs (analytic beamform power vs the
amortized prior net). Emits results/gifs/static_<case>.gif,
results/direction3/static_field.json, results/direction3/static_field_<case>.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

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
OUT_GIF = 'results/gifs/static_{}.gif'
OUT_PNG = 'results/direction3/static_field_{}.png'
OUT_JSON = 'results/direction3/static_field.json'

DEPTH = 50.0
SCAN_DT = 0.5          # fan revisit
M_PER_SCAN = 12
NOISE = 0.02
W_OBS = 20.0           # observation memory [s]
M_CAP = 600            # observation budget, matched across all cases
LEADS = np.round(np.arange(0.0, 5.001, 0.25), 3)

# Canonical geometry, taken from the episodes rather than invented here:
# frames/gt_grid_x spans 10..250 m and gt_grid_y -100..100 m (80 x 60), and
# the LiDAR returns of a frame span range 12..203 m over azimuth +-60 deg
# (configs/default.yaml: fov_h_deg 120). So the DOMAIN is that rectangle and
# the fan is the sensor's covered subset of it. Earlier versions of this
# script used a 20-100 m gate borrowed from the moving-platform scripts,
# where it existed only to keep a synthetic track inside a stereo patch.
DOM_X = (10.0, 250.0)
DOM_Y = (-100.0, 100.0)
DX, DY = 3.0, 3.4
R_FAN = (15.0, 200.0)
HALF_FOV = np.pi / 3   # +-60 deg


def fan_mask(X, Y, cx, cy, hvec):
    RX, RY = X - cx, Y - cy
    R = np.hypot(RX, RY)
    ca = (RX * hvec[0] + RY * hvec[1]) / np.maximum(R, 1e-9)
    return (R >= R_FAN[0]) & (R <= R_FAN[1]) & (ca >= np.cos(HALF_FOV)), R


def scene_sim(kind, seed=0):
    Hs, Tp = 2.0, 8.0
    kp = (2 * np.pi / Tp) ** 2 / G
    rng = np.random.default_rng(400 + seed)
    # Waves run toward -x (direction_deg=180) so they travel from the far end
    # of the domain toward the sensor at the origin: the sensor sees water
    # before that water reaches it, which is the only configuration in which
    # a stationary platform is worth anything.
    WDIR = 180.0
    if kind == 'linear':
        comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                                   direction_deg=WDIR,
                                   direction_spread_deg=25.0, seed=9000 + seed)
        kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
        A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
        om = omega_of(kx, ky, DEPTH)

        def make_eta():
            def eta(x, y, t):
                ps = kx[None] * np.atleast_1d(x)[:, None] \
                    + ky[None] * np.atleast_1d(y)[:, None] \
                    - om[None] * np.atleast_1d(t)[:, None] + ph[None]
                return (A[None] * np.cos(ps)).sum(1)
            return eta
        t0, cx, cy = W_OBS, 0.0, 0.0
        off = np.array([0.0, 0.0])
    else:
        Lx, Ly, Nx = 500.0, 400.0, 128

        def make_eta():
            s = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=128,
                                       depth=DEPTH, M=3))
            s.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=WDIR,
                                 direction_spread_deg=25.0, seed=9000 + seed)

            def eta(x, y, t):
                while s.t < t - 1e-9:
                    s.step(min(0.1, t - s.t))
                return s.get_eta_at(np.mod(np.atleast_1d(x), Lx),
                                    np.mod(np.atleast_1d(y), Ly))
            return eta
        # place the sensor so the whole 240 x 200 m domain sits inside the box
        t0, cx, cy = 10.0 + W_OBS, 100.0, 200.0
        off = np.array([cx, cy])

    hvec = np.array([1.0, 0.0])            # sensor looks +x, into the waves
    eta_obs = make_eta()
    P_all, z_all = [], []
    for t_s in np.arange(t0 - W_OBS, t0 + 1e-9, SCAN_DT):
        rr = rng.uniform(*R_FAN, M_PER_SCAN)
        aa = rng.uniform(-HALF_FOV, HALF_FOV, M_PER_SCAN)
        xs = cx + rr * np.cos(aa)
        ys = cy + rr * np.sin(aa)
        keep = ((xs >= cx + DOM_X[0]) & (xs <= cx + DOM_X[1]) &
                (ys >= cy + DOM_Y[0]) & (ys <= cy + DOM_Y[1]))
        if keep.sum() == 0:
            continue
        xs, ys = xs[keep], ys[keep]
        zz = eta_obs(xs, ys, t_s)
        P_all.append(np.column_stack([xs, ys, np.full(len(xs), t_s)]))
        z_all.append(zz + rng.normal(0, NOISE, len(xs)))

    gx = np.arange(cx + DOM_X[0], cx + DOM_X[1] + 0.1, DX)
    gy = np.arange(cy + DOM_Y[0], cy + DOM_Y[1] + 0.1, DY)
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    if len(z_all) > M_CAP:
        ii = np.random.default_rng(1).choice(len(z_all), M_CAP, replace=False)
        P_all, z_all = P_all[ii], z_all[ii]
    return dict(P=P_all, z=z_all, t0=t0,
                Hs=Hs, kp=kp, depth=DEPTH, center=(cx, cy), hvec=hvec,
                gx=gx, gy=gy, make_eta=make_eta, seed=seed,
                label={'linear': 'Linear sim', 'hos': 'HOS nonlinear sim'}[kind],
                sequential=(kind == 'hos'))


def scene_real(record='YS17', win=3):
    from realdata_analysis import directional_spread
    from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
    from d3_sa_multirecord import MRecord

    cfg = RECORDS[record]
    depth = cfg['depth']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(W_OBS / rec.dt))
    n_lead = int(round((LEADS[-1] + 1.0) / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    # GT lookup spans memory AND future: callers that resample observations
    # (d3_static_limit) query times before t0, and a future-only window
    # silently returns NaN for every one of them.
    Zall = rec.window(i0 - nfw + 1, nfw + n_lead + 1)
    K_T0 = nfw - 1                       # index of t0 inside Zall
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    _, _, pdir_ax, kp = directional_spread(np.nan_to_num(Zmem[-1]),
                                           rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    wv = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    hvec = -wv                                 # fan looks into the waves
    # Stand the platform at the DOWNWAVE edge of the patch, so the waves cross
    # the patch toward it and the fan covers as much of the patch as the
    # sensor's range allows. Sitting at the patch centre would throw away half
    # the domain behind the sensor.
    half = 0.5 * np.array([rec.gx.max() - rec.gx.min(),
                           rec.gy.max() - rec.gy.min()])
    cx, cy = np.array([cx, cy]) + wv * (half * np.abs(wv)).sum() * 0.85

    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        m, _ = fan_mask(Xg, Yg, cx, cy, hvec)
        m &= np.isfinite(Zmem[j])
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(M_PER_SCAN, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, NOISE, len(pick)))

    def make_eta():
        def eta(x, y, t):
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            # t is usually a scalar for a whole frame; zip would silently
            # truncate the query set to one point without this broadcast.
            t = np.broadcast_to(np.atleast_1d(t), x.shape)
            out = []
            for xx, yy, tt in zip(x, y, t):
                k = int(round((tt - t0) / rec.dt)) + K_T0
                if k < 0 or k >= Zall.shape[0]:
                    out.append(np.nan); continue
                out.append(bilinear(Zall[k], rec.gx, rec.gy, xx, yy))
            return np.array(out)
        return eta

    # Stereo frames arrive every 0.1 s, so 20 s of fan scans is 2400 points —
    # five times what the simulator scenes get. Cap to the same budget the
    # real-operator scripts use, or the real cases would be compared with a
    # different amount of information.
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    if len(z_all) > M_CAP:
        ii = np.random.default_rng(0).choice(len(z_all), M_CAP, replace=False)
        P_all, z_all = P_all[ii], z_all[ii]

    sq = max(1, int(round(3.0 / rec.dx)))
    return dict(P=P_all, z=z_all, t0=t0,
                Hs=Hs, kp=kp, depth=depth, center=(cx, cy), hvec=hvec,
                gx=rec.gx[::sq], gy=rec.gy[::sq], make_eta=make_eta, seed=i0,
                label=f'Real {record} (stationary)', sequential=False)


def solve(sc):
    Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
    npx = Xq.size
    Pq = np.vstack([np.column_stack([Xq.ravel(), Yq.ravel(),
                                     np.full(npx, sc['t0'] + l_)])
                    for l_ in LEADS])
    s0 = 0.05 * sc['Hs']
    print(f"  {len(sc['z'])} obs, {len(Pq)} queries", flush=True)
    fq_ws = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq, sigma0=s0,
                               kp_hint=sc['kp'], seed=sc['seed'])
    net = get_net(NET_CKPT)
    fq_wp = net_gamma_mmse(net, sc['P'], sc['z'], sc['depth'], Pq, sigma0=s0,
                           seed=sc['seed'], kp_hint=sc['kp'])
    shp = (len(LEADS), *Xq.shape)
    eta = sc['make_eta']()
    gt = np.empty(shp)
    for i, l_ in enumerate(LEADS):                 # ascending: HOS-safe
        gt[i] = eta(Xq.ravel(), Yq.ravel(), sc['t0'] + l_).reshape(Xq.shape)
    return Xq, Yq, gt, fq_ws.reshape(shp), fq_wp.reshape(shp)


def metrics(sc, Xq, Yq, gt, ws, wp):
    cx, cy = sc['center']
    inside, R = fan_mask(Xq, Yq, cx, cy, sc['hvec'])
    sg = float(np.nanstd(gt))
    out = {'label': sc['label'], 'n_obs': int(len(sc['z'])),
           'gt_std': sg, 'leads': LEADS.tolist(),
           'frac_domain_observed': float(inside.mean())}

    def nr(p, g, m):
        mm = m & np.isfinite(p) & np.isfinite(g)
        if mm.sum() < 16:
            return None
        return float(np.sqrt(np.mean((p[mm] - g[mm]) ** 2)) / sg)

    def cr(p, g, m):
        mm = m & np.isfinite(p) & np.isfinite(g)
        if mm.sum() < 16 or np.std(g[mm]) == 0 or np.std(p[mm]) == 0:
            return None
        return float(np.corrcoef(p[mm], g[mm])[0, 1])

    # Inside the rectangle the sensor's coverage carves out three regions:
    # the fan it actually sees; the far strip past its range gate (waves that
    # WILL enter the fan later); and the near/lateral blind wedges that no
    # scan ever touches. They fail for different reasons and must be scored
    # apart, or the rectangle's corners silently set the headline.
    beyond_range = (R > R_FAN[1]) & ~inside
    blind_near = ~inside & ~beyond_range

    rbins = [(10, 50), (50, 100), (100, 150), (150, 200), (200, 280)]
    out['frac_beyond_range'] = float(beyond_range.mean())
    out['frac_blind_near'] = float(blind_near.mean())
    for name, pr in (('WaveSplat', ws), ('WavePrior', wp)):
        e = {'nrmse_all': [], 'nrmse_in': [], 'nrmse_out': [],
             'corr_all': [], 'corr_in': [], 'corr_out': [],
             'nrmse_beyond_range': [], 'nrmse_blind_near': []}
        for i in range(len(LEADS)):
            e['nrmse_all'].append(nr(pr[i], gt[i], np.ones_like(inside)))
            e['nrmse_in'].append(nr(pr[i], gt[i], inside))
            e['nrmse_out'].append(nr(pr[i], gt[i], ~inside))
            e['nrmse_beyond_range'].append(nr(pr[i], gt[i], beyond_range))
            e['nrmse_blind_near'].append(nr(pr[i], gt[i], blind_near))
            e['corr_all'].append(cr(pr[i], gt[i], np.ones_like(inside)))
            e['corr_in'].append(cr(pr[i], gt[i], inside))
            e['corr_out'].append(cr(pr[i], gt[i], ~inside))
        e['by_range_nowcast'] = {
            f'{a}-{b}m': nr(pr[0], gt[0], (R >= a) & (R < b))
            for a, b in rbins}
        e['amp_ratio_nowcast'] = float(np.nanstd(pr[0]) / np.nanstd(gt[0]))
        out[name] = e
    return out, inside, R


def render(sc, Xq, Yq, gt, ws, wp, met, inside, path):
    vmax = 2 * 1.4 * float(np.nanstd(gt[0]))
    ext = [sc['gx'].min(), sc['gx'].max(), sc['gy'].min(), sc['gy'].max()]
    fig = plt.figure(figsize=(12.6, 5.2), dpi=80)
    gs = fig.add_gridspec(1, 3, wspace=0.08, left=0.05, right=0.93,
                          top=0.82, bottom=0.12)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ims = []
    for ax, arr, ti in zip(axs, [gt[0], ws[0], wp[0]],
                           ['Ground truth', 'WaveSplat (analytic prior)',
                            'WavePrior (learned prior)']):
        ims.append(ax.imshow(arr, origin='lower', extent=ext, cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear'))
        ax.contour(Xq, Yq, inside.astype(float), levels=[0.5],
                   colors='k', linewidths=0.9, linestyles='--')
        ax.set_title(ti, fontsize=10)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel('y [m]', fontsize=8)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    axs[0].plot(*sc['center'], marker='^', color='k', ms=8)
    cb = fig.colorbar(ims[-1], ax=axs, fraction=0.025, pad=0.012)
    cb.set_label('η [m]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    sup = fig.suptitle('', fontsize=11)
    sub = fig.text(0.5, 0.885, '', ha='center', fontsize=8.5, color='0.25')

    def upd(i):
        ims[0].set_data(gt[i]); ims[1].set_data(ws[i]); ims[2].set_data(wp[i])
        sup.set_text(f"{sc['label']} — U=0, lead {LEADS[i]:.2f} s "
                     f"(dashed = observed fan)")
        a, b = met['WaveSplat'], met['WavePrior']
        f = lambda v: '--' if v is None else f'{v:.3f}'
        sub.set_text(f"nRMSE in-fan  WS {f(a['nrmse_in'][i])} / "
                     f"WP {f(b['nrmse_in'][i])}      "
                     f"out-of-fan  WS {f(a['nrmse_out'][i])} / "
                     f"WP {f(b['nrmse_out'][i])}")
        return ims + [sup, sub]

    anim = animation.FuncAnimation(fig, upd, frames=len(LEADS), interval=300)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=3.5))
    plt.close(fig)
    print(f"  -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def figure(met, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), dpi=110)
    for ax, key, ti in ((axes[0], 'nrmse_in', 'inside observed fan'),
                        (axes[1], 'nrmse_out', 'outside the fan (unobserved)')):
        for name, col in (('WaveSplat', '#D08770'), ('WavePrior', '#4C9A6B')):
            ax.plot(LEADS, met[name][key], color=col, marker='o', ms=3,
                    label=name)
        ax.set_title(f'field nRMSE — {ti}', fontsize=10)
        ax.set_xlabel('forecast lead [s]', fontsize=9)
        ax.set_ylabel('nRMSE', fontsize=9)
        ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    ax = axes[2]
    bins = list(met['WaveSplat']['by_range_nowcast'])
    x = np.arange(len(bins))
    for i, (name, col) in enumerate((('WaveSplat', '#D08770'),
                                     ('WavePrior', '#4C9A6B'))):
        v = [met[name]['by_range_nowcast'][b] for b in bins]
        ax.bar(x + (i - 0.5) * 0.36, [np.nan if q is None else q for q in v],
               width=0.36, color=col, label=name)
    ax.axvspan(-0.5, 2.5, color='k', alpha=0.05)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, 'fan range gate', fontsize=7,
            ha='center', color='0.35')
    ax.set_xticks(x); ax.set_xticklabels(bins, fontsize=7.5)
    ax.set_title('nowcast nRMSE vs range from platform', fontsize=10)
    ax.set_xlabel('range', fontsize=9); ax.grid(alpha=0.25, axis='y')
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    fig.suptitle(met['label'] + '   (stationary platform, U = 0)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all',
                    choices=['linear', 'hos', 'real_YS17', 'real_LJ18g', 'all'])
    a = ap.parse_args()
    cases = ['linear', 'hos', 'real_YS17', 'real_LJ18g'] \
        if a.case == 'all' else [a.case]
    allm = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for case in cases:
        print(f"[{case}] stationary field...", flush=True)
        sc = scene_sim(case) if case in ('linear', 'hos') \
            else scene_real(case.split('_')[1])
        Xq, Yq, gt, ws, wp = solve(sc)
        met, inside, R = metrics(sc, Xq, Yq, gt, ws, wp)
        allm[case] = met
        render(sc, Xq, Yq, gt, ws, wp, met, inside, OUT_GIF.format(case))
        figure(met, OUT_PNG.format(case))
        for n in ('WaveSplat', 'WavePrior'):
            e = met[n]
            print(f"  {n:<10} nowcast: in-fan {e['nrmse_in'][0]:.3f} "
                  f"out {e['nrmse_out'][0]:.3f} all {e['nrmse_all'][0]:.3f} "
                  f"| corr in {e['corr_in'][0]:.3f} out {e['corr_out'][0]:.3f}"
                  f" | amp {e['amp_ratio_nowcast']:.3f}")
        with open(OUT_JSON, 'w') as f:
            json.dump(allm, f, indent=1)
    print(f"\n-> {OUT_JSON}")


if __name__ == '__main__':
    sys.exit(main())
