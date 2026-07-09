"""d3_compare_gifs.py — WaveSplat (analytic) vs WavePrior (learned) on the
three data classes: linear sim, HOS nonlinear sim, real stereo sea.

Both estimators solve the SAME physics MMSE over the SAME continuous
wavenumber bank from the SAME causal forward-fan LiDAR observations. The
only thing that differs is where the prior variance over the bank comes
from:

  WaveSplat  — analytic beamform power prior (hand-designed, no training)
  WavePrior  — amortized prior net (prior_net_v3, task-loss trained on
               synthetic seas only; every evaluation here is zero-shot)

So the comparison isolates the prior, not the forward model or the solver.

Renders results/gifs/compare_<case>.gif (GT | WaveSplat | WavePrior maps,
plus the encounter series both estimators feed to the controller) and writes
results/direction3/compare_metrics.json.

Usage: python3 scripts/d3_compare_gifs.py --case linear|hos|real|all
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
OUT_GIF = 'results/gifs/compare_{}.gif'
OUT_JSON = 'results/direction3/compare_metrics.json'

DEPTH = 50.0
SCAN_DT = 0.5
M_PER_SCAN = 12
NOISE = 0.02
SPINUP = 10.0
DT_HOS = 0.1

C_GT, C_WS, C_WP = '#5B7EC9', '#D08770', '#4C9A6B'


# ----------------------------------------------------------------- scenes
def scene_sim(kind, seed=0):
    """Linear-superposition or HOS sea, causal forward fan, ship at U."""
    U, Hs, Tp, W = 3.0, 2.0, 8.0, 20.0
    lead_max, lead_step = 20.0, 1.0
    leads = np.arange(0.0, lead_max + 1e-9, lead_step)
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
        sim = None

        def eta_at(x, y, t):
            ps = kx[None] * np.atleast_1d(x)[:, None] \
                + ky[None] * np.atleast_1d(y)[:, None] \
                - om[None] * np.atleast_1d(t)[:, None] + ph[None]
            return (A[None] * np.cos(ps)).sum(1)
        t0, cx, cy, wrap = W, 0.0, 0.0, None
    else:
        Lx, Ly, Nx = 500.0, 400.0, 128
        cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=128, depth=DEPTH, M=3)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9000 + seed)
        t0, cx, cy, wrap = SPINUP + W, Lx / 2, Ly / 2, (Lx, Ly)

        def eta_at(x, y, t):
            assert abs(sim.t - t) < 1e-6, (sim.t, t)
            return sim.get_eta_at(np.mod(np.atleast_1d(x), wrap[0]),
                                  np.mod(np.atleast_1d(y), wrap[1]))

    def ship_pos(t):
        return np.array([cx - U * (t - t0), cy])

    scan_ts = np.arange(t0 - W, t0 + 1e-9, SCAN_DT)
    P_all, z_all = [], []
    for t_s in scan_ts:
        if kind == 'hos':
            while sim.t < t_s - 1e-9:
                sim.step(DT_HOS)
        sp = ship_pos(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, M_PER_SCAN)
        xs = sp[0] - rr * np.cos(aa)
        ys = sp[1] + rr * np.sin(aa)
        zz = eta_at(xs, ys, sim.t if kind == 'hos' else t_s)
        P_all.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
        z_all.append(zz + rng.normal(0, NOISE, M_PER_SCAN))
    P, z = np.vstack(P_all), np.concatenate(z_all)

    x_lo = ship_pos(t0 + lead_max)[0] - 30.0
    x_hi = ship_pos(t0)[0] + 110.0
    gxq = np.arange(x_lo, x_hi, 4.0)
    gyq = np.arange(cy - 90.0, cy + 90.0, 4.0)
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = [np.column_stack([Xq.ravel(), Yq.ravel(), np.full(npx, t0 + l_)])
          for l_ in leads]
    Pq_enc = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    Pq = np.vstack(Pq + [Pq_enc])

    maps_gt, enc_gt = [], []
    for l_ in leads:
        t_ = t0 + l_
        if kind == 'hos':
            while sim.t < t_ - 1e-9:
                sim.step(DT_HOS)
        tt = sim.t if kind == 'hos' else t_
        maps_gt.append(eta_at(Xq.ravel(), Yq.ravel(), tt).reshape(Xq.shape))
        sp = ship_pos(t_)
        enc_gt.append(float(eta_at(sp[:1], sp[1:2], tt)[0]))

    return dict(P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq,
                gxq=gxq, gyq=gyq, depth=DEPTH, Hs=Hs, kp=kp, seed=seed,
                maps_gt=np.array(maps_gt), enc_gt=np.array(enc_gt),
                label={'linear': 'Linear sim (200-comp JONSWAP)',
                       'hos': 'HOS M=3 nonlinear sim'}[kind])


def scene_real(record='YS17', win=3, W=40.0, m_scan=12, m_cap=600,
               noise=0.02):
    """One window of the d3_falcon_real_operator protocol (same seeds)."""
    from realdata_analysis import directional_spread
    from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
    from d3_sa_multirecord import MRecord

    cfg = RECORDS[record]
    U, depth, lead_max = cfg['U'], cfg['depth'], cfg['lead_max']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(lead_max / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    spread, lam_p, pdir_ax, kp = directional_spread(
        np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    w = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    h = -w
    T_tr = W_max + lead_max

    def ship_pos(t):
        return np.array([cx, cy]) + U * (t - (t0 - W_max) - T_tr / 2) * h

    r0, r1 = cfg['r_fan']
    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        sp = ship_pos(t_j)
        RX, RY = Xg - sp[0], Yg - sp[1]
        R = np.hypot(RX, RY)
        ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
        m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(m_scan, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, noise, len(pick)))
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    m = P_all[:, 2] >= t0 - W
    P, z = P_all[m], z_all[m]
    if len(z) > m_cap:
        ii = np.random.default_rng(0).choice(len(z), m_cap, replace=False)
        P, z = P[ii], z[ii]

    leads = np.arange(0.0, lead_max + 1e-9, 1.0)
    sq = max(1, int(round(4.0 / rec.dx)))
    gxq, gyq = rec.gx[::sq], rec.gy[::sq]
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = [np.column_stack([Xq.ravel(), Yq.ravel(), np.full(npx, t0 + l_)])
          for l_ in leads]
    Pq_enc = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    Pq = np.vstack(Pq + [Pq_enc])

    maps_gt = np.array([Zfut[int(round(l_ / rec.dt))][::sq, ::sq]
                        for l_ in leads])
    enc_gt = np.array([bilinear(Zfut[int(round(l_ / rec.dt))], rec.gx, rec.gy,
                                px, py)
                       for (px, py, _t), l_ in zip(Pq_enc, leads)])

    return dict(P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq,
                gxq=gxq, gyq=gyq, depth=depth, Hs=Hs, kp=kp, seed=i0,
                maps_gt=maps_gt, enc_gt=enc_gt,
                label=f'Real stereo sea ({record}, win {win})')


# ----------------------------------------------------------------- solving
def solve(sc):
    """Run both estimators on the identical (P, z, Pq)."""
    P, z, Pq = sc['P'], sc['z'], sc['Pq']
    n, npx, nl = sc['npx'], sc['npx'], len(sc['leads'])
    s0 = 0.05 * sc['Hs']
    print(f"  WaveSplat (analytic prior): {len(z)} obs, {len(Pq)} queries...",
          flush=True)
    fq_ws = mmse_bank_estimate(P, z, sc['depth'], Pq, sigma0=s0,
                               kp_hint=sc['kp'], seed=sc['seed'])
    print("  WavePrior (learned prior)...", flush=True)
    net = get_net(NET_CKPT)
    fq_wp = net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0,
                           seed=sc['seed'], kp_hint=sc['kp'])
    out = {}
    for tag, fq in (('ws', fq_ws), ('wp', fq_wp)):
        out[tag] = dict(maps=fq[:npx * nl].reshape(nl, *sc['Xq'].shape),
                        enc=fq[npx * nl:])
    return out


# ----------------------------------------------------------------- metrics
def metrics(sc, sol):
    """Fixed normalizer = GT std pooled over all leads, so nRMSE is
    comparable across leads and across the two estimators."""
    gt = sc['maps_gt']
    leads = sc['leads']
    fin = np.isfinite(gt)
    sg = float(np.std(gt[fin]))
    egt = sc['enc_gt']
    efin = np.isfinite(egt)
    seg = float(np.std(egt[efin]))

    res = {'label': sc['label'], 'n_obs': int(len(sc['z'])),
           'Hs': float(sc['Hs']), 'gt_std': sg, 'leads': leads.tolist()}
    for tag, name in (('ws', 'WaveSplat'), ('wp', 'WavePrior')):
        mp, en = sol[tag]['maps'], sol[tag]['enc']
        per_lead, corr, amp = [], [], []
        for li in range(len(leads)):
            m = np.isfinite(gt[li]) & np.isfinite(mp[li])
            if m.sum() < 8:
                per_lead.append(None); corr.append(None); amp.append(None)
                continue
            g, p = gt[li][m], mp[li][m]
            per_lead.append(float(np.sqrt(np.mean((p - g) ** 2)) / sg))
            corr.append(float(np.corrcoef(p, g)[0, 1])
                        if np.std(p) > 0 and np.std(g) > 0 else 0.0)
            amp.append(float(np.std(p) / np.std(g)) if np.std(g) > 0 else None)

        def band(lo, hi):
            m = efin & (leads >= lo) & (leads <= hi)
            if m.sum() < 3:
                return None
            g, p = egt[m], en[m]
            return {'nrmse': float(np.sqrt(np.mean((p - g) ** 2)) / seg),
                    'corr': float(np.corrcoef(p, g)[0, 1])
                            if np.std(p) > 0 else 0.0,
                    'amp_ratio': float(np.std(p) / np.std(g))}

        ok = [v for v in per_lead if v is not None]
        okc = [v for v in corr if v is not None]
        res[name] = {
            'map_nrmse_nowcast': per_lead[0],
            'map_corr_nowcast': corr[0],
            'map_amp_ratio_nowcast': amp[0],
            'map_nrmse_mean_over_leads': float(np.mean(ok)) if ok else None,
            'map_corr_mean_over_leads': float(np.mean(okc)) if okc else None,
            'map_nrmse_per_lead': per_lead,
            'map_corr_per_lead': corr,
            'encounter_0-2s': band(0, 2),
            'encounter_0-10s': band(0, 10),
            'encounter_10-20s': band(10, 20),
        }
    return res


# ----------------------------------------------------------------- render
def render(sc, sol, met, path):
    gt, leads = sc['maps_gt'], sc['leads']
    ws, wp = sol['ws']['maps'], sol['wp']['maps']
    vmax = 2 * 1.4 * float(np.nanstd(gt[0]))
    ext = [sc['gxq'].min(), sc['gxq'].max(), sc['gyq'].min(), sc['gyq'].max()]

    fig = plt.figure(figsize=(12.6, 6.6), dpi=80)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1.0], hspace=0.36,
                          wspace=0.10, left=0.05, right=0.94, top=0.88,
                          bottom=0.10)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axT = fig.add_subplot(gs[1, :])
    ims = []
    titles = ['Ground truth',
              'WaveSplat — analytic prior',
              'WavePrior — learned prior']
    for ax, arr, ti in zip(axs, [gt[0], ws[0], wp[0]], titles):
        ims.append(ax.imshow(arr, origin='lower', extent=ext, cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear'))
        ax.set_title(ti, fontsize=10)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel('y [m]', fontsize=8)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    cb = fig.colorbar(ims[-1], ax=axs, fraction=0.025, pad=0.012)
    cb.set_label('η [m]', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    egt, ews, ewp = sc['enc_gt'], sol['ws']['enc'], sol['wp']['enc']
    axT.plot(leads, egt, color=C_GT, lw=2.0, label='truth')
    axT.plot(leads, ews, color=C_WS, lw=1.6, ls='--', label='WaveSplat')
    axT.plot(leads, ewp, color=C_WP, lw=1.6, ls='-.', label='WavePrior')
    axT.axvspan(0, 2, color='k', alpha=0.06)
    axT.text(0.15, 0.04, 'control band 0-2 s', fontsize=7,
             transform=axT.get_xaxis_transform(), color='0.35')
    vline = axT.axvline(leads[0], color='0.3', lw=1.0)
    axT.set_xlabel('forecast lead [s]', fontsize=8)
    axT.set_ylabel('η at ship [m]', fontsize=8)
    axT.legend(fontsize=7, ncol=3, loc='upper right', framealpha=0.9)
    axT.tick_params(labelsize=7)
    axT.set_xlim(leads[0], leads[-1])

    mw, mp_ = met['WaveSplat'], met['WavePrior']
    sup = fig.suptitle('', fontsize=11)
    sub = fig.text(0.5, 0.925, '', ha='center', fontsize=8.5, color='0.25')

    def upd(i):
        ims[0].set_data(gt[i])
        ims[1].set_data(ws[i])
        ims[2].set_data(wp[i])
        vline.set_xdata([leads[i], leads[i]])
        sup.set_text(f"{sc['label']}   —   lead {leads[i]:.0f} s "
                     f"({sc['n_obs'] if 'n_obs' in sc else len(sc['z'])} "
                     f"LiDAR points)")
        a = mw['map_nrmse_per_lead'][i]
        b = mp_['map_nrmse_per_lead'][i]
        ca = mw['map_corr_per_lead'][i]
        cb_ = mp_['map_corr_per_lead'][i]
        f = (lambda v: '--' if v is None else f'{v:.3f}')
        sub.set_text(f"map nRMSE   WaveSplat {f(a)} / WavePrior {f(b)}      "
                     f"corr   WaveSplat {f(ca)} / WavePrior {f(cb_)}")
        return ims + [vline, sup, sub]

    anim = animation.FuncAnimation(fig, upd, frames=len(leads),
                                   interval=420, blit=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=2.4))
    plt.close(fig)
    print(f"  -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def stats(case, reps, record='YS17'):
    """Repeat the scene over seeds (sim) or windows (real) and pool.

    A single scene is an illustration, not evidence: the GIF numbers move by
    O(0.05) nRMSE between seeds. This is what the reported metrics come from.
    """
    rows = []
    for r in range(reps):
        sc = scene_sim(case, seed=r) if case in ('linear', 'hos') \
            else scene_real(record, win=r)
        sc['n_obs'] = len(sc['z'])
        met = metrics(sc, solve(sc))
        rows.append(met)
        print(f"  [{case} rep {r}] WS {met['WaveSplat']['map_nrmse_nowcast']:.3f}"
              f" / WP {met['WavePrior']['map_nrmse_nowcast']:.3f}", flush=True)

    out = {'case': case, 'reps': reps}
    for name in ('WaveSplat', 'WavePrior'):
        agg = {}
        for key in ('map_nrmse_nowcast', 'map_corr_nowcast',
                    'map_amp_ratio_nowcast', 'map_nrmse_mean_over_leads',
                    'map_corr_mean_over_leads'):
            v = [m[name][key] for m in rows if m[name][key] is not None]
            agg[key] = {'mean': float(np.mean(v)), 'std': float(np.std(v))} \
                if v else None
        for bnd in ('encounter_0-2s', 'encounter_0-10s', 'encounter_10-20s'):
            for sub in ('nrmse', 'corr', 'amp_ratio'):
                v = [m[name][bnd][sub] for m in rows
                     if m[name][bnd] is not None]
                if v:
                    agg[f'{bnd}_{sub}'] = {'mean': float(np.mean(v)),
                                           'std': float(np.std(v))}
        out[name] = agg
    # paired per-rep difference on the headline metric
    d = [m['WaveSplat']['map_nrmse_nowcast'] - m['WavePrior']['map_nrmse_nowcast']
         for m in rows]
    out['paired_delta_map_nrmse_nowcast'] = {
        'mean': float(np.mean(d)), 'std': float(np.std(d)),
        'per_rep': [float(x) for x in d],
        'note': 'positive = WavePrior better'}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all',
                    choices=['linear', 'hos', 'real', 'all'])
    ap.add_argument('--record', default='YS17')
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--stats', type=int, default=0,
                    help='repeat over N seeds (sim) / windows (real), '
                         'pool metrics, skip the GIF')
    a = ap.parse_args()
    cases = ['linear', 'hos', 'real'] if a.case == 'all' else [a.case]

    if a.stats:
        path = 'results/direction3/compare_stats.json'
        allm = json.load(open(path)) if os.path.exists(path) else {}
        for case in cases:
            allm[case] = stats(case, a.stats, a.record)
        with open(path, 'w') as f:
            json.dump(allm, f, indent=1)
        print(f"\n-> {path}")
        return 0

    allm = {}
    if os.path.exists(OUT_JSON):
        allm = json.load(open(OUT_JSON))
    for case in cases:
        print(f"[{case}] building scene...", flush=True)
        sc = scene_sim(case) if case in ('linear', 'hos') \
            else scene_real(a.record, a.win)
        sc['n_obs'] = len(sc['z'])
        sol = solve(sc)
        met = metrics(sc, sol)
        allm[case] = met
        render(sc, sol, met, OUT_GIF.format(case))
        for name in ('WaveSplat', 'WavePrior'):
            e = met[name]
            b2 = e['encounter_0-2s']
            print(f"  {name:<10} map nRMSE(nowcast)={e['map_nrmse_nowcast']:.3f}"
                  f" corr={e['map_corr_nowcast']:.3f}"
                  f" | enc 0-2s nRMSE={b2['nrmse']:.3f}"
                  f" corr={b2['corr']:.3f}")

    with open(OUT_JSON, 'w') as f:
        json.dump(allm, f, indent=1)
    print(f"\n-> {OUT_JSON}")


if __name__ == '__main__':
    sys.exit(main())
