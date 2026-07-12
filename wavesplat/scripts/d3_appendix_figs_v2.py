"""Doctoral appendix figures v2 (two deliverables, user feedback 2026-07-12).

A. figs_appendix/appendix_<record>_v2.png  x5   (real 60-window gallery)
   Same 12 protocol windows per record as v1, but the hand/learned panels are
   now rendered on the FULL rectangular grid (the v1 truth-footprint mask is
   removed) with the truth footprint drawn as a thin black outline
   (isfinite-mask contour, level 0.5). This exposes the estimator's
   whole-field extrapolation that masking hid. corr numbers are READ from
   field_wide_eval.json / gate_promotion_eval.json (not recomputed); only the
   lead-0 field images are re-rendered (stride 2). v1 files are left untouched.

B. figs_appendix/appendix_synthetic.png + appendix_synth_eval.json
   Synthetic fixedT episodes: truth exists over the whole rectangular grid, so
   this is a footprint-free whole-field comparison at the DESIGN operating
   point (BF2-BF5, incl. two dir180 head-sea episodes). 6 representative
   episodes, one row each:
     [truth (global, lead 0) | hand | learned | per-lead global corr bars]
   Arms are identical to the real 60-window definition:
     hand    = mmse_gpu(P, z, depth, Q, sig0, kp_hint)   (default: val + EM)
     learned = net_gamma_mmse(..., use_em=False, smult_head=head)
   Obs = range-stratified ~2000 pts from a 12-frame / ~20 s window (deploy
   recipe). sig0 = 0.05*Hs (episode noise convention is not stored in the h5;
   this matches the real 60-window arm's convention and is documented here).
   Leads {0,2,5,8} s land on stored frames t_end+{0,2,5,8} <= 30 s.
"""
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
from d3_gpu_mmse import mmse_gpu, DEV
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_l1_s2head_train import SmultHead

OUT = 'results/direction3'
APP = f'{OUT}/figs_appendix'
DATE = '2026-07-12'
os.makedirs(APP, exist_ok=True)

# fixed palette (task spec)
C_HAND = '#2a78d6'   # physics-prior (hand) arm
C_NET = '#1baf7a'    # learned arm (net + s2-head)
C_INK = '#0b0b0b'    # truth / primary ink
C_INK2 = '#52514e'
C_GRID = '#e1e0d9'
C_BG = '#fcfcfb'
C_AXIS = '#c3c2b7'
C_MUT = '#898781'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': C_BG, 'axes.facecolor': C_BG,
    'savefig.facecolor': C_BG,
})

RECORD_ORDER = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
CK = 'results/direction3/prior_net_l1s2bw.pt'


def load_net_head():
    net = get_net(CK)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CK, map_location=DEVICE)['head'])
    head.eval()
    print(f'net/head loaded ({CK}); DEVICE {DEVICE}', flush=True)
    return net, head


def cell_extent(gx, gy):
    """imshow extent placing pixel CENTERS at gx,gy so a contour on (gx,gy)
    overlays exactly."""
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    return [gx[0] - dx / 2, gx[-1] + dx / 2, gy[0] - dy / 2, gy[-1] + dy / 2]


def footprint_contour(ax, gx, gy, T):
    finite = np.isfinite(T).astype(float)
    if 0 < finite.sum() < finite.size:
        ax.contour(gx, gy, finite, levels=[0.5], colors=[C_INK],
                   linewidths=0.8, zorder=6)


# =====================================================================  TASK A
def appendix_record_v2(record, Window, net, head, FIELD_K, GATE_K, leads):
    from d3_sa_multirecord import MRecord
    from d3_falcon_real_operator import RECORDS
    cfg = RECORDS[record]
    t0 = time.time()
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    print(f'  [{record}] MRecord loaded ({time.time()-t0:.1f}s)', flush=True)

    nrows = 12
    fig = plt.figure(figsize=(11.0, 2.05 * nrows), dpi=150)
    gs = GridSpec(nrows, 4, figure=fig, wspace=0.18, hspace=0.45,
                  left=0.135, right=0.985, top=0.955, bottom=0.02,
                  width_ratios=[1, 1, 1, 0.95])

    for wi in range(nrows):
        tw = time.time()
        w = Window(record, wi, rec=rec, net=net, head=head)
        gx, gy, truth, fh, fn = w.fields([0.0], stride=2)
        T = truth[0]
        Hfull = fh[0]           # FULL rectangular grid (no mask) -- v2 change
        Nfull = fn[0]
        vmax = np.nanpercentile(np.abs(T), 97)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ext = cell_extent(gx, gy)
        fr = FIELD_K[(record, wi)]
        gr = GATE_K[(record, wi)]

        axs = [fig.add_subplot(gs[wi, c]) for c in range(4)]
        panels = [(T, 'truth', False), (Hfull, 'hand', True),
                  (Nfull, 'learned', True)]
        for ax, (dat, ttl, outline) in zip(axs[:3], panels):
            ax.imshow(dat, origin='lower', extent=ext, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, aspect='auto',
                      interpolation='nearest')
            if outline:                     # truth footprint outline on pred
                footprint_contour(ax, gx, gy, T)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=8.5, color=C_INK2, pad=2)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            for s in ax.spines.values():
                s.set_edgecolor(C_AXIS)
        axs[0].set_ylabel(
            f'w{wi}\nfield h/l '
            f'{fr["field_hand"]:.2f}/{fr["field_learned"]:.2f}\n'
            f'track $\\rho$ h/l {gr["hand"]:.2f}/{gr["bw_s0_head"]:.2f}',
            fontsize=8.2, color=C_INK, rotation=0, ha='right', va='center',
            labelpad=10)

        m = axs[3]
        x = np.arange(len(leads))
        m.bar(x - 0.19, fr['per_lead_hand'], 0.36, color=C_HAND, label='hand')
        m.bar(x + 0.19, fr['per_lead_learned'], 0.36, color=C_NET,
              label='learned')
        m.set_ylim(0, 1.0)
        m.set_xticks(x)
        m.set_xticklabels([f'{int(L)}' for L in leads], fontsize=7)
        m.tick_params(axis='y', labelsize=7)
        m.set_title('field corr / lead(s)', fontsize=8, color=C_INK2, pad=2)
        m.grid(axis='y', linewidth=0.5)
        m.spines[['top', 'right']].set_visible(False)
        if wi == 0:
            m.legend(fontsize=6.5, frameon=False, loc='upper right',
                     handlelength=1.0, ncol=2, columnspacing=0.8)
        print(f'    w{wi} rendered ({time.time()-tw:.1f}s) '
              f'field h/l {fr["field_hand"]:.2f}/{fr["field_learned"]:.2f}',
              flush=True)

    fig.suptitle(
        f'Appendix {record} (v2) — 12 protocol windows: truth | hand | '
        f'learned fields (lead 0, RdBu_r)\n'
        f'hand/learned rendered on the FULL rectangular grid; black outline = '
        f'truth footprint\n'
        f'corr computed inside the outlined truth footprint',
        fontsize=10, color=C_INK, y=0.992)
    p = f'{APP}/appendix_{record}_v2.png'
    fig.savefig(p)
    plt.close(fig)
    print(f'  wrote {p}', flush=True)


def task_a(net, head):
    FIELD = json.load(open(f'{OUT}/field_wide_eval.json'))
    GATE = json.load(open(f'{OUT}/gate_promotion_eval.json'))
    FIELD_K = {(r['record'], r['wi']): r for r in FIELD}
    GATE_K = {(r['record'], r['wi']): r for r in GATE}
    leads = FIELD[0]['leads']
    from d3_external_figs import Window
    t0 = time.time()
    for r in RECORD_ORDER:
        appendix_record_v2(r, Window, net, head, FIELD_K, GATE_K, leads)
    print(f'TASK A gallery done in {(time.time()-t0)/60:.1f} min', flush=True)


# =====================================================================  TASK B
# representative episodes: BF2..BF5, two dir180 head-sea episodes.
SYNTH_EPISODES = [
    'episode_BF2_1comp_seed0_fixedT.h5',
    'episode_BF3_5comp_seed0_fixedT.h5',
    'episode_BF4_15comp_seed0_fixedT.h5',
    'episode_BF5_15comp_seed0_fixedT.h5',
    'episode_BF4_15comp_seed0_fixedT_dir180.h5',
    'episode_BF5_15comp_seed5_fixedT_dir180.h5',
]
SYNTH_LEADS = [0.0, 2.0, 5.0, 8.0]


def load_episode_obs(path, rng):
    """range-stratified ~2000 obs from a 12-frame / ~20 s window ending at
    t=22 s so leads {0,2,5,8} land on stored frames (<= 30 s)."""
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]        # [A, om, kx, ky, phi, Q]
        Hs = float(f['config'].attrs['Hs'])
        depth = float(f['config'].attrs['depth'])
        ncomp = int(f['config'].attrs['n_components'])
        sea = str(f['config'].attrs['sea_state'])
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        gz = f['frames/gt_wave_field'][:]            # (301, ny, nx)
        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])

        obs_i = np.linspace(20, 220, 12).astype(int)  # window ends t=22 s
        t_end = float(ts[obs_i[-1]])
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1],
                                      np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        rr = np.hypot(P[:, 0], P[:, 1])
        edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = []
        for j in range(12):
            idx = np.where(b == j)[0]
            k = min(len(idx), 167)
            if k:
                take.append(rng.choice(idx, k, replace=False))
        sel = np.concatenate(take)
        Ps, zs = P[sel], zv[sel]

        # truth frames at t_end + lead
        gt_leads = []
        for L in SYNTH_LEADS:
            idx = int(round((t_end + L) / (ts[1] - ts[0])))
            idx = min(idx, len(ts) - 1)
            gt_leads.append(gz[idx])
    return dict(Ps=Ps, zs=zs, Hs=Hs, depth=depth, ncomp=ncomp, sea=sea,
                kp=kp, gx=gx, gy=gy, t_end=t_end, gt_leads=gt_leads,
                dir180=('dir180' in os.path.basename(path)))


def synth_predict(ep, net, head):
    """fit both arms once, query the whole grid at all leads (one solve)."""
    Ps, zs, depth, kp = ep['Ps'], ep['zs'], ep['depth'], ep['kp']
    gx, gy, t_end = ep['gx'], ep['gy'], ep['t_end']
    sig0 = 0.05 * ep['Hs']
    X, Y = np.meshgrid(gx, gy)
    npix = X.size
    Q = np.vstack([
        np.column_stack([X.ravel(), Y.ravel(),
                         np.full(npix, t_end + L)]) for L in SYNTH_LEADS])
    th = time.time()
    ph = mmse_gpu(torch.tensor(Ps, device=DEV, dtype=torch.float32),
                  torch.tensor(zs, device=DEV, dtype=torch.float32),
                  depth, torch.tensor(Q, device=DEV, dtype=torch.float32),
                  sig0, kp_hint=kp).cpu().numpy()
    dt_h = time.time() - th
    tn = time.time()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, depth, Q, sig0, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head))
    dt_n = time.time() - tn
    shp = X.shape
    Hf = [ph[k * npix:(k + 1) * npix].reshape(shp) for k in range(len(SYNTH_LEADS))]
    Nf = [pn[k * npix:(k + 1) * npix].reshape(shp) for k in range(len(SYNTH_LEADS))]
    return Hf, Nf, dt_h, dt_n


def task_b(net, head):
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(11.0, 2.15 * len(SYNTH_EPISODES)), dpi=150)
    gs = GridSpec(len(SYNTH_EPISODES), 4, figure=fig, wspace=0.18,
                  hspace=0.40, left=0.145, right=0.985, top=0.945,
                  bottom=0.03, width_ratios=[1, 1, 1, 0.95])
    records = []
    for ri, fname in enumerate(SYNTH_EPISODES):
        path = f'data/{fname}'
        te = time.time()
        ep = load_episode_obs(path, rng)
        Hf, Nf, dt_h, dt_n = synth_predict(ep, net, head)

        corr_h, corr_n = [], []
        for k in range(len(SYNTH_LEADS)):
            g = ep['gt_leads'][k].ravel()
            corr_h.append(float(np.corrcoef(Hf[k].ravel(), g)[0, 1]))
            corr_n.append(float(np.corrcoef(Nf[k].ravel(), g)[0, 1]))

        T0 = ep['gt_leads'][0]
        vmax = np.nanpercentile(np.abs(T0), 97)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ext = cell_extent(ep['gx'], ep['gy'])
        axs = [fig.add_subplot(gs[ri, c]) for c in range(4)]
        for ax, dat, ttl in zip(axs[:3], [T0, Hf[0], Nf[0]],
                                ['truth (global)', 'hand', 'learned']):
            ax.imshow(dat, origin='lower', extent=ext, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, aspect='auto',
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=8.5, color=C_INK2, pad=2)
            for s in ax.spines.values():
                s.set_edgecolor(C_AXIS)
        dtag = 'dir180 (head sea)' if ep['dir180'] else 'dir0'
        axs[0].set_ylabel(
            f'{ep["sea"]}  {ep["ncomp"]}c\n{dtag}\n'
            f'Hs {ep["Hs"]:.1f} m\nfield$_0$ h/l '
            f'{corr_h[0]:.2f}/{corr_n[0]:.2f}',
            fontsize=8.0, color=C_INK, rotation=0, ha='right', va='center',
            labelpad=10)

        m = axs[3]
        x = np.arange(len(SYNTH_LEADS))
        m.bar(x - 0.19, corr_h, 0.36, color=C_HAND, label='hand')
        m.bar(x + 0.19, corr_n, 0.36, color=C_NET, label='learned')
        lo = min(0.0, min(corr_h + corr_n)) - 0.05
        m.set_ylim(lo, 1.0)
        if lo < 0:
            m.axhline(0, color=C_AXIS, lw=0.8)
        m.set_xticks(x)
        m.set_xticklabels([f'{int(L)}' for L in SYNTH_LEADS], fontsize=7)
        m.tick_params(axis='y', labelsize=7)
        m.set_title('global corr / lead(s)', fontsize=8, color=C_INK2, pad=2)
        m.grid(axis='y', linewidth=0.5)
        m.spines[['top', 'right']].set_visible(False)
        if ri == 0:
            m.legend(fontsize=6.5, frameon=False, loc='upper right',
                     handlelength=1.0, ncol=2, columnspacing=0.8)

        records.append(dict(
            episode=fname, sea=ep['sea'], n_comp=ep['ncomp'],
            dir180=ep['dir180'], Hs=ep['Hs'], depth=ep['depth'],
            kp=ep['kp'], n_obs=int(len(ep['zs'])), t_end=ep['t_end'],
            leads=SYNTH_LEADS,
            corr_hand=corr_h, corr_learned=corr_n,
            ms_hand=dt_h * 1e3, ms_learned=dt_n * 1e3))
        print(f'  [{fname}] {ep["sea"]} {ep["ncomp"]}c '
              f'{"dir180" if ep["dir180"] else "dir0"} n_obs {len(ep["zs"])} '
              f'hand {["%+.2f" % c for c in corr_h]} '
              f'learned {["%+.2f" % c for c in corr_n]} '
              f'({time.time()-te:.1f}s)', flush=True)

    fig.suptitle(
        'Appendix — synthetic fixedT episodes (design operating point '
        'BF2-BF5)\n'
        'truth exists on the whole rectangular grid: footprint-free '
        'whole-field comparison\n'
        'truth (global, lead 0) | hand | learned | per-lead global corr; '
        'obs = range-stratified ~2000 pts / 12-frame window',
        fontsize=10, color=C_INK, y=0.995)
    p = f'{APP}/appendix_synthetic.png'
    fig.savefig(p)
    plt.close(fig)
    print(f'  wrote {p}', flush=True)

    # ------- summary + JSON
    summ = _synth_summary(records)
    out = dict(date=DATE, leads=SYNTH_LEADS,
               sig0_convention='0.05*Hs (episode noise not stored in h5; '
               'matches real 60-window arm)',
               obs_recipe='range-stratified ~2000 pts, 12-frame/~20s window '
               'ending t=22s',
               arms=dict(hand='mmse_gpu(P,z,depth,Q,sig0,kp_hint) default '
                         '(val-select + EM)',
                         learned='net_gamma_mmse(...,use_em=False,'
                         'smult_head=head)'),
               episodes=records, summary=summ)
    json.dump(out, open(f'{OUT}/appendix_synth_eval.json', 'w'), indent=2)
    print(f'  wrote {OUT}/appendix_synth_eval.json', flush=True)
    return summ


def _synth_summary(records):
    H = np.array([r['corr_hand'] for r in records])     # (n_ep, n_lead)
    N = np.array([r['corr_learned'] for r in records])
    summ = dict(n_episodes=len(records))
    per_lead = []
    for k, L in enumerate(SYNTH_LEADS):
        per_lead.append(dict(
            lead=L, hand=float(H[:, k].mean()), learned=float(N[:, k].mean()),
            delta=float(N[:, k].mean() - H[:, k].mean()),
            learned_wins=int((N[:, k] > H[:, k]).sum())))
    summ['per_lead'] = per_lead
    summ['lead0_hand'] = float(H[:, 0].mean())
    summ['lead0_learned'] = float(N[:, 0].mean())
    summ['mean_all_hand'] = float(H.mean())
    summ['mean_all_learned'] = float(N.mean())
    return summ


def main():
    net, head = load_net_head()
    print('\n===== TASK B: synthetic fixedT appendix =====', flush=True)
    summ = task_b(net, head)
    print('\n--- TASK B summary (mean over 6 episodes) ---', flush=True)
    print(f"lead0  hand {summ['lead0_hand']:+.3f}  learned "
          f"{summ['lead0_learned']:+.3f}", flush=True)
    for pl in summ['per_lead']:
        print(f"lead {int(pl['lead'])}s  hand {pl['hand']:+.3f}  learned "
              f"{pl['learned']:+.3f}  delta {pl['delta']:+.3f}  "
              f"learned_wins {pl['learned_wins']}/6", flush=True)
    print(f"all-lead mean  hand {summ['mean_all_hand']:+.3f}  learned "
          f"{summ['mean_all_learned']:+.3f}", flush=True)

    print('\n===== TASK A: real 60-window gallery v2 (full grid + '
          'footprint outline) =====', flush=True)
    task_a(net, head)
    print('\nALL DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
