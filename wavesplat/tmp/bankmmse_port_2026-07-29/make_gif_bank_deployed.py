"""Animated real-sea (stereo) wave-field estimation by the DEPLOYED WaveSplat path
AFTER the 2026-07-29 bank-MMSE port.

Same protocol as tmp/real_gif_2026-07-27/make_real_estimation_gif.py (copied from
it verbatim except the deployed-arm bookkeeping): frozen scene_real window, shipped
DEFAULT_CONFIG through make_updater, gate allowed to act.  What changed is what the
REAL-DOMAIN GATE routes to: the ported bank-MMSE (observation-only parameters,
median-of-3 field ensemble) instead of a single v2_konly run.

Run from the wavesplat/ package root:
    python3 tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py [--record AA15a --win 5]
"""
import os, sys, io, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

OUTGIF = 'results/gifs/real_estimation_bankdeploy_2026-07-29.gif'
OUTJSON = 'tmp/bankmmse_port_2026-07-29/real_gif_info.json'
W_PX, DPI, FPS = 1200, 100, 5
TARGET_DT = 0.20          # animation time step [s] (snapped to the record frame rate)

INK, SEC, MUT, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, AQUA = '#2a78d6', '#1baf7a'
RECNAME = {'AA15a': 'Acqua Alta 2015-03-05', 'AA15b': 'Acqua Alta 2015-05-15',
           'AA14': 'Acqua Alta 2014-03-27', 'YS17': 'Yellow Sea 2017-05-13',
           'LJ18g': 'La Jument 2018-01-03'}

plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'text.color': INK,
                     'axes.edgecolor': MUT, 'font.family': 'sans-serif'})


# ---------------------------------------------------------------- data + estimate
def frames_from_scene(sc, t_shift):
    """Format adapter only (verbatim from tmp/blind_auto_real_2026-07-26/run_blind_auto.py
    plus a constant time offset, so absolute record time never enters the optimiser).
    No value altered, nothing filtered, no mean removed."""
    from wavesplat.data_loader import FrameData
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    out = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        out.append(FrameData(t=float(tt - t_shift),
                             x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
                             y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
                             z_obs=torch.tensor(z[m], dtype=torch.float32),
                             confidence=torch.ones(n, dtype=torch.float32),
                             incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return out


def dense_gt(record, win):
    """The stereo GT at the record's native frame rate over the same lead span, using
    the same i0 / window / subsample that scene_real uses (identical call chain)."""
    from d3_falcon_real_operator import RECORDS
    from d3_sa_multirecord import MRecord
    cfg = RECORDS[record]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])
    Zfut = rec.window(i0, n_lead + 1)
    sq = max(1, int(round(4.0 / rec.dx)))
    return dict(Z=Zfut[:, ::sq, ::sq], dt=float(rec.dt), i0=i0,
                gx=rec.gx[::sq], gy=rec.gy[::sq], lead_max=float(cfg['lead_max']))


def estimate(sc, seed=42):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = frames_from_scene(sc, t_shift)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=seed, device='cpu')
    set_seed(seed)
    upd = make_updater(cfg)
    t = time.time()
    upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                   y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
    return upd, t0, time.time() - t, cfg


def estimate_bank(sc, dense_t_abs):
    """Research arm: the frozen bank-MMSE call behind every published real-sea number
    (tmp/wp_real_2026-07-25/run_wp_real.py, rerun_collapse_fields.py, the real figures):

        mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05*Hs, kp_hint=kp, seed=i0)

    Called TWICE with identical (P, z, depth, sigma0, kp_hint, seed) and only Pq differing:
    once with the protocol Pq (to reproduce the published number) and once on the dense
    animation time grid. The estimator is linear in z and renders each query point
    independently (`out = A @ w_pred` per chunk), so the dense call cannot change the
    protocol values -- which is asserted, not assumed.
    """
    from d3_sa_nomp import mmse_bank_estimate
    npx, nl = sc['npx'], len(sc['leads'])
    s0 = 0.05 * sc['Hs']
    t = time.time()
    fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                            kp_hint=sc['kp'], seed=sc['seed'])
    maps_prot = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    Xq, Yq = sc['Xq'], np.meshgrid(sc['gxq'], sc['gyq'])[1]
    Pq_d = np.vstack([np.column_stack([Xq.ravel(), Yq.ravel(), np.full(npx, tt)])
                      for tt in dense_t_abs])
    fq_d = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq_d, sigma0=s0,
                              kp_hint=sc['kp'], seed=sc['seed'])
    maps_dense = fq_d.reshape(len(dense_t_abs), *sc['Xq'].shape)
    el = time.time() - t
    # assert the dense render reproduces the protocol render at the shared times
    t0a = float(sc['Pq'][0, 2])
    n_chk = 0
    for li, L in enumerate(sc['leads']):
        k = int(np.argmin(np.abs(np.asarray(dense_t_abs) - (t0a + float(L)))))
        if abs(dense_t_abs[k] - (t0a + float(L))) < 1e-6:
            assert np.allclose(maps_dense[k], maps_prot[li], atol=1e-9), (li, k)
            n_chk += 1
    print(f'  bank-MMSE: dense==protocol at {n_chk}/{nl} shared lead times, {el:.0f}s',
          flush=True)
    return maps_prot, maps_dense, el, n_chk


def estimate_learned(sc, dense_t_rel_fn, ckpt):
    """Research LEARNED arm, verbatim from tmp/mixed_retrain_2026-07-26/run_wp_real_new.py
    (Phase B) with the mixed_v1 checkpoint: beamform-conditioned set-diffusion proposal
    (4-sample BMA) + 80-iter polish + LS-fitted field, median over 5 diffusion seeds.

    Published number = median over the 5 seeds of the atoms_to_field BMA nowcast.
    DISPLAYED field = the fit_render_leads BMA of the MEDIAN seed, i.e. the same object
    the repo's own learned-arm GIF shows (tmp/figs_opus_real_2026-07-26/
    rerun_collapse_fields.py), so the picture and the number belong together.
    fit_render_leads evaluates every lead time independently from one LS fit, so
    rendering the dense animation grid cannot move the protocol lead values -- asserted.
    """
    import torch
    import d3_beamform_cond_diffusion as cd
    from d3_setdiff_prior import denorm_atoms
    from d3_beamform_net import KLIM, NK
    sys.path.insert(0, os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'))
    from run_wp_real_new import fit_render_leads, score_maps, SEEDS

    t_run = time.time()
    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=cd.DEVICE))
    model.eval()

    depth = sc['depth']
    P = sc['P'].astype(np.float32); z = sc['z'].astype(np.float32)
    if len(z) > cd.M_OBS:
        idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
        P, z = P[idx], z[idx]
    t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
    Bmap = cd._bmap_from_obs(Ps, z)
    tabs0 = float(sc['Pq'][0, 2])
    lead_ts = [tabs0 + float(L) - t0 for L in sc['leads']]
    dense_ts = [tabs0 + float(tt) - t0 for tt in dense_t_rel_fn]
    target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': lead_ts[0],
              'gt': sc['maps_gt'][0], 'Hs': sc['Hs']}
    Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))

    seed_now, seed_lead, seed_neff, f_prot, f_dense = [], [], [], [], []
    for sd in SEEDS:
        X = cd.sample(model, Bmap, abar, betas, n=4, seed=int(sd))
        preds_a2f, lf_prot, lf_dense, neffs = [], [], [], []
        for b in range(4):
            xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
            pa, _ = cd.atoms_to_field(xb, stats, Ps, z, depth, target)
            if pa is not None:
                preds_a2f.append(pa)
            kxi, kyi, ai = denorm_atoms(xb, stats)
            kxi = kxi.cpu().numpy(); kyi = kyi.cpu().numpy(); ai = ai.cpu().numpy()
            if ai.max() <= 0:
                continue
            keep = ai > 0.03 * ai.max()
            kxi, kyi, aii = kxi[keep], kyi[keep], ai[keep]
            if len(kxi) > 30:
                o = np.argsort(-aii)[:30]; kxi, kyi = kxi[o], kyi[o]
            neffs.append(len(kxi))
            lf_prot.append(fit_render_leads(kxi, kyi, Ps, z, depth, sc['gxq'], sc['gyq'],
                                            lead_ts, Uc=(0.0, 0.0)))
            lf_dense.append(fit_render_leads(kxi, kyi, Ps, z, depth, sc['gxq'], sc['gyq'],
                                             dense_ts, Uc=(0.0, 0.0)))
        bma_now = np.mean(preds_a2f, axis=0)
        m0 = np.isfinite(gt[0]) & np.isfinite(bma_now)
        seed_now.append(float(np.sqrt(np.mean((bma_now[m0] - gt[0][m0]) ** 2)) / sg))
        bl = np.mean(lf_prot, axis=0)
        _, lead_a, _ = score_maps(sc, bl)
        seed_lead.append(lead_a)
        seed_neff.append(float(np.mean(neffs)))
        f_prot.append(bl)
        f_dense.append(np.mean(lf_dense, axis=0))
        print(f'  [learned seed {sd}] nowcast {seed_now[-1]:.4f} lead {lead_a:.4f} '
              f'n_eff {seed_neff[-1]:.1f}', flush=True)

    i_med = int(np.argsort(seed_now)[len(seed_now) // 2])       # median seed
    now_med = float(np.median(seed_now))
    lead_med = float(np.median([v for v in seed_lead if v is not None]))
    maps_prot, maps_dense = f_prot[i_med], f_dense[i_med]
    # the dense render must reproduce the protocol render at the shared lead times
    n_chk = 0
    for li, L in enumerate(sc['leads']):
        k = int(np.argmin(np.abs(np.asarray(dense_t_rel_fn) - float(L))))
        if abs(dense_t_rel_fn[k] - float(L)) < 1e-6:
            assert np.allclose(maps_dense[k], maps_prot[li], atol=1e-9), (li, k)
            n_chk += 1
    el = time.time() - t_run
    info = dict(name='learned_waveprior_mixed_v1',
                entry='tmp/mixed_retrain_2026-07-26/run_wp_real_new.py (Phase B)',
                ckpt=ckpt, device=str(cd.DEVICE), diffusion_seeds=list(SEEDS),
                n_bma=4, polish_iters=80,
                learned_now_median_of_5=now_med,
                learned_now_seeds=[float(v) for v in seed_now],
                learned_lead_median_of_5=lead_med,
                displayed_seed=int(SEEDS[i_med]), displayed_seed_index=i_med,
                n_eff=float(np.mean(seed_neff)),
                dense_equals_protocol_at_shared_leads=int(n_chk),
                seconds=round(el, 1))
    print(f'  learned: median-of-5 nowcast {now_med:.4f} lead {lead_med:.4f} | '
          f'displayed seed {SEEDS[i_med]} | dense==protocol {n_chk}/{len(sc["leads"])} '
          f'| {el:.0f}s', flush=True)
    return maps_prot, maps_dense, info


# ------------------------------------------------------------------------ render
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default='AA15a')
    ap.add_argument('--win', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion_mixed_v1.pt',
                    help="checkpoint for --arm learned")
    ap.add_argument('--arm', default='deployed', choices=['deployed', 'bank', 'learned'],
                    help="'deployed' = wavesplat/ production path (auto -> real-domain "
                         "gate -> v2_konly); 'bank' = research mmse_bank_estimate")
    ap.add_argument('--out', default=OUTGIF, help='output .gif path')
    ap.add_argument('--note', default='', help='extra caveat line under the header')
    ap.add_argument('--info-key', default='', dest='info_key',
                    help='if set, store this run under additional_windows[<key>] in the '
                         'info JSON instead of overwriting its top level')
    a = ap.parse_args()

    import d3_compare_gifs as cg
    print(f'# scene {a.record} w{a.win}', flush=True)
    sc = cg.scene_real(a.record, win=a.win)
    D = dense_gt(a.record, a.win)
    assert np.allclose(D['gx'], sc['gxq']) and np.allclose(D['gy'], sc['gyq'])
    # sanity: the dense GT reproduces the protocol's 1 s lead maps bit-for-bit
    for li, L in enumerate(sc['leads']):
        j = int(round(float(L) / D['dt']))
        assert np.allclose(D['Z'][j], sc['maps_gt'][li], equal_nan=True), li

    target_dt = max(TARGET_DT, float(D['lead_max']) / 85.0)
    step = max(1, int(round(target_dt / D['dt'])))
    jj = list(range(0, D['Z'].shape[0], step))
    gx, gy = sc['gxq'], sc['gyq']
    sg = float(np.std(D['Z'][np.isfinite(D['Z'])]))
    tsec = [j * D['dt'] for j in jj]

    if a.arm == 'learned':
        maps_prot, maps_dense, arm_info = estimate_learned(sc, tsec, a.ckpt)
        preds = maps_dense
        maps_prot_pred = maps_prot
        init_s = arm_info['seconds']
        subtitle = ('Research LEARNED arm — WavePrior mixed_v1\n'
                    'set-diffusion proposal + polish + LS field (median seed of 5)')
        headline = 'research LEARNED (WavePrior mixed_v1) field estimate'
        cfgline = ('Frozen learned protocol: beamform-conditioned set-diffusion (4-sample '
                   'BMA) + 80-iter polish + LS field, median of 5 diffusion seeds.  '
                   'No new observation after t = 0.')
    elif a.arm == 'bank':
        t0a = float(sc['Pq'][0, 2])
        maps_prot, maps_dense, init_s, n_chk = estimate_bank(sc, [t0a + t for t in tsec])
        preds = maps_dense
        maps_prot_pred = maps_prot
        arm_info = dict(name='bank_mmse_research',
                        entry='scripts/d3_sa_nomp.mmse_bank_estimate',
                        call='mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05*Hs, '
                             'kp_hint=kp, seed=i0)',
                        sigma0=float(0.05 * sc['Hs']), kp_hint=float(sc['kp']),
                        seed=int(sc['seed']),
                        dense_equals_protocol_at_shared_leads=int(n_chk),
                        seconds=round(init_s, 1))
        subtitle = ('Research estimator — bank-MMSE\n'
                    'd3_sa_nomp.mmse_bank_estimate (frozen protocol call)')
        headline = 'research bank-MMSE field estimate'
        cfgline = ('Frozen research protocol: mmse_bank_estimate(P, z, depth, Pq, '
                   'sigma0 = 0.05·Hs, kp_hint, seed = i0).  No new observation after '
                   't = 0: all later frames are pure forecast.')
    else:
        upd, t0, init_s, cfg = estimate(sc, a.seed)
        prim = upd._primary
        assert prim.init_method == 'bank_mmse' and prim._auto_real_gate, (
            'expected the post-port gate route')
        print(f"  route={prim.init_method} real_gate={prim._auto_real_gate} "
              f"bank_members={len(prim._bank)} init {init_s:.0f}s", flush=True)
        gxt = torch.tensor(gx, dtype=torch.float32)
        gyt = torch.tensor(gy, dtype=torch.float32)
        preds = np.array([upd.predict_field(gxt, gyt, t0 + tt).cpu().numpy()
                          for tt in tsec])
        maps_prot_pred = np.array([upd.predict_field(gxt, gyt, t0 + float(L)).cpu().numpy()
                                   for L in sc['leads']])
        arm_info = dict(name='deployed_production_path_bank_mmse',
                        entry='wavesplat.optimizer.sequential_update.make_updater',
                        init_method_config=cfg['init_method'],
                        route_after_gate=prim.init_method,
                        real_domain_gate_fired=bool(prim._auto_real_gate),
                        bank_members=len(prim._bank),
                        bank_seeds=list(prim._bank_seeds),
                        bank_kp=[f.kp for f in prim._bank],
                        bank_val_misfit=[round(f.vr, 4) for f in prim._bank],
                        Hs_obs=float(prim._bank_Hs_obs), seed=a.seed,
                        init_seconds=round(init_s, 1))
        subtitle = ("WaveSplat estimate — deployed path (2026-07-29 port)\n"
                    "init_method='auto' → real-domain gate → bank-MMSE (median-of-3)")
        headline = 'WaveSplat field estimate'
        cfgline = (f"Shipped config, gate live → ported bank-MMSE, observation-only "
                   f"parameters (Hs, kp from the obs), median-of-{len(prim._bank)} "
                   f"field ensemble.  After t = 0 no new observation enters: everything "
                   f"past the first frame is pure forecast.")

    nrs, ccs = [], []
    for fi, j in enumerate(jj):
        g = D['Z'][j]; m = np.isfinite(g); p = preds[fi]
        nrs.append(float(np.sqrt(np.mean((p[m] - g[m]) ** 2)) / sg))
        ccs.append(float(np.corrcoef(p[m], g[m])[0, 1]))

    # protocol-grid scores (the 1 s leads scene_real defines), for the record
    prot_nr, prot_cc = [], []
    for li, L in enumerate(sc['leads']):
        p = maps_prot_pred[li]
        g = sc['maps_gt'][li]; m = np.isfinite(g)
        prot_nr.append(float(np.sqrt(np.mean((p[m] - g[m]) ** 2)) / sg))
        prot_cc.append(float(np.corrcoef(p[m], g[m])[0, 1]))

    vmax = float(np.nanpercentile(np.abs(D['Z']), 99))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#eeece6')
    ecmap = plt.get_cmap('inferno').copy(); ecmap.set_bad('#eeece6')
    err_all = np.abs(preds - D['Z'][jj])
    evmax = float(np.nanpercentile(err_all[np.isfinite(err_all)], 98))
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    obs = np.asarray(sc['P'], float)

    figw = W_PX / DPI
    frames_img = []
    for fi, j in enumerate(jj):
        g = D['Z'][j]; m = np.isfinite(g); p = preds[fi]
        fig = plt.figure(figsize=(figw, figw * 0.52), dpi=DPI)
        gs = fig.add_gridspec(1, 3, left=0.045, right=0.865, top=0.715, bottom=0.135,
                              wspace=0.12)
        axs = [fig.add_subplot(gs[0, c]) for c in range(3)]

        axs[0].imshow(np.ma.masked_invalid(g), origin='lower', extent=ext, cmap=cmap,
                      norm=norm, aspect='equal', interpolation='nearest')
        axs[0].scatter(obs[:, 0], obs[:, 1], s=1.6, c='#111111', alpha=0.30,
                       linewidths=0, zorder=3)
        axs[0].set_title('Stereo sea surface (truth)\n+ 600 LiDAR-fan samples, past 40 s',
                         fontsize=9.5, color=INK, pad=4)

        # masked to the stereo-valid region, as in the repo's other real-data field
        # figures, so the three panels are pixel-for-pixel comparable
        axs[1].imshow(np.ma.masked_invalid(np.where(m, p, np.nan)), origin='lower',
                      extent=ext, cmap=cmap, norm=norm, aspect='equal',
                      interpolation='nearest')
        axs[1].set_title(subtitle, fontsize=9.5, color=BLUE, pad=4)
        amp = float(np.std(p[m])) / float(np.std(g[m]))
        axs[1].text(0.97, 0.03, f'amp {amp:.1f}x truth', transform=axs[1].transAxes,
                    fontsize=8, color=SEC, va='bottom', ha='right',
                    bbox=dict(fc=SURF, ec='none', alpha=0.85, pad=2))

        axs[2].imshow(np.ma.masked_invalid(np.where(m, np.abs(p - g), np.nan)),
                      origin='lower', extent=ext, cmap=ecmap, vmin=0.0, vmax=evmax,
                      aspect='equal', interpolation='nearest')
        axs[2].set_title('|error|  (where stereo truth is valid)\n ',
                         fontsize=9.5, color=INK, pad=4)

        for k, ax in enumerate(axs):
            ax.set_xlabel('x [m]', fontsize=8, color=SEC)
            ax.tick_params(labelsize=7, colors=SEC)
            for s in ax.spines.values():
                s.set_color(MUT); s.set_linewidth(0.6)
            if k == 0:
                ax.set_ylabel('y [m]', fontsize=8, color=SEC)
            else:
                ax.set_yticklabels([])

        cax1 = fig.add_axes([0.885, 0.24, 0.012, 0.38])
        cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax1)
        cb1.ax.set_title('η [m]', fontsize=8, color=SEC, pad=5)
        cb1.ax.tick_params(labelsize=7, colors=SEC)
        cax2 = fig.add_axes([0.952, 0.24, 0.012, 0.38])
        cb2 = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(0, evmax), cmap=ecmap), cax=cax2)
        cb2.ax.set_title('|err| [m]', fontsize=8, color=SEC, pad=5)
        cb2.ax.tick_params(labelsize=7, colors=SEC)

        fig.suptitle(f"Real ocean data — {a.record} ({RECNAME.get(a.record, a.record)}), "
                     f"window {a.win}:  {headline} vs stereo truth",
                     fontsize=12.5, x=0.045, ha='left', y=0.975)
        fig.text(0.045, 0.905,
                 f"Bergamasco / Guimaraes 2020 IFREMER stereo; frozen d3_falcon_real_operator "
                 f"window (Hs {sc['Hs']:.2f} m, depth {sc['depth']:.0f} m, "
                 f"{len(sc['z'])} obs).  Zero-shot: nothing was tuned on this record.\n"
                 + cfgline + (f"\n{a.note}" if a.note else ''),
                 fontsize=8.6, color=SEC, ha='left', va='top', linespacing=1.5)
        lead = tsec[fi]
        fig.text(0.045, 0.788,
                 f"t = +{lead:5.2f} s" + ('   (nowcast)' if fi == 0 else '   (forecast lead)'),
                 fontsize=11, color=INK, ha='left', va='bottom', family='monospace')
        fig.text(0.865, 0.788,
                 f"nRMSE {nrs[fi]:.2f}    corr {ccs[fi]:+.2f}",
                 fontsize=11.5, color=BLUE, ha='right', va='bottom', family='monospace')

        axp = fig.add_axes([0.045, 0.035, 0.36, 0.011])
        axp.set_xlim(0, tsec[-1] if tsec[-1] > 0 else 1); axp.set_ylim(0, 1)
        axp.barh(0.5, lead, height=1.0, color=AQUA)
        axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values():
            s.set_color(GRID)
        fig.text(0.420, 0.030, f'lead 0 → {tsec[-1]:.0f} s', fontsize=8, color=SEC,
                 ha='left', va='bottom')
        fig.text(0.865, 0.030,
                 f"window mean over 0–{tsec[-1]:.0f} s:  nRMSE {np.mean(nrs):.2f}   "
                 f"corr {np.mean(ccs):+.2f}", fontsize=8.5, color=SEC, ha='right',
                 va='bottom')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI)
        plt.close(fig)
        buf.seek(0)
        frames_img.append(Image.open(buf).convert('RGB'))
        if fi % 10 == 0:
            print(f'  frame {fi+1}/{len(jj)}  t=+{lead:.2f}s  nRMSE {nrs[fi]:.3f}',
                  flush=True)

    pal = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE)
           for f in frames_img]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], loop=0,
                duration=int(round(1000 / FPS)), optimize=True, disposal=2)
    sz = os.path.getsize(a.out)
    dur = len(pal) / FPS
    print(f'wrote {a.out}: {len(pal)} frames, {frames_img[0].size[0]}x'
          f'{frames_img[0].size[1]} px, {FPS} fps, {dur:.1f} s, {sz/1e6:.2f} MB')

    info = dict(
        gif=a.out, size_bytes=sz, size_mb=round(sz / 1e6, 3), n_frames=len(pal),
        fps=FPS, duration_s=round(dur, 2),
        px=[frames_img[0].size[0], frames_img[0].size[1]],
        record=a.record, window=a.win, record_name=RECNAME.get(a.record),
        record_path=None, Hs=float(sc['Hs']), kp=float(sc['kp']),
        depth=float(sc['depth']), n_obs=int(len(sc['z'])), gt_std=sg,
        i0=D['i0'], record_dt=D['dt'], anim_dt=round(step * D['dt'], 4),
        lead_max_s=float(D['lead_max']),
        arm=a.arm, deployed=arm_info,
        scores_animation=dict(nowcast_nrmse=nrs[0], nowcast_corr=ccs[0],
                              mean_nrmse=float(np.mean(nrs)), mean_corr=float(np.mean(ccs)),
                              t_s=tsec, nrmse=nrs, corr=ccs),
        scores_protocol_1s_leads=dict(leads=[float(v) for v in sc['leads']],
                                      nrmse=prot_nr, corr=prot_cc,
                                      nowcast_nrmse=prot_nr[0], nowcast_corr=prot_cc[0],
                                      mean_nrmse=float(np.mean(prot_nr)),
                                      mean_corr=float(np.mean(prot_cc))),
    )
    info['note'] = a.note or None
    if a.info_key:
        base = json.load(open(OUTJSON)) if os.path.exists(OUTJSON) else {}
        base.setdefault('additional_windows', {})[a.info_key] = info
        json.dump(base, open(OUTJSON, 'w'), indent=1)
        print(f"wrote {OUTJSON} [additional_windows.{a.info_key}]")
    else:
        json.dump(info, open(OUTJSON, 'w'), indent=1)
        print('wrote', OUTJSON)


if __name__ == '__main__':
    main()
