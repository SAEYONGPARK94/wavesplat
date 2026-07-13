"""A-2 directional invariance: score ALL 30 dir180 (head-sea) synthetic
fixedT episodes with the two deployment arms and answer the registered
question:

  "Does the two-arm performance structure established at dir0 (both arms high
   corr + hand slightly ahead on clean synthetic seas) survive at dir180
   (head sea, encounter-causal)?"  -- judged on BOTH the whole-field corr
  (global grid, leads {0,2,5,8} s) AND an encounter-track corr.

Arms (identical to the real 60-window / appendix_synth definitions):
  hand    = mmse_gpu(P, z, depth, Q, sig0, kp_hint)          (default val+EM)
  learned = net_gamma_mmse(..., use_em=False, smult_head=head)
  ckpt    = results/direction3/prior_net_l1s2bw.pt

Obs recipe (matches appendix_synth_eval.json / real 60-window arm):
  range-stratified ~2000 pts from a 12-frame / ~20 s window ending t=22 s;
  sig0 = 0.05*Hs (episode noise is not stored in the h5; documented here).
  kp_hint = episode spectral peak |k| of the max-amplitude component.

Encounter track (SECONDARY):
  IMPORTANT -- the h5 stores NO vessel trajectory or speed convention (only
  the wave field grid, LiDAR points and timestamps; verified). The encounter
  track is therefore constructed EXTERNALLY from the codebase DESIGN point:
  U = 11.3 m/s (C-8 design speed), ship advancing +x into the dir180 head sea
  (dir180 waves propagate -x, so the bow faces +x toward the wave source).
  A fan of parallel hull tracks (y offsets) advances the space-time diagonal
  x(L)=x0+U*L at query time t_end+L; corr is taken along each diagonal
  (the encounter path) and averaged over tracks. This convention is flagged
  in the JSON (encounter_track.convention_stored_in_h5 = false).

python3 scripts/d3_a2_dir180_eval.py            # all 30, sequential
python3 scripts/d3_a2_dir180_eval.py --smoke    # first 2 (sanity)
"""
import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import torch
from d3_gpu_mmse import mmse_gpu, DEV
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_l1_s2head_train import SmultHead

OUT = 'results/direction3'
CK = 'results/direction3/prior_net_l1s2bw.pt'
DATE = '2026-07-12'
LEADS = [0.0, 2.0, 5.0, 8.0]                     # field corr leads

# --- encounter track (design convention; NOT stored in h5) ---
U_SHIP = 11.3                                     # C-8 design speed [m/s]
TRACK_LEADS = np.round(np.arange(0.0, 8.01, 0.2), 3)   # diagonal, lands on 0.1 s frames
TRACK_X0 = 60.0                                  # start x [m] (60 -> ~150 at 8 s, in grid)
TRACK_Y0 = np.linspace(-60.0, 60.0, 13)          # parallel hull tracks


def load_net_head():
    net = get_net(CK)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CK, map_location=DEVICE)['head'])
    head.eval()
    print(f'net/head loaded ({CK}); DEVICE {DEVICE}', flush=True)
    return net, head


def bilinear_frame(field, gx, gy, xq, yq):
    """bilinear sample of a single (ny,nx) frame at scattered (xq,yq)."""
    xq = np.clip(xq, gx[0], gx[-1])
    yq = np.clip(yq, gy[0], gy[-1])
    ix = np.clip(np.searchsorted(gx, xq) - 1, 0, len(gx) - 2)
    iy = np.clip(np.searchsorted(gy, yq) - 1, 0, len(gy) - 2)
    x0, x1 = gx[ix], gx[ix + 1]
    y0, y1 = gy[iy], gy[iy + 1]
    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)
    f00 = field[iy, ix]
    f10 = field[iy, ix + 1]
    f01 = field[iy + 1, ix]
    f11 = field[iy + 1, ix + 1]
    return ((f00 * (1 - tx) + f10 * tx) * (1 - ty)
            + (f01 * (1 - tx) + f11 * tx) * ty)


def load_episode(path, rng, n_obs=2000):
    """range-stratified obs (12-frame / ~20 s window ending t=22 s) plus
    the full gt field stack (for field-corr and track-corr truth).

    n_obs defaults to 2000 -- the frozen deployment recipe, unchanged.  Pass 600
    for the sparse regime the thesis assumes (50 returns/frame, configs
    `n_pts: 50`): the PI's rule is that learning may use as much data as it
    likes but validation runs at the originally assumed sparsity."""
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
        dt = float(ts[1] - ts[0])
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
            k = min(len(idx), max(-(-n_obs // 12), 1))
            if k:
                take.append(rng.choice(idx, k, replace=False))
        sel = np.concatenate(take)
        Ps, zs = P[sel], zv[sel]

        gt_leads = []
        for L in LEADS:
            idx = min(int(round((t_end + L) / dt)), len(ts) - 1)
            gt_leads.append(gz[idx])
    return dict(Ps=Ps, zs=zs, Hs=Hs, depth=depth, ncomp=ncomp, sea=sea,
                kp=kp, gx=gx, gy=gy, gz=gz, ts=ts, dt=dt, t_end=t_end,
                gt_leads=gt_leads,
                dir180=('dir180' in os.path.basename(path)))


def build_queries(ep):
    """field grid Q (all leads) + encounter-track Q (design convention)."""
    gx, gy, t_end = ep['gx'], ep['gy'], ep['t_end']
    X, Y = np.meshgrid(gx, gy)
    npix = X.size
    Qf = np.vstack([
        np.column_stack([X.ravel(), Y.ravel(),
                         np.full(npix, t_end + L)]) for L in LEADS])
    # encounter tracks: space-time diagonals x = x0 + U*L, one per y offset
    Qt_list, tr_truth = [], []
    for y0 in TRACK_Y0:
        xq = TRACK_X0 + U_SHIP * TRACK_LEADS
        yq = np.full_like(TRACK_LEADS, y0)
        tq = t_end + TRACK_LEADS
        Qt_list.append(np.column_stack([xq, yq, tq]))
        # truth along the diagonal (bilinear in space, exact frame in time)
        gt = np.array([
            bilinear_frame(ep['gz'][min(int(round(t / ep['dt'])),
                                        len(ep['ts']) - 1)],
                           gx, gy, np.array([x]), np.array([y]))[0]
            for x, y, t in zip(xq, yq, tq)])
        tr_truth.append(gt)
    Qt = np.vstack(Qt_list)
    return Qf, Qt, npix, X.shape, np.array(tr_truth)


def predict_arms(ep, Qf, Qt, net, head):
    """fit each arm once, query field + track in a single solve."""
    Ps, zs, depth, kp = ep['Ps'], ep['zs'], ep['depth'], ep['kp']
    sig0 = 0.05 * ep['Hs']
    Q = np.vstack([Qf, Qt])
    nf = len(Qf)
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
    return (ph[:nf], ph[nf:]), (pn[:nf], pn[nf:]), dt_h, dt_n


def score_episode(ep, net, head):
    Qf, Qt, npix, shp, tr_truth = build_queries(ep)
    (fh, th_), (fn, tn_), dt_h, dt_n = predict_arms(ep, Qf, Qt, net, head)

    # ---- field corr (global grid) per lead
    corr_h, corr_n = [], []
    for k in range(len(LEADS)):
        g = ep['gt_leads'][k].ravel()
        ph = fh[k * npix:(k + 1) * npix]
        pn = fn[k * npix:(k + 1) * npix]
        corr_h.append(float(np.corrcoef(ph, g)[0, 1]))
        corr_n.append(float(np.corrcoef(pn, g)[0, 1]))

    # ---- encounter-track corr along each space-time diagonal
    ntr, nl = tr_truth.shape
    th_m = th_.reshape(ntr, nl)
    tn_m = tn_.reshape(ntr, nl)
    tk_h_per_track, tk_n_per_track = [], []
    for i in range(ntr):
        g = tr_truth[i]
        if np.std(g) < 1e-9:
            continue
        tk_h_per_track.append(float(np.corrcoef(th_m[i], g)[0, 1]))
        tk_n_per_track.append(float(np.corrcoef(tn_m[i], g)[0, 1]))
    tk_h = float(np.mean(tk_h_per_track))
    tk_n = float(np.mean(tk_n_per_track))
    # pooled (all diagonal points)
    gp = tr_truth.ravel()
    tk_h_pool = float(np.corrcoef(th_.ravel(), gp)[0, 1])
    tk_n_pool = float(np.corrcoef(tn_.ravel(), gp)[0, 1])

    return dict(
        corr_hand=corr_h, corr_learned=corr_n,
        track_hand=tk_h, track_learned=tk_n,
        track_hand_pool=tk_h_pool, track_learned_pool=tk_n_pool,
        n_tracks=len(tk_h_per_track),
        ms_hand=dt_h * 1e3, ms_learned=dt_n * 1e3)


def summarize(records):
    H = np.array([r['corr_hand'] for r in records])
    N = np.array([r['corr_learned'] for r in records])
    TH = np.array([r['track_hand'] for r in records])
    TN = np.array([r['track_learned'] for r in records])
    per_lead = []
    for k, L in enumerate(LEADS):
        per_lead.append(dict(
            lead=L, hand=float(H[:, k].mean()), learned=float(N[:, k].mean()),
            delta=float(N[:, k].mean() - H[:, k].mean()),
            learned_wins=int((N[:, k] > H[:, k]).sum())))
    by_bf = {}
    for bf in sorted({r['sea'] for r in records}):
        m = np.array([r['sea'] == bf for r in records])
        by_bf[bf] = dict(
            n=int(m.sum()),
            field_hand=float(H[m].mean()), field_learned=float(N[m].mean()),
            field_lead0_hand=float(H[m, 0].mean()),
            field_lead0_learned=float(N[m, 0].mean()),
            track_hand=float(TH[m].mean()), track_learned=float(TN[m].mean()))
    return dict(
        n_episodes=len(records), per_lead=per_lead,
        field_mean_hand=float(H.mean()), field_mean_learned=float(N.mean()),
        field_lead0_hand=float(H[:, 0].mean()),
        field_lead0_learned=float(N[:, 0].mean()),
        track_mean_hand=float(TH.mean()), track_mean_learned=float(TN.mean()),
        field_learned_wins_all=int((N > H).sum()), field_cells=int(H.size),
        track_learned_wins=int((TN > TH).sum()),
        by_bf=by_bf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    eps = sorted(glob.glob('data/episode_*fixedT_dir180.h5'))
    if a.smoke:
        eps = eps[:2]
    print(f'{len(eps)} dir180 episodes', flush=True)

    net, head = load_net_head()
    rng = np.random.default_rng(11)              # same seed as appendix_synth
    records, t0 = [], time.time()
    for i, path in enumerate(eps):
        te = time.time()
        ep = load_episode(path, rng)
        s = score_episode(ep, net, head)
        rec = dict(episode=os.path.basename(path), sea=ep['sea'],
                   n_comp=ep['ncomp'], dir180=ep['dir180'], Hs=ep['Hs'],
                   depth=ep['depth'], kp=ep['kp'], n_obs=int(len(ep['zs'])),
                   t_end=ep['t_end'], leads=LEADS, **s)
        records.append(rec)
        print(f'  [{i+1}/{len(eps)}] {rec["episode"]:<42} {ep["sea"]} '
              f'{ep["ncomp"]:>2}c  field h/l '
              f'{["%+.2f" % c for c in s["corr_hand"]]}'
              f'{["%+.2f" % c for c in s["corr_learned"]]}  '
              f'track h/l {s["track_hand"]:+.2f}/{s["track_learned"]:+.2f}  '
              f'({time.time()-te:.1f}s)', flush=True)

    summ = summarize(records)
    out = dict(
        date=DATE, question='A-2 directional invariance (dir180 head sea)',
        leads=LEADS,
        sig0_convention='0.05*Hs (episode noise not stored in h5; '
                        'matches real 60-window arm)',
        obs_recipe='range-stratified ~2000 pts, 12-frame/~20s window '
                   'ending t=22s',
        arms=dict(hand='mmse_gpu(P,z,depth,Q,sig0,kp_hint) default '
                       '(val-select + EM)',
                  learned='net_gamma_mmse(...,use_em=False,smult_head=head)',
                  ckpt=CK),
        encounter_track=dict(
            convention_stored_in_h5=False,
            note='h5 stores no vessel trajectory/speed; track built from '
                 'codebase DESIGN point',
            U_ship=U_SHIP, heading='+x into dir180 head sea',
            x0=TRACK_X0, y_offsets=TRACK_Y0.tolist(),
            track_leads=TRACK_LEADS.tolist(),
            metric='mean over parallel hull tracks of corr along the '
                   'space-time encounter diagonal (x=x0+U*L, t=t_end+L)'),
        episodes=records, summary=summ)
    outp = f'{OUT}/a2_dir180_eval.json'
    json.dump(out, open(outp, 'w'), indent=2)
    print(f'\nwrote {outp}  ({(time.time()-t0)/60:.1f} min)', flush=True)

    # ---- console summary
    print('\n===== A-2 dir180 summary =====', flush=True)
    print(f'field mean  hand {summ["field_mean_hand"]:+.3f}  learned '
          f'{summ["field_mean_learned"]:+.3f}  '
          f'(learned wins {summ["field_learned_wins_all"]}/'
          f'{summ["field_cells"]} cells)', flush=True)
    for pl in summ['per_lead']:
        print(f'  lead {int(pl["lead"])}s  hand {pl["hand"]:+.3f}  learned '
              f'{pl["learned"]:+.3f}  delta {pl["delta"]:+.3f}  '
              f'learned_wins {pl["learned_wins"]}/{summ["n_episodes"]}',
              flush=True)
    print(f'track mean  hand {summ["track_mean_hand"]:+.3f}  learned '
          f'{summ["track_mean_learned"]:+.3f}  '
          f'(learned wins {summ["track_learned_wins"]}/{summ["n_episodes"]})',
          flush=True)
    print('\n--- by BF (field mean / lead0 / track) ---', flush=True)
    for bf, v in summ['by_bf'].items():
        print(f'  {bf} (n={v["n"]:>2})  field h/l '
              f'{v["field_hand"]:+.3f}/{v["field_learned"]:+.3f}  '
              f'lead0 {v["field_lead0_hand"]:+.3f}/'
              f'{v["field_lead0_learned"]:+.3f}  track '
              f'{v["track_hand"]:+.3f}/{v["track_learned"]:+.3f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
