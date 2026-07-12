"""A-3 speed invariance: sweep ship speed U over {1,3,6,9,12} m/s and score the
two deployment arms on the SAME synthetic head-sea episodes, answering the
registered question:

  "Is estimation performance (whole-field + encounter) invariant to ship speed
   U over 1-12 m/s -- for BOTH hand and learned arms?"

Rationale (frozen; interpretation belongs to the caller): the estimator lives in
earth-fixed coordinates and never sees U. Given the SAME observation cloud its
output is bit-identical regardless of what U produced it (proven by mmse_gpu /
net_gamma_mmse math -- psi = kx*x + ky*y - om*t, no U term). Therefore U can only
enter through (a) WHICH points the moving platform samples -- the synthetic
aperture the ship sweeps during the observation window -- and (b) the encounter
track geometry (how far ahead the ship reaches per lead). This script injects U
in exactly those two places and holds everything else (obs budget, fan geometry,
scan times, episode field) fixed.

How U enters the observation window (ship position moving):
  Ship advances +x into the dir180 head sea (waves propagate -x; bow faces the
  wave source at +x -- identical convention to d3_a2_dir180_eval). A forward fan
  (FOV 120 deg, range 20-150 m) samples the analytic linear field around the
  moving hull at N scan times over a W_obs window ending t_end. The hull
  mid-window position x_mid is FIXED across U, so the aggregate footprint stays
  centered on the grid and the ONLY thing that changes with U is the swept
  aperture width (U -> 0: compact fan; U -> 12: long synthetic aperture) plus the
  encounter-track reach. Matched budget: identical N_scan, pts/scan, fan, noise
  for every U.

Field truth is exact: fixedT episodes are pure linear superpositions of the
stored components (verified analytic vs gt_wave_field max err ~2e-16), so obs and
GT at any (x,y,t) come from eta = sum A cos(kx x + ky y - om t + phi).

Arms (identical to d3_a2_dir180_eval / real 60-window definitions):
  hand    = mmse_gpu(P, z, depth, Q, sig0, kp_hint)
  learned = net_gamma_mmse(..., use_em=False, smult_head=head)
  ckpt    = results/direction3/prior_net_l1s2bw.pt

Metrics per (episode, U, arm):
  field_corr[lead]      whole-grid corr at leads {0,2,5,8} s
  field_corr_incover    lead-0 corr over grid cells within 40 m of any obs
                        (aperture-coverage-controlled -> the clean invariance
                         probe: these cells are observed at EVERY U)
  track_corr            mean over parallel hull tracks of corr along the
                        space-time encounter diagonal x=x_end+U*L, t=t_end+L

python3 scripts/d3_a3_speed_sweep.py            # 2 seeds x 5 U, sequential
python3 scripts/d3_a3_speed_sweep.py --smoke    # 1 seed x U {1,12}
"""
import argparse
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

# --- speed grid + episodes (registered) ---
U_GRID = [1.0, 3.0, 6.0, 9.0, 12.0]
EPISODES = ['data/episode_BF4_15comp_seed0_fixedT_dir180.h5',
            'data/episode_BF4_15comp_seed1_fixedT_dir180.h5']

# --- observation window geometry (fixed across U) ---
T_END = 22.0                 # window end [s]; leads reach t_end+8 = 30 = episode span
W_OBS = 12.0                 # observation memory [s]
N_SCAN = 12                  # scan times in the window (matched budget)
M_PER_SCAN = 170             # pts per scan  -> ~2040 obs
X_MID = 72.0                 # hull x at window midpoint [m] (FIXED across U)
CY = 0.0                     # hull y (grid y-center)
R0, R1 = 20.0, 150.0         # forward fan range [m] (Falcon reaches 250 @10%)
AZ = np.radians(60.0)        # fan half-angle (FOV 120 deg)
NOISE = 0.02                 # LiDAR range noise 1sigma [m] (Falcon spec)
HEADING = +1.0               # ship advances +x into dir180 head sea

LEADS = [0.0, 2.0, 5.0, 8.0]                       # field-corr leads
R_COVER = 40.0                                     # in-aperture corr radius [m]

# --- encounter track (ship's own forward path at speed U) ---
TRACK_LEADS = np.round(np.arange(0.0, 8.01, 0.2), 3)
TRACK_Y0 = np.linspace(-40.0, 40.0, 9)             # parallel hull tracks


def load_net_head():
    net = get_net(CK)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CK, map_location=DEVICE)['head'])
    head.eval()
    print(f'net/head loaded ({CK}); DEVICE {DEVICE}', flush=True)
    return net, head


def load_episode(path):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]        # [A, om, kx, ky, phi, Q]
        Hs = float(f['config'].attrs['Hs'])
        depth = float(f['config'].attrs['depth'])
        ncomp = int(f['config'].attrs['n_components'])
        sea = str(f['config'].attrs['sea_state'])
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
    A, om, kx, ky, phi = (comp[:, 0], comp[:, 1], comp[:, 2],
                          comp[:, 3], comp[:, 4])
    kp = float(np.hypot(kx, ky)[np.argmax(A)])       # peak |k| of max-A comp

    def eta(x, y, t):
        """exact linear field; t is a scalar (one scan / one lead frame)."""
        x = np.ravel(x)[:, None]
        y = np.ravel(y)[:, None]
        ps = kx[None] * x + ky[None] * y - om[None] * t + phi[None]
        return (A[None] * np.cos(ps)).sum(1)

    return dict(eta=eta, Hs=Hs, depth=depth, ncomp=ncomp, sea=sea, kp=kp,
                gx=gx, gy=gy, path=path)


def ship_x(t, U):
    """hull x at time t; x_mid fixed at window midpoint, advances +x at U."""
    return X_MID + HEADING * U * (t - (T_END - W_OBS / 2.0))


def build_obs(ep, U, rng):
    """moving forward-fan aperture; matched budget, only U changes the sweep."""
    eta = ep['eta']
    scan_ts = np.linspace(T_END - W_OBS, T_END, N_SCAN)
    P, z = [], []
    for t_s in scan_ts:
        sx = ship_x(t_s, U)
        rr = rng.uniform(R0, R1, M_PER_SCAN)
        aa = rng.uniform(-AZ, AZ, M_PER_SCAN)
        xs = sx + HEADING * rr * np.cos(aa)          # forward = +x
        ys = CY + rr * np.sin(aa)
        zz = eta(xs, ys, t_s) + rng.normal(0, NOISE, M_PER_SCAN)
        P.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
        z.append(zz)
    return np.vstack(P), np.concatenate(z)


def build_queries(ep, U):
    """field grid (all leads) + encounter-track diagonals; exact truth."""
    gx, gy, eta = ep['gx'], ep['gy'], ep['eta']
    X, Y = np.meshgrid(gx, gy)
    npix = X.size
    Qf, gt_leads = [], []
    for L in LEADS:
        Qf.append(np.column_stack([X.ravel(), Y.ravel(),
                                   np.full(npix, T_END + L)]))
        gt_leads.append(eta(X.ravel(), Y.ravel(), T_END + L))
    Qf = np.vstack(Qf)

    x_end = ship_x(T_END, U)
    Qt, tr_truth = [], []
    for y0 in TRACK_Y0:
        xq = x_end + HEADING * U * TRACK_LEADS
        yq = np.full_like(TRACK_LEADS, y0)
        tq = T_END + TRACK_LEADS
        Qt.append(np.column_stack([xq, yq, tq]))
        tr_truth.append(np.array([eta(np.array([x]), np.array([y]), t)[0]
                                  for x, y, t in zip(xq, yq, tq)]))
    Qt = np.vstack(Qt)
    return Qf, Qt, npix, X.shape, np.array(gt_leads), np.array(tr_truth)


def in_cover_mask(ep, Ps):
    """grid cells (lead-0 nowcast) within R_COVER of any observation."""
    X, Y = np.meshgrid(ep['gx'], ep['gy'])
    gxy = np.column_stack([X.ravel(), Y.ravel()])
    oxy = Ps[:, :2]
    mind = np.full(len(gxy), np.inf)
    for i in range(0, len(gxy), 512):                # chunk to bound memory
        d = np.sqrt(((gxy[i:i + 512, None, :] - oxy[None]) ** 2).sum(2)).min(1)
        mind[i:i + 512] = d
    return mind <= R_COVER


def predict_arms(ep, Ps, zs, Qf, Qt, net, head):
    depth, kp = ep['depth'], ep['kp']
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


def _corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def score(ep, U, net, head, rng):
    Ps, zs = build_obs(ep, U, rng)
    Qf, Qt, npix, shp, gt_leads, tr_truth = build_queries(ep, U)
    cover = in_cover_mask(ep, Ps)
    (fh, th_), (fn, tn_), dt_h, dt_n = predict_arms(ep, Ps, zs, Qf, Qt, net,
                                                    head)
    # whole-field corr per lead + in-cover lead-0 corr
    fc_h, fc_n = [], []
    for k in range(len(LEADS)):
        g = gt_leads[k]
        fc_h.append(_corr(fh[k * npix:(k + 1) * npix], g))
        fc_n.append(_corr(fn[k * npix:(k + 1) * npix], g))
    g0 = gt_leads[0]
    ic_h = _corr(fh[:npix][cover], g0[cover])
    ic_n = _corr(fn[:npix][cover], g0[cover])
    # encounter-track corr along each space-time diagonal
    ntr, nl = tr_truth.shape
    th_m, tn_m = th_.reshape(ntr, nl), tn_.reshape(ntr, nl)
    tk_h = [_corr(th_m[i], tr_truth[i]) for i in range(ntr)
            if np.std(tr_truth[i]) > 1e-9]
    tk_n = [_corr(tn_m[i], tr_truth[i]) for i in range(ntr)
            if np.std(tr_truth[i]) > 1e-9]
    # aperture diagnostics
    ox = Ps[:, 0]
    return dict(
        U=U, n_obs=int(len(zs)), n_cover=int(cover.sum()),
        obs_x_lo=float(ox.min()), obs_x_hi=float(ox.max()),
        aperture_x=float(ox.max() - ox.min()),
        field_corr_hand=fc_h, field_corr_learned=fc_n,
        field_incover_hand=ic_h, field_incover_learned=ic_n,
        track_hand=float(np.mean(tk_h)), track_learned=float(np.mean(tk_n)),
        ms_hand=dt_h * 1e3, ms_learned=dt_n * 1e3)


def summarize(records):
    """per-U means across episodes (field lead-0, field-mean, in-cover, track)."""
    tbl = []
    for U in U_GRID:
        rs = [r for r in records if r['U'] == U]
        if not rs:
            continue
        FH = np.array([r['field_corr_hand'] for r in rs])
        FN = np.array([r['field_corr_learned'] for r in rs])
        tbl.append(dict(
            U=U, n=len(rs),
            field_lead0_hand=float(FH[:, 0].mean()),
            field_lead0_learned=float(FN[:, 0].mean()),
            field_mean_hand=float(FH.mean()),
            field_mean_learned=float(FN.mean()),
            incover_hand=float(np.nanmean([r['field_incover_hand'] for r in rs])),
            incover_learned=float(np.nanmean([r['field_incover_learned']
                                              for r in rs])),
            track_hand=float(np.nanmean([r['track_hand'] for r in rs])),
            track_learned=float(np.nanmean([r['track_learned'] for r in rs])),
            aperture_x=float(np.mean([r['aperture_x'] for r in rs]))))

    def spread(key):
        v = [row[key] for row in tbl]
        return float(max(v) - min(v)), float(np.std(v))

    invariance = {k: dict(zip(('range', 'std'), spread(k)))
                  for k in ('field_lead0_hand', 'field_lead0_learned',
                            'incover_hand', 'incover_learned',
                            'track_hand', 'track_learned')}
    return dict(per_U=tbl, invariance=invariance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    eps = EPISODES[:1] if a.smoke else EPISODES
    us = [1.0, 12.0] if a.smoke else U_GRID
    print(f'A-3 speed sweep: {len(eps)} episodes x {len(us)} U '
          f'{us}', flush=True)

    net, head = load_net_head()
    records, t0 = [], time.time()
    for path in eps:
        ep = load_episode(path)
        # one rng per episode -> the obs-sampling seed is identical across U,
        # so the fan pattern differs only through the hull displacement.
        for U in us:
            rng = np.random.default_rng(2026)
            te = time.time()
            s = score(ep, U, net, head, rng)
            rec = dict(episode=os.path.basename(path), sea=ep['sea'],
                       n_comp=ep['ncomp'], Hs=ep['Hs'], depth=ep['depth'],
                       kp=ep['kp'], leads=LEADS, **s)
            records.append(rec)
            print(f"  {os.path.basename(path):<38} U={U:>4.1f}  "
                  f"field0 h/l {s['field_corr_hand'][0]:+.3f}/"
                  f"{s['field_corr_learned'][0]:+.3f}  "
                  f"incov h/l {s['field_incover_hand']:+.3f}/"
                  f"{s['field_incover_learned']:+.3f}  "
                  f"track h/l {s['track_hand']:+.3f}/{s['track_learned']:+.3f}  "
                  f"ap={s['aperture_x']:.0f}m ({time.time()-te:.1f}s)",
                  flush=True)

    summ = summarize(records)
    out = dict(
        date=DATE,
        question='A-3 speed invariance (U 1-12 m/s, hand + learned)',
        U_grid=us, episodes=[os.path.basename(p) for p in eps],
        leads=LEADS,
        geometry=dict(
            note='U enters ONLY via moving-hull aperture + encounter-track '
                 'reach; estimator math is U-free (earth-fixed coords)',
            heading='+x into dir180 head sea (waves propagate -x)',
            t_end=T_END, W_obs=W_OBS, n_scan=N_SCAN, m_per_scan=M_PER_SCAN,
            x_mid_fixed=X_MID, fan_range=[R0, R1], fan_fov_deg=120.0,
            noise_m=NOISE, r_cover=R_COVER,
            obs_budget_matched=True),
        arms=dict(hand='mmse_gpu(P,z,depth,Q,sig0,kp_hint)',
                  learned='net_gamma_mmse(...,use_em=False,smult_head=head)',
                  ckpt=CK),
        sig0_convention='0.05*Hs',
        records=records, summary=summ)
    outp = f'{OUT}/a3_speed_sweep.json'
    json.dump(out, open(outp, 'w'), indent=2)
    print(f'\nwrote {outp}  ({(time.time()-t0)/60:.1f} min)', flush=True)

    # ---- console summary table
    print('\n===== A-3 speed-invariance summary (mean over episodes) =====',
          flush=True)
    print(f"{'U':>5} | {'field0 h':>9} {'field0 l':>9} | "
          f"{'incov h':>8} {'incov l':>8} | {'track h':>8} {'track l':>8} | "
          f"{'ap[m]':>6}", flush=True)
    for row in summ['per_U']:
        print(f"{row['U']:>5.1f} | {row['field_lead0_hand']:>9.3f} "
              f"{row['field_lead0_learned']:>9.3f} | "
              f"{row['incover_hand']:>8.3f} {row['incover_learned']:>8.3f} | "
              f"{row['track_hand']:>8.3f} {row['track_learned']:>8.3f} | "
              f"{row['aperture_x']:>6.0f}", flush=True)
    print('\n--- invariance (max-min range across U; small = invariant) ---',
          flush=True)
    for k, v in summ['invariance'].items():
        print(f"  {k:<22} range {v['range']:.3f}  std {v['std']:.3f}",
              flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
