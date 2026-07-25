"""B0 mechanism kill-test — why does the learned prior beat the analytic prior on HOS?

The leading (untested) hypothesis: the net wins because the HOS free-wave spectrum
differs from the linear-JONSWAP shape the hand prior assumes. Discriminating test:
give the analytic MMSE the TRUE free-wave spectrum as its prior (oracle) and see if it
catches the net.

  - oracle catches/beats net  ⟹  mechanism = free-wave spectral shape (publishable)
  - net still beats oracle     ⟹  net's edge is NOT just spectral shape (mechanism open)

HOS scene built with KNOWN components (HOSSimulator.initialize_from_components) so the
oracle spectrum is exact. Same solver/bank/downstream for all three arms (fair).
Lead-averaged map nRMSE; lower = better.
"""
import argparse
import json
import time

import numpy as np
import d3_compare_gifs as cg  # side effects: chdir repo root, sys.path
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from wavesim.hos_simulator import HOSSimulator, HOSConfig

G, DEPTH, SCAN_DT = cg.G, cg.DEPTH, cg.SCAN_DT
M_PER_SCAN, NOISE, SPINUP, DT_HOS = cg.M_PER_SCAN, cg.NOISE, cg.SPINUP, cg.DT_HOS
NET_CKPT = cg.NET_CKPT


def hos_scene_known(Hs, Tp, seed, nc=60, U=3.0, W=20.0, lead_max=20.0, lead_step=1.0):
    """d3_compare_gifs.scene_sim's `hos` branch, but initialized from KNOWN
    components so the true free-wave spectrum is available for the oracle."""
    leads = np.arange(0.0, lead_max + 1e-9, lead_step)
    kp = (2 * np.pi / Tp) ** 2 / G
    rng = np.random.default_rng(400 + seed)
    comps = cg.jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH,
                                  direction_spread_deg=25.0, seed=9000 + seed)
    Ctrue = np.array([[c.kx, c.ky, c.A] for c in comps])

    Lx, Ly, Nx = 500.0, 400.0, 128
    sim = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=128, depth=DEPTH, M=3))
    sim.initialize_from_components(comps)
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
        while sim.t < t_s - 1e-9:
            sim.step(DT_HOS)
        sp = ship_pos(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, M_PER_SCAN)
        xs = sp[0] - rr * np.cos(aa)
        ys = sp[1] + rr * np.sin(aa)
        zz = eta_at(xs, ys, sim.t)
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

    maps_gt = []
    for l_ in leads:
        t_ = t0 + l_
        while sim.t < t_ - 1e-9:
            sim.step(DT_HOS)
        maps_gt.append(eta_at(Xq.ravel(), Yq.ravel(), sim.t).reshape(Xq.shape))
    return dict(P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq, gxq=gxq, gyq=gyq,
                depth=DEPTH, Hs=Hs, kp=kp, seed=seed,
                maps_gt=np.array(maps_gt), Ctrue=Ctrue)


def lead_nrmse(fq, sc):
    npx, nl = sc['npx'], len(sc['leads'])
    maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    gt = sc['maps_gt']
    sg = float(np.std(gt[np.isfinite(gt)]))
    per = []
    for li in range(nl):
        m = np.isfinite(gt[li]) & np.isfinite(maps[li])
        if m.sum() >= 8:
            per.append(np.sqrt(np.mean((maps[li][m] - gt[li][m]) ** 2)) / sg)
    return float(np.mean(per)), float(per[0])


def run_seed(Hs, Tp, seed, nc, net):
    sc = hos_scene_known(Hs, Tp, seed, nc=nc)
    s0 = 0.05 * sc['Hs']
    P, z, Pq = sc['P'], sc['z'], sc['Pq']
    fq = {
        'hand': mmse_bank_estimate(P, z, sc['depth'], Pq, sigma0=s0,
                                   kp_hint=sc['kp'], seed=seed),
        'net': net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0,
                              seed=seed, kp_hint=sc['kp']),
        'oracle': net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0,
                                 seed=seed, kp_hint=sc['kp'],
                                 oracle_components=sc['Ctrue']),
    }
    return {k: lead_nrmse(v, sc) for k, v in fq.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--Hs', type=float, default=2.0)
    ap.add_argument('--Tp', type=float, default=8.0)
    ap.add_argument('--nc', type=int, default=60)
    a = ap.parse_args()
    net = get_net(NET_CKPT)

    rows = []
    for r in range(a.reps):
        t0 = time.time()
        res = run_seed(a.Hs, a.Tp, r, a.nc, net)
        rows.append(res)
        print(f"[{r:2d}/{a.reps}] hand {res['hand'][0]:.3f} net {res['net'][0]:.3f} "
              f"oracle {res['oracle'][0]:.3f} | {time.time()-t0:.0f}s", flush=True)

    def arr(k): return np.array([row[k][0] for row in rows])  # lead-avg
    hand, net_, orac = arr('hand'), arr('net'), arr('oracle')
    out = {'Hs': a.Hs, 'Tp': a.Tp, 'nc': a.nc, 'reps': a.reps,
           'means_lead': {'hand': float(hand.mean()), 'net': float(net_.mean()),
                          'oracle': float(orac.mean())},
           'per_seed_lead': {'hand': hand.tolist(), 'net': net_.tolist(),
                             'oracle': orac.tolist()}}
    # paired deltas (+ = first better/lower)
    out['net_minus_hand'] = float((hand - net_).mean())   # + = net better than hand
    out['oracle_minus_hand'] = float((hand - orac).mean())  # + = oracle better than hand
    out['net_minus_oracle'] = float((orac - net_).mean())   # + = net STILL better than oracle
    out['net_beats_oracle_seeds'] = int(np.sum(net_ < orac))
    with open('tmp/G1_waveprior_power/mechanism.json', 'w') as f:
        json.dump(out, f, indent=1)
    print("\n-> tmp/G1_waveprior_power/mechanism.json")
    print("\n=== MECHANISM VERDICT (lead-avg nRMSE, lower better) ===")
    print(f"  hand   {hand.mean():.3f}")
    print(f"  net    {net_.mean():.3f}   (net beats hand by {(hand-net_).mean():+.3f})")
    print(f"  oracle {orac.mean():.3f}   (oracle beats hand by {(hand-orac).mean():+.3f})")
    print(f"  net vs oracle: net better by {(orac-net_).mean():+.3f}  "
          f"(net<oracle on {int(np.sum(net_<orac))}/{a.reps} seeds)")
    if (orac - net_).mean() <= 0.005:
        print("  ⟹ oracle catches/beats net ⟹ mechanism = free-wave SPECTRAL SHAPE")
    else:
        print("  ⟹ net STILL beats oracle ⟹ edge is NOT just spectral shape (mechanism OPEN)")


if __name__ == '__main__':
    main()
