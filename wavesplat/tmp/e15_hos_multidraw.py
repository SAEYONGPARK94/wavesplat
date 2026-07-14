"""E-15 — the HOS stage, done properly: many seas AND many observation draws.

What E-12e -> E-14c established:
  * the frozen HOS leg rested on TWO seas (Tp, direction, spread all constant;
    only Hs differed), so cluster-robust inference was not even defined;
  * each cell drew its 600 observations exactly ONCE, and re-drawing them moves
    delta by sd ~0.05 -- more than half the claimed +0.091;
  * the frozen run's draw for the one cell we could test sat at the 95th
    percentile of its own draw distribution (+0.068 vs a true mean of +0.007);
  * E-13 (15 seas, 1 draw each) puts the effect at +0.050, E-14 (4 seas, 6 draws)
    at +0.040.

So the HOS effect is real but ~2x smaller than published, and the steepness
mechanism (G2) could not be tested because the per-sea noise (0.049) is as large
as the true sea-to-sea variation (0.063).

THE FIX -- and it is nearly free. In d3_h7_hos_arena.scan_hits the expensive part
(grid eval -> mesh -> RAYCAST -> slope) is DETERMINISTIC in the surface snapshot;
the rng is used for exactly one line, the return-power lottery

    pw = A_of(s, r) * rng.exponential(1.0, len(idx));  keep = pw > T_MANUAL_50

So we raycast each sea ONCE, cache (hit positions, range, amplitude), and then a
new observation draw costs an exponential lottery plus the two estimator calls.
Extra draws become ~free instead of ~6 min each.

CORRECTNESS GATE (runs first, aborts on failure):
  Re-implementing the draw by hand is only legitimate if it consumes the rng in
  EXACTLY the order d3_a1_field_hos.run_cell does. So we replay the frozen cell
  (Tp 8, Hs 1.2, seed 0) with np.random.default_rng(7) through the CACHED path
  and require it to return the frozen value +0.0677 to 1e-9. If the cached path
  is bit-identical there, it is bit-identical everywhere.

--- PRE-REGISTERED GATES ---------------------------------------------------
G1  (re-confirm) pooled delta > 0 with sea-mean t(k-1) AND Webb wild cluster
    bootstrap-t both < 0.05, clustering on the SEA. Expected to pass (E-13 did).
    Reported at the DRAW-AVERAGED per-sea delta, so the per-sea noise is now
    sd/sqrt(D) instead of sd.
G2  (the real target) does the learning gain grow with steepness Hs/lam_p?
    Spearman rho over the k seas, on draw-averaged per-sea deltas.
    PASS -> the thesis mechanism ("learning wins where the linear physics prior
            is most wrong") is confirmed and becomes a quantitative law.
    FAIL -> with the noise now removed, a flat/absent dose-response means the HOS
            win does NOT come from nonlinearity. The mechanism story must change,
            and the paper must say what the gain does track instead.
    Either way, report the residual: what DOES order the seas?
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

os.environ.setdefault('A1_N_OBS', '600')

import torch                                                    # noqa: E402
import d3_f1_hos_door1 as F1                                    # noqa: E402
import d3_a1_field_hos as A1                                    # noqa: E402
from d3_f1_hos_door1 import hos_run, sea_components, DT         # noqa: E402
from d3_h7_hos_arena import (surface_snap, z_at, SCAN_REL,      # noqa: E402
                             SPINUP, NOISE, _GX, _GY, _O, _D)
from d3_h7_hos_arena import slope_at                            # noqa: E402
from d3_g0b_selection_aware import A_of                         # noqa: E402
from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H            # noqa: E402
from wavesim.lidar_simulator import build_mesh, raycast_surface  # noqa: E402
from wavesim.raydrop_threshold import T_MANUAL_50               # noqa: E402
from d3_gpu_mmse import mmse_gpu                                # noqa: E402
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE   # noqa: E402
from d3_l1_s2head_train import SmultHead                        # noqa: E402
from d3_cluster_se import cluster_se                            # noqa: E402

G = 9.81
OUT = 'results/direction3/e15_hos_multidraw.json'
CKPT = 'results/direction3/prior_net_l1s2bw.pt'
DEPTH = 50.0
LEADS = A1.LEADS
N_OBS = A1.N_OBS

def _lam(Tp, depth=DEPTH):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(80):
        k = om ** 2 / (G * np.tanh(k * depth))
    return float(2 * np.pi / k)


# The sea grid. E-13 used 15 seas at 1 draw each; the draw noise (sd ~0.05) then
# swamped the true sea-to-sea variation (sd ~0.063) and G2 could not be tested.
# Draws are now nearly free (the raycast is cached), so we can afford BOTH more
# seas (for G2's power over the steepness axis) and many draws per sea (to kill
# the per-sea noise). Steepness is capped at 0.045 to stay inside HOS M=3's
# validity, and floored at 0.006 so the low end is not all-linear.
# The two FROZEN seas (Tp 8, Hs 1.2 / 3.0) are in the grid as a bridge.
def _build_grid(steep_lo=0.006, steep_hi=0.045):
    out = []
    for Tp in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0):
        lam = _lam(Tp)
        for Hs in (1.2, 2.0, 3.0, 4.0, 5.0):
            st = Hs / lam
            if steep_lo <= st <= steep_hi:
                out.append((Tp, Hs))
    return out


SEAS = _build_grid()


def lam_p(Tp, depth=DEPTH):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(80):
        k = om ** 2 / (G * np.tanh(k * depth))
    return float(2 * np.pi / k)


def build_sea(Tp, Hs, seed):
    """Run the HOS sim ONCE and cache everything that does not depend on rng."""
    F1.TP = float(Tp)
    sim = hos_run(seed, 3, Hs=float(Hs))
    while sim.t < SPINUP - 1e-9:
        sim.step(min(DT, SPINUP - sim.t))
    t_end = SPINUP + 20.0

    scans = []                       # per SCAN_REL: the deterministic raycast
    for trel in SCAN_REL:
        tgt = t_end + trel
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        snap = surface_snap(sim)
        # --- verbatim scan_hits, minus the one rng line -----------------------
        Z = z_at(snap, _GX.ravel(), _GY.ravel()).reshape(_GX.shape)
        V, T = build_mesh(_GX, _GY, Z)
        hm, hp, _ = raycast_surface(_O, _D, V, T, range_min=R_MIN,
                                    range_max=R_MAX, batch_size=8000)
        idx = np.where(hm)[0]
        r = np.hypot(hp[idx, 0], hp[idx, 1])
        ok = (r > R_MIN) & (r < R_MAX)
        idx, r = idx[ok], r[ok]
        s = slope_at(snap, hp[idx, 0], hp[idx, 1])
        scans.append(dict(trel=float(trel), amp=A_of(s, r), r=r,
                          xy=hp[idx, :2].copy(), z=hp[idx, 2].copy()))

    leads = []                       # cached lead snapshots for the truth
    for l_ in LEADS:
        tgt = t_end + l_
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        leads.append(surface_snap(sim))

    S = sea_components(seed, Hs=float(Hs))
    kp = float(np.hypot(S['kx'], S['ky'])[np.argmax(S['A'])])
    return dict(scans=scans, leads=leads, kp=kp)


def one_draw(sea, rng, net, head):
    """A fresh 600-point observation draw + both arms. rng consumption order is
    byte-for-byte d3_a1_field_hos.run_cell (see the correctness gate)."""
    P, zv, rr = [], [], []
    for sc in sea['scans']:                        # 1) the return-power lottery
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep],
                                  np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep])
        rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))        # 2) measurement noise

    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)   # 3) range-stratified 600
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zs = P[sel], zv[sel]

    rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0  # 4) queries
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy))
        Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    Q, gt = np.vstack(Q), np.concatenate(gt)

    kp = sea['kp']
    with torch.no_grad():
        ph = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32),
                      DEPTH,
                      torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      NOISE, kp_hint=kp).cpu().numpy()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head))
    hand = float(np.corrcoef(ph, gt)[0, 1])
    learned = float(np.corrcoef(pn, gt)[0, 1])
    return hand, learned, int(len(zs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=8)
    ap.add_argument('--seas', type=int, default=len(SEAS))
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    # ---------------- correctness gate --------------------------------------
    print('correctness gate: replay the frozen cell (Tp 8, Hs 1.2, seed 0) with '
          'rng(7) through the CACHED path...', flush=True)
    t0 = time.time()
    sea0 = build_sea(8.0, 1.2, 0)
    # NOTE: net_gamma_mmse gets seed=0 here and run_cell passes seed=<cell seed>,
    # which is also 0 for this cell -- so the replay is exact.
    h, n_, nobs = one_draw(sea0, np.random.default_rng(7), net, head)
    d = n_ - h
    fz = [x for x in json.load(
        open('results/direction3/a1_field_hos_sparse600.json'))
        if x['seed'] == 0 and abs(x['Hs'] - 1.2) < 1e-9][0]
    fd = fz['learned'] - fz['hand']
    print(f"  cached path : hand {h:+.4f}  net {n_:+.4f}  d {d:+.4f}  "
          f"n_obs {nobs}  [{time.time()-t0:.0f}s]")
    print(f"  frozen JSON : hand {fz['hand']:+.4f}  net {fz['learned']:+.4f}  "
          f"d {fd:+.4f}  n_obs {fz['n_obs']}")
    if abs(d - fd) > 1e-9:
        print(f"\n  *** GATE FAILED: |{d:.6f} - {fd:.6f}| = {abs(d-fd):.2e} "
              f"> 1e-9. The cached draw does NOT reproduce run_cell. Aborting.")
        return 1
    print('  *** GATE PASSED — the cached path is bit-identical to run_cell.\n')

    # ---------------- the grid ----------------------------------------------
    rows = []
    for si, (Tp, Hs) in enumerate(SEAS[:a.seas]):
        t0 = time.time()
        sea = build_sea(Tp, Hs, 0) if not (Tp == 8.0 and Hs == 1.2) else sea0
        t_build = time.time() - t0
        lam = lam_p(Tp)
        ds = []
        for r in range(a.draws):
            rng = np.random.default_rng(900_000 + 7919 * si + 31 * r)
            h, n_, nobs = one_draw(sea, rng, net, head)
            ds.append(n_ - h)
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=float(Tp),
                             Hs=float(Hs), lam_p=lam,
                             steepness=float(Hs) / lam, draw=r,
                             hand=h, learned=n_, delta=n_ - h, n_obs=nobs))
        ds = np.array(ds)
        print(f"[Tp {Tp:4.1f} Hs {Hs:3.1f}] steep {Hs/lam:.4f}  "
              f"delta {ds.mean():+.4f} +- {ds.std(ddof=1)/np.sqrt(len(ds)):.4f} "
              f"(sd {ds.std(ddof=1):.4f}, {a.draws} draws)  "
              f"sign {(ds>0).sum()}/{len(ds)}  "
              f"[build {t_build:.0f}s + draws {time.time()-t0-t_build:.0f}s]",
              flush=True)
        json.dump(rows, open(a.out, 'w'), indent=1)

    # ---------------- inference ---------------------------------------------
    seas = sorted({r['sea'] for r in rows})
    k = len(seas)
    sea_mean = np.array([np.mean([r['delta'] for r in rows if r['sea'] == s])
                         for s in seas])
    sea_steep = np.array([[r['steepness'] for r in rows
                           if r['sea'] == s][0] for s in seas])
    within_sd = np.array([np.std([r['delta'] for r in rows if r['sea'] == s],
                                 ddof=1) for s in seas])

    d_all = np.array([r['delta'] for r in rows])
    g_all = np.array([r['sea'] for r in rows])

    def cr_t(y):
        mu_ = y.mean()
        acc = sum((y[g_all == s].sum() - (g_all == s).sum() * mu_) ** 2
                  for s in seas)
        n_ = len(y)
        c = (k / (k - 1)) * ((n_ - 1) / (n_ - k)) if (k > 1 and n_ > k) else 1.0
        se_ = np.sqrt(c * acc / n_ ** 2)
        return mu_ / se_ if se_ > 0 else np.inf

    rng = np.random.default_rng(0)
    WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5),
                     np.sqrt(.5), 1., np.sqrt(1.5)])
    idx = {s: i for i, s in enumerate(seas)}
    ci = np.array([idx[s] for s in g_all])
    t_obs = cr_t(d_all)
    wb = np.array([cr_t(d_all * rng.choice(WEBB, k)[ci]) for _ in range(10000)])
    p_wild = float((np.abs(wb) >= abs(t_obs)).mean())
    t_sea, p_sea = stats.ttest_1samp(sea_mean, 0.0)

    print(f"\n{'='*80}\nE-15  {k} seas x {a.draws} draws = {len(rows)} cells"
          f"\n{'='*80}")
    print(f"  pooled delta            {d_all.mean():+.4f}")
    print(f"  sea sign                {(sea_mean>0).sum()}/{k}")
    print(f"  sea-mean t({k-1})         {t_sea:+.2f}   p {p_sea:.4f}")
    print(f"  wild boot-t (Webb)      t {t_obs:+.2f}   p {p_wild:.4f}")
    g1 = bool(d_all.mean() > 0 and p_sea < 0.05 and p_wild < 0.05)
    print(f"  *** G1: {'PASS' if g1 else 'FAIL'}")
    print(f"\n  within-sea draw sd (mean over seas)  {within_sd.mean():.4f}")
    print(f"  between-sea sd of draw-averaged Δ    {sea_mean.std(ddof=1):.4f}")
    print(f"  SE of each sea's mean (sd/sqrt(D))   "
          f"{(within_sd/np.sqrt(a.draws)).mean():.4f}")

    rho, p_rho = stats.spearmanr(sea_steep, sea_mean)
    pr, p_pr = stats.pearsonr(sea_steep, sea_mean)
    sl = np.polyfit(sea_steep, sea_mean, 1)[0]
    g2 = bool(p_rho < 0.05 and rho > 0)
    print(f"\n  G2 — steepness dose-response (draw-averaged, {k} seas)")
    print(f"     Spearman rho {rho:+.3f} (p {p_rho:.4f})   "
          f"Pearson r {pr:+.3f} (p {p_pr:.4f})   slope {sl:+.2f}")
    print(f"  *** G2: {'PASS — the mechanism is confirmed' if g2 else 'FAIL — the gain does NOT track steepness'}")

    print(f"\n{'sea':<14}{'steep':>8}{'delta':>9}{'SE':>8}{'sign':>7}")
    order = np.argsort(sea_steep)
    for i in order:
        n_pos = sum(1 for r in rows
                    if r['sea'] == seas[i] and r['delta'] > 0)
        print(f"{seas[i]:<14}{sea_steep[i]:8.4f}{sea_mean[i]:+9.4f}"
              f"{within_sd[i]/np.sqrt(a.draws):8.4f}"
              f"{f'{n_pos}/{a.draws}':>7}")

    # what DOES order the seas, if not steepness?
    print('\n  residual diagnostics — what else tracks the gain?')
    for nm, v in (('Hs', np.array([[r['Hs'] for r in rows
                                    if r['sea'] == s][0] for s in seas])),
                  ('Tp', np.array([[r['Tp'] for r in rows
                                    if r['sea'] == s][0] for s in seas])),
                  ('lam_p', np.array([[r['lam_p'] for r in rows
                                       if r['sea'] == s][0] for s in seas]))):
        rr_, pp_ = stats.spearmanr(v, sea_mean)
        print(f"     {nm:<8} Spearman rho {rr_:+.3f}  p {pp_:.4f}")

    json.dump({'rows': rows,
               'summary': dict(
                   k_seas=k, draws=a.draws, n_cells=len(rows),
                   delta=float(d_all.mean()),
                   sea_means={s: float(m) for s, m in zip(seas, sea_mean)},
                   t_sea=float(t_sea), p_sea=float(p_sea),
                   t_wild=float(t_obs), p_wild=p_wild, G1_pass=g1,
                   spearman_rho=float(rho), spearman_p=float(p_rho),
                   pearson_r=float(pr), pearson_p=float(p_pr),
                   slope=float(sl), G2_pass=g2,
                   within_sea_sd=float(within_sd.mean()),
                   between_sea_sd=float(sea_mean.std(ddof=1)))},
              open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
