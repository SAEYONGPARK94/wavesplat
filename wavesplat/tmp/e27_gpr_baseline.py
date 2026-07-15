"""E-27 (closing #B, second baseline) — the aperture law vs a SECOND well-specified
classical: a dispersion-consistent Gaussian Process (GPR), a different prior class.

WHY. E-16..E-21 showed the aperture law (learned gain Δlin = net − classical grows
as the aperture shrinks, with a short-λ sign-FLIP where the field is well-determined)
for classical = hand-MMSE (linear bank + kp + spectrum γ prior). E-23 showed it
generalises across TWO learned priors, but the second classical there (ridge) is only
weakly regularised, and E-22 showed it does NOT hold vs RELAX — because RELAX/ridge are
PRIOR-LESS / near-optimal nowhere, so the net beats them EVERYWHERE (no flip is even
possible). The examiner's residual objection: "you only ever had ONE genuinely
well-specified classical (hand-MMSE); the flip may be specific to its exact form."

To close it we need a SECOND, genuinely DIFFERENT, well-specified prior-based classical.
This is it: a dispersion-consistent GP.

  Prior class          : hand  = sparse ARD / SBL-EM over a dense kp-band bank
                         GPR   = a Gaussian random-wave field with a dispersion-
                                 consistent space-time covariance kernel
                                   K(Δx,Δt) = Σ_n ½ A_n² cos(k_n·Δx − ω_n Δt)
                                 over an ESTIMATED component set (RELAX greedy search),
                                 GP posterior mean with a measurement-noise floor σ.
These are genuinely different objects (dense ARD weight-space regression vs a fixed
function-space Gaussian random-wave covariance, and a different component estimator).
Both are correct-physics: every atom obeys the dispersion shell, so on the LINEAR field
a well-behaved learned prior should NOT beat GPR where the field is well-determined.

The GP FORECASTS: the kernel carries the −ω_n Δt phase, so querying at future leads
(0/2/5/8 s) propagates the field forward through dispersion — exactly like hand/net's
design matrix. The corr metric pools all four leads, so a nowcast-only estimator would
collapse (3/4 of the query points are true forecasts).

  python3 tmp/e27_gpr_baseline.py                 # full run (do NOT launch here)
  python3 tmp/e27_gpr_baseline.py --selftest      # <2 s pure-numpy forecast logic check
  python3 tmp/e27_gpr_baseline.py --smoke         # 2 seas × 1 phase (still builds seas)

--- PRE-REGISTERED PASS (sea-level, on the LINEAR field) --------------------------
Let Δlin(sea) = mean_corr(net) − mean_corr(GPR) over draws, per sea; aperture_ratio
ar = R_APERTURE / λ (HIGH ar = short λ = well-determined).
  P1  aperture law : Spearman(ar, Δlin_netGPR) < 0 with p < 0.10  (gain grows as
                     aperture shrinks — same sign as the net−hand result).
  P2  the FLIP     : short-λ (ar ≥ median) mean Δlin_netGPR < 0 < long-λ mean Δlin.
  W1  WELL-SPECIFIED GATE (the crux): at short λ, mean_corr(GPR) ≥ mean_corr(hand) − τ
                     with τ = 0.05, i.e. GPR is genuinely near-optimal where the field
                     is well-determined. If W1 FAILS, GPR is NOT well-specified (it
                     behaves like ridge/RELAX) and P1/P2 are null-by-construction —
                     report honestly that the flip needs a genuinely near-optimal prior
                     and GPR-as-built is not one.
  OVERALL PASS = P1 AND P2 AND W1.
A net−hand reference (same draws, same grid) is reported as a positive control that
the grid reproduces the known aperture law.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from e18_decorrelated_2factor import build_cell, _lam, R_APERTURE  # noqa: E402

OUT = 'results/direction3/e27_gpr_baseline.json'
TPS = [6.0, 7.0, 8.0, 9.0, 11.0, 13.0]
STEEP = 0.025
HS_CAP = 5.0                       # cap longest-λ Hs so M3 does not blow up (NaN corner)
TAU = 0.05                         # W1 well-specified tolerance (corr units)
G = 9.81


# ----------------------------------------------------------------------------- #
# GP kernel + posterior mean (copied verbatim from scripts/spacetime_gp_30ep.py  #
# build_K_spacetime / gp_mean, inlined so this file needs no EpisodeDataLoader).  #
# ----------------------------------------------------------------------------- #
def _build_K_spacetime(P1, P2, A, om, kx, ky):
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    dt = P1[:, 2:3] - P2[None, :, 2]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        K += 0.5 * A[n] ** 2 * np.cos(kx[n] * dx + ky[n] * dy - om[n] * dt)
    return K


def _gp_mean(K_oo, K_po, z, sigma):
    N = len(z)
    Amat = K_oo + (sigma ** 2 + 1e-6) * np.eye(N)
    try:
        alpha = np.linalg.solve(Amat, z)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(Amat, z, rcond=None)[0]
    return K_po @ alpha


def gpr_at_Q(P, z, Q, depth, seed=0, noise=0.05):
    """Dispersion-consistent GP reconstruction at arbitrary query points Q
    (forecasts via the −ω Δt phase in the space-time kernel).

    Component set: RELAX greedy search (estimate_klim + relax_cv) — a genuinely
    different estimator from hand's dense SBL-EM bank. Amplitudes: LS fit. The
    GP prior is the Gaussian random-wave covariance Σ_n ½A_n² cos(k·Δx − ωΔt)
    with a measurement-noise floor σ = noise (the Bayesian shrinkage that makes
    this prior-based, unlike raw RELAX-LS in E-22).
    """
    from estimator_improved import estimate_klim, relax_cv
    from d18_relax_estimator import ls_fit, beamform_top, omega
    klim = estimate_klim(P, z, depth)
    kx, ky = relax_cv(P, z, depth, klim, seed=seed)
    if len(kx) == 0:
        bx, by = beamform_top(P, z, depth, klim=klim)
        kx, ky = np.array([bx]), np.array([by])
    kx, ky = np.asarray(kx, float), np.asarray(ky, float)
    _, _, amp = ls_fit(kx, ky, P, z, depth)             # per-component amplitude
    k = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = omega(k, depth)
    K_oo = _build_K_spacetime(P, P, amp, om, kx, ky)
    K_qo = _build_K_spacetime(Q, P, amp, om, kx, ky)
    return _gp_mean(K_oo, K_qo, z, sigma=noise)


def draw3(sea, rng, net, head):
    """One 600-point observation draw, three arms (hand-MMSE, net, GPR) at the
    arena queries (4 forecast leads). Byte-identical observation pipeline to
    E-15/E-22 one_draw; only the third arm changes (RELAX-LS -> GPR)."""
    import torch
    from e15_hos_multidraw import N_OBS, LEADS
    from d3_h7_hos_arena import z_at
    from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H
    from wavesim.raydrop_threshold import T_MANUAL_50
    from d3_gpu_mmse import mmse_gpu
    from d3_prior_net_eval import net_gamma_mmse, DEVICE
    NOISE, DEPTH = 0.05, 50.0
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep], np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zs = P[sel], zv[sel]
    rf = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, 800)) * R_MAX
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
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32), DEPTH,
                      torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      NOISE, kp_hint=kp).cpu().numpy()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head))
    try:
        pg = gpr_at_Q(Ps, zs, Q, DEPTH, seed=0, noise=NOISE)
    except Exception:                                      # noqa: BLE001
        pg = np.full(len(gt), np.nan)

    def c(p):
        return float(np.corrcoef(p, gt)[0, 1]) if np.std(p) > 1e-9 else np.nan
    return c(ph), c(pn), c(pg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])   # >=3 phases
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--workers', type=int, default=11)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    tps = TPS[:2] if a.smoke else TPS
    seeds = a.seeds[:1] if a.smoke else a.seeds
    seas = [(Tp, min(round(STEEP * _lam(Tp), 3), HS_CAP)) for Tp in tps]
    tasks = [(Tp, Hs, sd, order) for (Tp, Hs) in seas
             for sd in seeds for order in (1, 3)]
    print(f"E-27 GPR baseline: {len(seas)} seas × {len(seeds)} phases ×2 orders = "
          f"{len(tasks)} cells (hand/net/GPR); Hs capped at {HS_CAP} m", flush=True)
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
            if sea is None:
                print(f"  !! build {task} -> {note}", flush=True)
    print(f"  built {sum(v is not None for v in built.values())}/{len(tasks)} in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    import torch
    from e15_hos_multidraw import CKPT
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    net = get_net(CKPT); head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head']); head.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp); ar = R_APERTURE / lam
        for sd in seeds:
            sl, s3 = built[(Tp, Hs, sd, 1)], built[(Tp, Hs, sd, 3)]
            if sl is None:
                continue
            H1, N1, G1, H3, N3, G3 = ([] for _ in range(6))
            for rr in range(a.draws):
                rng = np.random.default_rng(830000 + 31 * rr)
                h, n_, g = draw3(sl, rng, net, head)
                H1.append(h); N1.append(n_); G1.append(g)
                if s3 is not None:
                    rng = np.random.default_rng(830000 + 31 * rr)
                    h, n_, g = draw3(s3, rng, net, head)
                    H3.append(h); N3.append(n_); G3.append(g)
            m = lambda v: float(np.nanmean(v)) if len(v) else float('nan')
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, lam=lam,
                             aperture_ratio=ar, seed=sd,
                             hand_lin=m(H1), net_lin=m(N1), gpr_lin=m(G1),
                             hand_m3=m(H3), net_m3=m(N3), gpr_m3=m(G3),
                             gng_lin=m(N1) - m(G1), gnh_lin=m(N1) - m(H1),
                             gng_m3=m(N3) - m(G3), gnh_m3=m(N3) - m(H3)))
            print(f"[Tp{Tp:g} λ{lam:.0f} ar{ar:.2f} ph{sd}] LIN corr hand {m(H1):+.2f} "
                  f"net {m(N1):+.2f} GPR {m(G1):+.2f} | net-GPR {m(N1)-m(G1):+.3f} "
                  f"net-hand {m(N1)-m(H1):+.3f}", flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    _analyse(rows, a.out)
    return 0


def _analyse(rows, out):
    fr = [r for r in rows if np.isfinite(r['gng_lin']) and np.isfinite(r['gnh_lin'])
          and np.isfinite(r['gpr_lin']) and np.isfinite(r['hand_lin'])]
    seas_u = sorted({r['sea'] for r in fr})
    if len(seas_u) < 4:
        print(f"\ntoo few finite seas ({len(seas_u)}) for the aperture-law test")
        return
    sm = lambda k, s: np.nanmean([r[k] for r in fr if r['sea'] == s])
    ar = np.array([[r['aperture_ratio'] for r in fr if r['sea'] == s][0] for s in seas_u])
    gng_lin = np.array([sm('gng_lin', s) for s in seas_u])
    gnh_lin = np.array([sm('gnh_lin', s) for s in seas_u])
    gng_m3 = np.array([sm('gng_m3', s) for s in seas_u])
    hand_lin = np.array([sm('hand_lin', s) for s in seas_u])
    gpr_lin = np.array([sm('gpr_lin', s) for s in seas_u])
    hi = ar >= np.median(ar)                              # short λ = well-determined

    r_ng, p_ng = stats.spearmanr(ar, gng_lin)
    r_nh, p_nh = stats.spearmanr(ar, gnh_lin)             # reference control
    flip = bool(gng_lin[hi].mean() < 0 < gng_lin[~hi].mean())
    # W1: is GPR genuinely near-optimal where the field is well-determined?
    gpr_short, hand_short = float(gpr_lin[hi].mean()), float(hand_lin[hi].mean())
    w1 = bool(gpr_short >= hand_short - TAU)
    p1 = bool(r_ng < 0 and p_ng < 0.10)
    overall = bool(p1 and flip and w1)

    print(f"\n{'='*74}\nE-27 GPR baseline ({len(seas_u)} seas, sea-level, LINEAR field)\n{'='*74}")
    print(f"  {'sea':<14}{'ar':>6}{'hand':>7}{'gpr':>7}{'net':>7}"
          f"{'net-GPR':>9}{'net-hand':>10}")
    net_lin = np.array([sm('net_lin', s) for s in seas_u])
    for i in np.argsort(-ar):
        print(f"  {seas_u[i]:<14}{ar[i]:6.2f}{hand_lin[i]:+7.2f}{gpr_lin[i]:+7.2f}"
              f"{net_lin[i]:+7.2f}{gng_lin[i]:+9.3f}{gnh_lin[i]:+10.3f}")
    print(f"\n  P1 aperture law : Spearman(ar, net−GPR on LINEAR) {r_ng:+.3f} p {p_ng:.4f}"
          f"  -> {'PASS' if p1 else 'fail'}")
    print(f"     (reference)    Spearman(ar, net−hand on LINEAR) {r_nh:+.3f} p {p_nh:.4f}")
    print(f"  P2 flip         : short-λ net−GPR {gng_lin[hi].mean():+.3f} < 0 < "
          f"long-λ {gng_lin[~hi].mean():+.3f}  -> {'PASS' if flip else 'fail'}")
    print(f"  W1 well-spec    : short-λ corr GPR {gpr_short:+.3f} vs hand {hand_short:+.3f} "
          f"(τ={TAU})  -> {'PASS — GPR near-optimal' if w1 else 'FAIL — GPR not well-specified'}")
    print(f"  net−GPR on M3   : {np.nanmean(gng_m3):+.3f} (learned still beats GPR on nonlinear)")
    print(f"\n  *** OVERALL PASS (P1 & P2 & W1): {overall}")
    if not w1:
        print("      -> W1 failed: GPR behaves like ridge/RELAX (net beats it even where "
              "well-determined). The flip requires a genuinely near-optimal prior; report "
              "that GPR-as-built does not close the objection (need a better-specified 2nd "
              "classical, e.g. denser component set / ML amplitudes).")
    json.dump({'rows': rows, 'summary': dict(
        n_seas=len(seas_u),
        spearman_ap_net_gpr_lin=[float(r_ng), float(p_ng)],
        spearman_ap_net_hand_lin=[float(r_nh), float(p_nh)],
        flip=flip, gng_short=float(gng_lin[hi].mean()), gng_long=float(gng_lin[~hi].mean()),
        w1_well_specified=w1, gpr_short_corr=gpr_short, hand_short_corr=hand_short,
        gng_m3_mean=float(np.nanmean(gng_m3)), P1=p1, overall_pass=overall)},
        open(out, 'w'), indent=2)
    print(f"\n-> {out}")


# --------------------------------------------------------------------------- #
# Pure-numpy logic check (<2 s, no seas, no torch, no GPU): does the           #
# dispersion GP FORECAST? Build a known 2-wave field, sparse obs at t∈[-20,0], #
# query at leads {0,2,5,8}, and require high corr INCLUDING at forecast leads. #
# --------------------------------------------------------------------------- #
def selftest():
    from d18_relax_estimator import omega
    rng = np.random.default_rng(0)
    depth = 50.0
    # two dispersion-consistent plane waves
    kx = np.array([0.06, -0.02]); ky = np.array([0.01, 0.05])
    k = np.sqrt(kx ** 2 + ky ** 2); om = omega(k, depth)
    A = np.array([1.0, 0.6]); ph = np.array([0.3, 1.7])

    def field(x, y, t):
        s = np.zeros_like(x)
        for n in range(2):
            s += A[n] * np.cos(kx[n] * x + ky[n] * y - om[n] * t + ph[n])
        return s
    # sparse obs over the buffer window t∈[-20,0]
    ts = np.repeat(np.linspace(-20, 0, 12), 50)
    ox = rng.uniform(-200, 200, ts.size); oy = rng.uniform(-200, 200, ts.size)
    P = np.column_stack([ox, oy, ts])
    z = field(ox, oy, ts) + rng.normal(0, 0.05, ts.size)
    leads = [0.0, 2.0, 5.0, 8.0]
    print("selftest — dispersion GP forecast (2 known waves, 600 sparse obs):")
    ok = True
    for l_ in leads:
        qx = rng.uniform(-200, 200, 400); qy = rng.uniform(-200, 200, 400)
        Q = np.column_stack([qx, qy, np.full(400, l_)])
        gt = field(qx, qy, l_)
        # GP with the TRUE components (isolates the kernel/forecast math from RELAX)
        K_oo = _build_K_spacetime(P, P, A, om, kx, ky)
        K_qo = _build_K_spacetime(Q, P, A, om, kx, ky)
        mu = _gp_mean(K_oo, K_qo, z, sigma=0.05)
        cc = float(np.corrcoef(mu, gt)[0, 1])
        good = cc > 0.9
        ok &= good
        print(f"   lead {l_:4.1f}s  corr {cc:+.4f}  {'ok' if good else 'LOW'}")
    print(f"  *** forecast logic {'PASS' if ok else 'FAIL'} "
          f"(GP propagates the field forward through dispersion)")
    return 0 if ok else 1


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
