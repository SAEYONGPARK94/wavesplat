"""E-16 -- the DECISIVE experiment: physics vs aperture, operator held FIXED.

BACKGROUND (E-15, section 14.5). The stage slope is
    clean-linear   -0.046   (physics correct   -> learning loses)
    HOS            +0.072   (physics misspec'd  -> learning wins)
E-15 also found the *magnitude* of the HOS gain tracks wavelength (lam_p,
rho +0.41), i.e. aperture/wavelength ratio, NOT steepness (rho -0.21). That
opened a threat: maybe the clean-vs-HOS slope is ALSO an aperture artifact,
because the clean-linear seas cluster at short wavelengths (47/76/125 m) while
the HOS grid is longer. The matched-wavelength table in 14.5 argued no (sign
flips at matched lam), BUT the two stages used DIFFERENT observation operators
(wavesim Falcon episodes vs HOS-arena fan). So that comparison was suggestive,
not a paired intervention.

THE DECISIVE MOVE (registered as "next #1"). Run LINEAR physics through the
SAME HOS-arena operator, at the SAME wavelengths, paired sea-by-sea. The HOS
arena already supports M=1 (linear evolution, full spectral grid) vs M=3
(3rd-order nonlinear) -- same JONSWAP init (same seed -> same phases), same fan
raycast operator, same kp_hint, same 600-obs recipe. The ONLY thing that
differs between the two orders at a given (Tp, Hs) is the wave physics.

  Delta(M)  =  learned_rho - hand_rho   (the learning gain, as in E-15)

If, with the operator and wavelength held FIXED (paired seas):
  * Delta(M3) > 0   (learning wins on misspecified physics)   -- E-15 says +0.072
  * Delta(M1) <= 0  (learning loses / ties on correct physics) -- matches -0.046
  * paired dd(sea) = Delta(M3) - Delta(M1) > 0, significant
then the stage slope is driven by PHYSICS (misspecification), not aperture,
because aperture is identical between M1 and M3 at a matched wavelength.

BONUS -- the clean steepness test. E-15's G2 (does the gain track steepness?)
FAILED because wavelength confounded it. The paired difference dd(sea) removes
wavelength/aperture exactly (M1 and M3 share it), so dd-vs-steepness is the
CLEAN nonlinearity dose-response. At low steepness bound energy ~ (ka)^2 ~ 1-2%
=> M1 ~ M3 => dd ~ 0; at high steepness they diverge => dd larger, IF the
mechanism is nonlinearity.

--- PRE-REGISTERED GATES ---------------------------------------------------
G0  correctness: my build_sea(M=3) must reproduce the frozen cell (Tp8, Hs1.2,
    seed0) delta +0.0677 to 1e-9 -- proves this copy is faithful to E-15.
G1  (decisive) paired dd(sea) = Delta(M3) - Delta(M1) > 0, with sea-mean t(k-1)
    AND Webb wild cluster bootstrap-t both < 0.05, clustering on the SEA.
    PASS -> stage slope is physics, not aperture. Narrative survives cleanly.
    FAIL -> if dd ~ 0 with Delta(M1) also > 0, the "learning wins" story is an
            operator/aperture effect, not physics. Narrative must change.
G2  sign of Delta(M1): is learning losing/tied on CORRECT physics? (expect <=0,
    mirroring the -0.046 clean-linear leg). Reported with its own CI.
G3  (mechanism, exploratory) does dd(sea) grow with steepness Hs/lam_p?
    Spearman rho over seas. This is the confound-free steepness dose-response.
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

import d3_f1_hos_door1 as F1                                    # noqa: E402
from d3_f1_hos_door1 import hos_run, sea_components, DT         # noqa: E402
from d3_h7_hos_arena import (surface_snap, z_at, SCAN_REL,      # noqa: E402
                             SPINUP, _GX, _GY, _O, _D, slope_at)
from d3_g0b_selection_aware import A_of                         # noqa: E402
from d3_door1b_field_bias import R_MIN, R_MAX                   # noqa: E402
from wavesim.lidar_simulator import build_mesh, raycast_surface  # noqa: E402
from wavesim.raydrop_threshold import T_MANUAL_50               # noqa: E402
from d3_prior_net_eval import get_net, DEVICE                   # noqa: E402
from d3_l1_s2head_train import SmultHead                        # noqa: E402
# reuse E-15 verbatim: the observation draw + both arms, and the sea grid
from e15_hos_multidraw import one_draw, lam_p, SEAS, LEADS, CKPT  # noqa: E402

OUT = 'results/direction3/e16_physics_operator.json'


def build_sea_linear(Tp, Hs, seed):
    """The CORRECT-PHYSICS arm: the analytic linear free-wave field, the exact
    model the hand MMSE assumes (free waves on the dispersion shell, NO bound
    harmonics), pushed through the SAME fan operator as M=3.

    Why not hos_run(M=1)? HOSSimulator sets dealiasing_factor=ceil((M+1)/2)=1 for
    M=1 (wavesim/hos_simulator.py:60) -> NO dealiasing, yet _rhs still evaluates
    the nonlinear BC products (grad_eta^2, |grad phi|^2). So M=1 aliases and
    blows up (NaN) at Hs>=2, exactly the high-steepness seas the dd-vs-steepness
    test needs; and M=1 is not even cleanly linear. The analytic field is the
    unambiguous linear control: eta = sum A cos(k.x - w t + phi) from the same
    JONSWAP components (same seed -> matched sea), evaluated by the same
    hos_eval/z_at/slope_at path. Stable everywhere.
    """
    F1.TP = float(Tp)
    S = sea_components(seed, Hs=float(Hs))
    kx, ky = S['kx'], S['ky']
    t_end = SPINUP + 20.0

    def lin_snap(t):
        a = S['A'] * np.exp(1j * (-S['om'] * t + S['ph']))
        return (kx, ky, a)              # (kx, ky, complex amp) -> z_at/slope_at

    scans = []
    for trel in SCAN_REL:
        snap = lin_snap(t_end + trel)
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
    leads = [lin_snap(t_end + l_) for l_ in LEADS]
    kp = float(np.hypot(kx, ky)[np.argmax(S['A'])])
    return dict(scans=scans, leads=leads, kp=kp)


def build_sea_M(Tp, Hs, seed, M):
    """VERBATIM e15_hos_multidraw.build_sea, except the HOS order M is a
    parameter (e15 hard-codes M=3). Everything the rng/operator touches is
    byte-identical, so build_sea_M(...,3) == e15.build_sea(...) (G0 checks it).
    """
    F1.TP = float(Tp)
    sim = hos_run(seed, M, Hs=float(Hs))
    while sim.t < SPINUP - 1e-9:
        sim.step(min(DT, SPINUP - sim.t))
    t_end = SPINUP + 20.0

    scans = []
    for trel in SCAN_REL:
        tgt = t_end + trel
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        snap = surface_snap(sim)
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

    leads = []
    for l_ in LEADS:
        tgt = t_end + l_
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        leads.append(surface_snap(sim))

    S = sea_components(seed, Hs=float(Hs))
    kp = float(np.hypot(S['kx'], S['ky'])[np.argmax(S['A'])])
    return dict(scans=scans, leads=leads, kp=kp)


def cr_t(y, g_all, seas, k):
    mu_ = y.mean()
    acc = sum((y[g_all == s].sum() - (g_all == s).sum() * mu_) ** 2
              for s in seas)
    n_ = len(y)
    c = (k / (k - 1)) * ((n_ - 1) / (n_ - k)) if (k > 1 and n_ > k) else 1.0
    se_ = np.sqrt(c * acc / n_ ** 2)
    return mu_ / se_ if se_ > 0 else np.inf


def wild_p(y, g_all, seas, k, t_obs, nboot=10000, seed=0):
    rng = np.random.default_rng(seed)
    WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5),
                     np.sqrt(.5), 1., np.sqrt(1.5)])
    idx = {s: i for i, s in enumerate(seas)}
    ci = np.array([idx[s] for s in g_all])
    wb = np.array([cr_t(y * rng.choice(WEBB, k)[ci], g_all, seas, k)
                   for _ in range(nboot)])
    return float((np.abs(wb) >= abs(t_obs)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=8)
    ap.add_argument('--seas', type=int, default=len(SEAS))
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    import torch
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    # ---------------- G0 correctness gate -----------------------------------
    print('G0 gate: build_sea_M(M=3) must reproduce frozen (Tp8,Hs1.2,seed0)...',
          flush=True)
    t0 = time.time()
    sea0 = build_sea_M(8.0, 1.2, 0, 3)
    h, n_, nobs = one_draw(sea0, np.random.default_rng(7), net, head)
    d = n_ - h
    fz = [x for x in json.load(
        open('results/direction3/a1_field_hos_sparse600.json'))
        if x['seed'] == 0 and abs(x['Hs'] - 1.2) < 1e-9][0]
    fd = fz['learned'] - fz['hand']
    print(f"  build_sea_M(M3): hand {h:+.4f} net {n_:+.4f} d {d:+.4f} "
          f"n_obs {nobs} [{time.time()-t0:.0f}s]")
    print(f"  frozen JSON    : hand {fz['hand']:+.4f} net {fz['learned']:+.4f} "
          f"d {fd:+.4f} n_obs {fz['n_obs']}")
    if abs(d - fd) > 1e-9:
        print(f"  *** G0 FAILED: |{d:.6f}-{fd:.6f}|={abs(d-fd):.2e} > 1e-9. "
              "Copy diverges from E-15. Aborting.")
        return 1
    print('  *** G0 PASSED -- build_sea_M is bit-identical to E-15.\n')

    # ---------------- reuse E-15's M=3 (32 seas x 12 draws, committed) -------
    # The M=3 raycast builds cost ~5 min each; re-running all 32 would be ~3 h
    # of pure duplication. E-15 already measured M=3 with the SAME build_sea
    # code (G0 proves this copy reproduces its frozen anchor bit-for-bit). So we
    # reuse E-15's per-sea draw-averaged Delta(M3) and only compute M=1 fresh,
    # then VALIDATE the reuse with a 4-sea M=3 bridge (rebuilt here, must match
    # E-15's stored value within Monte-Carlo error).
    e15 = json.load(open('results/direction3/e15_hos_multidraw.json'))
    m3_stored = e15['summary']['sea_means']          # {sea: draw-avg Delta(M3)}
    # per-sea within-sea sd of M3 (for the bridge tolerance)
    m3_rows = e15['rows']
    m3_within_sd = {}
    for s in m3_stored:
        dl = [r['delta'] for r in m3_rows if r['sea'] == s]
        m3_within_sd[s] = float(np.std(dl, ddof=1)) if len(dl) > 1 else 0.1

    rows = []
    seas_list = SEAS[:a.seas]
    # bridge seas: 4 spanning the lam range (short/mid/long), rebuilt at M=3
    lams_all = [(lam_p(Tp), i) for i, (Tp, Hs) in enumerate(seas_list)]
    bridge_idx = sorted({0, len(seas_list)//3, 2*len(seas_list)//3,
                         len(seas_list)-1})

    # ---- LINEAR (analytic free-wave) fresh on all seas + M=3 bridge ----
    for si, (Tp, Hs) in enumerate(seas_list):
        t0 = time.time()
        sea = build_sea_linear(Tp, Hs, 0)
        lam = lam_p(Tp)
        ds = []
        for r in range(a.draws):
            rng = np.random.default_rng(900_000 + 7919 * si + 31 * r)
            hh, nn, nob = one_draw(sea, rng, net, head)
            ds.append(nn - hh)
            rows.append(dict(order=1, sea=f'Tp{Tp:g}_Hs{Hs:g}',
                             Tp=float(Tp), Hs=float(Hs), lam_p=lam,
                             steepness=float(Hs) / lam, draw=r,
                             hand=hh, learned=nn, delta=nn - hh, n_obs=nob))
        ds = np.array(ds)
        tag = ''
        if si in bridge_idx:
            sea3 = sea0 if (Tp == 8.0 and Hs == 1.2) else build_sea_M(Tp, Hs, 0, 3)
            d3 = np.array([one_draw(
                sea3, np.random.default_rng(900_000 + 7919 * si + 31 * r),
                net, head) for r in range(a.draws)])
            d3d = d3[:, 1] - d3[:, 0]
            key = f'Tp{Tp:g}_Hs{Hs:g}'
            se = np.sqrt(m3_within_sd[key]**2 / 12 + d3d.std(ddof=1)**2 / a.draws)
            ok = abs(d3d.mean() - m3_stored[key]) <= 3 * se
            tag = (f"  |BRIDGE M3 fresh {d3d.mean():+.4f} vs stored "
                   f"{m3_stored[key]:+.4f} d {abs(d3d.mean()-m3_stored[key]):.4f} "
                   f"3se {3*se:.4f} {'OK' if ok else '!!MISMATCH'}|")
            for r in range(a.draws):
                rows.append(dict(order=3, sea=key, Tp=float(Tp), Hs=float(Hs),
                                 lam_p=lam, steepness=float(Hs)/lam, draw=r,
                                 hand=float(d3[r, 0]), learned=float(d3[r, 1]),
                                 delta=float(d3d[r]), n_obs=int(d3[r, 2]),
                                 bridge=True))
        print(f"[LIN Tp{Tp:4.1f} Hs{Hs:3.1f}] steep {Hs/lam:.4f} lam {lam:5.1f}  "
              f"delta {ds.mean():+.4f} (sd {ds.std(ddof=1):.4f}) "
              f"sign {(ds>0).sum()}/{len(ds)} [{time.time()-t0:.0f}s]{tag}",
              flush=True)
        json.dump(rows, open(a.out, 'w'), indent=1)

    # ---------------- inference: paired physics effect ----------------------
    seas = sorted(set(m3_stored) & {f"Tp{Tp:g}_Hs{Hs:g}"
                                    for Tp, Hs in seas_list})
    k = len(seas)
    dM3 = np.array([m3_stored[s] for s in seas])
    dM1 = np.array([np.mean([r['delta'] for r in rows
                             if r['sea'] == s and r['order'] == 1])
                    for s in seas])
    dd = dM3 - dM1                      # paired physics effect, lambda removed
    steep = np.array([[r['steepness'] for r in rows if r['sea'] == s][0]
                      for s in seas])
    lamv = np.array([[r['lam_p'] for r in rows if r['sea'] == s][0]
                     for s in seas])

    # cluster-robust on the PAIRED per-draw difference is not defined (draws are
    # not paired across orders); use the k sea-level paired means -> t(k-1).
    t3, p3 = stats.ttest_1samp(dM3, 0.0)
    t1, p1 = stats.ttest_1samp(dM1, 0.0)
    tdd, pdd = stats.ttest_1samp(dd, 0.0)
    # Webb wild bootstrap at the sea level (k values), one cluster per sea
    rng = np.random.default_rng(0)
    WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5),
                     np.sqrt(.5), 1., np.sqrt(1.5)])

    def wild_sea(y):
        # Webb wild cluster bootstrap-t, NULL IMPOSED (each sea = one cluster).
        # yb = y*w has E[yb]=0 under H0, so the bootstrap t-dist is centered at
        # 0 and P(|t*|>=|t_obs|) is a valid p-value. (The earlier form
        # yb = mu+(y-mu)*w left the statistic centered at t_obs -> p~=0.5 always,
        # a false-negative bug found in adversarial review 2026-07-15.)
        mu = y.mean()
        t_o = mu / (y.std(ddof=1) / np.sqrt(len(y)))
        boot = []
        for _ in range(10000):
            w = rng.choice(WEBB, len(y))
            yb = y * w
            seb = yb.std(ddof=1) / np.sqrt(len(y))
            boot.append(yb.mean() / seb if seb > 0 else 0.0)
        boot = np.array(boot)
        return t_o, float((np.abs(boot) >= abs(t_o)).mean())

    tdd_w, pdd_w = wild_sea(dd)

    g1 = bool(dd.mean() > 0 and pdd < 0.05 and pdd_w < 0.05)

    rho_dd, prho_dd = stats.spearmanr(steep, dd)   # confound-free steepness
    rho_lam_dd, prho_lam_dd = stats.spearmanr(lamv, dd)
    rho_lam_m1, prho_lam_m1 = stats.spearmanr(lamv, dM1)
    rho_lam_m3, prho_lam_m3 = stats.spearmanr(lamv, dM3)

    print(f"\n{'='*78}\nE-16  physics vs aperture, operator FIXED  ({k} seas x "
          f"{a.draws} draws x 2 orders)\n{'='*78}")
    print(f"  Delta(M3) mean  {dM3.mean():+.4f}   t({k-1}) {t3:+.2f} p {p3:.4f}"
          f"   sign {(dM3>0).sum()}/{k}")
    print(f"  Delta(M1) mean  {dM1.mean():+.4f}   t({k-1}) {t1:+.2f} p {p1:.4f}"
          f"   sign {(dM1>0).sum()}/{k}   [G2: learning on CORRECT physics]")
    print(f"  paired dd mean  {dd.mean():+.4f}   t({k-1}) {tdd:+.2f} p {pdd:.4f}"
          f"   wildboot t {tdd_w:+.2f} p {pdd_w:.4f}   sign {(dd>0).sum()}/{k}")
    print(f"  *** G1 (decisive, dd>0): {'PASS' if g1 else 'FAIL'}")
    print(f"\n  G3 mechanism  dd vs steepness  Spearman {rho_dd:+.3f} "
          f"p {prho_dd:.4f}")
    print(f"     controls    dd  vs lam  {rho_lam_dd:+.3f} p {prho_lam_dd:.4f}")
    print(f"                 M1  vs lam  {rho_lam_m1:+.3f} p {prho_lam_m1:.4f}")
    print(f"                 M3  vs lam  {rho_lam_m3:+.3f} p {prho_lam_m3:.4f}")

    print(f"\n{'sea':<13}{'steep':>8}{'lam':>7}{'d(M3)':>9}{'d(M1)':>9}{'dd':>9}")
    order = np.argsort(steep)
    for i in order:
        print(f"{seas[i]:<13}{steep[i]:8.4f}{lamv[i]:7.1f}"
              f"{dM3[i]:+9.4f}{dM1[i]:+9.4f}{dd[i]:+9.4f}")

    json.dump({'rows': rows, 'summary': dict(
        k_seas=k, draws=a.draws,
        delta_M3=float(dM3.mean()), t_M3=float(t3), p_M3=float(p3),
        delta_M1=float(dM1.mean()), t_M1=float(t1), p_M1=float(p1),
        paired_dd=float(dd.mean()), t_dd=float(tdd), p_dd=float(pdd),
        t_dd_wild=float(tdd_w), p_dd_wild=float(pdd_w), G1_pass=g1,
        sign_M3=int((dM3 > 0).sum()), sign_M1=int((dM1 > 0).sum()),
        sign_dd=int((dd > 0).sum()),
        rho_dd_steep=float(rho_dd), p_dd_steep=float(prho_dd),
        rho_lam_dd=float(rho_lam_dd), rho_lam_M1=float(rho_lam_m1),
        rho_lam_M3=float(rho_lam_m3),
        sea_means={'M3': {s: float(v) for s, v in zip(seas, dM3)},
                   'M1': {s: float(v) for s, v in zip(seas, dM1)}})},
        open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
